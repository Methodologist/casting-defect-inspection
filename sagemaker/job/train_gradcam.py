"""Fine-tune ResNet-18 per dataset, test on held-out images, write Grad-CAM heatmaps.

Runs on a local machine (uses an NVIDIA GPU when present; see train_local_cnn.py) or inside the AWS PyTorch
training container as a SageMaker training job (see launch.py). --local-check runs a tiny smoke test.

Inputs (SageMaker channels):
  images   s3://<bucket>/images/<dataset>/{train,test}/<label>/*   the same splits Rekognition used
  heldout  s3://<bucket>/heldout/<dataset>/<label>/*               images the models never trained on

Outputs (/opt/ml/output/data, uploaded by SageMaker as output.tar.gz):
  <dataset>/predictions.json      verdict, confidence and heatmap file for every held-out image
  <dataset>/heatmaps/<image>.png  transparent overlay, same size as the original image (the original is untouched)
  <dataset>/metrics.json          held-out accuracy, confusion matrix, training history, hardware and speed
  <dataset>/model.pt              the trained weights (used by layer_maps.py for the site's network diagram)

Heatmaps show Grad-CAM evidence for the positive class (defective / damaged), scaled on one shared scale per
dataset, so a good part should look mostly clear instead of every image glowing somewhere.
"""
import argparse
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms

DATASETS = {"casting": ("defective", "ok"), "packages": ("damaged", "intact")}  # (positive, negative)
EXT = {".jpg", ".jpeg", ".png"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def files(folder):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in EXT) if Path(folder).exists() else []


def load_split(root, labels):
    return [(p, i) for i, label in enumerate(labels) for p in files(Path(root) / label)]


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


def build_model(pretrained):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def hardware():
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine() or "CPU"


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(DEVICE)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def latency_ms(model, x, device, runs=50):
    """Median time to classify one image (batch of 1), after warm-up, the way a station checks one part at a time."""
    model = model.to(device).eval()
    x = x.to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    with torch.no_grad():
        for i in range(runs + 10):
            sync()
            t = time.perf_counter()
            model(x)
            sync()
            if i >= 10:
                times.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(times), 2)


class GradCAM:
    """Grad-CAM on the last convolutional block (layer4) of ResNet-18."""

    def __init__(self, model):
        self.model, self.acts, self.grads = model, None, None
        model.layer4.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        model.layer4.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, cls):
        self.model.zero_grad()
        logits = self.model(x.to(DEVICE))
        logits[0, cls].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1))[0]
        return cam.detach().cpu().numpy(), F.softmax(logits, 1)[0].detach().cpu().numpy()


def overlay_png(cam, scale, size, out_path):
    """Transparent heat overlay: clear where there is no evidence, yellow to red where it is strong."""
    v = np.clip(cam / scale, 0, 1) if scale > 0 else np.zeros_like(cam)
    small = Image.fromarray((v * 255).astype(np.uint8))
    v = np.asarray(small.resize(size, Image.BILINEAR), dtype=np.float32) / 255
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (230 * (1 - v)).astype(np.uint8)       # yellow (weak) -> red (strong)
    rgba[..., 2] = 0
    rgba[..., 3] = (np.clip((v - 0.25) / 0.75, 0, 1) * 190).astype(np.uint8)  # weak evidence stays clear
    Image.fromarray(rgba, "RGBA").save(out_path, optimize=True)


def run_dataset(name, args, out_root):
    pos, neg = DATASETS[name]
    labels = [pos, neg]  # index 0 = positive class
    img_root, held_root = Path(args.images) / name, Path(args.heldout) / name
    train, test, held = load_split(img_root / "train", labels), load_split(img_root / "test", labels), \
        load_split(held_root, labels)
    if args.local_check:
        rng = random.Random(0)
        train, test, held = (rng.sample(s, min(len(s), 16)) for s in (train, test, held))
    print(f"[{name}] train {len(train)}, test {len(test)}, held-out {len(held)}", flush=True)
    if not (train and test and held):
        raise SystemExit(f"[{name}] missing images: train {len(train)}, test {len(test)}, held-out {len(held)}")

    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(15), transforms.ColorJitter(0.15, 0.15),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize(MEAN, STD)])
    seed_everything(42)
    gen = torch.Generator().manual_seed(42)  # same shuffle order every run
    dl = lambda items, tf, shuffle: torch.utils.data.DataLoader(
        ImageSet(items, tf), batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        generator=gen, persistent_workers=args.workers > 0, pin_memory=DEVICE.type == "cuda")
    train_dl, test_dl = dl(train, train_tf, True), dl(test, eval_tf, False)

    model = build_model(pretrained=not args.local_check).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_acc, best_state, history = -1, None, []
    train_began = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        began, seen, loss_sum = time.time(), 0, 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        sched.step()
        acc = evaluate(model, test_dl)
        history.append({"epoch": epoch, "train_loss": round(loss_sum / seen, 4), "test_accuracy": round(acc, 4)})
        print(f"[{name}] epoch {epoch}: train_loss={loss_sum / seen:.4f} test_accuracy={acc:.4f} "
              f"({time.time() - began:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc, best_epoch, best_state = acc, epoch, {k: v.clone() for k, v in model.state_dict().items()}
    train_seconds = round(time.time() - train_began, 1)
    model.load_state_dict(best_state)
    model.eval()

    sample = eval_tf(Image.open(held[0][0]).convert("RGB")).unsqueeze(0)
    speed = {"cpu": latency_ms(model, sample, torch.device("cpu"))}
    if DEVICE.type == "cuda":
        speed["gpu"] = latency_ms(model, sample, DEVICE)
    model.to(DEVICE)
    print(f"[{name}] time per image: " + ", ".join(f"{k} {v} ms" for k, v in speed.items()), flush=True)

    out = out_root / name
    (out / "heatmaps").mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model)
    raw = []
    for path, y in held:
        img = Image.open(path).convert("RGB")
        c, probs = cam(eval_tf(img).unsqueeze(0), 0)  # evidence for the positive class
        raw.append((path, y, img.size, c, probs))
    pos_peaks = [c.max() for _, y, _, c, _ in raw if y == 0]
    scale = float(np.percentile(pos_peaks, 90)) if pos_peaks else 1.0

    preds, cm = [], {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for path, y, size, c, probs in raw:
        p = int(np.argmax(probs))
        key = ("tp" if p == 0 else "fn") if y == 0 else ("fp" if p == 0 else "tn")
        cm[key] += 1
        heat = f"heatmaps/{path.stem}.png"
        overlay_png(c, scale, size, out / heat)
        preds.append({"file": path.name, "true_label": labels[y], "predicted": labels[p],
                      "confidence": round(float(probs[p]) * 100, 1), "correct": p == y, "heatmap": heat,
                      "evidence": round(float(min(c.max() / scale, 1.0)) if scale > 0 else 0.0, 3)})
    n = len(preds)
    metrics = {"dataset": name, "model": "ResNet-18 (ImageNet weights), fine-tuned", "epochs": args.epochs,
               "best_epoch": best_epoch, "best_test_accuracy": round(best_acc, 4),
               "train_images": len(train), "test_images": len(test), "heldout_images": n,
               "trained_on": "Amazon SageMaker AI" if os.environ.get("SM_CURRENT_HOST") else "local",
               "hardware": hardware(), "cpu": platform.processor() or platform.machine(),
               "torch": torch.__version__, "train_seconds": train_seconds, "latency_ms": speed,
               "heldout_accuracy": round((cm["tp"] + cm["tn"]) / n, 4), **cm,
               "recall": round(cm["tp"] / max(cm["tp"] + cm["fn"], 1), 4),
               "precision": round(cm["tp"] / max(cm["tp"] + cm["fp"], 1), 4), "history": history}
    (out / "predictions.json").write_text(json.dumps(preds, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model.pt")  # for layer_maps.py
    print(f"[{name}] held-out accuracy {metrics['heldout_accuracy']:.3f} "
          f"(tp {cm['tp']} fn {cm['fn']} fp {cm['fp']} tn {cm['tn']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="casting,packages")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SM_NUM_CPUS", "2")))
    ap.add_argument("--images", default=os.environ.get("SM_CHANNEL_IMAGES", "/opt/ml/input/data/images"))
    ap.add_argument("--heldout", default=os.environ.get("SM_CHANNEL_HELDOUT", "/opt/ml/input/data/heldout"))
    ap.add_argument("--output", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    ap.add_argument("--local-check", action="store_true",
                    help="tiny smoke test: 16 images per split, random weights, no download")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Device: {DEVICE} ({hardware()}), torch {torch.__version__}", flush=True)
    out_root = Path(args.output)
    for name in args.datasets.split(","):
        run_dataset(name.strip(), args, out_root)


if __name__ == "__main__":
    main()
