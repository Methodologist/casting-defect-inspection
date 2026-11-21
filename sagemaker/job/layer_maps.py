"""Feature maps for the results site's "Inside our network" diagram.

Runs one held-out image (the correctly caught defect with the strongest heatmap evidence) through the trained CNN
and saves what each stage computed:
  - first filters and stages 1 to 3: the 4 feature maps with the most spatial contrast
  - stage 4: the 4 feature maps that added the most to the "defective" heatmap (Grad-CAM weight x activation)
Each map is drawn at its real resolution (56, 28, 14 or 7 pixels square, enlarged with square pixels), so the
pictures get blockier stage by stage, exactly as the network's view does.

    python layer_maps.py <results/<dataset>/cnn> <held-out folder with one subfolder per label>

Writes <cnn dir>/layers/: <stage>.png (2x2 grid), pooled.png (the 512 averaged numbers) and layers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gradcam import DATASETS, MEAN, STD, build_model  # noqa: E402

STAGES = ("maxpool", "layer1", "layer2", "layer3", "layer4")
# Dark purple -> orange -> pale yellow ("inferno"-like): readable on light and dark pages, no color meaning implied.
RAMP = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21],
                 [250, 193, 39], [252, 255, 164]], dtype=np.float32) / 255


def colorize(v):
    i = np.clip(v, 0, 1) * (len(RAMP) - 1)
    lo = np.clip(np.floor(i).astype(int), 0, len(RAMP) - 2)
    f = (i - lo)[..., None]
    return ((RAMP[lo] * (1 - f) + RAMP[lo + 1] * f) * 255).astype(np.uint8)


def normalize(a):
    a = a - a.min()
    return a / a.max() if a.max() > 0 else a


def tile(a, size):
    """One feature map, colored, enlarged with square pixels (no smoothing) so its real resolution shows."""
    return Image.fromarray(colorize(normalize(a)), "RGB").resize((size, size), Image.NEAREST)


def grid(maps, size=112, gap=6):
    out = Image.new("RGBA", (2 * size + gap, 2 * size + gap), (0, 0, 0, 0))
    for k, a in enumerate(maps[:4]):
        out.paste(tile(a, size), ((k % 2) * (size + gap), (k // 2) * (size + gap)))
    return out


def main():
    cnn_dir, held_dir = Path(sys.argv[1]), Path(sys.argv[2])
    metrics = json.loads((cnn_dir / "metrics.json").read_text())
    preds = json.loads((cnn_dir / "predictions.json").read_text())
    pos, neg = DATASETS[metrics["dataset"]]
    hits = [p for p in preds if p["true_label"] == pos and p["correct"]]
    if not hits:
        sys.exit(f"No correctly caught {pos} image to draw.")
    ex = max(hits, key=lambda p: p["evidence"])

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(cnn_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    acts, grads = {}, {}
    for name in STAGES:
        getattr(model, name).register_forward_hook(lambda m, i, o, name=name: acts.__setitem__(name, o))
    model.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("layer4", go[0]))

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    x = tf(Image.open(held_dir / pos / ex["file"]).convert("RGB")).unsqueeze(0)
    logits = model(x)
    logits[0, 0].backward()  # index 0 = positive class, as in training
    probs = torch.softmax(logits, 1)[0].tolist()

    out = cnn_dir / "layers"
    out.mkdir(exist_ok=True)
    info = {"file": ex["file"], "positive": pos, "negative": neg, "probs": {pos: probs[0], neg: probs[1]},
            "stages": {}}
    for name in STAGES:
        a = acts[name][0].detach()
        if name == "layer4":
            weight = grads["layer4"][0].mean(dim=(1, 2))           # how much each map's strength moves "defective"
            pick = torch.topk(weight * a.mean(dim=(1, 2)), 4).indices  # biggest contributors to the heatmap
        else:
            pick = torch.topk(a.flatten(1).std(1), 4).indices       # maps with the most going on
        grid([a[c].numpy() for c in pick]).save(out / f"{name}.png", optimize=True)
        info["stages"][name] = {"maps": int(a.shape[0]), "size": int(a.shape[1]), "shown": pick.tolist()}

    pooled = acts["layer4"][0].detach().mean(dim=(1, 2)).numpy()  # what the average step hands to the final layer
    cells = normalize(pooled).reshape(16, 32)
    Image.fromarray(colorize(cells), "RGB").resize((32 * 8, 16 * 8), Image.NEAREST).save(out / "pooled.png")
    (out / "layers.json").write_text(json.dumps(info, indent=1))
    print(f"[{metrics['dataset']}] layer maps for {ex['file']} saved to {out}")


if __name__ == "__main__":
    main()
