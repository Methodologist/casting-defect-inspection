"""Train our own CNN (ResNet-18 + Grad-CAM heatmaps) locally, on exactly the images Rekognition used.

1. Copies each dataset's train / test / held-out images (from common.splits(), the same split as steps 2 and 3)
   into data/_staged/.
2. Runs sagemaker/job/train_gradcam.py with the GPU environment in .venv-gpu (an NVIDIA GPU is used if present).
3. Saves predictions, metrics, heatmaps and the trained weights to results/<dataset>/cnn/, then draws the
   feature maps for the site's network diagram (sagemaker/job/layer_maps.py) into results/<dataset>/cnn/layers/.

No AWS calls, no cost. Downloads (the ImageNet starting weights) go to .cache/ in this folder.

    .venv\\Scripts\\python train_local_cnn.py                  # both datasets
    .venv\\Scripts\\python train_local_cnn.py --datasets casting
"""
import argparse
import os
import shutil
import subprocess
import sys

import common
from common import DATASETS, ROOT

STAGE = ROOT / "data" / "_staged"
OUT = ROOT / "data" / "_cnn_output"
GPU_PYTHON = ROOT / ".venv-gpu" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
JOB = ROOT / "sagemaker" / "job" / "train_gradcam.py"
LAYER_MAPS = ROOT / "sagemaker" / "job" / "layer_maps.py"


def stage(name):
    os.environ["VISION_DATASET"] = name
    sp = common.splits()
    for split in ("train", "test", "holdout"):
        for path, label in sp[split]:
            dest = (STAGE / "heldout" / name / label if split == "holdout"
                    else STAGE / "images" / name / split / label) / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    print(f"[{name}] staged train {len(sp['train'])}, test {len(sp['test'])}, held-out {len(sp['holdout'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not GPU_PYTHON.exists():
        sys.exit("No .venv-gpu found. See the README section 'Our own CNN' for the one-time install.")

    names = [n.strip() for n in args.datasets.split(",")]
    for folder in (STAGE, OUT):
        shutil.rmtree(folder, ignore_errors=True)
    for name in names:
        stage(name)

    env = dict(os.environ, TORCH_HOME=str(ROOT / ".cache" / "torch"), PYTHONUNBUFFERED="1")
    cmd = [str(GPU_PYTHON), "-I", str(JOB), "--datasets", ",".join(names), "--epochs", str(args.epochs),
           "--workers", str(args.workers), "--images", str(STAGE / "images"), "--heldout", str(STAGE / "heldout"),
           "--output", str(OUT)]
    subprocess.run(cmd, env=env, check=True)

    for name in names:
        dest = ROOT / "results" / name / "cnn"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(OUT / name, dest)
        print(f"Saved results/{name}/cnn/")
        # Feature maps for the site's "Inside our network" diagram.
        subprocess.run([str(GPU_PYTHON), "-I", str(LAYER_MAPS), str(dest), str(STAGE / "heldout" / name)],
                       env=env, check=True)
    print("Next: python build_site.py, then python deploy_site.py")


if __name__ == "__main__":
    main()
