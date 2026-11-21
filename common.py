"""Shared setup for the inspection scripts: settings, AWS clients, the datasets and their fixed splits."""
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("VISION_DATA", ROOT / "data"))

# Credentials come from the AWS CLI profile named in settings.json (created with `aws login --profile ...`).
# Nothing secret is stored in this folder.

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError:
    sys.exit("boto3 is not installed. Run: .venv\\Scripts\\pip install -r requirements.txt")

EXTENSIONS = {".jpeg": "image/jpeg", ".jpg": "image/jpeg", ".png": "image/png"}
SEED = 42

# Each dataset lives in data/<folder>/. "classes" maps the dataset's folder names to the labels used here;
# the first label is the positive class (the one a line must not miss).
#   split "given":  the dataset's own train/ and test/ folders are kept apart, with fixed counts per class.
#   split "pooled": every image of a class is pooled, then shuffled once (seeded) into train/test/holdout.
DATASETS = {
    "casting": {
        "folder": "casting",
        "title": "cast pump impellers",
        "classes": {"def_front": "defective", "ok_front": "ok"},
        "positive_plural": "defective parts",
        "negative_plural": "good parts",
        "split": "given",
        "train_per_class": 500,   # from the dataset's train folder
        "test_per_class": 100,    # from the dataset's test folder
        "holdout_per_class": 50,  # also from the test folder, never shown to Custom Labels
        "baseline_per_class": 25,
    },
    "packages": {
        "folder": "packages",
        "title": "shipping cases",
        "classes": {"damaged": "damaged", "intact": "intact"},
        "positive_plural": "damaged cases",
        "negative_plural": "intact cases",
        "split": "pooled",
        "fractions": {"train": 0.65, "test": 0.20},  # the rest is holdout
        "baseline_per_class": 25,
    },
}

TRAINING_PER_HOUR = 1.00
INFERENCE_PER_HOUR = 4.00


def settings():
    path = ROOT / "settings.json"
    if not path.exists():
        sys.exit("settings.json is missing. Copy settings.example.json to settings.json and put your bucket name in it.")
    s = json.loads(path.read_text())
    if os.environ.get("VISION_DATASET"):  # pick a dataset for one command without editing settings.json
        s["dataset"] = os.environ["VISION_DATASET"]
        s.pop("project", None)
    s.setdefault("dataset", "casting")
    if s["dataset"] not in DATASETS:
        sys.exit(f"settings.json has dataset '{s['dataset']}'. Use one of: {', '.join(DATASETS)}.")
    s.setdefault("region", "us-east-2")
    s.setdefault("profile", None)
    s.setdefault("project", f"{s['dataset']}-inspection")
    s.setdefault("version", "v1")
    bucket = s.get("bucket", "")
    if not bucket or "YOUR" in bucket:
        sys.exit("Put your S3 bucket name in settings.json.")
    if "custom-labels" not in bucket:
        sys.exit("The bucket name must contain 'custom-labels' so the AmazonRekognitionCustomLabelsFullAccess "
                 "policy covers it (for example custom-labels-casting-demo).")
    return s


def dataset():
    """The dataset chosen in settings.json, with its name and labels filled in."""
    name = settings()["dataset"]
    ds = dict(DATASETS[name], name=name)
    ds["labels"] = tuple(dict.fromkeys(ds["classes"].values()))
    ds["positive"], ds["negative"] = ds["labels"]
    return ds


def results_dir():
    path = ROOT / "results" / settings()["dataset"]
    path.mkdir(parents=True, exist_ok=True)
    return path


_session = None


def client(name):
    """An AWS client using the profile and region from settings.json."""
    global _session
    if _session is None:
        s = settings()
        _session = boto3.Session(profile_name=s["profile"], region_name=s["region"])
    return _session.client(name)


def find_images(ds):
    """Returns {(split, label): [paths]} from the dataset's train/valid/test folders, sorted by filename."""
    root = DATA / ds["folder"]
    if not root.exists():
        sys.exit(f"No data folder at {root}. Download the {ds['name']} dataset and extract it there (see README).")
    found = {}
    for folder in root.rglob("*"):
        if not folder.is_dir() or folder.name not in ds["classes"]:
            continue
        split = folder.parent.name.lower()
        if split not in ("train", "valid", "test"):
            continue  # e.g. the casting dataset also ships an unsplit casting_512x512 copy; skip it
        by_name = found.setdefault((split, ds["classes"][folder.name]), {})
        for f in folder.iterdir():
            if f.suffix.lower() in EXTENSIONS:
                by_name.setdefault(f.name, f)
    for label in ds["labels"]:
        if not any(found.get((split, label)) for split in ("train", "valid", "test")):
            sys.exit(f"No images for '{label}' under {root}. Expected folders named "
                     f"{', '.join(ds['classes'])} inside train/valid/test folders.")
    return {key: [by_name[n] for n in sorted(by_name)] for key, by_name in found.items()}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(paths, exclude=()):
    """Drops exact duplicate images (same bytes, any filename), keeping the first, and any whose bytes are in exclude."""
    seen, out = set(exclude), []
    for p in paths:
        d = _digest(p)
        if d not in seen:
            seen.add(d)
            out.append(p)
    return out


def splits():
    """The same images every run: train and test go to Custom Labels, holdout is never shown to it.

    No image in test or holdout is a byte-for-byte copy of a training image. The casting dataset ships a few
    such copies across its own train and test folders; they are dropped from test and holdout.
    """
    ds = dataset()
    images = find_images(ds)
    out = {"train": [], "test": [], "holdout": [], "baseline": []}
    if ds["split"] == "given":
        # Only a train image with the same file size as some test image can be a copy of it, so hash just those.
        test_sizes = {p.stat().st_size for l in ds["labels"] for p in images.get(("test", l), [])}
        every_train_image = {_digest(p) for l in ds["labels"] for p in images.get(("train", l), [])
                             if p.stat().st_size in test_sizes}
    for label in ds["labels"]:
        rng = random.Random(f"{SEED}-{label}")
        if ds["split"] == "given":
            train_pool = images.get(("train", label), [])
            train = rng.sample(train_pool, min(ds["train_per_class"], len(train_pool)))
            pool = _unique(images.get(("test", label), []), exclude=every_train_image)
            rng.shuffle(pool)
            need = ds["test_per_class"] + ds["holdout_per_class"]
            if len(pool) < need:
                sys.exit(f"Only {len(pool)} test images for '{label}'; need {need}.")
            test = pool[:ds["test_per_class"]]
            holdout = pool[ds["test_per_class"]:need]
        else:
            by_name = {}
            for split in ("train", "valid", "test"):
                for p in images.get((split, label), []):
                    by_name.setdefault(p.name, p)
            pool = _unique(by_name[n] for n in sorted(by_name))
            rng.shuffle(pool)
            n_train = round(len(pool) * ds["fractions"]["train"])
            n_test = round(len(pool) * ds["fractions"]["test"])
            train, test, holdout = pool[:n_train], pool[n_train:n_train + n_test], pool[n_train + n_test:]
        out["train"] += [(p, label) for p in train]
        out["test"] += [(p, label) for p in test]
        out["holdout"] += [(p, label) for p in holdout]
        out["baseline"] += [(p, label) for p in holdout[:ds["baseline_per_class"]]]
    return out


def find_project(rek, name):
    found = rek.describe_projects(ProjectNames=[name]).get("ProjectDescriptions", [])
    return found[0] if found else None


def find_version(rek, project_arn, version):
    found = rek.describe_project_versions(ProjectArn=project_arn, VersionNames=[version]).get(
        "ProjectVersionDescriptions", [])
    return found[0] if found else None


def stop_model(rek, project_arn, version):
    """Stops a running model and waits until AWS reports it stopped. Safe to call when it is not running."""
    v = find_version(rek, project_arn, version)
    if v is None:
        return
    while v["Status"] == "STARTING":
        print("  The model is still starting. Waiting so it can be stopped...")
        time.sleep(30)
        v = find_version(rek, project_arn, version)
    if v["Status"] == "RUNNING":
        print(f"Stopping model {version}...")
        rek.stop_project_version(ProjectVersionArn=v["ProjectVersionArn"])
    deadline = time.time() + 30 * 60
    while v["Status"] in ("RUNNING", "STOPPING") and time.time() < deadline:
        time.sleep(20)
        v = find_version(rek, project_arn, version)
    if v["Status"] in ("RUNNING", "STOPPING"):
        print(f"WARNING: model {version} still shows {v['Status']}. Run stop_model.py again in a few minutes, "
              "or stop it in the Rekognition console. It bills until it stops.")
    else:
        print(f"Model {version} is {v['Status']}. Nothing is being billed for inference.")


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path.relative_to(ROOT)}")


def score(rows, pos):
    """Confusion counts for held-out rows ({"true_label", "predicted"}), with `pos` as the class a line must catch.

    A row whose prediction is anything other than `pos` (including "none") counts as a pass.
    """
    tp = sum(r["true_label"] == pos and r["predicted"] == pos for r in rows)
    fn = sum(r["true_label"] == pos and r["predicted"] != pos for r in rows)
    fp = sum(r["true_label"] != pos and r["predicted"] == pos for r in rows)
    tn = len(rows) - tp - fn - fp
    return {"true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0,
            "accuracy": round((tp + tn) / len(rows), 3) if rows else 0}


def pct(n, total):
    return f"{100 * n / total:.0f}%" if total else "-"


def minutes(seconds):
    return f"{seconds / 60:.0f} min"


def explain(e):
    err = e.response.get("Error", {})
    code, msg = err.get("Code", ""), err.get("Message", "")
    hint = ""
    if "verified" in msg.lower():
        hint = "\nThis is an account-level restriction on new AWS accounts, not a missing permission."
    elif "AccessDenied" in code or code == "UnauthorizedOperation":
        hint = ("\nYour AWS identity is missing a permission. If this worked before, check whether the account hit a "
                "spend limit or ran out of Free plan credits. If you use an IAM user, it needs "
                "AmazonRekognitionCustomLabelsFullAccess and AmazonRekognitionReadOnlyAccess, and the bucket name "
                "must contain 'custom-labels'.")
    elif code in ("ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId", "SignatureDoesNotMatch",
                  "UnrecognizedClientException", "InvalidAccessKeyId"):
        hint = f"\nYour AWS sign-in has expired or is invalid. Run: aws login --profile {settings()['profile'] or 'default'}"
    elif code == "NoSuchBucket":
        hint = "\nCheck the bucket name in settings.json and that it exists in the same region."
    return f"AWS error {code}: {msg}{hint}"


def run(main):
    try:
        main()
    except NoCredentialsError:
        sys.exit(f"No AWS credentials found. Run: aws login --profile {settings()['profile'] or 'default'} (see README).")
    except ClientError as e:
        sys.exit(explain(e))
    except BotoCoreError as e:
        sys.exit(f"AWS error: {e}\nIf this is about credentials or a login session, run: "
                 f"aws login --profile {settings()['profile'] or 'default'}")
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
