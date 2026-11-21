"""Shared setup for the casting inspection scripts: settings, AWS clients, the dataset and its fixed splits."""
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("VISION_DATA", ROOT / "data"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Credentials come from the AWS CLI profile named in settings.json (created with `aws login --profile ...`).
# Nothing secret is stored in this folder.

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError:
    sys.exit("boto3 is not installed. Run: .venv\\Scripts\\pip install -r requirements.txt")

# The Kaggle casting dataset sorts images into def_front (defective) and ok_front folders.
FOLDERS = {"def_front": "defective", "ok_front": "ok"}
LABELS = ("defective", "ok")
POSITIVE = "defective"
EXTENSIONS = {".jpeg": "image/jpeg", ".jpg": "image/jpeg", ".png": "image/png"}

TRAIN_PER_CLASS = 500     # Custom Labels training images per class, from the dataset's train folder
TEST_PER_CLASS = 100      # Custom Labels test images per class, from the dataset's test folder
HOLDOUT_PER_CLASS = 50    # images the model never sees, used in step 3
BASELINE_PER_CLASS = 25   # step 1 uses the first 25 holdout images per class
SEED = 42

TRAINING_PER_HOUR = 1.00
INFERENCE_PER_HOUR = 4.00


def settings():
    path = ROOT / "settings.json"
    if not path.exists():
        sys.exit("settings.json is missing. Copy settings.example.json to settings.json and put your bucket name in it.")
    s = json.loads(path.read_text())
    s.setdefault("region", "us-east-2")
    s.setdefault("profile", None)
    s.setdefault("project", "casting-inspection")
    s.setdefault("version", "v1")
    bucket = s.get("bucket", "")
    if not bucket or "YOUR" in bucket:
        sys.exit("Put your S3 bucket name in settings.json.")
    if "custom-labels" not in bucket:
        sys.exit("The bucket name must contain 'custom-labels' so the AmazonRekognitionCustomLabelsFullAccess "
                 "policy covers it (for example custom-labels-casting-demo).")
    return s


_session = None


def client(name):
    """An AWS client using the profile and region from settings.json."""
    global _session
    if _session is None:
        s = settings()
        _session = boto3.Session(profile_name=s["profile"], region_name=s["region"])
    return _session.client(name)


def find_images():
    """Returns {(split, label): [paths]} for the train and test folders, sorted by filename."""
    if not DATA.exists():
        sys.exit(f"No data folder at {DATA}. Download the Kaggle casting dataset and extract it there (see README).")
    found = {}
    for folder in DATA.rglob("*"):
        if not folder.is_dir() or folder.name not in FOLDERS:
            continue
        split = folder.parent.name.lower()
        if split not in ("train", "test"):
            continue  # the dataset also ships an unsplit casting_512x512 copy; skip it
        by_name = found.setdefault((split, FOLDERS[folder.name]), {})
        for f in folder.iterdir():
            if f.suffix.lower() in EXTENSIONS:
                by_name.setdefault(f.name, f)
    for split in ("train", "test"):
        for label in LABELS:
            if not found.get((split, label)):
                sys.exit(f"No {split} images for '{label}' under {DATA}. Expected folders like "
                         f"casting_data/{split}/def_front and casting_data/{split}/ok_front.")
    return {key: [by_name[n] for n in sorted(by_name)] for key, by_name in found.items()}


def splits():
    """The same images every run: train and test go to Custom Labels, holdout is never shown to it."""
    images = find_images()
    out = {"train": [], "test": [], "holdout": [], "baseline": []}
    for label in LABELS:
        rng = random.Random(f"{SEED}-{label}")
        train_pool = images[("train", label)]
        train = rng.sample(train_pool, min(TRAIN_PER_CLASS, len(train_pool)))
        pool = list(images[("test", label)])
        rng.shuffle(pool)
        if len(pool) < TEST_PER_CLASS + HOLDOUT_PER_CLASS:
            sys.exit(f"Only {len(pool)} test images for '{label}'; need {TEST_PER_CLASS + HOLDOUT_PER_CLASS}.")
        test = pool[:TEST_PER_CLASS]
        holdout = pool[TEST_PER_CLASS:TEST_PER_CLASS + HOLDOUT_PER_CLASS]
        out["train"] += [(p, label) for p in train]
        out["test"] += [(p, label) for p in test]
        out["holdout"] += [(p, label) for p in holdout]
        out["baseline"] += [(p, label) for p in holdout[:BASELINE_PER_CLASS]]
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


def pct(n, total):
    return f"{100 * n / total:.0f}%" if total else "-"


def minutes(seconds):
    return f"{seconds / 60:.0f} min"


def explain(e):
    err = e.response.get("Error", {})
    code, msg = err.get("Code", ""), err.get("Message", "")
    hint = ""
    if "AccessDenied" in code or code == "UnauthorizedOperation":
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
