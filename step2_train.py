"""Step 2: upload images, build the Custom Labels datasets and train a defective/ok model.

Safe to re-run: it skips images already uploaded and picks up a training run already in progress.
Training is billed at about $1 an hour of training time; 1,000 images usually takes about an hour.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import (EXTENSIONS, LABELS, RESULTS, ROOT, TRAINING_PER_HOUR, client, find_project, find_version,
                    minutes, run, settings, splits, utc_now)

ATTR = "quality"  # the label attribute name inside the manifest; any name works


def manifest_line(bucket, key, label, stamp):
    return json.dumps({
        "source-ref": f"s3://{bucket}/{key}",
        ATTR: LABELS.index(label),
        f"{ATTR}-metadata": {"confidence": 1, "job-name": "labeling-job/casting-quality", "class-name": label,
                             "human-annotated": "yes", "creation-date": stamp,
                             "type": "groundtruth/image-classification"},
    })


def upload_images(s3, bucket, items):
    existing = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="images/"):
        existing.update(o["Key"] for o in page.get("Contents", []))
    todo = [(path, key) for path, _, key in items if key not in existing]
    print(f"Uploading {len(todo)} images ({len(items) - len(todo)} already in S3)...")

    def put(job):
        path, key = job
        s3.put_object(Bucket=bucket, Key=key, Body=path.read_bytes(), ContentType=EXTENSIONS[path.suffix.lower()])

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(put, todo), 1):
            if i % 50 == 0 or i == len(todo):
                print(f"\r  {i}/{len(todo)}", end="", flush=True)
    if todo:
        print()


def ensure_project(rek, name):
    project = find_project(rek, name)
    if project is None:
        print(f"Creating Custom Labels project '{name}'...")
        rek.create_project(ProjectName=name)
    while True:
        project = find_project(rek, name)
        if project and project["Status"] == "CREATED":
            return project
        time.sleep(5)


def ensure_datasets(rek, s, project):
    have = {d["DatasetType"]: d["DatasetArn"] for d in project.get("Datasets", [])}
    for kind in ("TRAIN", "TEST"):
        if kind not in have:
            print(f"Creating the {kind.lower()} dataset from its manifest...")
            have[kind] = rek.create_dataset(
                ProjectArn=project["ProjectArn"], DatasetType=kind,
                DatasetSource={"GroundTruthManifest": {"S3Object": {
                    "Bucket": s["bucket"], "Name": f"manifests/{kind.lower()}.manifest"}}})["DatasetArn"]
    for kind, arn in have.items():
        while True:
            d = rek.describe_dataset(DatasetArn=arn)["DatasetDescription"]
            if d["Status"] in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
                stats = d.get("DatasetStats", {})
                print(f"  {kind.lower()} dataset ready: {stats.get('TotalEntries', '?')} images, "
                      f"{stats.get('ErrorEntries', 0)} with errors")
                break
            if d["Status"].endswith("FAILED"):
                sys.exit(f"The {kind.lower()} dataset failed: {d.get('StatusMessage', '')}\n"
                         "Delete the project in the Rekognition console (Custom Labels > Projects) and run this again.")
            time.sleep(10)


def report(s3, v):
    secs = v.get("BillableTrainingTimeInSeconds", 0)
    print(f"\nTraining finished. Billable training time {minutes(secs)}, "
          f"about ${secs / 3600 * TRAINING_PER_HOUR:.2f}.")
    ev = v.get("EvaluationResult", {})
    loc = ev.get("Summary", {}).get("S3Object")
    if not loc:
        print(f"Overall F1 score: {ev.get('F1Score')}")
        return
    summary = json.loads(s3.get_object(Bucket=loc["Bucket"], Key=loc["Name"])["Body"].read())
    out = RESULTS / "step2_evaluation.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out.relative_to(ROOT)}")

    agg = summary.get("AggregatedEvaluationResults", {})
    print(f"\nOn the {summary.get('EvaluationDetails', {}).get('NumberOfTestingImages', '?')} test images:")
    print(f"  F1 {agg.get('F1Score', ev.get('F1Score', 0)):.3f}   precision {agg.get('Precision', 0):.3f}   "
          f"recall {agg.get('Recall', 0):.3f}")
    print(f"\n{'Label':<12}{'Images':>8}{'Precision':>11}{'Recall':>8}{'F1':>7}{'Threshold':>11}")
    for r in summary.get("LabelEvaluationResults", []):
        m = r.get("Metrics", {})
        print(f"{r.get('Label', '?'):<12}{r.get('NumberOfTestingImages', '?'):>8}{m.get('Precision', 0):>11.3f}"
              f"{m.get('Recall', 0):>8.3f}{m.get('F1Score', 0):>7.3f}{m.get('Threshold', 0):>11.3f}")
    print("\nScreenshot the model's Evaluation page in the Rekognition console for the interview. "
          "Next: python step3_test.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true", help="start training and exit; re-run later to see results")
    args = ap.parse_args()
    s = settings()
    rek, s3 = client("rekognition"), client("s3")

    project = find_project(rek, s["project"])
    v = find_version(rek, project["ProjectArn"], s["version"]) if project else None
    if v is None:
        sp = splits()
        stamp = utc_now()
        items = {split: [(p, label, f"images/{split}/{label}/{p.name}") for p, label in sp[split]]
                 for split in ("train", "test")}
        upload_images(s3, s["bucket"], items["train"] + items["test"])
        for split, rows in items.items():
            body = "\n".join(manifest_line(s["bucket"], key, label, stamp) for _, label, key in rows) + "\n"
            (RESULTS / f"{split}.manifest").write_text(body)
            s3.put_object(Bucket=s["bucket"], Key=f"manifests/{split}.manifest", Body=body.encode())
        print("Uploaded train and test manifests.")
        project = ensure_project(rek, s["project"])
        ensure_datasets(rek, s, project)
        rek.create_project_version(
            ProjectArn=project["ProjectArn"], VersionName=s["version"],
            OutputConfig={"S3Bucket": s["bucket"], "S3KeyPrefix": f"output/{s['version']}/"})
        print(f"Training started for model {s['version']}.")
        v = find_version(rek, project["ProjectArn"], s["version"])

    if v["Status"] not in ("TRAINING_COMPLETED", "STOPPED", "RUNNING", "STARTING", "STOPPING"):
        if args.no_wait:
            print(f"Model {s['version']} is {v['Status']}. Run this again later to see the results.")
            return
        print("Waiting for training. You can close this window; re-running picks up where it left off.")
        began = time.time()
        while v["Status"] == "TRAINING_IN_PROGRESS":
            print(f"\r  training... {minutes(time.time() - began)} waited", end="", flush=True)
            time.sleep(60)
            v = find_version(rek, project["ProjectArn"], s["version"])
        print()
        if v["Status"] != "TRAINING_COMPLETED":
            sys.exit(f"Training ended as {v['Status']}: {v.get('StatusMessage', '')}")
    report(s3, v)


if __name__ == "__main__":
    run(main)
