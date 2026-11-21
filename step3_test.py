"""Step 3: start the trained model, test it on held-out images it never saw, then stop it.

A running model bills about $4 an hour, from start until stopped. This script always stops it at the end,
including after an error or Ctrl+C. If the window or PC dies mid-run, run stop_model.py.
"""
import argparse
import json
import sys
import time

from common import (INFERENCE_PER_HOUR, ROOT, client, dataset, find_project, find_version, minutes, pct,
                    results_dir, run, score, settings, splits, stop_model, write_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()
    s = settings()
    ds = dataset()
    pos, neg = ds["positive"], ds["negative"]
    rek = client("rekognition")
    project = find_project(rek, s["project"])
    v = find_version(rek, project["ProjectArn"], s["version"]) if project else None
    if v is None or v["Status"] not in ("TRAINING_COMPLETED", "STOPPED", "RUNNING"):
        sys.exit("No trained model ready yet. Run step2_train.py first"
                 + (f" (model is {v['Status']})." if v else "."))
    arn = v["ProjectVersionArn"]
    images = splits()["holdout"]

    if not args.yes:
        answer = input(f"This starts the model, billed at about ${INFERENCE_PER_HOUR:.0f}/hour until it stops "
                       "(it stops automatically at the end; expect about $2 to $4 total). Continue? [y/N] ")
        if answer.strip().lower() != "y":
            return

    began = time.time()
    try:
        if v["Status"] != "RUNNING":
            print("Starting the model. This often takes 10 to 20 minutes...")
            rek.start_project_version(ProjectVersionArn=arn, MinInferenceUnits=1, MaxInferenceUnits=1)
            while True:
                v = find_version(rek, project["ProjectArn"], s["version"])
                if v["Status"] == "RUNNING":
                    break
                if v["Status"] == "FAILED":
                    sys.exit(f"The model failed to start: {v.get('StatusMessage', '')}")
                print(f"\r  {v['Status'].lower()}... {minutes(time.time() - began)}", end="", flush=True)
                time.sleep(30)
            print(f"\nRunning after {minutes(time.time() - began)}.")

        print(f"Testing on {len(images)} held-out images...")
        rows = []
        for i, (path, truth) in enumerate(images, 1):
            resp = rek.detect_custom_labels(ProjectVersionArn=arn, Image={"Bytes": path.read_bytes()},
                                            MinConfidence=0)
            found = resp.get("CustomLabels", [])
            best = max(found, key=lambda x: x["Confidence"]) if found else None
            rows.append({"file": path.name, "true_label": truth,
                         "predicted": best["Name"] if best else "none",
                         "confidence": f"{best['Confidence']:.1f}" if best else "",
                         "correct": "yes" if best and best["Name"] == truth else "no"})
            print(f"\r  {i}/{len(images)}", end="", flush=True)
        print()
    finally:
        stop_model(rek, project["ProjectArn"], s["version"])
        ran = time.time() - began
        print(f"The model ran about {minutes(ran)}, roughly ${ran / 3600 * INFERENCE_PER_HOUR:.2f}.")

    out = results_dir()
    write_csv(out / "step3_holdout.csv", rows)
    counts = score(rows, pos)
    tp, fn, fp, tn = (counts[k] for k in ("true_positive", "false_negative", "false_positive", "true_negative"))
    precision, recall = counts["precision"], counts["recall"]
    summary = {"dataset": ds["name"], "positive_class": pos, "images": len(rows), **counts}
    (out / "step3_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nHeld-out results ({len(rows)} {ds['title']} the model never saw), '{pos}' as the positive class:")
    print(f"{'':>20}{'predicted ' + pos:>20}{'predicted ' + neg:>20}")
    print(f"{'actually ' + pos:>20}{tp:>20}{fn:>20}")
    print(f"{'actually ' + neg:>20}{fp:>20}{tn:>20}")
    print(f"\nAccuracy {pct(tp + tn, len(rows))}   precision {precision:.3f}   recall {recall:.3f}")
    print(f"Missed {ds['positive_plural']} (the costly mistake on a line): {fn}. "
          f"False rejects of {ds['negative_plural']}: {fp}.")
    misses = [r["file"] for r in rows if r["correct"] == "no"]
    if misses:
        print("Wrong calls, worth opening to see why: " + ", ".join(misses))
    print(f"Saved {(out / 'step3_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    run(main)
