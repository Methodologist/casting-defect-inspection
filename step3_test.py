"""Step 3: start the trained model, test it on 100 held-out images it never saw, then stop it.

A running model bills about $4 an hour, from start until stopped. This script always stops it at the end,
including after an error or Ctrl+C. If the window or PC dies mid-run, run stop_model.py.
"""
import argparse
import json
import sys
import time

from common import (INFERENCE_PER_HOUR, LABELS, POSITIVE, RESULTS, ROOT, client, find_project, find_version,
                    minutes, pct, run, settings, splits, stop_model, write_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()
    s = settings()
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

    write_csv(RESULTS / "step3_holdout.csv", rows)
    tp = sum(r["true_label"] == POSITIVE and r["predicted"] == POSITIVE for r in rows)
    fn = sum(r["true_label"] == POSITIVE and r["predicted"] != POSITIVE for r in rows)
    fp = sum(r["true_label"] != POSITIVE and r["predicted"] == POSITIVE for r in rows)
    tn = len(rows) - tp - fn - fp
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    accuracy = (tp + tn) / len(rows)
    summary = {"images": len(rows), "true_positive": tp, "false_negative": fn, "false_positive": fp,
               "true_negative": tn, "precision": round(precision, 3), "recall": round(recall, 3),
               "accuracy": round(accuracy, 3)}
    (RESULTS / "step3_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nHeld-out results ({len(rows)} images the model never saw), 'defective' as the positive class:")
    print(f"{'':>22}{'predicted defective':>21}{'predicted ok':>14}")
    print(f"{'actually defective':>22}{tp:>21}{fn:>14}")
    print(f"{'actually ok':>22}{fp:>21}{tn:>14}")
    print(f"\nAccuracy {pct(tp + tn, len(rows))}   precision {precision:.3f}   recall {recall:.3f}")
    print(f"Missed defects (the costly mistake on a line): {fn}. False rejects of good parts: {fp}.")
    misses = [r["file"] for r in rows if r["correct"] == "no"]
    if misses:
        print("Wrong calls, worth opening to see why: " + ", ".join(misses))
    print(f"Saved {(RESULTS / 'step3_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    run(main)
