"""Step 1: off-the-shelf Rekognition (DetectLabels) on good and defective castings.

The point: a general model describes the part ("metal", "gear") but has no idea what a casting defect is.
50 images, covered by the free tier (1,000 a month), otherwise about 5 cents.
"""
from collections import Counter

from common import LABELS, RESULTS, client, pct, run, splits, write_csv


def main():
    rek = client("rekognition")
    images = splits()["baseline"]
    print(f"Sending {len(images)} held-out images to Rekognition DetectLabels...")
    seen = {label: Counter() for label in LABELS}
    rows = []
    for i, (path, label) in enumerate(images, 1):
        resp = rek.detect_labels(Image={"Bytes": path.read_bytes()}, MaxLabels=10, MinConfidence=50)
        found = [(x["Name"], x["Confidence"]) for x in resp["Labels"]]
        seen[label].update({name for name, _ in found})
        rows.append({"file": path.name, "true_label": label,
                     "rekognition_labels": "; ".join(f"{n} {c:.0f}%" for n, c in found)})
        print(f"\r  {i}/{len(images)}", end="", flush=True)
    print()
    write_csv(RESULTS / "step1_baseline.csv", rows)

    total = {label: sum(1 for _, l in images if l == label) for label in LABELS}
    names = sorted(set(seen["ok"]) | set(seen["defective"]),
                   key=lambda n: -(seen["ok"][n] + seen["defective"][n]))
    print(f"\nHow often each label came back:\n{'Label':<28}{'OK parts':>10}{'Defective':>11}")
    for n in names[:15]:
        print(f"{n:<28}{pct(seen['ok'][n], total['ok']):>10}{pct(seen['defective'][n], total['defective']):>11}")

    gaps = [n for n in names
            if abs(seen["ok"][n] / total["ok"] - seen["defective"][n] / total["defective"]) >= 0.3]
    if gaps:
        print(f"\nLabels that lean toward one class: {', '.join(gaps)}. That is a weak hint, not an inspection: "
              "none of them says 'defective', and none would hold up as a pass/fail rule.")
    else:
        print("\nNo label separates good parts from defective ones. The general model sees the part, "
              "not the defect. That is the case for a custom model (step 2).")


if __name__ == "__main__":
    run(main)
