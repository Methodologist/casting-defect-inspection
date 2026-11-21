"""Step 1: off-the-shelf Rekognition (DetectLabels) on good and bad examples from the chosen dataset.

The point: a general model describes the object ("metal", "gear", "box") but has no idea what a defect is.
50 images, covered by the free tier (1,000 a month), otherwise about 5 cents.
"""
from collections import Counter

from common import client, dataset, pct, results_dir, run, splits, write_csv


def main():
    ds = dataset()
    pos, neg = ds["positive"], ds["negative"]
    rek = client("rekognition")
    images = splits()["baseline"]
    print(f"Sending {len(images)} held-out images of {ds['title']} to Rekognition DetectLabels...")
    seen = {label: Counter() for label in ds["labels"]}
    rows = []
    for i, (path, label) in enumerate(images, 1):
        resp = rek.detect_labels(Image={"Bytes": path.read_bytes()}, MaxLabels=10, MinConfidence=50)
        found = [(x["Name"], x["Confidence"]) for x in resp["Labels"]]
        seen[label].update({name for name, _ in found})
        rows.append({"file": path.name, "true_label": label,
                     "rekognition_labels": "; ".join(f"{n} {c:.0f}%" for n, c in found)})
        print(f"\r  {i}/{len(images)}", end="", flush=True)
    print()
    write_csv(results_dir() / "step1_baseline.csv", rows)

    total = {label: sum(1 for _, l in images if l == label) for label in ds["labels"]}
    names = sorted(set(seen[neg]) | set(seen[pos]), key=lambda n: -(seen[neg][n] + seen[pos][n]))
    print(f"\nHow often each label came back:\n{'Label':<28}{neg.capitalize():>10}{pos.capitalize():>11}")
    for n in names[:15]:
        print(f"{n:<28}{pct(seen[neg][n], total[neg]):>10}{pct(seen[pos][n], total[pos]):>11}")

    gaps = [n for n in names if abs(seen[neg][n] / total[neg] - seen[pos][n] / total[pos]) >= 0.3]
    if gaps:
        print(f"\nLabels that lean toward one class: {', '.join(gaps)}. That is a weak hint, not an inspection: "
              "none of them would hold up as a pass/fail rule.")
    else:
        print(f"\nNo label separates {ds['negative_plural']} from {ds['positive_plural']}. The general model sees "
              "the object, not the defect. That is the case for a custom model (step 2).")


if __name__ == "__main__":
    run(main)
