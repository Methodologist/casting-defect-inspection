"""Held-out scoring and the training manifest format."""
import json

from common import score
from step2_train import manifest_line


def rows(*pairs):
    return [{"true_label": t, "predicted": p} for t, p in pairs]


def test_counts_misses_and_false_rejects_separately():
    r = rows(("bad", "bad"), ("bad", "bad"), ("bad", "good"),   # 2 caught, 1 missed
             ("good", "good"), ("good", "bad"))                 # 1 passed, 1 falsely rejected
    s = score(r, "bad")
    assert (s["true_positive"], s["false_negative"], s["false_positive"], s["true_negative"]) == (2, 1, 1, 1)
    assert s["accuracy"] == 0.6
    assert s["precision"] == round(2 / 3, 3)
    assert s["recall"] == round(2 / 3, 3)


def test_no_prediction_counts_as_a_pass():
    # If the model returns no label, the part is not rejected: a defective part with no label is a miss.
    s = score(rows(("bad", "none"), ("good", "none")), "bad")
    assert s["false_negative"] == 1 and s["true_negative"] == 1


def test_perfect_and_empty_runs():
    assert score(rows(("bad", "bad"), ("good", "good")), "bad")["accuracy"] == 1.0
    assert score([], "bad") == {"true_positive": 0, "false_negative": 0, "false_positive": 0, "true_negative": 0,
                                "precision": 0, "recall": 0, "accuracy": 0}


def test_matches_the_published_packages_result():
    # 33 caught, 6 missed, 4 false rejects, 45 passed: the packages held-out result in the README.
    r = rows(*[("damaged", "damaged")] * 33, *[("damaged", "intact")] * 6,
             *[("intact", "damaged")] * 4, *[("intact", "intact")] * 45)
    assert score(r, "damaged")["accuracy"] == 0.886


def test_manifest_line_uses_ground_truth_classification_format():
    ds = {"name": "casting", "labels": ("defective", "ok")}
    line = json.loads(manifest_line(ds, "my-bucket", "images/casting/train/ok/a.jpeg", "ok", "2026-01-01T00:00:00"))
    assert line["source-ref"] == "s3://my-bucket/images/casting/train/ok/a.jpeg"
    assert line["quality"] == 1  # index of "ok" in the label list
    meta = line["quality-metadata"]
    assert meta["class-name"] == "ok"
    assert meta["type"] == "groundtruth/image-classification"
    assert meta["human-annotated"] == "yes"
