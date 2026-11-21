"""The train/test/held-out split: no leaks, no overlaps, same result every run.

Uses a tiny fake dataset written to a temporary folder. The "images" are just distinct bytes with a .jpg name,
which is all the split logic looks at, so no real data or AWS access is needed.
"""
import hashlib

import pytest

import common


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def use_dataset(monkeypatch, tmp_path, name, config):
    monkeypatch.setattr(common, "DATA", tmp_path)
    monkeypatch.setitem(common.DATASETS, name, dict(config, folder=name))
    monkeypatch.setattr(common, "settings", lambda: {"dataset": name})


GIVEN = {
    "title": "toy parts", "classes": {"bad_dir": "bad", "good_dir": "good"}, "split": "given",
    "train_per_class": 3, "test_per_class": 2, "holdout_per_class": 1, "baseline_per_class": 1,
}


@pytest.fixture
def given_dataset(monkeypatch, tmp_path):
    root = tmp_path / "toy_given"
    for folder in GIVEN["classes"]:
        for i in range(5):
            write(root / "train" / folder / f"tr_{i}.jpg", f"{folder}-train-{i}".encode())
        # Exactly as many unique test-folder images as test + held out need (2 + 1), so any copy that slipped
        # through would have to displace one of them.
        for i in range(3):
            write(root / "test" / folder / f"te_{i}.jpg", f"{folder}-test-{i}".encode())
        # The source dataset's quirk: training images also shipped in the test folder under other names.
        write(root / "test" / folder / "te_copy_a.jpg", f"{folder}-train-4".encode())
        write(root / "test" / folder / "te_copy_b.jpg", f"{folder}-train-3".encode())
        # An unsplit copy of the dataset (like casting_512x512) that must be ignored entirely.
        write(root / "casting_512x512" / folder / "extra.jpg", f"{folder}-unsplit".encode())
    use_dataset(monkeypatch, tmp_path, "toy_given", GIVEN)
    return root


def test_given_split_counts(given_dataset):
    s = common.splits()
    for label in ("bad", "good"):
        assert sum(l == label for _, l in s["train"]) == 3
        assert sum(l == label for _, l in s["test"]) == 2
        assert sum(l == label for _, l in s["holdout"]) == 1


def test_copies_of_training_images_never_reach_test_or_holdout(given_dataset):
    s = common.splits()
    # Compare against every image in the train folders, not only the ones sampled for training.
    train_folder = {digest(p) for p in given_dataset.glob("train/*/*.jpg")}
    for split in ("test", "holdout"):
        assert not {digest(p) for p, _ in s[split]} & train_folder
    for label in ("bad", "good"):
        assert {p.name for split in ("test", "holdout") for p, l in s[split] if l == label} == \
            {"te_0.jpg", "te_1.jpg", "te_2.jpg"}


def test_unique_drops_same_bytes_under_any_name_and_excluded_bytes(tmp_path):
    a = write(tmp_path / "a.jpg", b"one")
    b = write(tmp_path / "b.jpg", b"one")    # same bytes as a
    c = write(tmp_path / "c.jpg", b"two")
    d = write(tmp_path / "d.jpg", b"three")  # matches the exclude list
    assert common._unique([a, b, c, d], exclude={digest(d)}) == [a, c]


def test_piles_do_not_overlap(given_dataset):
    s = common.splits()
    train, test, holdout = ({p for p, _ in s[k]} for k in ("train", "test", "holdout"))
    assert not train & test and not train & holdout and not test & holdout


def test_baseline_images_come_from_holdout(given_dataset):
    s = common.splits()
    assert {p for p, _ in s["baseline"]} <= {p for p, _ in s["holdout"]}


def test_unsplit_folder_is_ignored(given_dataset):
    s = common.splits()
    assert all("casting_512x512" not in str(p) for split in s.values() for p, _ in split)


def test_split_is_the_same_every_run(given_dataset):
    assert common.splits() == common.splits()


POOLED = {
    "title": "toy cases", "classes": {"damaged": "damaged", "intact": "intact"}, "split": "pooled",
    "fractions": {"train": 0.6, "test": 0.2}, "baseline_per_class": 1,
}


@pytest.fixture
def pooled_dataset(monkeypatch, tmp_path):
    root = tmp_path / "toy_pooled"
    for label in POOLED["classes"]:
        for i in range(10):
            folder = ("train", "valid", "test")[i % 3]
            write(root / folder / label / f"{label}_{i}.jpg", f"{label}-{i}".encode())
        write(root / "valid" / label / "renamed_copy.jpg", f"{label}-0".encode())  # same bytes, new name
        write(root / "test" / label / f"{label}_1.jpg", f"{label}-1".encode())   # same name in two folders
    use_dataset(monkeypatch, tmp_path, "toy_pooled", POOLED)
    return root


def test_pooled_split_drops_duplicates_and_follows_fractions(pooled_dataset):
    s = common.splits()
    for label in ("damaged", "intact"):  # 10 unique images per class: 6 / 2 / 2
        assert sum(l == label for _, l in s["train"]) == 6
        assert sum(l == label for _, l in s["test"]) == 2
        assert sum(l == label for _, l in s["holdout"]) == 2


def test_pooled_piles_share_no_image_bytes(pooled_dataset):
    s = common.splits()
    train, test, holdout = ({digest(p) for p, _ in s[k]} for k in ("train", "test", "holdout"))
    assert not train & test and not train & holdout and not test & holdout


def test_missing_class_folder_is_reported(monkeypatch, tmp_path):
    write(tmp_path / "toy_given" / "train" / "bad_dir" / "a.jpg", b"a")
    use_dataset(monkeypatch, tmp_path, "toy_given", GIVEN)
    with pytest.raises(SystemExit, match="No images for 'good'"):
        common.splits()
