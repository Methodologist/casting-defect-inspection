"""Builds the static results site in site/ from results/<dataset>/ and the held-out images.

Only numbers and labels are copied out of the AWS result files, never ARNs, account IDs or bucket paths, and
the finished site is scanned for those before it is written. Images are copied byte-for-byte (unmodified), as
the casting dataset's CC BY-NC-ND license requires. site/ is git-ignored.
"""
import csv
import html
import json
import re
import shutil
import sys
import urllib.request
from datetime import date

import common
from common import DATASETS, ROOT

SITE = ROOT / "site"
GITHUB_USER = "Methodologist"
REPO_URL = f"https://github.com/{GITHUB_USER}/casting-defect-inspection"

CREDITS = {
    "casting": ('"casting product image data for quality inspection" by Ravirajsinh Dabhi, Kaggle',
                "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product",
                "CC BY-NC-ND 4.0", "https://creativecommons.org/licenses/by-nc-nd/4.0/"),
    "packages": ('"Damaged vs good packages", provided by a Roboflow user on Roboflow Universe',
                 "https://universe.roboflow.com/aadhavs-first-workspace/damaged-vs-good-packages",
                 "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
}

# Plain-language notes on the off-the-shelf baseline, written after reading each step 1 result.
BASELINE_NOTES = {
    "casting": ("The general model recognizes the object (coil, rotor, spoke) on nearly every image, good or "
                "defective. Only \"Hole\" leans toward defective parts, which fits blowhole and pinhole defects, "
                "but it would miss most defects and still flag good parts. It is a hint, not an inspection."),
    "packages": ("The labels that lean toward damaged (carton, cardboard, package delivery) describe the kind of "
                 "box, not damage: many intact photos are printed retail packaging, many damaged ones are plain "
                 "shipping cartons. That is a warning about this web-collected dataset. A model could score well "
                 "by learning \"brown carton\" instead of \"damage\", so its results below deserve skepticism, and "
                 "a production system would be trained on the customer's own cases, photographed in the cell."),
}


def github_profile():
    """The author's public GitHub name and photo, looked up when the site is built (falls back to the login)."""
    p = {"name": GITHUB_USER, "url": f"https://github.com/{GITHUB_USER}",
         "avatar": f"https://github.com/{GITHUB_USER}.png?size=96"}
    try:
        req = urllib.request.Request(f"https://api.github.com/users/{GITHUB_USER}",
                                     headers={"Accept": "application/vnd.github+json", "User-Agent": "build-site"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        p["name"] = data.get("name") or GITHUB_USER
        if data.get("avatar_url"):
            p["avatar"] = data["avatar_url"] + ("&" if "?" in data["avatar_url"] else "?") + "s=96"
    except (OSError, ValueError):
        print("Could not reach GitHub for the profile name; using the login instead.")
    return p


def author_badge(p):
    return (f'<a class="author" href="{esc(p["url"])}" rel="author">'
            f'<img src="{esc(p["avatar"])}" alt="" width="38" height="38">'
            f'<span><b>{esc(p["name"])}</b><small>github.com/{esc(GITHUB_USER)}</small></span></a>')


def esc(x):
    return html.escape(str(x), quote=True)


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def load(name):
    """Everything the page shows for one dataset, as plain numbers, labels and image paths."""
    common.settings = (lambda base: (lambda: dict(base, dataset=name, project=f"{name}-inspection")))(
        json.loads((ROOT / "settings.json").read_text()))
    ds = common.dataset()
    out_dir = ROOT / "results" / name
    d = {"name": name, "ds": ds, "baseline": None, "eval": None, "holdout": None}

    rows = read_csv(out_dir / "step1_baseline.csv")
    if rows:
        totals = {l: sum(r["true_label"] == l for r in rows) for l in ds["labels"]}
        counts = {}
        for r in rows:
            for item in filter(None, (x.strip() for x in r["rekognition_labels"].split(";"))):
                label = item.rsplit(" ", 1)[0]
                counts.setdefault(label, {l: 0 for l in ds["labels"]})[r["true_label"]] += 1
        top = sorted(counts, key=lambda n: -sum(counts[n].values()))[:12]
        d["baseline"] = {"n": len(rows), "rows": [
            {"label": n, **{l: counts[n][l] / totals[l] for l in ds["labels"]}} for n in top]}

    ev = read_json(out_dir / "step2_evaluation.json")
    if ev:
        agg = ev.get("AggregatedEvaluationResults", {})
        d["eval"] = {
            "images": ev.get("EvaluationDetails", {}).get("NumberOfTestingImages"),
            "f1": agg.get("F1Score"), "precision": agg.get("Precision"), "recall": agg.get("Recall"),
            "labels": [{"label": r.get("Label"), "images": r.get("NumberOfTestingImages"),
                        **{k.lower(): r.get("Metrics", {}).get(k) for k in ("Precision", "Recall", "F1Score")}}
                       for r in ev.get("LabelEvaluationResults", [])]}

    rows = read_csv(out_dir / "step3_holdout.csv")
    summary = read_json(out_dir / "step3_summary.json")
    if rows and summary:
        paths = {p.name: p for p, _ in common.splits()["holdout"]}
        images = []
        for r in rows:
            src = paths.get(r["file"])
            if src is None:
                continue
            rel = f"img/{name}/{src.name}"
            images.append({"src": rel, "file": src, "truth": r["true_label"], "pred": r["predicted"],
                           "conf": r["confidence"], "ok": r["correct"] == "yes"})
        d["holdout"] = {"summary": summary, "images": images}

    # Our own CNN (ResNet-18 + Grad-CAM), optional: a SageMaker run if one was fetched, else the local run.
    d["cnn"] = None
    for cnn_dir in (out_dir / "sagemaker", out_dir / "cnn"):
        preds, metrics = read_json(cnn_dir / "predictions.json"), read_json(cnn_dir / "metrics.json")
        if preds and metrics and d["holdout"]:
            break
    else:
        return d
    srcs = {im["file"].name: im["src"] for im in d["holdout"]["images"]}
    if {r["file"] for r in preds} != set(srcs):
        sys.exit(f"{cnn_dir} was not tested on the same held-out images as Rekognition. Re-run train_local_cnn.py.")
    rows = []
    for r in preds:
        heat_file = cnn_dir / r["heatmap"]
        rows.append({"src": srcs[r["file"]], "heat_file": heat_file, "heat": f"heat/{name}/{heat_file.name}",
                     "truth": r["true_label"], "pred": r["predicted"], "conf": r["confidence"], "ok": r["correct"],
                     "evidence": r.get("evidence", 0)})
    summary = {"images": metrics["heldout_images"], "true_positive": metrics["tp"],
               "false_negative": metrics["fn"], "false_positive": metrics["fp"], "true_negative": metrics["tn"],
               "accuracy": metrics["heldout_accuracy"], "recall": metrics["recall"],
               "precision": metrics["precision"]}
    layers = read_json(cnn_dir / "layers" / "layers.json")
    if layers and layers["file"] not in srcs:
        layers = None
    d["cnn"] = {"metrics": metrics, "rows": rows, "summary": summary, "dir": cnn_dir, "layers": layers}
    return d


def cnn_where(m):
    """Where our CNN was trained, in plain words."""
    if m.get("trained_on") == "Amazon SageMaker AI":
        return "in an Amazon SageMaker AI training job"
    return f'locally on an {m["hardware"]}' if m.get("hardware", "").startswith("NVIDIA") else "locally on a PC"


def pct(x, digits=0):
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def kpis(d, s):
    ds = d["ds"]
    if not s:
        return f'<p class="pending">Held-out test for {esc(ds["title"])} not run yet.</p>'
    pos_total = s["true_positive"] + s["false_negative"]
    neg_total = s["false_positive"] + s["true_negative"]
    tiles = [
        ("Held-out accuracy", pct(s["accuracy"], 1), f'{s["images"]} images never seen in training'),
        (f'{ds["positive_plural"].capitalize()} caught', f'{s["true_positive"]} of {pos_total}',
         f'{s["false_negative"]} missed (recall {pct(s["recall"], 1)})'),
        (f'{ds["negative_plural"].capitalize()} falsely rejected', f'{s["false_positive"]} of {neg_total}',
         f'precision {pct(s["precision"], 1)}'),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{esc(a)}</div><div class="kpi-value">{esc(b)}</div>'
        f'<div class="kpi-sub">{esc(c)}</div></div>' for a, b, c in tiles) + "</div>"


def baseline_table(d):
    ds, b = d["ds"], d["baseline"]
    if not b:
        return '<p class="pending">Baseline not run yet.</p>'
    neg, pos = ds["negative"], ds["positive"]
    head = (f'<tr><th>Label Rekognition returned</th><th>{esc(neg.capitalize())} images</th>'
            f'<th>{esc(pos.capitalize())} images</th></tr>')
    body = ""
    for r in b["rows"]:
        lean = abs(r[pos] - r[neg]) >= 0.25
        body += (f'<tr{" class=lean" if lean else ""}><td>{esc(r["label"])}</td>'
                 + "".join(f'<td><div class="bar"><span class="track"><span class="fill {cls}" '
                           f'style="width:{100 * r[l]:.0f}%"></span></span>'
                           f'<span class="val">{pct(r[l])}</span></div></td>'
                           for l, cls in ((neg, "neg"), (pos, "pos")))
                 + "</tr>")
    return (f'<div class="tablewrap"><table class="baseline"><thead>{head}</thead><tbody>{body}</tbody></table></div>'
            f'<p class="caption">Share of {b["n"]} held-out images on which off-the-shelf DetectLabels returned each '
            f'label. Highlighted rows differ by 25 points or more between classes.</p>'
            f'<p>{esc(BASELINE_NOTES.get(d["name"], ""))}</p>')


def confusion(d, s):
    if not s:
        return ""
    pos, neg = d["ds"]["positive"], d["ds"]["negative"]
    return (f'<div class="tablewrap"><table class="confusion"><thead><tr><th></th><th>Predicted {esc(pos)}</th>'
            f'<th>Predicted {esc(neg)}</th></tr></thead><tbody>'
            f'<tr><th>Actually {esc(pos)}</th><td class="good">{s["true_positive"]}<small>caught</small></td>'
            f'<td class="bad">{s["false_negative"]}<small>missed</small></td></tr>'
            f'<tr><th>Actually {esc(neg)}</th><td class="warn">{s["false_positive"]}<small>false reject</small></td>'
            f'<td class="good">{s["true_negative"]}<small>passed</small></td></tr></tbody></table></div>')


def eval_table(d):
    e = d["eval"]
    if not e:
        return '<p class="pending">Training not finished yet.</p>'
    rows = "".join(f'<tr><td>{esc(r["label"])}</td><td>{esc(r["images"])}</td><td>{pct(r["precision"], 1)}</td>'
                   f'<td>{pct(r["recall"], 1)}</td><td>{pct(r["f1score"], 1)}</td></tr>' for r in e["labels"])
    return (f'<div class="tablewrap"><table class="eval"><thead><tr><th>Label</th><th>Test images</th><th>Precision</th><th>Recall</th>'
            f'<th>F1</th></tr></thead><tbody>{rows}</tbody></table></div>'
            f'<p class="caption">AWS\'s own evaluation after training, on the {esc(e["images"])}-image test set '
            f'(overall F1 {pct(e["f1"], 1)}), scored at a confidence cut-off AWS picks for each label. The held-out '
            f'numbers above come from separate images the model never saw during training or evaluation, and simply '
            f'take its top guess for each image.</p>')


def heat_data(data):
    """Heatmap library items, classes interleaved like the verdict gallery."""
    items = []
    for d in data:
        if not d["cnn"]:
            continue
        by_class = [[r for r in d["cnn"]["rows"] if r["truth"] == label] for label in d["ds"]["labels"]]
        for n in range(max((len(c) for c in by_class), default=0)):
            for c in by_class:
                if n < len(c):
                    r = c[n]
                    items.append({"d": d["name"], "src": r["src"], "h": r["heat"], "t": r["truth"], "p": r["pred"],
                                  "c": r["conf"], "ok": r["ok"], "pos": r["truth"] == d["ds"]["positive"]})
    return items


def compare_table(data):
    """Both models side by side on the same held-out images. Shown whichever model is selected."""
    runs = [d for d in data if d["cnn"] and d["holdout"]]
    if not runs:
        return ""
    rows = ""
    for d in runs:
        r, c = d["holdout"]["summary"], d["cnn"]["summary"]
        pos_n, neg_n = r["true_positive"] + r["false_negative"], r["false_positive"] + r["true_negative"]
        rows += (f'<tr><td>{esc(d["ds"]["title"].capitalize())}<small>{r["images"]} held-out images</small></td>'
                 f'<td>{pct(r["accuracy"], 1)}</td><td>{pct(c["accuracy"], 1)}</td>'
                 f'<td>{r["false_negative"]} / {c["false_negative"]}<small>of {pos_n}</small></td>'
                 f'<td>{r["false_positive"]} / {c["false_positive"]}<small>of {neg_n}</small></td></tr>')
    return (f'<h3>Side by side</h3><div class="tablewrap"><table class="eval compare"><thead><tr><th>Dataset</th>'
            f'<th>Amazon Rekognition</th><th>Our own CNN</th><th>Missed (Rekognition / CNN)</th>'
            f'<th>False rejects (Rekognition / CNN)</th></tr></thead><tbody>{rows}</tbody></table></div>'
            f'<p class="caption">Accuracy on the same held-out images for both models, trained on the same training '
            f'images.</p>')


def cnn_details(d):
    c = d["cnn"]
    if not c:
        return '<p class="pending">Not run yet.</p>'
    m = c["metrics"]
    lat = m.get("latency_ms", {})
    speed = []
    if "gpu" in lat:
        speed.append(f'{lat["gpu"]:.1f} ms on the GPU')
    if "cpu" in lat:
        speed.append(f'{lat["cpu"]:.1f} ms on the CPU')
    rows = [("Trained", cnn_where(m)), ("Training images", f'{m.get("train_images", "–")}'),
            ("Training time", f'{m["train_seconds"]:.0f} seconds, {m["epochs"]} passes over the images'
             if "train_seconds" in m else "–"),
            ("Pass kept", f'{m.get("best_epoch", "–")} of {m["epochs"]} (best test-set accuracy, '
                          f'{pct(m["best_test_accuracy"], 1)})'),
            ("Time to classify one image", ", ".join(speed) or "–")]
    return ('<div class="tablewrap"><table class="eval"><tbody>' + "".join(
        f'<tr><th>{esc(a)}</th><td>{esc(b)}</td></tr>' for a, b in rows) + '</tbody></table></div>')


def slab(x, cy, w, h, depth, cls):
    """A 3D-looking block: front face plus top and side faces, for one stage of the network."""
    top, d = cy - h / 2, depth
    return (f'<path class="{cls} side" d="M{x + w},{top} l{d},{-d} v{h} l{-d},{d} z"/>'
            f'<path class="{cls} top" d="M{x},{top} l{d},{-d} h{w} l{-d},{d} z"/>'
            f'<rect class="{cls}" x="{x}" y="{top}" width="{w}" height="{h}"/>')


def network_diagram(data):
    """Our ResNet-18, drawn left to right with one real held-out casting and its Grad-CAM heatmap."""
    d = next((d for d in data if d["cnn"] and d["name"] == "casting"), None) or next((d for d in data if d["cnn"]), None)
    if not d:
        return ""
    pos, neg = d["ds"]["positive"], d["ds"]["negative"]
    hits = [r for r in d["cnn"]["rows"] if r["truth"] == pos and r["ok"]]
    if not hits:
        return ""
    ex = max(hits, key=lambda r: r["evidence"])
    lay = d["cnn"]["layers"]
    if lay:  # the same photo as the step-by-step cards below
        ex = next((r for r in hits if r["src"].endswith("/" + lay["file"])), ex)
    p_pos = float(ex["conf"]) if ex["pred"] == pos else 100 - float(ex["conf"])
    cy = 165
    stages = [  # (x, width, height, grid, filters, what it tends to respond to)
        (238, 14, 124, "56×56", 64, "edges, spots"),
        (318, 22, 94, "28×28", 128, "textures"),
        (406, 34, 66, "14×14", 256, "shapes, parts"),
        (510, 52, 40, "7×7", 512, "defect-like|patterns"),
    ]
    s = []
    arrow = lambda x1, x2: f'<path class="flow" d="M{x1},{cy} H{x2}" marker-end="url(#arr)"/>'
    # Input photo
    s.append(f'<image href="{esc(ex["src"])}" x="16" y="{cy - 62}" width="124" height="124" '
             f'preserveAspectRatio="xMidYMid slice"/><rect class="frame" x="16" y="{cy - 62}" width="124" height="124"/>')
    s.append(f'<text x="78" y="{cy + 82}" class="c b">Photo in</text>'
             f'<text x="78" y="{cy + 98}" class="c m">resized to 224 × 224</text>')
    s.append(arrow(144, 168))
    # First convolution
    s.append(slab(172, cy, 10, 140, 8, "stem"))
    s.append(f'<text x="182" y="{cy + 98}" class="c m">first filters</text>')
    s.append(arrow(194, 232))
    # Four stages
    prev_end = None
    for i, (x, w, h, grid, filt, what) in enumerate(stages, 1):
        if prev_end:
            s.append(arrow(prev_end, x - 4))
        s.append(slab(x, cy, w, h, 10, f"st{i}"))
        mid = x + w / 2 + 5
        s.append(f'<text x="{mid}" y="{cy - h / 2 - 18}" class="c b">Stage {i}</text>'
                 f'<text x="{mid}" y="{cy + h / 2 + 22}" class="c">{grid} · {filt}</text>'
                 + "".join(f'<text x="{mid}" y="{cy + h / 2 + 38 + 15 * k}" class="c m">{line}</text>'
                           for k, line in enumerate(what.split("|"))))
        prev_end = x + w + 14
    # Average pool and final layer
    s.append(arrow(prev_end, 600))
    s.append(f'<rect class="pool" x="604" y="{cy - 40}" width="12" height="80"/>'
             f'<text x="610" y="{cy - 52}" class="c b">Average</text>'
             f'<text x="610" y="{cy + 58}" class="c">512</text><text x="610" y="{cy + 74}" class="c m">numbers</text>')
    s.append(arrow(620, 648))
    s.append(f'<rect class="fc" x="652" y="{cy - 26}" width="40" height="52" rx="6"/>'
             f'<text x="672" y="{cy + 5}" class="c b inv">512→2</text>'
             f'<text x="672" y="{cy - 38}" class="c b">Final layer</text>')
    s.append(arrow(696, 722))
    # Verdict bars
    for j, (label, p) in enumerate(((pos, p_pos), (neg, 100 - p_pos))):
        y = cy - 34 + j * 40
        s.append(f'<text x="728" y="{y}" class="m">{esc(label)}</text>'
                 f'<rect class="track" x="728" y="{y + 6}" width="120" height="14" rx="3"/>'
                 f'<rect class="{"bar-pos" if j == 0 else "bar-neg"}" x="728" y="{y + 6}" '
                 f'width="{max(1.5, 1.2 * p):.1f}" height="14" rx="3"/>'
                 f'<text x="856" y="{y + 18}" class="b">{p:.0f}%</text>')
    s.append(f'<text x="728" y="{cy - 56}" class="b">Verdict</text>')
    # Grad-CAM branch from stage 4 to the heatmap
    x4 = stages[-1][0] + stages[-1][2] / 2
    s.append(f'<path class="cam" d="M{x4},{cy + 80} V{cy + 140} H{972} V{cy + 66}" marker-end="url(#arr2)"/>'
             f'<text x="{(x4 + 972) / 2}" y="{cy + 134}" class="c cam-t">Grad-CAM: which spots in stage 4 pushed '
             f'the verdict toward "{esc(pos)}"</text>')
    s.append(f'<image href="{esc(ex["src"])}" x="910" y="{cy - 62}" width="124" height="124" '
             f'preserveAspectRatio="xMidYMid slice"/>'
             f'<image href="{esc(ex["heat"])}" x="910" y="{cy - 62}" width="124" height="124" '
             f'preserveAspectRatio="xMidYMid slice"/><rect class="frame" x="910" y="{cy - 62}" width="124" height="124"/>'
             f'<text x="972" y="{cy - 74}" class="c b">Where it looked</text>')
    # Brackets: what was pre-trained vs new
    s.append('<path class="brk" d="M172,36 v-8 H572 v8"/>'
             '<text x="372" y="18" class="c m">Pre-trained on ImageNet (everyday photos), then fine-tuned on our photos</text>'
             '<path class="brk" d="M652,36 v-8 H692 v8"/><text x="672" y="18" class="c m">New for our task</text>')
    svg = ('<svg class="netfig" viewBox="0 0 1050 330" role="img" aria-labelledby="nettitle netdesc">'
           '<title id="nettitle">Our ResNet-18 network, from photo to verdict and heatmap</title>'
           f'<desc id="netdesc">A held-out {esc(d["ds"]["title"][:-1])} photo enters the network, passes a first set of '
           'filters and four stages that shrink the image from 56 by 56 to 7 by 7 while growing from 64 to 512 '
           f'filters, is averaged into 512 numbers, and a final layer outputs {esc(pos)} {p_pos:.0f} percent. '
           'Grad-CAM reads stage 4 to draw the heatmap of where the model looked.</desc>'
           '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker>'
           '<marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="camhead"/></marker></defs>'
           + "".join(s) + '</svg>')
    return (f'<h3>Inside our network</h3><p class="sub">The shape of the network at a glance, following one real '
            f'held-out {esc(d["ds"]["title"][:-1])}. At each stage the image gets smaller while the number of filters '
            'grows: early stages respond to simple things like edges and spots, later ones to larger patterns. Step by '
            'step below, each stage shows what it actually computed for this photo.</p>'
            f'<div class="netwrap">{svg}</div>'
            '<p class="caption">Block height shows the image size at each stage (56×56 down to 7×7); block width, the '
            'number of filters (64 up to 512). ResNet-18: 18 layers with learned weights (1 first layer, 16 in the four '
            'stages, 1 final layer).</p>')


# (stage key, title, learned numbers, what it does, what the pictures show). The "Here:" notes describe the feature
# maps drawn for the current casting example (results/casting/cnn/layers); re-check them if the model is retrained.
LAYER_STEPS = [
    ("maxpool", "First filters", "9.5 thousand",
     "64 simple detectors slide over the photo, each looking for one basic thing: an edge at some angle, a "
     "light-to-dark change, a spot. The image is then halved twice in size.",
     "Here: overall brightness and the outlines of the rings."),
    ("layer1", "Stage 1", "148 thousand",
     "Combines those edges into slightly richer patterns: corners, short curves, fine texture.",
     "Here: fine edges and the machining marks on the metal."),
    ("layer2", "Stage 2", "526 thousand",
     "Halves the image again and doubles the detectors to 128. Typically responds to surface texture: smooth "
     "metal versus rough or pitted.",
     "Here: fewer, sparser responses. Most of the part looks normal."),
    ("layer3", "Stage 3", "2.1 million",
     "Halves again, 256 detectors. Each spot now covers a larger area, so it can respond to larger shapes: the "
     "rim, the hub, a dent.",
     "Here: single bright spots, including one on the notch in the rim."),
    ("layer4", "Stage 4", "8.4 million",
     "The last and largest stage: 512 detectors on a 7×7 grid. After our training, some of them respond to "
     "defect-like patterns. Each of the 49 spots covers a big chunk of the photo.",
     "These four are the maps that pushed hardest toward \"{pos}\": all lit up on the damaged side."),
]


def network_cards(data):
    """Inside our network: one real held-out image, step by step, with the real feature maps from each stage."""
    d = next((d for d in data if d["cnn"] and d["cnn"]["layers"] and d["name"] == "casting"), None) or \
        next((d for d in data if d["cnn"] and d["cnn"]["layers"]), None)
    if not d:
        return ""
    lay, rows = d["cnn"]["layers"], d["cnn"]["rows"]
    pos, neg = lay["positive"], lay["negative"]
    ex = next(r for r in rows if r["src"].endswith("/" + lay["file"]))
    thing = d["ds"]["title"][:-1]  # "cast pump impeller"
    pretrained = '<span class="ptag">pre-trained, then fine-tuned</span>'
    cards = [f'<li class="pcard"><div class="ph"><span class="pn">1</span><b>Photo in</b></div>'
             f'<img src="{esc(ex["src"])}" alt="The held-out {esc(thing)} photo" loading="lazy">'
             f'<div class="pdim">224 × 224 pixels · 3 colors</div>'
             f'<p>A held-out photo the network never trained on, resized, with its brightness values adjusted to the '
             f'range the network learned from.</p></li>']
    for n, (key, title, weights, does, shows) in enumerate(LAYER_STEPS, 2):
        st = lay["stages"][key]
        cards.append(
            f'<li class="pcard"><div class="ph"><span class="pn">{n}</span><b>{esc(title)}</b>{pretrained}</div>'
            f'<img class="maps" src="net/{d["name"]}/{key}.png" alt="Four of the {st["maps"]} feature maps from '
            f'{esc(title.lower())}, each {st["size"]} by {st["size"]}" loading="lazy">'
            f'<div class="pdim">{st["size"]}×{st["size"]} grid · {st["maps"]} maps · {weights} learned numbers</div>'
            f'<p>{esc(does)}</p><p class="pshow">{esc(shows.format(pos=pos))}</p></li>')
    p_pos = 100 * lay["probs"][pos]
    cards.append(
        f'<li class="pcard"><div class="ph"><span class="pn">7</span><b>Average, then vote</b>'
        f'<span class="ptag new">new for our task</span></div>'
        f'<img class="maps" src="net/{d["name"]}/pooled.png" alt="512 averaged values, one square each" '
        f'loading="lazy"><div class="pdim">512 numbers → 2 scores · 1,026 learned numbers</div>'
        f'<p>Each of stage 4\'s 512 maps is averaged into one number: how much of that pattern is in the photo, '
        f'anywhere (one square each above). The final layer is a weighted vote: each number pushes toward '
        f'"{esc(pos)}" or "{esc(neg)}".</p><p class="pshow">This layer replaced the original one, which chose among '
        f'1,000 everyday categories.</p></li>')
    cards.append(
        f'<li class="pcard"><div class="ph"><span class="pn">8</span><b>Verdict and where it looked</b></div>'
        f'<div class="pframe"><img src="{esc(ex["src"])}" alt="" loading="lazy">'
        f'<img class="pheat" src="{esc(ex["heat"])}" alt="Heatmap over the photo" loading="lazy"></div>'
        f'<div class="pbars"><span>{esc(pos)}</span><span class="pbar"><span class="pos" style="width:{p_pos:.1f}%">'
        f'</span></span><b>{p_pos:.0f}%</b><span>{esc(neg)}</span><span class="pbar"><span class="neg" '
        f'style="width:{100 - p_pos:.1f}%"></span></span><b>{100 - p_pos:.0f}%</b></div>'
        f'<p>Grad-CAM asks stage 4 which spots pushed the score toward "{esc(pos)}", and stretches that 7×7 '
        f'answer over the photo. That is why the heat is a soft blob, not an outline.</p></li>')
    inside = [
        ("3×3 filters", "the pattern detectors: small grids of learned numbers that slide over the maps"),
        ("Batch normalization", "rescales the numbers after each filter layer so training stays stable"),
        ("ReLU", "keeps positive responses (\"pattern here\") and sets the rest to zero"),
        ("Shortcut", "adds each block's input back to its output, so a block only learns a correction. The "
                     "\"Res\" in ResNet, and what lets deep networks train well"),
    ]
    return (
        f'<h3 class="h-step">Step by step</h3><p class="sub">The same {esc(thing)}, start to finish. The pictures '
        f'are real: what our trained network computed for this photo at each stage. A stage produces one '
        f'<i>feature map</i> per detector (bright = strong response), and four are shown for each. They are drawn '
        f'at their true resolution, so they get blockier as the network shrinks the image and looks for bigger '
        f'patterns.</p>'
        f'<ol class="pipe">{"".join(cards)}</ol>'
        f'<div class="inside"><b>Inside each stage:</b> two blocks, each with ' + "".join(
            f'<span class="chip"><b>{esc(a)}</b> {esc(b)}</span>' for a, b in inside) + '</div>'
        f'<p class="caption">ResNet-18: 18 layers with learned weights, about 11.2 million numbers in all. Training '
        f'started from weights learned on ImageNet (everyday photos), so the early stages already knew edges and '
        f'textures; all 11.2 million numbers were then adjusted on our photos, the deepest stages and the new final '
        f'layer the most. The stage descriptions are the typical roles in networks like this; the pictures are '
        f'what this one actually did. Stages 1 to 3 show the four maps with the most going on; stage 4 shows the four '
        f'that added most to the heatmap.</p>')


def edge_note(data):
    """One measured speed line for the robot-cell section, if our CNN has run."""
    for d in data:
        lat = (d["cnn"] or {}).get("metrics", {}).get("latency_ms", {})
        if lat:
            parts = [f'{lat["gpu"]:.0f} ms with a {d["cnn"]["metrics"]["hardware"]}' if "gpu" in lat else "",
                     f'{lat["cpu"]:.0f} ms on the CPU alone' if "cpu" in lat else ""]
            return (' <i>Measured here:</i> our own CNN classifies one image in about '
                    + " and ".join(p for p in parts if p) + ', with no network involved.')
    return ""


def gallery_data(data):
    """Gallery items with the classes interleaved (good, bad, good, bad...) so no screen shows only one class."""
    items = []
    for d in data:
        if not d["holdout"]:
            continue
        by_class = [[im for im in d["holdout"]["images"] if im["truth"] == label] for label in d["ds"]["labels"]]
        for n in range(max(len(c) for c in by_class)):
            for c in by_class:
                if n < len(c):
                    im = c[n]
                    items.append({"d": d["name"], "src": im["src"], "t": im["truth"], "p": im["pred"],
                                  "c": im["conf"], "ok": im["ok"], "pos": im["truth"] == d["ds"]["positive"]})
    return items


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Automated Visual Inspection</title>
<meta name="description" content="A pass/fail vision inspection pipeline on Amazon Rekognition Custom Labels, tested on cast impellers and shipping cases.">
<style>
:root{--bg:#eef1f5;--surface:#fff;--ink:#0b0f14;--muted:#3a4350;--line:#b9c1cc;--accent:#0b4fb3;
--hero:#0b1f3a;--hero-ink:#fff;--hero-muted:#c9d6ea;--hero-link:#9cc3ff;
--good:#0d6b37;--good-bg:#cdeed9;--bad:#a3140b;--bad-bg:#fbd3cf;--warn:#7a4500;--warn-bg:#fde3b3;
--neg:#4a5666;--pos:#0b5fe0;--track:#dfe4ea;--lean:#ffe9a8}
@media (prefers-color-scheme:dark){:root{--bg:#07090c;--surface:#12161c;--ink:#f4f6f9;--muted:#c3cbd6;
--line:#3a4452;--accent:#7fb0ff;--hero:#000;--hero-ink:#fff;--hero-muted:#b8c4d6;--hero-link:#9cc3ff;
--good:#6be39c;--good-bg:#0d3320;--bad:#ff8f80;--bad-bg:#45140f;--warn:#ffc266;--warn-bg:#3d2a08;
--neg:#8e9aab;--pos:#5b9bff;--track:#232a33;--lean:#3a300c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent);font-weight:600}
.wrap{max-width:1040px;margin:0 auto;padding:0 16px}
header{padding:52px 0 32px;background:var(--hero);color:var(--hero-ink);border-bottom:4px solid var(--pos)}
header a{color:var(--hero-link)}
h1{font-size:clamp(28px,4vw,42px);line-height:1.12;margin:0 0 10px;letter-spacing:-.015em}
.lede{color:var(--hero-muted);font-size:18px;max-width:760px;margin:0}
.meta{margin-top:18px;font-size:14px;color:var(--hero-muted)}
section{padding:40px 0;border-bottom:1px solid var(--line)}
section:nth-of-type(even){background:var(--surface)}
h2{font-size:24px;margin:0 0 6px}
h3{font-size:18px;margin:28px 0 10px}
.sub{color:var(--muted);margin:0 0 18px;max-width:760px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 18px}
.tabs button{font:inherit;font-size:14px;padding:6px 14px;border-radius:999px;border:1px solid var(--line);
background:var(--surface);color:var(--ink);cursor:pointer}
.tabs button{font-weight:600;border-width:2px}
.tabs button[aria-pressed=true]{background:var(--ink);border-color:var(--ink);color:var(--surface)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:8px 0 4px}
.kpi{background:var(--surface);border:1px solid var(--line);border-left:5px solid var(--pos);border-radius:10px;padding:16px}
.kpi-label{font-size:13px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.kpi-value{font-size:32px;font-weight:750;margin:4px 0 2px;font-variant-numeric:tabular-nums}
.kpi-sub{font-size:13px;color:var(--muted)}
.ds-block{margin-bottom:28px}
.ds-title{font-weight:650;font-size:18px;margin:0 0 6px}
table{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
thead th{font-size:13px;color:var(--ink);font-weight:700;background:var(--track)}
tr.lean td{background:var(--lean)}
.bar{display:flex;align-items:center;gap:8px;min-width:120px}
.track{position:relative;flex:1;height:18px;background:var(--track);border:1px solid var(--line);border-radius:4px}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.fill.neg{background:var(--neg)}.fill.pos{background:var(--pos)}
.val{width:3.2em;text-align:right;font-size:13px;font-weight:700;font-variant-numeric:tabular-nums}
.caption{font-size:13px;color:var(--muted);margin:8px 0 12px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}}
.confusion td{font-size:28px;font-weight:750;text-align:center}
.confusion td small{display:block;font-size:12px;font-weight:500;color:var(--muted)}
.confusion td.good{background:var(--good-bg);color:var(--good)}
.confusion td.bad{background:var(--bad-bg);color:var(--bad)}
.confusion td.warn{background:var(--warn-bg);color:var(--warn)}
.pending{color:var(--muted);font-style:italic}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
.card img{display:block;width:100%;aspect-ratio:1;object-fit:cover;background:var(--bg)}
.card .info{padding:8px 10px}
.card .verdict{font-weight:700}
.card.ok{border-top:4px solid var(--good)}.card.ok .verdict{color:var(--good)}
.card.miss{border:2px solid var(--bad);border-top-width:4px}.card.miss .verdict{color:var(--bad)}
.card .truth{color:var(--muted)}
.card .open{display:block;width:100%;padding:0;border:0;background:none;cursor:zoom-in}
.card .frame{position:relative;display:block}
.card .frame .heat{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;pointer-events:none;background:transparent}
.vframe{position:relative;line-height:0}
.viewer .vheat{position:absolute;inset:0;background:transparent;pointer-events:none}
.legend{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--muted);margin:0 0 12px}
.legend .ramp{width:140px;height:12px;border-radius:3px;border:1px solid var(--line);background:linear-gradient(90deg,rgba(255,230,0,.25),rgba(255,115,0,.6),rgba(255,0,0,.85))}
.card .open:focus-visible{outline:3px solid var(--pos);outline-offset:-3px}
dialog.viewer{position:fixed;inset:0;margin:0;border:0;padding:0;background:transparent;max-width:none;max-height:none;
width:100%;height:100%;color:#fff;overflow:hidden}
dialog.viewer::backdrop{background:rgba(5,8,12,.94)}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:10px}
@media (max-width:600px){.bar{min-width:84px}th,td{padding:7px 8px}.kpi-value{font-size:28px}}
.viewer .stage{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:56px 64px}
.viewer img{width:min(82vw,78vh);height:min(82vw,78vh);object-fit:contain;background:#000;border-radius:8px;image-rendering:auto}
.viewer .vcap{font-size:16px;text-align:center}
.viewer .vcap b{font-size:18px}
.viewer .vcap .vok{color:#6be39c}.viewer .vcap .vmiss{color:#ff8f80}
.viewer .vmeta{color:#c3cbd6;font-size:13px}
.viewer button{position:absolute;font:inherit;font-size:26px;line-height:1;color:#fff;background:rgba(255,255,255,.12);
border:1px solid rgba(255,255,255,.35);border-radius:999px;width:48px;height:48px;cursor:pointer}
.viewer button:hover,.viewer button:focus-visible{background:rgba(255,255,255,.28)}
.viewer .vclose{top:14px;right:14px}.viewer .vprev{left:12px;top:50%}.viewer .vnext{right:12px;top:50%}
@media (max-width:600px){.viewer .stage{padding:64px 12px}.viewer .vprev,.viewer .vnext{top:auto;bottom:14px}}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 20px}
.step{background:var(--surface);border:1px solid var(--line);border-top:4px solid var(--pos);border-radius:10px;padding:14px;position:relative}
.step b{display:block;margin-bottom:4px}
.step span{font-size:13px;color:var(--muted)}
ul.points{padding-left:20px;max-width:800px}ul.points li{margin:6px 0}
footer{padding:28px 0 48px;font-size:13px;color:var(--muted)}
.count{color:var(--muted);font-size:14px;margin:0 0 12px}
body[data-model=rek] .m-cnn,body[data-model=cnn] .m-rek{display:none}
.modelbar{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:2px solid var(--line);
box-shadow:0 2px 8px rgba(0,0,0,.08)}
.modelbar .wrap{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding-top:10px;padding-bottom:10px}
.modelbar .label{font-weight:700;font-size:14px}
.modelbar .tabs{margin:0}
.modelbar .tabs button{font-size:15px;padding:8px 16px}
.modelbar .what{font-size:13px;color:var(--muted);flex-basis:100%}
@media (max-width:600px){.modelbar .what{display:none}.modelbar .wrap{padding-top:8px;padding-bottom:8px}}
.compare td small{display:block;font-size:12px;color:var(--muted)}
.compare td:nth-child(2),.compare td:nth-child(3){font-weight:700}
.netwrap{overflow-x:auto;-webkit-overflow-scrolling:touch;background:var(--surface);border:1px solid var(--line);
border-radius:10px;padding:12px 8px}
.netfig{display:block;width:100%;min-width:760px;height:auto}
.netfig text{font:13px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;fill:var(--ink)}
.netfig .c{text-anchor:middle}.netfig .b{font-weight:700}.netfig .m{fill:var(--muted);font-size:12px}
.netfig .inv{fill:#fff}
.netfig .frame{fill:none;stroke:var(--line);stroke-width:1.5}
.netfig .flow{stroke:var(--muted);stroke-width:2;fill:none}.netfig .arrowhead{fill:var(--muted)}
.netfig .cam{stroke:#e8590c;stroke-width:2;stroke-dasharray:6 4;fill:none}.netfig .camhead{fill:#e8590c}
.netfig .cam-t{fill:#c2410c;font-weight:600;font-size:12px}
.netfig .brk{fill:none;stroke:var(--muted);stroke-width:1.5}
.netfig .stem,.netfig .st1,.netfig .st2,.netfig .st3,.netfig .st4{stroke:var(--hero);stroke-width:1}
.netfig .stem{fill:#9db8e8}.netfig .st1{fill:#7fa6ea}.netfig .st2{fill:#5a8de6}.netfig .st3{fill:#3874e0}
.netfig .st4{fill:#0b5fe0}
.netfig .top{filter:brightness(1.18)}.netfig .side{filter:brightness(.78)}
.netfig .pool{fill:var(--neg)}.netfig .fc{fill:var(--hero);stroke:var(--line)}
.netfig .track{fill:var(--track);stroke:var(--line)}.netfig .bar-pos{fill:var(--bad)}.netfig .bar-neg{fill:var(--neg)}
@media (prefers-color-scheme:dark){.netfig .cam-t{fill:#ff9f5a}.netfig .fc{fill:#1d3a66}}
.pipe{list-style:none;margin:8px 0 18px;padding:0;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px 26px}
@media (max-width:900px){.pipe{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:520px){.pipe{grid-template-columns:1fr}}
.pcard{position:relative;background:var(--surface);border:1px solid var(--line);border-top:4px solid var(--pos);
border-radius:10px;padding:12px;font-size:14px}
.pcard::after{content:"→";position:absolute;right:-21px;top:118px;font-size:20px;font-weight:700;color:var(--muted)}
.pcard:nth-child(4n)::after,.pcard:last-child::after{content:none}
@media (max-width:900px){.pcard:nth-child(2n)::after{content:none}.pcard:nth-child(4n+1)::after,.pcard:nth-child(4n+3)::after{content:"→"}}
@media (max-width:520px){.pcard::after,.pcard:nth-child(4n+1)::after,.pcard:nth-child(4n+3)::after{content:none}}
.pcard .ph{display:flex;align-items:center;flex-wrap:wrap;gap:4px 8px;margin-bottom:8px}
.pcard .pn{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:999px;background:var(--ink);
color:var(--surface);font-size:13px;font-weight:700}
.pcard .ptag{font-size:11px;font-weight:600;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:1px 7px}
.pcard .ptag.new{color:var(--warn);border-color:var(--warn);background:var(--warn-bg)}
.pcard img{display:block;width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:var(--bg)}
.pcard img.maps{object-fit:contain;background:transparent;image-rendering:pixelated}
.pcard img[src$="pooled.png"]{aspect-ratio:2}
.pcard .pdim{font-size:12px;font-weight:700;margin:8px 0 4px;font-variant-numeric:tabular-nums}
.pcard p{margin:4px 0;color:var(--ink);line-height:1.45}
.pcard p.pshow{color:var(--muted);font-size:13px}
.pframe{position:relative}.pframe .pheat{position:absolute;inset:0;background:transparent}
.pbars{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:4px 8px;font-size:13px;margin:8px 0}
.pbar{height:12px;background:var(--track);border:1px solid var(--line);border-radius:3px;overflow:hidden}
.pbar span{display:block;height:100%}.pbar .pos{background:var(--bad)}.pbar .neg{background:var(--neg)}
.inside{font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:4px 0 6px}
.inside .chip{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:6px 10px;color:var(--ink);
flex:1 1 220px}
.inside .chip b{display:block;font-size:13px}
.author{display:inline-flex;align-items:center;gap:10px;margin:0 0 18px;padding:5px 14px 5px 5px;border-radius:999px;
background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.28);color:var(--hero-ink);text-decoration:none;
font-weight:400;line-height:1.2}
.author:hover,.author:focus-visible{background:rgba(255,255,255,.18)}
.author img{width:38px;height:38px;border-radius:999px;border:2px solid rgba(255,255,255,.7);background:#fff}
.author b{display:block;font-size:15px}.author small{display:block;font-size:12px;color:var(--hero-muted)}
.badge{display:inline-block;font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;border:1px solid var(--line);
background:var(--track);color:var(--ink);vertical-align:middle;margin-left:6px}
</style>
</head>
<body data-model="rek">
<header><div class="wrap">
__AUTHOR__
<h1>__TITLE__</h1>
<p class="lede">A pass/fail camera check of the kind a robot cell runs: train a custom vision model on labeled
photos, then measure it on images it has never seen, counting missed defects and false rejects separately.</p>
<p class="meta">Two datasets: cast pump impellers (main run) and damaged vs intact shipping cases ·
<a href="__REPO__">Source code on GitHub</a> · Built __DATE__</p>
</div></header>
__MODELBAR__
<section><div class="wrap">
<h2>Results on held-out images <span class="badge m-rek">Amazon Rekognition</span><span class="badge m-cnn">Our own CNN</span></h2>
<p class="sub">Each model was tested on photos kept aside from training and from AWS's own evaluation, so these
numbers show how it does on parts it has truly never seen.</p>
__KPIS__
__COMPARE__
</div></section>

<section><div class="wrap">
<h2>Why off-the-shelf AI isn't enough</h2>
<p class="sub">Step 1 sent held-out images to Rekognition's general-purpose label detector. It describes what it
sees, but it has no concept of a defect.</p>
__BASELINE__
</div></section>

<section><div class="wrap">
<h2>The custom model <span class="badge m-rek">Amazon Rekognition</span><span class="badge m-cnn">Our own CNN</span></h2>
<div class="m-rek">
<p class="sub">Rekognition Custom Labels, trained on labeled images, then tested on separate held-out images.
AWS chooses and tunes the model itself and does not publish its design. On a line, a missed defect and a false
reject cost different things, so they are shown separately.</p>
__MODEL__
</div>
__CNN_MODEL__
</div></section>

<section><div class="wrap">
<h2>Every held-out image <span class="badge m-rek">Amazon Rekognition</span><span class="badge m-cnn">Our own CNN</span></h2>
<div class="m-rek">
<p class="sub">The model's verdict and confidence for each image it was tested on. Images are shown unmodified;
the verdict sits beside them.</p>
<div class="tabs" id="gtabs" role="group" aria-label="Dataset"></div>
<div class="tabs" id="gfilters" role="group" aria-label="Show"></div>
<p class="count" id="gcount"></p>
<div class="gallery" id="gallery"></div>
</div>
<div class="m-cnn">
<p class="sub">Our CNN's verdict for each held-out image, with a Grad-CAM heatmap: the parts of the photo that
pushed it toward __POS_WORDS__. The heat is a separate layer over the unmodified photo; switch it off to compare.
It shows where the model found evidence, not a traced outline of the defect.</p>
<p class="legend"><span>Less evidence</span><span class="ramp"></span><span>More evidence for a defect</span></p>
<div class="tabs" id="htabs" role="group" aria-label="Dataset"></div>
<div class="tabs" id="hfilters" role="group" aria-label="Show"></div>
<p class="count" id="hcount"></p>
<div class="gallery" id="hgallery"></div>
</div>
</div></section>

<section><div class="wrap">
<h2>From demo to robot cell</h2>
<p class="sub">How the same approach would fit a production inspection station.</p>
<div class="flow">
<div class="step"><b>1. Capture</b><span>Fixed camera and lighting at the station, triggered as each part or case arrives.</span></div>
<div class="step"><b>2. Classify</b><span>The trained model returns pass or fail with a confidence score.</span></div>
<div class="step"><b>3. Signal</b><span>The verdict goes to the PLC or robot controller.</span></div>
<div class="step"><b>4. Act</b><span>Pass continues down the line; fail goes to a reject chute or a review lane.</span></div>
</div>
<ul class="points">
<li><b>Train in the cloud, decide on the line.</b> A cloud call adds network delay, often a few hundred
milliseconds, which can be too slow for a fast conveyor. The cloud is a quick, cheap way to collect data, train and
prove a model; production checks typically run on vision built into the cell.__EDGE_NOTE__</li>
<li><b>Running cost.</b> A hosted Custom Labels model bills about $4 an hour while running, roughly $35,000 a year
around the clock. Fine for a pilot, but a reason to run inference at the edge in production.</li>
<li><b>Train on the customer's parts.</b> Accuracy depends on photos that match the real station: same camera,
angle and lighting. The public datasets here prove the method; a production model is retrained on the customer's own
images. (The casting dataset's license is also non-commercial.)</li>
<li><b>Tune the threshold to the cost of a miss.</b> Lowering the confidence needed to reject catches more defects
at the price of more false rejects. Where a missed defect reaches a customer, that trade is usually worth it.</li>
</ul>
<p><a href="__REPO__/blob/main/docs/factory-guide.md">From demo to factory: the simplest working setup</a>, a
step-by-step guide from camera station to labeling to PLC.</p>
</div></section>

<section><div class="wrap">
<h2>Method and credits</h2>
<ul class="points">
<li>Casting: 1,000 training, 200 test and 100 held-out images, balanced between defective and ok. Test and
held-out images that were byte-for-byte copies of training images (the source dataset contains a few) were removed.</li>
<li>Limitation: the casting dataset's 7,348 photos were made by its author from 1,300 originals by rotating,
flipping and shifting them. A rotated copy of a training part can still appear among the held-out images, so the
100% result may be partly flattered. A fair production test uses parts the model has never seen at any angle.</li>
<li>Packages: 581 photos pooled and split 65/20/15 into train, test and held out with a fixed seed, after removing
exact duplicates.</li>
__CNN_METHOD__
<li>Costs for the whole project: a few dollars of AWS credits for training and testing. Models are stopped when not
in use.</li>
__CREDITS__
</ul>
</div></section>

<footer><div class="wrap">__BUILT_WITH__
Images remain the property of their creators and are used under the licenses listed above.</div></footer>

<dialog class="viewer" id="viewer" aria-label="Enlarged image">
<div class="stage" id="vstage">
<div class="vframe"><img id="vimg" alt=""><img id="vheat" class="vheat" alt="" hidden></div>
<div class="vcap" id="vcap"></div>
</div>
<button class="vclose" id="vclose" aria-label="Close">✕</button>
<button class="vprev" id="vprev" aria-label="Previous image">‹</button>
<button class="vnext" id="vnext" aria-label="Next image">›</button>
</dialog>

<script>
const ITEMS = __ITEMS__, NAMES = __NAMES__, HEAT = __HEAT_ITEMS__;
const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;
function button(label, pressed, onClick) {
  const b = document.createElement("button"); b.type = "button"; b.textContent = label;
  b.setAttribute("aria-pressed", pressed); b.onclick = onClick; return b;
}

// One gallery = dataset tabs + filters + grid. Used for the verdict gallery and the heatmap library.
function makeGallery(items, ids, withHeat) {
  const tabs = document.getElementById(ids.tabs), filters = document.getElementById(ids.filters),
        grid = document.getElementById(ids.grid), count = document.getElementById(ids.count);
  if (!grid) return null;
  const names = NAMES.filter(n => items.some(i => i.d === n[0]));
  const st = {d: names.length ? names[0][0] : null, show: "all", heat: true};
  let shown = [];
  function render() {
    tabs.replaceChildren(...names.map(([key, title]) =>
      button(title, st.d === key, () => { st.d = key; render(); })));
    const [, , posLabel, negLabel] = names.find(n => n[0] === st.d) || [];
    const opts = [["all", "All"], ["pos", `Actually ${posLabel}`], ["neg", `Actually ${negLabel}`],
                  ["wrong", "Mistakes only"]];
    filters.replaceChildren(...opts.map(([k, label]) => button(label, st.show === k, () => { st.show = k; render(); })));
    if (withHeat) filters.append(button(st.heat ? "Heatmap: on" : "Heatmap: off", st.heat,
                                        () => { st.heat = !st.heat; render(); }));
    const all = items.filter(i => i.d === st.d);
    shown = all.filter(i => st.show === "all" || (st.show === "pos" && i.pos) ||
                            (st.show === "neg" && !i.pos) || (st.show === "wrong" && !i.ok));
    const nPos = all.filter(i => i.pos).length;
    count.textContent = all.length
      ? `Showing ${shown.length} of ${all.length} · ${cap(posLabel)}: ${nPos}, ${negLabel}: ${all.length - nPos} · ` +
        `${all.filter(i => !i.ok).length} wrong calls`
      : "Not run yet.";
    grid.replaceChildren(...shown.map((i, n) => {
      const c = document.createElement("figure"); c.className = "card " + (i.ok ? "ok" : "miss"); c.style.margin = 0;
      const open = document.createElement("button"); open.className = "open"; open.type = "button";
      open.setAttribute("aria-label", `Enlarge image ${n + 1}: actually ${i.t}, predicted ${i.p}`);
      const frame = document.createElement("span"); frame.className = "frame";
      const img = document.createElement("img"); img.src = i.src; img.loading = "lazy";
      img.alt = `Held-out image, actually ${i.t}`; frame.append(img);
      if (withHeat && st.heat && i.h) {
        const h = document.createElement("img"); h.src = i.h; h.loading = "lazy"; h.alt = ""; h.className = "heat";
        frame.append(h);
      }
      open.append(frame);
      open.onclick = () => show(shown, n, open, withHeat && st.heat);
      const info = document.createElement("figcaption"); info.className = "info";
      const v = document.createElement("div"); v.className = "verdict";
      v.textContent = (i.ok ? "✓ " : "✗ ") + `Predicted ${i.p}` + (i.c ? ` · ${Math.round(+i.c)}%` : "");
      const t = document.createElement("div"); t.className = "truth"; t.textContent = `Actually ${i.t}`;
      info.append(v, t); c.append(open, info); return c;
    }));
  }
  render();
  return () => shown;
}

// Enlarged view: click an image; arrows or ←/→ to move, Esc / ✕ / click outside to close.
const viewer = document.getElementById("viewer"), vimg = document.getElementById("vimg"),
      vheat = document.getElementById("vheat"), vcap = document.getElementById("vcap");
let current = [], at = 0, opener = null, heatOn = false;
function paint() {
  const i = current[at];
  vimg.src = i.src; vimg.alt = `Held-out image ${at + 1} of ${current.length}, actually ${i.t}`;
  vheat.hidden = !(heatOn && i.h);
  if (heatOn && i.h) vheat.src = i.h;
  vcap.replaceChildren();
  const b = document.createElement("b"); b.className = i.ok ? "vok" : "vmiss";
  b.textContent = (i.ok ? "✓ Correct: " : "✗ Wrong: ") + `predicted ${i.p}` + (i.c ? ` · ${Math.round(+i.c)}% confident` : "");
  const t = document.createElement("div"); t.textContent = `Actually ${i.t}`;
  const m = document.createElement("div"); m.className = "vmeta";
  m.textContent = `Image ${at + 1} of ${current.length} · original resolution, shown enlarged` +
                  (heatOn && i.h ? " · heatmap drawn as a separate layer" : "");
  vcap.append(b, t, m);
}
function show(list, n, from, withHeat) {
  current = list; at = n; opener = from || null; heatOn = !!withHeat; paint();
  if (!viewer.open) viewer.showModal();
}
function step(d) { if (current.length) { at = (at + d + current.length) % current.length; paint(); } }
document.getElementById("vclose").onclick = () => viewer.close();
document.getElementById("vprev").onclick = () => step(-1);
document.getElementById("vnext").onclick = () => step(1);
viewer.addEventListener("click", e => { if (e.target === viewer || e.target.id === "vstage") viewer.close(); });
viewer.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
  if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
});
viewer.addEventListener("close", () => { if (opener) opener.focus(); });

const verdictShown = makeGallery(ITEMS, {tabs: "gtabs", filters: "gfilters", grid: "gallery", count: "gcount"}, false);
const heatShown = makeGallery(HEAT, {tabs: "htabs", filters: "hfilters", grid: "hgallery", count: "hcount"}, true);

// Model switch: Amazon Rekognition vs our own CNN. The choice is kept in the link (#cnn) so it can be shared.
const MODELS = [["rek", "Amazon Rekognition"], ["cnn", "Our own CNN"]], mtabs = document.getElementById("mtabs");
function setModel(m, remember) {
  document.body.dataset.model = m;
  if (mtabs) mtabs.replaceChildren(...MODELS.map(([k, t]) => button(t, k === m, () => setModel(k, true))));
  if (remember) history.replaceState(null, "", m === "cnn" ? "#cnn" : location.pathname + location.search);
}
setModel(mtabs && ["#cnn", "#preview-heat"].includes(location.hash) ? "cnn" : "rek", false);
if (location.hash === "#preview-viewer" && verdictShown) show(verdictShown(), 0);
if (location.hash === "#preview-heat" && heatShown && heatShown().length) show(heatShown(), 0, null, true);
</script>
</body>
</html>
"""

FORBIDDEN = [
    (re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])"), "a 12-digit number (looks like an AWS account ID)"),
    (re.compile(r"arn:aws", re.I), "an AWS ARN"),
    (re.compile(r"s3://", re.I), "an S3 path"),
    (re.compile(r"amazonaws\.com", re.I), "an AWS endpoint hostname"),
]


def privacy_check(text, bucket, terms=()):
    """Exits if the page would expose AWS identifiers, the bucket name, or a private term (any letter case)."""
    problems = [why for rx, why in FORBIDDEN if rx.search(text)]
    if any(t.lower() in text.lower() for t in terms if t):
        problems.append("a term listed in private_terms")
    if bucket and bucket in text:
        problems.append("the training bucket's name")
    if problems:
        sys.exit("Refusing to build: the page would contain " + ", ".join(problems) + ".")


def main():
    data = [load(name) for name in DATASETS]
    local = json.loads((ROOT / "settings.json").read_text())
    bucket = local.get("bucket", "")
    # Names that must never appear on the public page, kept in the git-ignored settings.json ("private_terms").
    private_terms = local.get("private_terms", [])

    title_of = lambda d: esc(d["ds"]["title"].capitalize())
    rek_summary = lambda d: (d["holdout"] or {}).get("summary")
    cnn_summary = lambda d: (d["cnn"] or {}).get("summary")
    cnn_runs = [d for d in data if d["cnn"]]

    kpi_html = ('<div class="m-rek">' + "".join(
        f'<div class="ds-block"><div class="ds-title">{title_of(d)}</div>{kpis(d, rek_summary(d))}</div>'
        for d in data) + '</div>')
    if cnn_runs:
        kpi_html += ('<div class="m-cnn">' + "".join(
            f'<div class="ds-block"><div class="ds-title">{title_of(d)}</div>{kpis(d, cnn_summary(d))}</div>'
            for d in data) + '</div>')
    base_html = "".join(f'<h3>{title_of(d)}</h3>{baseline_table(d)}' for d in data)
    model_html = "".join(f'<h3>{title_of(d)}</h3><div class="grid2"><div>{confusion(d, rek_summary(d))}'
                         f'</div><div>{eval_table(d)}</div></div>' for d in data)

    modelbar = cnn_model = cnn_method = ""
    if cnn_runs:
        where = cnn_where(cnn_runs[0]["cnn"]["metrics"])
        modelbar = (
            '<nav class="modelbar" aria-label="Choose a model"><div class="wrap"><span class="label">Model:</span>'
            '<div class="tabs" id="mtabs" role="group" aria-label="Model"></div>'
            '<span class="what m-rek">Amazon Rekognition Custom Labels: a managed AWS service that picks, trains and '
            'hosts the model for you.</span>'
            f'<span class="what m-cnn">Our own CNN: a ResNet-18 fine-tuned on the same images, trained {esc(where)}, '
            'with heatmaps showing where it looks.</span></div></nav>')
        cnn_model = (
            '<div class="m-cnn"><p class="sub">A ResNet-18 convolutional neural network, starting from weights '
            'pre-trained on ImageNet (general photos) and fine-tuned on the same training images Rekognition used. '
            'After each pass over the training images it was scored on the same test set, and the best pass was '
            'kept. Then it was tested once on the held-out images. Unlike the Rekognition model, every part of it is '
            'visible, which is what makes the heatmaps possible.</p>'
            + network_diagram(data) + network_cards(data)
            + "".join(f'<h3>{title_of(d)}</h3><div class="grid2"><div>{confusion(d, cnn_summary(d))}</div>'
                      f'<div>{cnn_details(d)}</div></div>' for d in data) + '</div>')
        cnn_method = (f'<li>Our own CNN: ResNet-18 (PyTorch, ImageNet starting weights), fine-tuned {esc(where)} on '
                      'the same training images, with the same test and held-out images. Images resized to 224×224; '
                      'training used random flips, small rotations and brightness changes. Heatmaps: Grad-CAM on the '
                      'last convolutional block, for the defect class, on one shared scale per dataset so a clean '
                      'part stays mostly clear.</li>')
    pos_words = " / ".join(f'"{esc(d["ds"]["positive"])}"' for d in cnn_runs) or '"defective"'
    credits_html = "".join(f'<li>{esc(d["ds"]["title"].capitalize())}: <a href="{esc(CREDITS[d["name"]][1])}">'
                           f'{esc(CREDITS[d["name"]][0])}</a>, licensed <a href="{esc(CREDITS[d["name"]][3])}">'
                           f'{esc(CREDITS[d["name"]][2])}</a>.</li>' for d in data)
    items = gallery_data(data)
    names = [[d["name"], d["ds"]["title"].capitalize(), d["ds"]["positive"], d["ds"]["negative"]]
             for d in data if d["holdout"]]

    used_sagemaker = any(d["cnn"]["metrics"].get("trained_on") == "Amazon SageMaker AI" for d in cnn_runs)
    if used_sagemaker:
        title = "Automated visual inspection with Amazon Rekognition and Amazon SageMaker AI"
        built_with = ("Built with Amazon Rekognition Custom Labels, Amazon SageMaker AI (PyTorch) and Amazon S3, "
                      "hosted on AWS Amplify Hosting.")
    elif cnn_runs:
        title = "Automated visual inspection: Amazon Rekognition vs. our own CNN"
        built_with = ("Built with Amazon Rekognition Custom Labels, Amazon S3 and PyTorch, hosted on AWS Amplify "
                      "Hosting.")
    else:
        title = "Automated visual inspection with Amazon Rekognition"
        built_with = "Built with Amazon Rekognition Custom Labels and Amazon S3, hosted on AWS Amplify Hosting."

    page = (PAGE.replace("__REPO__", REPO_URL).replace("__DATE__", date.today().strftime("%B %Y"))
            .replace("__TITLE__", title).replace("__BUILT_WITH__", built_with)
            .replace("__AUTHOR__", author_badge(github_profile()))
            .replace("__MODELBAR__", modelbar).replace("__COMPARE__", compare_table(data))
            .replace("__KPIS__", kpi_html).replace("__BASELINE__", base_html).replace("__MODEL__", model_html)
            .replace("__CNN_MODEL__", cnn_model).replace("__CNN_METHOD__", cnn_method)
            .replace("__POS_WORDS__", pos_words).replace("__EDGE_NOTE__", edge_note(data))
            .replace("__CREDITS__", credits_html)
            .replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
            .replace("__HEAT_ITEMS__", json.dumps(heat_data(data)).replace("</", "<\\/"))
            .replace("__NAMES__", json.dumps(names).replace("</", "<\\/")))
    privacy_check(page, bucket, private_terms)

    if SITE.exists():
        shutil.rmtree(SITE)
    for d in data:
        for im in (d["holdout"] or {}).get("images", []):
            dest = SITE / im["src"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(im["file"], dest)  # byte-for-byte: no resizing, cropping or re-encoding
        if d["cnn"] and d["cnn"]["layers"]:
            (SITE / "net" / d["name"]).mkdir(parents=True, exist_ok=True)
            for png in (d["cnn"]["dir"] / "layers").glob("*.png"):
                shutil.copyfile(png, SITE / "net" / d["name"] / png.name)
        for r in (d["cnn"] or {}).get("rows", []):
            dest = SITE / r["heat"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r["heat_file"], dest)  # separate overlay layer; the photo itself is never altered
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(page, encoding="utf-8")
    n_img = sum(1 for p in SITE.rglob("*") if p.is_file()) - 1
    print(f"Built {SITE.relative_to(ROOT)}/index.html with {n_img} images. Privacy check passed.")
    for d in data:
        state = ("held-out test done" if d["holdout"] else "training done" if d["eval"]
                 else "baseline only" if d["baseline"] else "nothing yet")
        state += (f", our CNN: {pct(d['cnn']['summary']['accuracy'], 1)} held-out, {len(d['cnn']['rows'])} heatmaps"
                  if d["cnn"] else ", our CNN: not run yet")
        print(f"  {d['name']}: {state}")


if __name__ == "__main__":
    main()
