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
from datetime import date

import common
from common import DATASETS, ROOT

SITE = ROOT / "site"
REPO_URL = "https://github.com/Methodologist/casting-defect-inspection"

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

    # SageMaker Grad-CAM run (optional): only shown once real results exist.
    sm_dir = out_dir / "sagemaker"
    preds, metrics = read_json(sm_dir / "predictions.json"), read_json(sm_dir / "metrics.json")
    d["sm"] = None
    if preds and metrics and d["holdout"]:
        srcs = {im["file"].name: im["src"] for im in d["holdout"]["images"]}
        rows = []
        for r in preds:
            heat_file = sm_dir / r["heatmap"]
            if r["file"] in srcs and heat_file.exists():
                rows.append({"src": srcs[r["file"]], "heat_file": heat_file,
                             "heat": f"heat/{name}/{heat_file.name}", "truth": r["true_label"],
                             "pred": r["predicted"], "conf": r["confidence"], "ok": r["correct"]})
        d["sm"] = {"metrics": metrics, "rows": rows}
    return d


def pct(x, digits=0):
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def kpis(d):
    ds, h = d["ds"], d["holdout"]
    if not h:
        return f'<p class="pending">Held-out test for {esc(ds["title"])} not run yet.</p>'
    s = h["summary"]
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


def confusion(d):
    ds, h = d["ds"], d["holdout"]
    if not h:
        return ""
    s, pos, neg = h["summary"], ds["positive"], ds["negative"]
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
        if not d["sm"]:
            continue
        by_class = [[r for r in d["sm"]["rows"] if r["truth"] == label] for label in d["ds"]["labels"]]
        for n in range(max((len(c) for c in by_class), default=0)):
            for c in by_class:
                if n < len(c):
                    r = c[n]
                    items.append({"d": d["name"], "src": r["src"], "h": r["heat"], "t": r["truth"], "p": r["pred"],
                                  "c": r["conf"], "ok": r["ok"], "pos": r["truth"] == d["ds"]["positive"]})
    return items


def heat_section(data):
    runs = [d for d in data if d["sm"]]
    if not runs:
        return ""
    rows = ""
    for d in runs:
        m, s = d["sm"]["metrics"], (d["holdout"] or {}).get("summary", {})
        rows += (f'<tr><td>{esc(d["ds"]["title"].capitalize())}</td>'
                 f'<td>{pct(s.get("accuracy"), 1)}</td><td>{pct(m["heldout_accuracy"], 1)}</td>'
                 f'<td>{s.get("false_negative", "–")} / {m["fn"]}</td><td>{s.get("false_positive", "–")} / {m["fp"]}</td></tr>')
    return f'''<section><div class="wrap">
<h2>Where the model looks</h2>
<p class="sub">A second model, a ResNet-18 fine-tuned in an Amazon SageMaker AI training job on the same training
images, with Grad-CAM heatmaps showing which parts of each held-out image pushed it toward
"{esc(runs[0]["ds"]["positive"])}". Heat is drawn as a separate layer over the unmodified photo; switch it off to
compare.</p>
<div class="tablewrap"><table class="eval"><thead><tr><th>Dataset</th><th>Rekognition held-out accuracy</th>
<th>SageMaker ResNet-18 held-out accuracy</th><th>Missed (Rekognition / SageMaker)</th>
<th>False rejects (Rekognition / SageMaker)</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="caption">Same held-out images for both models. The heatmaps show where this model found evidence; they
explain its decision, not a guaranteed defect outline.</p>
<p class="legend"><span>Less evidence</span><span class="ramp"></span><span>More evidence for a defect</span></p>
<div class="tabs" id="htabs" role="group" aria-label="Dataset"></div>
<div class="tabs" id="hfilters" role="group" aria-label="Show"></div>
<p class="count" id="hcount"></p>
<div class="gallery" id="hgallery"></div>
</div></section>
'''


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
</style>
</head>
<body>
<header><div class="wrap">
<h1>__TITLE__</h1>
<p class="lede">A pass/fail camera check of the kind a robot cell runs: train a custom vision model on labeled
photos, then measure it on images it has never seen, counting missed defects and false rejects separately.</p>
<p class="meta">Two datasets: cast pump impellers (main run) and damaged vs intact shipping cases ·
<a href="__REPO__">Source code on GitHub</a> · Built __DATE__</p>
</div></header>

<section><div class="wrap">
<h2>Results on held-out images</h2>
<p class="sub">Each model was tested on photos kept aside from training and from AWS's own evaluation, so these
numbers show how it does on parts it has truly never seen.</p>
__KPIS__
</div></section>

<section><div class="wrap">
<h2>Why off-the-shelf AI isn't enough</h2>
<p class="sub">Step 1 sent held-out images to Rekognition's general-purpose label detector. It describes what it
sees, but it has no concept of a defect.</p>
__BASELINE__
</div></section>

<section><div class="wrap">
<h2>The custom model</h2>
<p class="sub">Rekognition Custom Labels, trained on labeled images, then tested on separate held-out images.
On a line, a missed defect and a false reject cost different things, so they are shown separately.</p>
__MODEL__
</div></section>

<section><div class="wrap">
<h2>Every held-out image</h2>
<p class="sub">The model's verdict and confidence for each image it was tested on. Images are shown unmodified;
the verdict sits beside them.</p>
<div class="tabs" id="gtabs" role="group" aria-label="Dataset"></div>
<div class="tabs" id="gfilters" role="group" aria-label="Show"></div>
<p class="count" id="gcount"></p>
<div class="gallery" id="gallery"></div>
</div></section>
__HEAT_SECTION__
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
prove a model; production checks typically run on vision built into the cell.</li>
<li><b>Running cost.</b> A hosted Custom Labels model bills about $4 an hour while running, roughly $35,000 a year
around the clock. Fine for a pilot, but a reason to run inference at the edge in production.</li>
<li><b>Train on the customer's parts.</b> Accuracy depends on photos that match the real station: same camera,
angle and lighting. The public datasets here prove the method; a production model is retrained on the customer's own
images. (The casting dataset's license is also non-commercial.)</li>
<li><b>Tune the threshold to the cost of a miss.</b> Lowering the confidence needed to reject catches more defects
at the price of more false rejects. Where a missed defect reaches a customer, that trade is usually worth it.</li>
</ul>
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

    kpi_html = "".join(f'<div class="ds-block"><div class="ds-title">{esc(d["ds"]["title"].capitalize())}</div>'
                       f'{kpis(d)}</div>' for d in data)
    base_html = "".join(f'<h3>{esc(d["ds"]["title"].capitalize())}</h3>{baseline_table(d)}' for d in data)
    model_html = "".join(f'<h3>{esc(d["ds"]["title"].capitalize())}</h3><div class="grid2"><div>{confusion(d) or ""}'
                         f'</div><div>{eval_table(d)}</div></div>' for d in data)
    credits_html = "".join(f'<li>{esc(d["ds"]["title"].capitalize())}: <a href="{esc(CREDITS[d["name"]][1])}">'
                           f'{esc(CREDITS[d["name"]][0])}</a>, licensed <a href="{esc(CREDITS[d["name"]][3])}">'
                           f'{esc(CREDITS[d["name"]][2])}</a>.</li>' for d in data)
    items = gallery_data(data)
    names = [[d["name"], d["ds"]["title"].capitalize(), d["ds"]["positive"], d["ds"]["negative"]]
             for d in data if d["holdout"]]

    used_sagemaker = any(d["sm"] for d in data)
    title = ("Automated visual inspection with Amazon Rekognition and Amazon SageMaker AI" if used_sagemaker
             else "Automated visual inspection with Amazon Rekognition")
    built_with = ("Built with Amazon Rekognition Custom Labels, Amazon SageMaker AI and Amazon S3, hosted on AWS "
                  "Amplify Hosting." if used_sagemaker else
                  "Built with Amazon Rekognition Custom Labels and Amazon S3, hosted on AWS Amplify Hosting.")

    page = (PAGE.replace("__REPO__", REPO_URL).replace("__DATE__", date.today().strftime("%B %Y"))
            .replace("__TITLE__", title).replace("__BUILT_WITH__", built_with)
            .replace("__KPIS__", kpi_html).replace("__BASELINE__", base_html).replace("__MODEL__", model_html)
            .replace("__HEAT_SECTION__", heat_section(data))
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
        for r in (d["sm"] or {}).get("rows", []):
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
        state += f", SageMaker heatmaps: {len(d['sm']['rows'])}" if d["sm"] else ", SageMaker heatmaps: not yet"
        print(f"  {d['name']}: {state}")


if __name__ == "__main__":
    main()
