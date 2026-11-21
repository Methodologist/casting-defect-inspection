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
                 + "".join(f'<td><div class="bar"><span class="fill {cls}" style="width:{100 * r[l]:.0f}%"></span>'
                           f'<span class="val">{pct(r[l])}</span></div></td>'
                           for l, cls in ((neg, "neg"), (pos, "pos")))
                 + "</tr>")
    return (f'<table class="baseline"><thead>{head}</thead><tbody>{body}</tbody></table>'
            f'<p class="caption">Share of {b["n"]} held-out images on which off-the-shelf DetectLabels returned each '
            f'label. Highlighted rows differ by 25 points or more between classes.</p>'
            f'<p>{esc(BASELINE_NOTES.get(d["name"], ""))}</p>')


def confusion(d):
    ds, h = d["ds"], d["holdout"]
    if not h:
        return ""
    s, pos, neg = h["summary"], ds["positive"], ds["negative"]
    return (f'<table class="confusion"><thead><tr><th></th><th>Predicted {esc(pos)}</th>'
            f'<th>Predicted {esc(neg)}</th></tr></thead><tbody>'
            f'<tr><th>Actually {esc(pos)}</th><td class="good">{s["true_positive"]}<small>caught</small></td>'
            f'<td class="bad">{s["false_negative"]}<small>missed</small></td></tr>'
            f'<tr><th>Actually {esc(neg)}</th><td class="warn">{s["false_positive"]}<small>false reject</small></td>'
            f'<td class="good">{s["true_negative"]}<small>passed</small></td></tr></tbody></table>')


def eval_table(d):
    e = d["eval"]
    if not e:
        return '<p class="pending">Training not finished yet.</p>'
    rows = "".join(f'<tr><td>{esc(r["label"])}</td><td>{esc(r["images"])}</td><td>{pct(r["precision"], 1)}</td>'
                   f'<td>{pct(r["recall"], 1)}</td><td>{pct(r["f1score"], 1)}</td></tr>' for r in e["labels"])
    return (f'<table class="eval"><thead><tr><th>Label</th><th>Test images</th><th>Precision</th><th>Recall</th>'
            f'<th>F1</th></tr></thead><tbody>{rows}</tbody></table>'
            f'<p class="caption">AWS\'s own evaluation after training, on the {esc(e["images"])}-image test set '
            f'(overall F1 {pct(e["f1"], 1)}). The held-out numbers above come from separate images the model never '
            f'saw during training or evaluation.</p>')


def gallery_data(data):
    items = []
    for d in data:
        if d["holdout"]:
            for im in d["holdout"]["images"]:
                items.append({"d": d["name"], "src": im["src"], "t": im["truth"], "p": im["pred"],
                              "c": im["conf"], "ok": im["ok"]})
    return items


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Automated Visual Inspection</title>
<meta name="description" content="A pass/fail vision inspection pipeline on Amazon Rekognition Custom Labels, tested on cast impellers and shipping cases.">
<style>
:root{--bg:#f6f7f9;--surface:#fff;--ink:#16191d;--muted:#5b6470;--line:#dde1e6;--accent:#1f5fbf;
--good:#1d7a46;--good-bg:#e5f4ea;--bad:#b42318;--bad-bg:#fdecea;--warn:#9a5b00;--warn-bg:#fdf3e1;
--neg:#8a94a3;--pos:#1f5fbf;--lean:#fff8e6}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--surface:#171b21;--ink:#e8ebef;--muted:#9aa4b1;
--line:#2a3039;--accent:#6ea2ff;--good:#5cc98a;--good-bg:#12291c;--bad:#ff8a7a;--bad-bg:#331714;
--warn:#f0b35a;--warn-bg:#2e2412;--neg:#6c7684;--pos:#6ea2ff;--lean:#262113}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}
.wrap{max-width:1040px;margin:0 auto;padding:0 16px}
header{padding:48px 0 24px;border-bottom:1px solid var(--line);background:var(--surface)}
h1{font-size:clamp(28px,4vw,40px);line-height:1.15;margin:0 0 8px;letter-spacing:-.01em}
.lede{color:var(--muted);font-size:18px;max-width:760px;margin:0}
.meta{margin-top:16px;font-size:14px;color:var(--muted)}
section{padding:36px 0;border-bottom:1px solid var(--line)}
h2{font-size:24px;margin:0 0 6px}
h3{font-size:18px;margin:28px 0 10px}
.sub{color:var(--muted);margin:0 0 18px;max-width:760px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 18px}
.tabs button{font:inherit;font-size:14px;padding:6px 14px;border-radius:999px;border:1px solid var(--line);
background:var(--surface);color:var(--ink);cursor:pointer}
.tabs button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:8px 0 4px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}
.kpi-label{font-size:13px;color:var(--muted)}
.kpi-value{font-size:30px;font-weight:650;margin:4px 0 2px;font-variant-numeric:tabular-nums}
.kpi-sub{font-size:13px;color:var(--muted)}
.ds-block{margin-bottom:28px}
.ds-title{font-weight:650;font-size:18px;margin:0 0 6px}
table{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
thead th{font-size:13px;color:var(--muted);font-weight:600}
tr.lean td{background:var(--lean)}
.bar{position:relative;height:22px;background:var(--bg);border-radius:4px;min-width:90px}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:4px;opacity:.85}
.fill.neg{background:var(--neg)}.fill.pos{background:var(--pos)}
.val{position:relative;padding-left:8px;font-size:13px;line-height:22px;font-weight:600}
.caption{font-size:13px;color:var(--muted);margin:8px 0 12px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}}
.confusion td{font-size:26px;font-weight:650;text-align:center}
.confusion td small{display:block;font-size:12px;font-weight:500;color:var(--muted)}
.confusion td.good{background:var(--good-bg);color:var(--good)}
.confusion td.bad{background:var(--bad-bg);color:var(--bad)}
.confusion td.warn{background:var(--warn-bg);color:var(--warn)}
.pending{color:var(--muted);font-style:italic}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
.card img{display:block;width:100%;aspect-ratio:1;object-fit:cover;background:var(--bg)}
.card .info{padding:8px 10px}
.card .verdict{font-weight:650}
.card.ok .verdict{color:var(--good)}.card.miss{border-color:var(--bad)}.card.miss .verdict{color:var(--bad)}
.card .truth{color:var(--muted)}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 20px}
.step{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;position:relative}
.step b{display:block;margin-bottom:4px}
.step span{font-size:13px;color:var(--muted)}
ul.points{padding-left:20px;max-width:800px}ul.points li{margin:6px 0}
footer{padding:28px 0 48px;font-size:13px;color:var(--muted)}
.count{color:var(--muted);font-size:14px;margin:0 0 12px}
</style>
</head>
<body>
<header><div class="wrap">
<h1>Automated visual inspection with Amazon Rekognition</h1>
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
<div class="tabs" id="gtabs" role="group" aria-label="Filter images"></div>
<p class="count" id="gcount"></p>
<div class="gallery" id="gallery"></div>
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
<li>Packages: 581 photos pooled and split 65/20/15 into train, test and held out with a fixed seed, after removing
exact duplicates.</li>
<li>Costs for the whole project: a few dollars of AWS credits for training and testing. Models are stopped when not
in use.</li>
__CREDITS__
</ul>
</div></section>

<footer><div class="wrap">Built with Amazon Rekognition Custom Labels and Amazon S3, hosted on Amazon CloudFront.
Images remain the property of their creators and are used under the licenses listed above.</div></footer>

<script>
const ITEMS = __ITEMS__;
const NAMES = __NAMES__;
const tabs = document.getElementById("gtabs"), grid = document.getElementById("gallery"),
      count = document.getElementById("gcount");
let state = {d: NAMES.length ? NAMES[0][0] : null, wrong: false};
function button(label, pressed, onClick) {
  const b = document.createElement("button"); b.textContent = label;
  b.setAttribute("aria-pressed", pressed); b.onclick = onClick; return b;
}
function render() {
  tabs.replaceChildren();
  NAMES.forEach(([key, title]) => tabs.append(button(title, state.d === key, () => { state.d = key; render(); })));
  tabs.append(button("Mistakes only", state.wrong, () => { state.wrong = !state.wrong; render(); }));
  const shown = ITEMS.filter(i => i.d === state.d && (!state.wrong || !i.ok));
  const all = ITEMS.filter(i => i.d === state.d);
  count.textContent = all.length ? `${shown.length} of ${all.length} images · ${all.filter(i => !i.ok).length} wrong calls`
                                 : "Held-out test not run yet.";
  grid.replaceChildren(...shown.map(i => {
    const c = document.createElement("figure"); c.className = "card " + (i.ok ? "ok" : "miss"); c.style.margin = 0;
    const img = document.createElement("img"); img.src = i.src; img.loading = "lazy";
    img.alt = `Held-out image, actually ${i.t}`;
    const info = document.createElement("figcaption"); info.className = "info";
    const v = document.createElement("div"); v.className = "verdict";
    v.textContent = (i.ok ? "✓ " : "✗ ") + `Predicted ${i.p}` + (i.c ? ` · ${Math.round(+i.c)}%` : "");
    const t = document.createElement("div"); t.className = "truth"; t.textContent = `Actually ${i.t}`;
    info.append(v, t); c.append(img, info); return c;
  }));
}
render();
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


def privacy_check(text, bucket):
    problems = [why for rx, why in FORBIDDEN if rx.search(text)]
    # Names that must never appear on the public page, kept in the git-ignored settings.json ("private_terms").
    terms = json.loads((ROOT / "settings.json").read_text()).get("private_terms", [])
    if any(t.lower() in text.lower() for t in terms if t):
        problems.append("a term listed in private_terms")
    if bucket and bucket in text:
        problems.append("the training bucket's name")
    if problems:
        sys.exit("Refusing to build: the page would contain " + ", ".join(problems) + ".")


def main():
    data = [load(name) for name in DATASETS]
    bucket = json.loads((ROOT / "settings.json").read_text()).get("bucket", "")

    kpi_html = "".join(f'<div class="ds-block"><div class="ds-title">{esc(d["ds"]["title"].capitalize())}</div>'
                       f'{kpis(d)}</div>' for d in data)
    base_html = "".join(f'<h3>{esc(d["ds"]["title"].capitalize())}</h3>{baseline_table(d)}' for d in data)
    model_html = "".join(f'<h3>{esc(d["ds"]["title"].capitalize())}</h3><div class="grid2"><div>{confusion(d) or ""}'
                         f'</div><div>{eval_table(d)}</div></div>' for d in data)
    credits_html = "".join(f'<li>{esc(d["ds"]["title"].capitalize())}: <a href="{esc(CREDITS[d["name"]][1])}">'
                           f'{esc(CREDITS[d["name"]][0])}</a>, licensed <a href="{esc(CREDITS[d["name"]][3])}">'
                           f'{esc(CREDITS[d["name"]][2])}</a>.</li>' for d in data)
    items = gallery_data(data)
    names = [[d["name"], d["ds"]["title"].capitalize()] for d in data if d["holdout"]]

    page = (PAGE.replace("__REPO__", REPO_URL).replace("__DATE__", date.today().strftime("%B %Y"))
            .replace("__KPIS__", kpi_html).replace("__BASELINE__", base_html).replace("__MODEL__", model_html)
            .replace("__CREDITS__", credits_html)
            .replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
            .replace("__NAMES__", json.dumps(names).replace("</", "<\\/")))
    privacy_check(page, bucket)

    if SITE.exists():
        shutil.rmtree(SITE)
    for d in data:
        for im in (d["holdout"] or {}).get("images", []):
            dest = SITE / im["src"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(im["file"], dest)  # byte-for-byte: no resizing, cropping or re-encoding
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(page, encoding="utf-8")
    n_img = sum(1 for p in SITE.rglob("*") if p.is_file()) - 1
    print(f"Built {SITE.relative_to(ROOT)}/index.html with {n_img} images. Privacy check passed.")
    for d in data:
        state = ("held-out test done" if d["holdout"] else "training done" if d["eval"]
                 else "baseline only" if d["baseline"] else "nothing yet")
        print(f"  {d['name']}: {state}")


if __name__ == "__main__":
    main()
