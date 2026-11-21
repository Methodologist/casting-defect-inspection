"""Builds docs/project-overview.pdf from the project's results/ files. Usage: python make_overview_pdf.py <project_root>"""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(sys.argv[1])
OUT = ROOT / "docs" / "project-overview.pdf"
REPO = "github.com/Methodologist/casting-defect-inspection"
SITE = "https://main.d1vk1voghz7z5z.amplifyapp.com/"

NAVY = colors.HexColor("#0b1f3a")
BLUE = colors.HexColor("#0b5fe0")
INK = colors.HexColor("#0b0f14")
MUTED = colors.HexColor("#3a4350")
LINE = colors.HexColor("#b9c1cc")
TINT = colors.HexColor("#eef1f5")


def load(name):
    d = ROOT / "results" / name
    ev = json.loads((d / "step2_evaluation.json").read_text())
    s = json.loads((d / "step3_summary.json").read_text())
    return ev["AggregatedEvaluationResults"]["F1Score"], ev["EvaluationDetails"]["NumberOfTestingImages"], s


cast_f1, cast_test, cast = load("casting")
pack_f1, pack_test, pack = load("packages")
cast_cnn = json.loads((ROOT / "results" / "casting" / "cnn" / "metrics.json").read_text())
pack_cnn = json.loads((ROOT / "results" / "packages" / "cnn" / "metrics.json").read_text())
TITLE = "Automated Visual Inspection: Amazon Rekognition vs. Our Own CNN"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6, leading=13.4,
                      textColor=INK, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11, textColor=MUTED)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=9,
                    spaceAfter=4, textColor=NAVY)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=8.8, leading=11.5, spaceAfter=0, alignment=TA_LEFT)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def header_band(canvas, doc):
    w, h = letter
    canvas.saveState()
    if doc.page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 1.55 * inch, w, 1.55 * inch, stroke=0, fill=1)
        canvas.setFillColor(BLUE)
        canvas.rect(0, h - 1.55 * inch - 4, w, 4, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(0.75 * inch, h - 0.5 * inch, "SHOWCASE PROJECT  ·  PROJECT OVERVIEW")
        size = 20
        while canvas.stringWidth(TITLE, "Helvetica-Bold", size) > w - 1.5 * inch:
            size -= 0.5
        canvas.setFont("Helvetica-Bold", size)
        canvas.drawString(0.75 * inch, h - 0.86 * inch, TITLE)
        canvas.setFont("Helvetica", 10.5)
        canvas.setFillColor(colors.HexColor("#c9d6ea"))
        canvas.drawString(0.75 * inch, h - 1.14 * inch,
                          "A pass/fail camera check: a managed cloud model and our own network, tested on images "
                          "neither ever saw.")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 0.5 * inch, f"{REPO}  ·  November 2026")
    canvas.drawRightString(w - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def pct(x):
    return f"{100 * x:.1f}%"


def result_row(title, f1, n_test, s, c, train):
    pos = s["true_positive"] + s["false_negative"]
    neg = s["false_positive"] + s["true_negative"]
    return [Paragraph(title, cellb), Paragraph(train, cell), Paragraph(f"{pct(f1)} F1<br/>({n_test} images)", cell),
            Paragraph(f"<b>{pct(s['accuracy'])}</b>", cell), Paragraph(f"<b>{pct(c['heldout_accuracy'])}</b>", cell),
            Paragraph(f"{s['false_negative']} / {c['fn']} of {pos}", cell),
            Paragraph(f"{s['false_positive']} / {c['fp']} of {neg}", cell)]


results = Table(
    [[Paragraph(h, cellb) for h in ("Dataset", "Images: train / test / held out", "AWS evaluation on test set",
                                     "Held-out: Rekognition", "Held-out: our CNN", "Defects missed (Rek. / CNN)",
                                     "Good items rejected (Rek. / CNN)")],
     result_row("Cast pump impellers", cast_f1, cast_test, cast, cast_cnn, "1,000 / 200 / 100"),
     result_row("Shipping cases (damaged vs intact)", pack_f1, pack_test, pack, pack_cnn, "377 / 116 / 88")],
    colWidths=[1.2 * inch, 1.0 * inch, 0.95 * inch, 0.9 * inch, 0.8 * inch, 1.0 * inch, 1.05 * inch])
results.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 1, NAVY),
    ("LINEBELOW", (0, 1), (-1, -1), 0.5, LINE), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))

steps = Table([[Paragraph(f"<b>{a}</b><br/>{b}", cell) for a, b in (
    ("1. Baseline", "Off-the-shelf Rekognition labels held-out images. It names the object, not the defect."),
    ("2. Train two models", "Rekognition Custom Labels (managed by AWS) and our own ResNet-18 CNN (PyTorch, local "
                            "GPU) learn pass/fail from the same labeled photos."),
    ("3. Held-out test", "Both models check the same images they never saw. The cloud model is stopped right after."),
    ("4. Publish", "Results are turned into a web page and hosted on AWS Amplify, with no account details exposed."))]],
    colWidths=[1.75 * inch] * 4)
steps.setStyle(TableStyle([
    ("BOX", (0, 0), (0, 0), 0.5, LINE), ("BOX", (1, 0), (1, 0), 0.5, LINE), ("BOX", (2, 0), (2, 0), 0.5, LINE),
    ("BOX", (3, 0), (3, 0), 0.5, LINE), ("LINEABOVE", (0, 0), (-1, 0), 3, BLUE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))

story = [
    Spacer(1, 0.95 * inch),
    Paragraph("What this is", h2),
    Paragraph("A portfolio project showing how a camera-based pass/fail inspection, the kind a robot cell runs at the "
              "end of a line, can be prototyped quickly and cheaply on AWS. It is a demonstration built on public "
              "datasets, not a production system. The same code runs on two problems: spotting defective cast pump "
              "impellers, and telling damaged shipping cases from intact ones.", body),
    Paragraph(f'<b>Interactive results:</b> <link href="{SITE}" color="#0b5fe0">{SITE.replace("https://", "")}</link>'
              f' (switch between both models; every held-out image with its verdict and heatmap) &nbsp;·&nbsp; '
              f'<b>Code:</b> '
              f'<link href="https://{REPO}" color="#0b5fe0">{REPO}</link>', body),
    Paragraph("How it works", h2),
    steps,
    Spacer(1, 4),
    Paragraph("Results", h2),
    results,
    Paragraph("Held-out images were kept apart from both training and AWS's own evaluation. A missed defect and a "
              "falsely rejected good item cost different things on a line, so they are counted separately.", small),
    Paragraph("What the results show", h2),
    *bullets([
        "<b>Off-the-shelf AI is not an inspector.</b> The general model returned the same labels (coil, rotor, "
        "spoke) for good and defective castings. A custom-trained model was needed.",
        f"<b>Consistent images make the job easy.</b> The casting photos come from one camera setup, and the model "
        f"got all {cast['images']} held-out parts right. One caveat: the dataset's 7,348 photos are rotated and "
        f"flipped copies of 1,300 originals, so a held-out part may have been seen at another angle, and the score "
        f"may be partly flattered.",
        "<b>Messy data needs skepticism.</b> The shipping-case photos were collected from the web, and the baseline "
        "showed that damaged and intact photos also differ in box type (plain cartons vs printed retail boxes). On "
        f"held-out images the model scored {pct(pack['accuracy'])}, below AWS's {pct(pack_f1)} test score, and its "
        "mistakes cluster on boxes whose type doesn't match the usual pattern (damaged food boxes passed, intact food "
        "boxes rejected). It partly learned box type, not just damage: the reason a production model is trained on "
        "consistent photos from the real station.",
        f"<b>Our own CNN matched it on clean data.</b> A ResNet-18 fine-tuned on the same photos also scored "
        f"{pct(cast_cnn['heldout_accuracy'])} on castings, and its Grad-CAM heatmaps land on the damaged rim of "
        f"defective parts. On shipping cases it scored {pct(pack_cnn['heldout_accuracy'])}, missing "
        f"{pack_cnn['fn']} of 39 damaged cases; some of those photos look undamaged (an open carton, a takeout box), so "
        f"the labels are part of the problem. It classifies one image in about {cast_cnn['latency_ms']['gpu']:.0f} ms "
        f"on a GTX 1070 GPU, {cast_cnn['latency_ms']['cpu']:.0f} ms on the CPU.",
    ]),
    Paragraph("From demo to production", h2),
    *bullets([
        "<b>Train in the cloud, decide on the line.</b> A cloud call adds network delay, and a hosted model bills "
        "about $4 an hour while running. Production checks typically run on vision hardware in the cell.",
        "<b>Train on the real station.</b> Accuracy depends on photos from the actual camera, angle and lighting, so "
        "a production model is retrained on the customer's own parts.",
        "<b>Tune for the cost of a miss.</b> The rejection threshold can trade a few extra false rejects for fewer "
        "defects reaching a customer.",
    ]),
    Paragraph("Doing it carefully", h2),
    *bullets([
        "<b>No leaks between training and testing.</b> The casting dataset contains a few byte-identical images in both "
        "its train and test folders. They were found and removed before testing.",
        "<b>No stored credentials.</b> AWS access uses short-lived browser sign-in (aws login), not access keys.",
        "<b>Cost control.</b> The whole project used about $3 to $4 of AWS credits. Models are stopped as soon as "
        "each test ends, and a safety script stops anything left running.",
        "<b>Privacy.</b> The published page is checked automatically for account IDs, resource names and storage paths "
        "before it goes live.",
        "<b>Automated tests.</b> GitHub Actions checks on every push that the split can't leak training images into "
        "testing, that misses and false rejects are counted correctly, and that the privacy check blocks what it "
        "should.",
    ]),
    KeepTogether([
        Paragraph("Built with", h2),
        Paragraph("Amazon Rekognition and Rekognition Custom Labels, Amazon S3, AWS Amplify Hosting, Python with "
                  "boto3, the AWS CLI, PyTorch (ResNet-18, Grad-CAM) on an NVIDIA GPU, and pytest with GitHub "
                  "Actions.", body),
        Paragraph("Data and credits", h2),
        *bullets([
            "Casting: \"casting product image data for quality inspection\" by Ravirajsinh Dabhi (Kaggle), "
            "licensed CC BY-NC-ND 4.0. Used for non-commercial demonstration only.",
            "Shipping cases: \"Damaged vs good packages\", provided by a Roboflow user on Roboflow Universe, "
            "licensed CC BY 4.0.",
        ]),
        Paragraph("Next step", h2),
        Paragraph("Run the same CNN training as an Amazon SageMaker AI training job (the script is ready), and "
                  "improve the shipping-case model by cleaning the labels and correcting its lean toward \"intact\", "
                  "checking every change on the test set only, so the held-out images stay a fair final test.", body),
        Paragraph(f"Source code, setup steps and full method: {REPO} · Interactive results: "
                  f"{SITE.replace('https://', '').rstrip('/')}", small),
    ]),
]

OUT.parent.mkdir(exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.8 * inch, title="Automated Visual Inspection: Project Overview",
                        author="Methodologist", subject="Showcase project overview")
doc.build(story, onFirstPage=header_band, onLaterPages=header_band)
print(f"Wrote {OUT}")
