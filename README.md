# Casting defect inspection with Amazon Rekognition

> **Showcase project.** A portfolio demonstration of camera-based pass/fail inspection on AWS, built on public
> datasets. It is not a production system and is not maintained as a library.
>
> - **[Interactive results](https://main.d1vk1voghz7z5z.amplifyapp.com/)**: every held-out image with the
>   model's verdict; click any image to inspect it.
> - **[Project overview (PDF)](docs/project-overview.pdf)**: a two-page summary of what was built, the
>   results and the lessons.

## Results at a glance

Measured on held-out images the model never saw during training or AWS's own evaluation.

| Dataset | Held-out accuracy | Defects missed | Good items rejected | AWS test-set F1 |
|---|---|---|---|---|
| Cast pump impellers | **100%** (100 images) | 0 of 50 | 0 of 50 | 100% (200 images) |
| Shipping cases, damaged vs intact | **88.6%** (88 images) | 6 of 39 | 4 of 49 | 95.7% (116 images) |

- **Off-the-shelf AI isn't an inspector:** general Rekognition labels (coil, rotor, spoke) appear on good and
  defective castings alike.
- **Consistent images make the job easy:** the casting photos come from one camera setup, and every held-out part
  was classified correctly.
- **Messy data needs skepticism:** the web-collected shipping-case photos also differ in box type, and the model's
  held-out mistakes cluster where box type and damage disagree. It partly learned the shortcut, which is why a
  production model is trained on consistent photos from the real station.

The whole project used about $3 to $4 of AWS credits; models are stopped as soon as each test ends.

## About the project

A small computer vision project: tell good parts from bad ones, the kind of pass/fail check a camera does
in a robot cell. The main run inspects cast pump impellers. The same pipeline then runs unchanged on
damaged vs intact shipping cases, the check that matters before a robot depalletizes or ships a load.

1. **Baseline.** Off-the-shelf Rekognition (DetectLabels) describes the object but can't judge it.
2. **Custom model.** Rekognition Custom Labels is trained on labeled images and evaluated on a test set.
3. **Held-out test.** The trained model is checked on images it never saw. Missed defects and false
   rejects are counted separately, because on a line they cost different things.

| Dataset (`"dataset"` in settings.json) | Images used: train / test / held out | Classes |
|---|---|---|
| `casting` (main run) | 1,000 / 200 / 100 | defective, ok |
| `packages` | 377 / 116 / 88 | damaged, intact |

No test or held-out image is a copy of a training image. The casting dataset ships a few byte-identical
copies across its own train and test folders; those are dropped from test and held-out sets. See
[Data and credits](#data-and-credits) for sources and licenses.

## One-time setup (about 20 minutes)

You need the [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
and Python 3.10 or newer. No access keys are created or stored anywhere in this project.

**1. Sign in with the AWS CLI.** This opens a browser. The credentials last 12 hours and renew for up
to 90 days without signing in again.

```powershell
aws configure set region us-east-2 --profile vision-inspection
aws login --region us-east-2 --profile vision-inspection
```

Use the Region your AWS account works in. Rekognition Custom Labels is only offered in some Regions;
this project was built in us-east-2 (Ohio). If you use the new AWS experience, your project's Region
is under AWS Settings > your project > Overview > Additional info.

**2. Watch the spend.** On the Free plan the account pauses when credits run out instead of billing
you, so keep an eye on remaining credits in AWS Settings > Billing. On a paid account, set a $20
monthly budget alert under Billing and Cost Management > Budgets.

**3. Make the S3 bucket** in the same Region, with the default settings (public access stays
blocked). In the console, choose bucket type **General purpose**, not Directory. The name must contain
`custom-labels`, for example `custom-labels-casting-demo`. That matches the
`AmazonRekognitionCustomLabelsFullAccess` policy, in case you run this as an IAM user. One bucket holds
both datasets, each under its own prefix.

```powershell
aws s3 mb s3://custom-labels-casting-demo --region us-east-2 --profile vision-inspection
```

Copy `settings.example.json` to `settings.json` and put your bucket name, Region and profile in it.
`settings.json` is git-ignored.

**4. Get the images** (links under [Data and credits](#data-and-credits)) and extract each one into its
own folder. `data/` is git-ignored. The exact depth inside each folder doesn't matter.

| Dataset | Download as | Extract into | Expected folders inside |
|---|---|---|---|
| casting | Kaggle zip | `data/casting/` | `train/def_front`, `train/ok_front`, `test/...` |
| packages | Roboflow, "Folder Structure" format | `data/packages/` | `train/damaged`, `train/intact`, `valid/...`, `test/...` |

**5. Install boto3** (the AWS SDK for Python), from this folder in PowerShell. The `[crt]` extra is
what lets boto3 use `aws login` credentials.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Running it

Run everything from this folder with `.venv\Scripts\python`. The `"dataset"` setting in `settings.json`
picks which dataset the steps run on (`casting` or `packages`). Each dataset gets its own Custom Labels
project (`casting-inspection`, `packages-inspection`), its own S3 prefix and its own `results/<dataset>/`
folder, so the two runs never overwrite each other. Do the full run on `casting` first, then switch.

| Command | What it does | Time | Cost (casting run) |
|---|---|---|---|
| `check_setup.py` | Checks the sign-in, bucket, Rekognition access and dataset split. Spends nothing. | 1 min | $0 |
| `step1_baseline.py` | DetectLabels on 50 held-out images, tallies what it sees for each class | 2 min | free tier, else about 5 cents |
| `step2_train.py` | Uploads the train and test images, builds the datasets, trains, prints precision/recall/F1 | about 1 to 2 hours, mostly waiting | about $1 to $3 |
| `step3_test.py` | Starts the model, tests the held-out images, **stops it**, prints the confusion matrix | 10 to 40 min | about $0.50 to $4, depending on how long the model takes to start (about $0.50 here) |
| `stop_model.py` | Shows every project's models and stops anything running, whichever dataset is selected | 1 to 5 min | $0 |

You can close the window during `step2_train.py`. Running it again picks up where it left off.

**The one way to waste money:** a running model bills about $4 an hour until it stops. `step3_test.py`
always stops it, even after an error or Ctrl+C. If the PC restarts mid-run, run `stop_model.py`. When
in doubt, run `stop_model.py`; it's always safe.

Results land in `results/<dataset>/` (git-ignored). For a presentation, screenshot the model's Evaluation
page in the Rekognition console (Custom Labels > Projects > `casting-inspection` or `packages-inspection` >
the model). The console opens in one Region at a time; make sure it is the one in `settings.json`.

## The results site

`build_site.py` turns `results/` into a static page in `site/` (git-ignored): held-out results, the
off-the-shelf baseline, confusion matrices, and a gallery of every held-out image that opens full size on click.
Only numbers and labels are copied out of the AWS result files, and the build refuses to run if the page would
contain an account ID, ARN, S3 path, the bucket name, or any term listed under `"private_terms"` in
`settings.json`. Images are copied byte for byte, never edited.

`deploy_site.py` publishes `site/` with AWS Amplify Hosting (HTTPS, free tier at this size) and prints the
link. Re-running it publishes the current build to the same link.

```powershell
.venv\Scripts\python build_site.py
.venv\Scripts\python deploy_site.py
```

## Phase 2: where the model looks (SageMaker, in progress)

`sagemaker/` holds a second experiment: fine-tune a ResNet-18 on the same training images in an Amazon
SageMaker AI training job, test it on the same held-out images, and draw a Grad-CAM heatmap for each one,
showing which parts of the photo pushed the model toward "defective" or "damaged". The results site adds a
heatmap library (heat drawn as a separate, switchable layer over the unmodified photo) once real results exist.

```powershell
.venv\Scripts\pip install -r sagemaker\requirements-launch.txt
.venv\Scripts\python sagemaker\launch.py role       # least-privilege execution role (one bucket, three prefixes)
.venv\Scripts\python sagemaker\launch.py upload     # held-out images to s3://<bucket>/heldout/
.venv\Scripts\python sagemaker\launch.py preflight  # quota, role, pinned container, bucket Region (free)
.venv\Scripts\python sagemaker\launch.py start      # one ml.m5.xlarge job, capped at 1 hour
.venv\Scripts\python sagemaker\launch.py fetch      # waits, then saves results to results/<dataset>/sagemaker/
```

New AWS accounts start with a SageMaker training quota of 0; request "ml.m5.xlarge for training job usage" = 1
in Service Quotas first. Status: the code is written and checked locally; the cloud run is waiting on that quota.

## Cleaning up

A stopped model costs nothing to keep, so you can start it again for a live demo with `step3_test.py`.
When you're done with the project for good:
- Delete both projects in the Rekognition console (Custom Labels > Projects).
- Empty and delete the bucket in S3.
- Delete the Amplify app `inspection-results` (this takes the results site offline).
- Delete the IAM role `vision-inspection-sagemaker-training`, if you ran phase 2.

## Data and credits

- **Casting:** "casting product image data for quality inspection" on Kaggle,
  [ravirajsinh45/real-life-industrial-dataset-of-casting-product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product).
  By Ravirajsinh Dabhi, licensed
  [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (attribution, non-commercial, no
  derivatives). Top-view photos of submersible pump impellers, sorted into `def_front` and `ok_front`.
  Because of the non-commercial term, this project is a demonstration only: a production system would be
  trained on the customer's own parts.
- **Packages:** "Damaged vs good packages", provided by a Roboflow user on Roboflow Universe,
  [aadhavs-first-workspace/damaged-vs-good-packages](https://universe.roboflow.com/aadhavs-first-workspace/damaged-vs-good-packages),
  licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 581 images, resized to 224×224 by
  Roboflow. The original split is re-pooled here and divided 65/20/15 into train, test and held out.

No images are stored in this repository. Download them from the links above.

## Why no labeling tool is used

Both datasets arrive already sorted into one folder per class, so no labeling job (for example SageMaker
Ground Truth) is needed. The labels come from those folder names and are written in Ground Truth's manifest
format, which Custom Labels reads directly.
