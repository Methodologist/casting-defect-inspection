# Casting defect inspection with Amazon Rekognition

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
| `step3_test.py` | Starts the model, tests the held-out images, **stops it**, prints the confusion matrix | 20 to 40 min | about $2 to $4 |
| `stop_model.py` | Shows every project's models and stops anything running, whichever dataset is selected | 1 to 5 min | $0 |

You can close the window during `step2_train.py`. Running it again picks up where it left off.

**The one way to waste money:** a running model bills about $4 an hour until it stops. `step3_test.py`
always stops it, even after an error or Ctrl+C. If the PC restarts mid-run, run `stop_model.py`. When
in doubt, run `stop_model.py`; it's always safe.

Results land in `results/<dataset>/` (git-ignored). For a presentation, screenshot the model's Evaluation
page in the Rekognition console (Custom Labels > Projects > `casting-inspection` or `packages-inspection` >
the model). The console opens in one Region at a time; make sure it is the one in `settings.json`.

## Cleaning up

A stopped model costs nothing to keep, so you can start it again for a live demo with `step3_test.py`.
When you're done with the project for good:
- Delete both projects in the Rekognition console (Custom Labels > Projects).
- Empty and delete the bucket in S3.

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

## Why Ground Truth isn't used

SageMaker Ground Truth is closed to new AWS customers. The labels here come from the dataset's folder
names and are written in Ground Truth's manifest format, which Custom Labels reads directly.
