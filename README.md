# Casting defect inspection with Amazon Rekognition

A small computer vision project: tell good cast pump impellers from defective ones, the kind of pass/fail
check a camera does in a robot cell.

1. **Baseline.** Off-the-shelf Rekognition (DetectLabels) describes the part but can't judge it.
2. **Custom model.** Rekognition Custom Labels is trained on 1,000 labeled images and evaluated on 200 more.
3. **Held-out test.** The trained model is checked on 100 images it never saw. Missed defects and false
   rejects are counted separately, because on a line they cost different things.

Data: the "casting product image data for quality inspection" dataset on Kaggle (top-view photos of
submersible pump impellers, sorted into `def_front` and `ok_front`).

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
blocked). The name must contain `custom-labels`, for example `custom-labels-casting-demo`. That
matches the `AmazonRekognitionCustomLabelsFullAccess` policy, in case you run this as an IAM user.

```powershell
aws s3 mb s3://custom-labels-casting-demo --region us-east-2 --profile vision-inspection
```

Copy `settings.example.json` to `settings.json` and put your bucket name, Region and profile in it.
`settings.json` is git-ignored.

**4. Get the images.** Download the dataset zip from Kaggle and extract it into `data/` here, so you
end up with `data/casting_data/casting_data/train/def_front` and so on. The exact depth doesn't matter.

**5. Install boto3** (the AWS SDK for Python), from this folder in PowerShell. The `[crt]` extra is
what lets boto3 use `aws login` credentials.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Running it

Run everything from this folder with `.venv\Scripts\python`.

| Command | What it does | Time | Cost |
|---|---|---|---|
| `check_setup.py` | Checks the sign-in, bucket, Rekognition access and dataset. Spends nothing. | 1 min | $0 |
| `step1_baseline.py` | DetectLabels on 50 images, tallies what it sees for good vs defective | 2 min | free tier, else about 5 cents |
| `step2_train.py` | Uploads 1,200 images, builds the datasets, trains, prints precision/recall/F1 | about 1 to 2 hours, mostly waiting | about $1 to $3 |
| `step3_test.py` | Starts the model, tests 100 unseen images, **stops it**, prints the confusion matrix | 20 to 40 min | about $2 to $4 |
| `stop_model.py` | Shows each model's state and stops anything running | 1 to 5 min | $0 |

You can close the window during `step2_train.py`. Running it again picks up where it left off.

**The one way to waste money:** a running model bills about $4 an hour until it stops. `step3_test.py`
always stops it, even after an error or Ctrl+C. If the PC restarts mid-run, run `stop_model.py`. When
in doubt, run `stop_model.py`; it's always safe.

Results land in `results/`. For a presentation, screenshot the model's Evaluation page in the Rekognition
console (Custom Labels > Projects > casting-inspection > the model).

## Cleaning up

A stopped model costs nothing to keep, so you can start it again for a live demo with `step3_test.py`.
When you're done with the project for good:
- Delete the project in the Rekognition console (Custom Labels > Projects).
- Empty and delete the bucket in S3.

## Why Ground Truth isn't used

SageMaker Ground Truth is closed to new AWS customers. The labels here come from the dataset's folder
names and are written in Ground Truth's manifest format, which Custom Labels reads directly.
