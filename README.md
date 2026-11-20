# Casting defect inspection with Amazon Rekognition

A small computer vision project: tell good cast pump impellers from defective ones, the kind of pass/fail
check a camera does in a robot cell.

1. **Baseline.** Off-the-shelf Rekognition (DetectLabels) describes the part but can't judge it.
2. **Custom model.** Rekognition Custom Labels is trained on 1,000 labeled images and evaluated on 200 more.
3. **Held-out test.** The trained model is checked on 100 images it never saw. Missed defects and false
   rejects are counted separately, because on a line they cost different things.

Data: the "casting product image data for quality inspection" dataset on Kaggle (top-view photos of
submersible pump impellers, sorted into `def_front` and `ok_front`).

## One-time setup (about 30 minutes)

**1. Set a budget alert first.** In the AWS console, go to Billing > Budgets > Create budget. Choose a
monthly cost budget of $20 with an email alert.

**2. Make an IAM user for this project.** Don't use the root account's keys.
- IAM > Users > Create user, named `vision-project`. It doesn't need console access.
- Attach two policies: `AmazonRekognitionCustomLabelsFullAccess` and `AmazonRekognitionReadOnlyAccess`.
- Open the user > Security credentials > Create access key > "Local code".
- Copy `.aws/credentials.example` to `.aws/credentials` and paste the key ID and secret into it. That
  file is git-ignored. Never paste the secret anywhere else.

**3. Make the S3 bucket.** S3 > Create bucket, region **US East (N. Virginia) us-east-1**, with the
default settings (public access stays blocked). The name must contain `custom-labels`, for example
`custom-labels-casting-demo`, because the Custom Labels policy only covers buckets named that way.
Copy `settings.example.json` to `settings.json` and put the bucket name in it.

**4. Get the images.** Download the dataset zip from Kaggle and extract it into `data/` here, so you
end up with `data/casting_data/casting_data/train/def_front` and so on. The exact depth doesn't matter.

**5. Install boto3** (the AWS SDK for Python), from this folder in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Running it

Run everything from this folder with `.venv\Scripts\python`.

| Command | What it does | Time | Cost |
|---|---|---|---|
| `check_setup.py` | Checks the key, bucket, Rekognition access and dataset. Spends nothing. | 1 min | $0 |
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
- Deactivate the IAM user's access key.

## Why Ground Truth isn't used

SageMaker Ground Truth is closed to new AWS customers. The labels here come from the dataset's folder
names and are written in Ground Truth's manifest format, which Custom Labels reads directly.
