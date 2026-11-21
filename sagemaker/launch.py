"""Runs the Grad-CAM heatmap job on SageMaker, one visible step at a time. Only `start` costs money.

    python sagemaker/launch.py role       create/update the job's execution role (prints its permissions)
    python sagemaker/launch.py upload     upload held-out images to s3://<bucket>/heldout/<dataset>/<label>/
    python sagemaker/launch.py preflight  check quota, role trust, container image and bucket Region (free)
    python sagemaker/launch.py start      launch the training job (billed while it runs, capped by MAX_RUNTIME)
    python sagemaker/launch.py fetch      wait for the job, then download results to results/<dataset>/sagemaker/

Everything stays in the one private training bucket from settings.json (prefixes images/, heldout/, sagemaker/).
Job details are kept in results/sagemaker_job.json (git-ignored).
"""
import io
import json
import re
import sys
import tarfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

ROOT = common.ROOT
JOB_DIR = Path(__file__).resolve().parent / "job"
STATE = ROOT / "results" / "sagemaker_job.json"
ROLE_NAME = "vision-inspection-sagemaker-training"
INSTANCE_TYPE = "ml.m5.xlarge"          # CPU; quota "ml.m5.xlarge for training job usage" must be >= 1
QUOTA_CODE = "L-CCE2AFA6"
MAX_RUNTIME_SECONDS = 3600              # hard spend cap: SageMaker stops the job after 1 hour
IMAGE_REPO = "pytorch-training"
IMAGE_TAG = "2.5.1-cpu-py311"           # same PyTorch/Python as the local check
IMAGE_DIGEST = "sha256:2e54be2c47e2b18b238bc8370c5f135cc0462f63a99cf3b0191f779fb98ec9ef"
DLC_ACCOUNT = "763104351884"            # AWS Deep Learning Containers registry (owned by AWS)


def ctx():
    s = common.settings()
    sts = common.client("sts").get_caller_identity()
    return s, sts["Account"], s["region"], s["bucket"]


def image_uri(region):
    return f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/{IMAGE_REPO}@{IMAGE_DIGEST}"


def role_policies(account, region, bucket):
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}
    perms = {"Version": "2012-10-17", "Statement": [
        {"Sid": "ReadInputs", "Effect": "Allow", "Action": "s3:GetObject",
         "Resource": [f"arn:aws:s3:::{bucket}/images/*", f"arn:aws:s3:::{bucket}/heldout/*",
                      f"arn:aws:s3:::{bucket}/sagemaker/*"]},
        {"Sid": "ListInputs", "Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}",
         "Condition": {"StringLike": {"s3:prefix": ["images/*", "heldout/*", "sagemaker/*"]}}},
        {"Sid": "WriteOutputs", "Effect": "Allow", "Action": "s3:PutObject",
         "Resource": f"arn:aws:s3:::{bucket}/sagemaker/*"},
        {"Sid": "Logs", "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
         "Resource": f"arn:aws:logs:{region}:{account}:log-group:/aws/sagemaker/TrainingJobs*"},
        {"Sid": "Metrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*",
         "Condition": {"StringLike": {"cloudwatch:namespace": "/aws/sagemaker/*"}}},
        {"Sid": "PullTrainingImage", "Effect": "Allow",
         "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
         "Resource": f"arn:aws:ecr:{region}:{DLC_ACCOUNT}:repository/{IMAGE_REPO}"},
        {"Sid": "EcrAuth", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
    ]}
    return trust, perms


def cmd_role():
    s, account, region, bucket = ctx()
    iam = common.client("iam")
    trust, perms = role_policies(account, region, bucket)
    try:
        iam.get_role(RoleName=ROLE_NAME)
        iam.update_assume_role_policy(RoleName=ROLE_NAME, PolicyDocument=json.dumps(trust))
        print(f"Role {ROLE_NAME} exists; trust policy refreshed.")
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                        Description="SageMaker training for the vision-inspection Grad-CAM job (one bucket only)",
                        MaxSessionDuration=3600)
        print(f"Created role {ROLE_NAME}.")
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="vision-inspection-training",
                        PolicyDocument=json.dumps(perms))
    print("Permissions: read images/, heldout/ and sagemaker/ in the training bucket; write sagemaker/ only; "
          "write this job's logs and metrics; pull the AWS PyTorch container. Nothing else.")


def heldout_items():
    import os
    original = os.environ.get("VISION_DATASET")
    out = []
    try:
        for name in common.DATASETS:
            os.environ["VISION_DATASET"] = name  # the supported per-command dataset override
            out += [(name, label, p) for p, label in common.splits()["holdout"]]
    finally:
        if original is None:
            os.environ.pop("VISION_DATASET", None)
        else:
            os.environ["VISION_DATASET"] = original
    return out


def cmd_upload():
    s, _, _, bucket = ctx()
    s3 = common.client("s3")
    items = heldout_items()
    existing = {o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="heldout/")
                for o in page.get("Contents", [])}
    todo = [(f"heldout/{n}/{label}/{p.name}", p) for n, label, p in items
            if f"heldout/{n}/{label}/{p.name}" not in existing]
    for key, p in todo:
        s3.put_object(Bucket=bucket, Key=key, Body=p.read_bytes(), ContentType=common.EXTENSIONS[p.suffix.lower()])
    print(f"Held-out images in S3: {len(items)} ({len(todo)} uploaded now).")


def cmd_preflight():
    s, account, region, bucket = ctx()
    ok = True
    q = common.client("service-quotas").get_service_quota(ServiceCode="sagemaker", QuotaCode=QUOTA_CODE)["Quota"]
    print(f"{'OK ' if q['Value'] >= 1 else 'NO '} quota '{q['QuotaName']}' = {q['Value']:.0f} (need 1)")
    ok &= q["Value"] >= 1
    try:
        doc = common.client("iam").get_role(RoleName=ROLE_NAME)["Role"]["AssumeRolePolicyDocument"]
        trusted = "sagemaker.amazonaws.com" in json.dumps(doc)
        print(f"{'OK ' if trusted else 'NO '} role {ROLE_NAME} trusts SageMaker")
        ok &= trusted
    except Exception:
        print(f"NO  role {ROLE_NAME} not found (run: launch.py role)")
        ok = False
    common.client("ecr").describe_images(registryId=DLC_ACCOUNT, repositoryName=IMAGE_REPO,
                                         imageIds=[{"imageDigest": IMAGE_DIGEST}])
    print(f"OK  container {IMAGE_REPO}:{IMAGE_TAG} (pinned by digest) exists in {region}")
    where = common.client("s3").get_bucket_location(Bucket=bucket).get("LocationConstraint") or "us-east-1"
    print(f"{'OK ' if where == region else 'NO '} bucket is in {where}")
    ok &= where == region
    n = common.client("s3").list_objects_v2(Bucket=bucket, Prefix="heldout/").get("KeyCount", 0)
    print(f"{'OK ' if n else 'NO '} held-out images uploaded ({n}{'+' if n == 1000 else ''})")
    ok &= n > 0
    print("\nReady to start." if ok else "\nNot ready yet.")
    return ok


def cmd_start():
    if not cmd_preflight():
        sys.exit("Preflight failed; nothing was started.")
    import boto3
    from sagemaker.core import Attribution, set_attribution
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.training.configs import Compute, InputData, OutputDataConfig, SourceCode, StoppingCondition
    from sagemaker.train import ModelTrainer

    set_attribution(Attribution.SAGEMAKER_AGENT_PLUGIN)
    s, account, region, bucket = ctx()
    boto_session = boto3.Session(profile_name=s["profile"], region_name=region)
    sm = boto_session.client("sagemaker")
    session = Session(boto_session=boto_session, sagemaker_client=sm, default_bucket=bucket,
                      default_bucket_prefix="sagemaker")
    prefix = f"inspection-gradcam-{uuid.uuid4().hex[:10]}"
    trainer = ModelTrainer(
        training_image=image_uri(region),
        source_code=SourceCode(source_dir=str(JOB_DIR), entry_script="train_gradcam.py"),
        compute=Compute(instance_type=INSTANCE_TYPE, instance_count=1, volume_size_in_gb=30),
        output_data_config=OutputDataConfig(s3_output_path=f"s3://{bucket}/sagemaker/output/"),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_SECONDS),
        role=f"arn:aws:iam::{account}:role/{ROLE_NAME}",
        sagemaker_session=session,
        base_job_name=prefix,
    )
    trainer.train(input_data_config=[
        InputData(channel_name="images", data_source=f"s3://{bucket}/images/"),
        InputData(channel_name="heldout", data_source=f"s3://{bucket}/heldout/"),
    ], wait=False, logs=False)
    for _ in range(30):
        jobs = [j for j in sm.list_training_jobs(NameContains=prefix, SortBy="CreationTime", SortOrder="Descending",
                                                 MaxResults=10)["TrainingJobSummaries"]
                if j["TrainingJobName"].startswith(prefix)]
        if jobs:
            break
        time.sleep(2)
    else:
        sys.exit("The job was submitted but did not appear in the job list; check the SageMaker console.")
    name = jobs[0]["TrainingJobName"]
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"job_name": name}, indent=2))
    print(f"Started training job {name} on {INSTANCE_TYPE}, capped at {MAX_RUNTIME_SECONDS // 60} minutes.")


def cmd_fetch():
    if not STATE.exists():
        sys.exit("No job started yet (run: launch.py start).")
    name = json.loads(STATE.read_text())["job_name"]
    sm, s3 = common.client("sagemaker"), common.client("s3")
    last = None
    while True:
        d = sm.describe_training_job(TrainingJobName=name)
        status = f"{d['TrainingJobStatus']} / {d.get('SecondaryStatus', '')}"
        if status != last:
            print(f"{time.strftime('%H:%M:%S')}  {status}", flush=True)
            last = status
        if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(30)
    billable = d.get("BillableTimeInSeconds", 0)
    print(f"Billable time: {billable // 60} min {billable % 60} s.")
    if d["TrainingJobStatus"] != "Completed":
        sys.exit(f"Job ended as {d['TrainingJobStatus']}: {d.get('FailureReason', '')}")
    m = re.match(r"s3://([^/]+)/(.+)$", d["ModelArtifacts"]["S3ModelArtifacts"].replace("model.tar.gz", "output.tar.gz"))
    body = s3.get_object(Bucket=m.group(1), Key=m.group(2))["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not member.isfile() or ".." in parts or parts[0] not in common.DATASETS:
                continue
            dest = ROOT / "results" / parts[0] / "sagemaker" / Path(*parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(member).read())
    for name_ in common.DATASETS:
        mfile = ROOT / "results" / name_ / "sagemaker" / "metrics.json"
        if mfile.exists():
            mt = json.loads(mfile.read_text())
            print(f"{name_}: held-out accuracy {mt['heldout_accuracy']:.3f} "
                  f"(tp {mt['tp']} fn {mt['fn']} fp {mt['fp']} tn {mt['tn']})")
    print("Saved to results/<dataset>/sagemaker/. Next: python build_site.py, then python deploy_site.py")


if __name__ == "__main__":
    commands = {"role": cmd_role, "upload": cmd_upload, "preflight": cmd_preflight, "start": cmd_start,
                "fetch": cmd_fetch}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    common.run(commands[sys.argv[1]])
