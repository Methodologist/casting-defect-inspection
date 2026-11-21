"""Step 0: checks the AWS key, the bucket, Rekognition access and the dataset before anything costs money."""
from common import client, dataset, run, settings, splits


def main():
    s = settings()
    ds = dataset()
    print(f"Dataset {ds['name']} ({ds['title']}), region {s['region']}, profile {s['profile'] or 'default'}, "
          f"bucket {s['bucket']}, project {s['project']}")

    who = client("sts").get_caller_identity()
    print("OK  AWS sign-in works")
    if who["Arn"].endswith(":root"):
        print("    WARNING: these are the root user's access keys. Use `aws login` instead (see README).")

    s3 = client("s3")
    where = s3.get_bucket_location(Bucket=s["bucket"]).get("LocationConstraint") or "us-east-1"
    if where != s["region"]:
        raise SystemExit(f"The bucket is in {where} but settings.json says {s['region']}. Make them match.")
    s3.put_object(Bucket=s["bucket"], Key="setup-check.txt", Body=b"ok")
    print("OK  bucket exists in the right region and is writable")

    client("rekognition").describe_projects(MaxResults=1)
    print("OK  Rekognition Custom Labels is reachable")

    sp = splits()
    for split in ("train", "test", "holdout"):
        counts = ", ".join(f"{sum(1 for _, l in sp[split] if l == label)} {label}" for label in ds["labels"])
        print(f"OK  {split:<8} {len(sp[split]):>4} images ({counts})")
    print("\nAll set. Next: python step1_baseline.py")


if __name__ == "__main__":
    run(main)
