"""Step 0: checks the AWS key, the bucket, Rekognition access and the dataset before anything costs money."""
from common import LABELS, client, find_images, run, settings, splits


def main():
    s = settings()
    print(f"Region {s['region']}, bucket {s['bucket']}, project {s['project']}")

    who = client("sts").get_caller_identity()
    print(f"OK  AWS key works (account {who['Account']})")
    if who["Arn"].endswith(":root"):
        print("    WARNING: this is the root user's key. Make an IAM user for this project instead (see README).")

    s3 = client("s3")
    where = s3.get_bucket_location(Bucket=s["bucket"]).get("LocationConstraint") or "us-east-1"
    if where != s["region"]:
        raise SystemExit(f"The bucket is in {where} but settings.json says {s['region']}. Make them match.")
    s3.put_object(Bucket=s["bucket"], Key="setup-check.txt", Body=b"ok")
    print("OK  bucket exists in the right region and is writable")

    client("rekognition").describe_projects(MaxResults=1)
    print("OK  Rekognition Custom Labels is reachable")

    images = find_images()
    for split in ("train", "test"):
        print(f"OK  dataset {split}: " + ", ".join(f"{len(images[(split, l)])} {l}" for l in LABELS))
    sp = splits()
    print(f"    Will use {len(sp['train'])} training, {len(sp['test'])} test and {len(sp['holdout'])} holdout images.")
    print("\nAll set. Next: python step1_baseline.py")


if __name__ == "__main__":
    run(main)
