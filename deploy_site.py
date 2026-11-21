"""Publishes site/ with AWS Amplify Hosting (HTTPS on AWS's CDN), creating the app on first run.

Uses Amplify's manual deploy: the site is zipped and uploaded straight to Amplify, so no public S3 bucket is
involved and the training bucket is never exposed. The app ID is kept in site.json, which is git-ignored.
Re-running publishes the current site/ as a new deployment. Cost at presentation-sized traffic: about $0
(within Amplify Hosting's free tier).

(CloudFront + a private S3 bucket would also work, but new AWS accounts can be blocked from creating CloudFront
resources until AWS Support verifies them.)
"""
import io
import json
import sys
import time
import urllib.request
import zipfile

from common import ROOT, client, run

SITE = ROOT / "site"
STATE = ROOT / "site.json"
APP_NAME = "inspection-results"
BRANCH = "main"


def zip_site():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SITE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(SITE).as_posix())
    return buf.getvalue()


def main():
    if not (SITE / "index.html").exists():
        sys.exit("No site/index.html. Run build_site.py first.")
    amp = client("amplify")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}

    if "app_id" not in state:
        print("Creating the Amplify Hosting app...")
        app = amp.create_app(name=APP_NAME, platform="WEB", description="Inspection results site")["app"]
        state = {"app_id": app["appId"], "domain": app["defaultDomain"]}
        STATE.write_text(json.dumps(state, indent=2))
    try:
        amp.get_branch(appId=state["app_id"], branchName=BRANCH)
    except amp.exceptions.NotFoundException:
        amp.create_branch(appId=state["app_id"], branchName=BRANCH, stage="PRODUCTION")

    body = zip_site()
    dep = amp.create_deployment(appId=state["app_id"], branchName=BRANCH)
    req = urllib.request.Request(dep["zipUploadUrl"], data=body, method="PUT",
                                 headers={"Content-Type": "application/zip"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status not in (200, 201):
            sys.exit(f"Upload failed with HTTP {r.status}.")
    print(f"Uploaded the site ({len(body) / 1e6:.1f} MB zipped).")
    amp.start_deployment(appId=state["app_id"], branchName=BRANCH, jobId=dep["jobId"])

    status = "PENDING"
    for _ in range(60):
        status = amp.get_job(appId=state["app_id"], branchName=BRANCH, jobId=dep["jobId"])["job"]["summary"]["status"]
        if status in ("SUCCEED", "FAILED", "CANCELLED"):
            break
        time.sleep(10)
    if status != "SUCCEED":
        sys.exit(f"Amplify deployment ended as {status}. Check the Amplify console for the app '{APP_NAME}'.")

    url = f"https://{BRANCH}.{state['domain']}/"
    for _ in range(12):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                if r.status == 200:
                    print(f"Live: {url}")
                    return
        except Exception:
            pass
        time.sleep(10)
    print(f"Deployed, but the page didn't answer yet. Try in a minute: {url}")


if __name__ == "__main__":
    run(main)
