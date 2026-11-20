"""Safety switch: shows each model's state and stops any that is running. Run it any time you're unsure."""
from common import client, find_project, run, settings, stop_model


def main():
    s = settings()
    rek = client("rekognition")
    project = find_project(rek, s["project"])
    if project is None:
        print("No project yet, so nothing is running.")
        return
    versions = rek.describe_project_versions(ProjectArn=project["ProjectArn"]).get("ProjectVersionDescriptions", [])
    for v in versions:
        print(f"Model {v['ProjectVersionArn'].split('/')[-2]}: {v['Status']}")
    live = [v for v in versions if v["Status"] in ("STARTING", "RUNNING", "STOPPING")]
    if not live:
        print("Nothing is running, so nothing is being billed for inference.")
    for v in live:
        stop_model(rek, project["ProjectArn"], v["ProjectVersionArn"].split("/")[-2])


if __name__ == "__main__":
    run(main)
