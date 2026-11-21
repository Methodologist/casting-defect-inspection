"""Safety switch: shows each model's state and stops any that is running. Run it any time you're unsure.

It checks every dataset's project, not just the one selected in settings.json, so switching datasets can't
leave a forgotten model billing in the background.
"""
from common import DATASETS, client, find_project, run, settings, stop_model


def main():
    s = settings()
    rek = client("rekognition")
    names = dict.fromkeys([s["project"]] + [f"{name}-inspection" for name in DATASETS])
    any_project = False
    for name in names:
        project = find_project(rek, name)
        if project is None:
            continue
        any_project = True
        versions = rek.describe_project_versions(ProjectArn=project["ProjectArn"]).get(
            "ProjectVersionDescriptions", [])
        for v in versions:
            print(f"{name} / model {v['ProjectVersionArn'].split('/')[-2]}: {v['Status']}")
        for v in versions:
            if v["Status"] in ("STARTING", "RUNNING", "STOPPING"):
                stop_model(rek, project["ProjectArn"], v["ProjectVersionArn"].split("/")[-2])
    if not any_project:
        print("No projects yet, so nothing is running.")
    else:
        print("Checked every project. Any warning above means a model may still be billing.")


if __name__ == "__main__":
    run(main)
