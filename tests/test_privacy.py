"""The results site's privacy check: the build must refuse to publish anything that identifies the AWS account."""
import pytest

from build_site import privacy_check

BUCKET = "custom-labels-example-bucket"


@pytest.mark.parametrize("text, reason", [
    ("Owner 123456789012 trained this", "12-digit number"),
    ("model arn:aws:rekognition:us-east-2:project/x", "AWS ARN"),
    ("images at s3://somewhere/images/", "S3 path"),
    ("see https://s3.us-east-2.amazonaws.com/x", "endpoint hostname"),
    (f"stored in {BUCKET}", "bucket's name"),
    ("Prepared for Example Robotics", "private_terms"),
    ("prepared for EXAMPLE ROBOTICS", "private_terms"),  # any letter case
])
def test_blocks_identifying_text(text, reason):
    with pytest.raises(SystemExit, match=reason):
        privacy_check(text, BUCKET, ["Example Robotics"])


@pytest.mark.parametrize("text", [
    "<p>Held-out accuracy 88.6% on 88 images</p>",
    "a 13-digit number 1234567890123 is not an account ID",
    "an 11-digit number 12345678901 is not either",
    "Hosted on AWS Amplify Hosting, built with Amazon Rekognition",
])
def test_allows_clean_text(text):
    privacy_check(text, BUCKET, ["Example Robotics"])  # must not exit


def test_empty_settings_do_not_block_everything():
    # An empty private term or bucket name would otherwise "match" every page.
    privacy_check("anything", "", [""])
