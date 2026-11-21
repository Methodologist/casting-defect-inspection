"""Lets the tests import the project's scripts (common.py, build_site.py, ...) from the repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
