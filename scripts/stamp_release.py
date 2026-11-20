#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Stamp a model folder's config.json before it is uploaded: the sha256 of every weight file (checked when the model
loads) and the oldest yhatmodels that runs it. Run it with the yhatmodels the release was checked with, then record the
golden outputs (tests/golden/make_golden.py) from the same folder.

    PYTHONPATH=src python scripts/stamp_release.py MODEL_FOLDER [--requires 0.1.0]
"""

import argparse
import hashlib
import json
from pathlib import Path

import yhatmodels
from yhatmodels.hub import CONFIG, read_config, weight_files

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("--requires", default=yhatmodels.__version__, help="the oldest yhatmodels that runs this model")
a = ap.parse_args()

config = read_config(a.folder)
config["sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in weight_files(a.folder)}
config["requires_yhatmodels"] = a.requires
(a.folder / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
print(f"{a.folder / CONFIG}: sha256 for {len(config['sha256'])} file(s), requires yhatmodels >= {a.requires}")
