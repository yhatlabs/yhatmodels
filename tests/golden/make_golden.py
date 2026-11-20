#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Write the expected outputs of a released model for tests/test_golden.py (CPU, fp32), tied to its weights' digest.
Run once per release, from a checkout whose outputs were checked against the research code:
    PYTHONPATH=src python tests/golden/make_golden.py MODEL_FOLDER"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cases import run

from yhatmodels import ChakraTS
from yhatmodels.hub import read_config, weight_files

folder = Path(sys.argv[1])
torch.manual_seed(0)
model = ChakraTS.from_pretrained(folder, device="cpu")
digest = hashlib.sha256(b"".join(f.read_bytes() for f in weight_files(folder))).hexdigest()
name = read_config(folder)["model_name"]
out = Path(__file__).parent / f"{name}.npz"
np.savez_compressed(out, weights_sha256=np.array(digest), **run(model))
print(f"wrote {out} ({out.stat().st_size // 1024} KB) for weights {digest[:12]}")
