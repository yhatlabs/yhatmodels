"""A released model's outputs on fixed inputs, against those recorded for its exact weights (tests/golden/<model>.npz),
so no change to yhatmodels can alter a released model's forecasts unnoticed. Needs the weights: set
YHATMODELS_TEST_MODEL to the model's folder (skipped otherwise). YHATMODELS_TEST_DEVICE=cuda / mps runs it on a GPU: the
outputs were recorded on CPU in fp32, and CUDA runs in bf16, so it gets a wider tolerance."""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from yhatmodels import ChakraTS
from yhatmodels.hub import read_config, weight_files

sys.path.insert(0, str(Path(__file__).parent / "golden"))
from cases import run

FOLDER = os.environ.get("YHATMODELS_TEST_MODEL")
DEVICE = os.environ.get("YHATMODELS_TEST_DEVICE", "cpu")
TOLERANCE = 3e-2 if DEVICE.startswith("cuda") else 1e-4  # bf16 on CUDA


@pytest.mark.skipif(not FOLDER, reason="set YHATMODELS_TEST_MODEL to a released model's folder")
def test_released_model_reproduces_its_recorded_forecasts():
    folder = Path(FOLDER)
    golden = np.load(Path(__file__).parent / "golden" / f"{read_config(folder)['model_name']}.npz")
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in weight_files(folder))).hexdigest()
    assert digest == str(golden["weights_sha256"]), "these are not the weights the golden outputs were recorded for"
    got = run(ChakraTS.from_pretrained(folder, device=DEVICE))
    for name, value in got.items():
        np.testing.assert_allclose(value, golden[name], rtol=TOLERANCE, atol=TOLERANCE, err_msg=name)
