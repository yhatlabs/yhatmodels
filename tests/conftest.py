import json

import pytest
import torch
from safetensors.torch import save_file

from yhatmodels.ts.architectures.chakrats_v1 import ChakraTSNet, Config

TINY = {"patch": 8, "d": 32, "heads": 2, "ff": 64, "schedule": ["t", "tv", "t", "tv"], "unet": True, "time_channel": True,
        "xpos": True, "xpos_span": 256, "patches": [2, 4], "out_bound": 10.0}  # fmt: skip


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A randomly initialised model in the released folder layout (config.json + model.safetensors)."""
    torch.manual_seed(0)
    net = ChakraTSNet(Config.from_dict(TINY))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.02 * torch.randn_like(p))
    d = tmp_path_factory.mktemp("tiny")
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, d / "model.safetensors")
    config = {"architecture": "chakrats_v1", "model_name": "tiny", "model": TINY,
              "inference": {"max_context": 512, "presets": {"fast": {"multi_scale": False}, "pro": {"multi_scale": True}}}}  # fmt: skip
    (d / "config.json").write_text(json.dumps(config))
    return d
