import json
import warnings

import numpy as np
import pytest
import torch

from yhatmodels import ChakraTS
from yhatmodels.ts import tune as tn


@pytest.fixture(scope="module")
def model(tiny_model_dir):
    return ChakraTS.from_pretrained(tiny_model_dir, device="cpu")


def _series(n=3, T=400):
    r = np.random.default_rng(0)
    t = np.arange(T)
    return [(10 + 2 * np.sin(t / 24 * 2 * np.pi + i) + r.normal(0, 0.3, T)).astype(np.float32) for i in range(n)]


def test_training_learns_and_merging_keeps_the_function(model):
    w = tn._Windows([(y, None, len(y)) for y in _series()], 16, model.net.cfg.patch, np.random.default_rng(0))
    batch = w.batch(4)
    import copy

    net = copy.deepcopy(model.net)
    wrapped = tn._wrap(net, 4)
    opt = torch.optim.AdamW([p for _, lo in wrapped for p in (lo.A, lo.B)], lr=1e-2)
    first = float(tn._loss(net, *batch, "cpu").detach())
    for _ in range(60):
        loss = tn._loss(net, *batch, "cpu")
        opt.zero_grad()
        loss.backward()
        opt.step()
    trained = float(tn._loss(net, *batch, "cpu").detach())
    assert trained < 0.7 * first
    tn._merge(wrapped)
    assert float(tn._loss(net, *batch, "cpu").detach()) == pytest.approx(trained, rel=1e-4)
    assert not any(isinstance(m, torch.nn.utils.parametrize.ParametrizationList) for m in net.modules())


def test_tune_keeps_the_base_model_unchanged_and_saves_a_derivative(model, tmp_path):
    ys = _series()
    before = model.forecast(ys, 16)
    tuned = model.tune(ys, 16, steps=20, min_gain=-1.0)  # keep whatever it learned: this tests the path, not the gain
    assert tuned.report["used"] and tuned.report["steps"] >= 1 and tuned.info["name"] == "tiny-tuned"
    after = model.forecast(ys, 16)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))  # the pretrained model is untouched
    (model.folder / "LICENSE.md").write_text("licence text")
    (model.folder / "NOTICE").write_text("tiny notice\n")
    folder = tuned.save(tmp_path / "tuned")
    assert sorted(p.name for p in folder.iterdir()) == ["LICENSE.md", "NOTICE", "config.json", "model.safetensors"]
    assert "was fine-tuned from tiny" in (folder / "NOTICE").read_text()
    config = json.loads((folder / "config.json").read_text())
    assert config["base_model"] == "tiny" and "model.safetensors" in config["sha256"] and config["info"]["tune"]["used"]
    back = ChakraTS.from_pretrained(folder, device="cpu")
    assert all(np.allclose(a, b) for a, b in zip(back.forecast(ys, 16), tuned.forecast(ys, 16), strict=True))


def test_a_tune_that_does_not_help_returns_the_pretrained_model(model):
    ys = _series()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = model.tune(ys, 16, steps=10, min_gain=10.0)  # no tune beats the base by 1000%
    assert not out.report["used"] and out.net is model.net
    assert any("did not beat the pretrained model" in str(w.message) for w in caught)


def test_too_little_history_is_refused(model):
    with pytest.raises(ValueError, match="not enough history to tune"):
        model.tune([np.arange(20, dtype=np.float32)], 16, steps=5)
