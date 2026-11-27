import hashlib
import json
import shutil

import numpy as np
import pytest

import yhatmodels
from yhatmodels import ChakraTS
from yhatmodels.errors import LicenseNotAcceptedError, ModelLoadError
from yhatmodels.licensing import is_accepted
from yhatmodels.ts.architectures.chakrats_v1 import Config


def test_another_repositorys_licence_is_accepted_on_this_machine(tmp_path, monkeypatch):
    monkeypatch.setenv("YHATMODELS_CACHE", str(tmp_path))
    monkeypatch.delenv("YHATMODELS_ACCEPT_LICENSE", raising=False)
    yhatmodels.accept_license("someone/closed-model")
    assert is_accepted("someone/closed-model") and not is_accepted("someone/another-model")


def test_the_environment_variable_accepts_too(tmp_path, monkeypatch):
    monkeypatch.setenv("YHATMODELS_CACHE", str(tmp_path))
    monkeypatch.setenv("YHATMODELS_ACCEPT_LICENSE", "1")
    assert is_accepted("someone/closed-model")


def test_a_folder_missing_its_weights_is_refused(tiny_model_dir, tmp_path):
    shutil.copy(tiny_model_dir / "config.json", tmp_path / "config.json")
    with pytest.raises(ModelLoadError, match="model.safetensors"):
        ChakraTS.from_pretrained(tmp_path)


def test_an_unknown_architecture_is_refused(tiny_model_dir, tmp_path):
    shutil.copy(tiny_model_dir / "model.safetensors", tmp_path / "model.safetensors")
    config = json.loads((tiny_model_dir / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**config, "architecture": "something_else"}))
    with pytest.raises(ModelLoadError, match="unknown architecture"):
        ChakraTS.from_pretrained(tmp_path)


def test_settings_the_architecture_does_not_implement_are_refused():
    with pytest.raises(ValueError):
        Config.from_dict({"arm": "encoder"})
    with pytest.raises(ValueError, match="unknown to this yhatmodels"):  # a released config with a key we do not know
        Config.from_dict({"patch": 8, "block": 32})
    assert Config.from_dict({"patch": 8, "block": 32, "mtp_frac": 0.25}, strict=False).patch == 8  # research configs


def _copy(src, dst, **config_changes):
    dst.mkdir(exist_ok=True)
    shutil.copy(src / "model.safetensors", dst / "model.safetensors")
    config = {**json.loads((src / "config.json").read_text()), **config_changes}
    (dst / "config.json").write_text(json.dumps(config))
    return dst


def test_weights_under_an_open_licence_need_no_acceptance(tiny_model_dir, tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setenv("YHATMODELS_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("YHATMODELS_ACCEPT_LICENSE", raising=False)
    open_repo = _copy(tiny_model_dir, tmp_path / "open", license="apache-2.0")
    closed_repo = _copy(tiny_model_dir, tmp_path / "closed", license="some-research-licence")
    repos = {"someone/open-model": open_repo, "someone/closed-model": closed_repo}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_hub(repos))
    assert ChakraTS.from_pretrained("someone/open-model", device="cpu").config["license"] == "apache-2.0"
    with pytest.raises(LicenseNotAcceptedError):
        ChakraTS.from_pretrained("someone/closed-model", device="cpu")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_hub(repos, cached={"someone/closed-model"}))
    assert ChakraTS.from_pretrained("someone/closed-model", device="cpu")  # downloaded before: accepted before


def fake_hub(repos, cached=()):
    """snapshot_download over local folders: `cached` repositories are in the local cache, the others download."""

    def snapshot_download(repo, local_files_only=False, **kw):
        if local_files_only and repo not in cached:
            from huggingface_hub.utils import LocalEntryNotFoundError

            raise LocalEntryNotFoundError("not in the cache")
        return str(repos[repo])

    return snapshot_download


def test_list_models_names_the_catalog():
    assert [m["name"] for m in yhatmodels.list_models("forecasting")] == ["ChakraTS-OS-10M"]
    assert yhatmodels.list_models()[0]["license"] == "chakra-nc-1.0"


def test_weights_are_checked_against_their_digest(tiny_model_dir, tmp_path):
    good = hashlib.sha256((tiny_model_dir / "model.safetensors").read_bytes()).hexdigest()
    assert ChakraTS.from_pretrained(_copy(tiny_model_dir, tmp_path / "ok", sha256={"model.safetensors": good}), device="cpu")
    with pytest.raises(ModelLoadError, match="sha256"):
        ChakraTS.from_pretrained(_copy(tiny_model_dir, tmp_path / "bad", sha256={"model.safetensors": "0" * 64}), device="cpu")


def test_a_model_for_a_newer_yhatmodels_asks_for_an_upgrade(tiny_model_dir, tmp_path):
    with pytest.raises(ModelLoadError, match="pip install -U yhatmodels"):
        ChakraTS.from_pretrained(_copy(tiny_model_dir, tmp_path / "new", requires_yhatmodels="99.0"), device="cpu")
    assert ChakraTS.from_pretrained(_copy(tiny_model_dir, tmp_path / "same", requires_yhatmodels=yhatmodels.__version__), device="cpu")


def test_sharded_weights_load_like_one_file(tiny_model_dir, tmp_path):
    from safetensors.torch import load_file, save_file

    state = load_file(tiny_model_dir / "model.safetensors")
    keys = sorted(state)
    d = tmp_path / "sharded"
    d.mkdir()
    halves = {"model-00001-of-00002.safetensors": keys[: len(keys) // 2], "model-00002-of-00002.safetensors": keys[len(keys) // 2 :]}
    for name, ks in halves.items():
        save_file({k: state[k] for k in ks}, d / name)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: n for n, ks in halves.items() for k in ks}}))
    shutil.copy(tiny_model_dir / "config.json", d / "config.json")
    x = np.sin(np.arange(80) / 4).astype(np.float32)
    one = ChakraTS.from_pretrained(tiny_model_dir, device="cpu").forecast(x, 5)
    assert np.array_equal(ChakraTS.from_pretrained(d, device="cpu").forecast(x, 5), one)


def test_a_model_is_only_loaded_by_its_own_task():
    from yhatmodels import registry

    with pytest.raises(ModelLoadError, match="is a forecasting model, not a tabular one"):
        registry.get("chakrats_v1", "tabular")
    with pytest.raises(ValueError, match="unknown tasks"):
        registry.register(registry.Architecture("x", ("classification",), lambda settings: None))


def test_files_a_model_lists_must_come_with_it(tiny_model_dir, tmp_path):
    d = _copy(tiny_model_dir, tmp_path / "with-files", files=["preprocessor.json"])
    with pytest.raises(ModelLoadError, match="preprocessor.json"):
        ChakraTS.from_pretrained(d, device="cpu")
    (d / "preprocessor.json").write_text("{}")
    assert ChakraTS.from_pretrained(d, device="cpu").folder == d


def test_info_describes_the_model_for_a_model_list(tiny_model_dir, tmp_path):
    d = _copy(tiny_model_dir, tmp_path / "info", info={"covariates": ["past", "known", "categorical"], "multivariate": True})
    info = ChakraTS.from_pretrained(d, device="cpu").info
    assert info["task"] == "forecasting" and info["parameters"] > 0 and info["multivariate"] is True
    assert info["patch_sizes"] == [2, 4, 8] and info["default_preset"] == "fast" and info["presets"] == ["fast", "pro"]


def test_the_hubs_set_a_token_hint_is_hidden_during_downloads(caplog):
    import logging

    from yhatmodels.hub import _without_token_hint

    log = logging.getLogger("huggingface_hub.utils._http")
    with caplog.at_level(logging.WARNING, logger="huggingface_hub.utils._http"), _without_token_hint():
        log.warning("You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits.")
        log.warning("another Hub warning")
    log.warning("unauthenticated requests, after the download")
    assert [r.getMessage() for r in caplog.records] == ["another Hub warning", "unauthenticated requests, after the download"]
