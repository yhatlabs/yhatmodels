import pytest
import torch

from yhatmodels import ChakraTS
from yhatmodels.cli import main
from yhatmodels.errors import ModelLoadError


def test_a_device_this_machine_lacks_is_refused_up_front(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no NVIDIA GPU"):
        ChakraTS.from_pretrained(tiny_model_dir, device="cuda")
    with pytest.raises(ValueError, match="is not a device"):
        ChakraTS.from_pretrained(tiny_model_dir, device="gpu")
    assert ChakraTS.from_pretrained(tiny_model_dir, device="cpu").device == "cpu"


def test_a_folder_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(ModelLoadError, match="no such folder.*ChakraTS-OS-10M"):
        ChakraTS.from_pretrained(str(tmp_path / "missing"))
    with pytest.raises(ModelLoadError, match="no such folder"):
        ChakraTS.from_pretrained("./my-tuned")


def test_an_unknown_model_name_says_what_a_model_can_be(monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import RepositoryNotFoundError

    def not_found(repo, local_files_only=False, **kw):
        if local_files_only:
            from huggingface_hub.utils import LocalEntryNotFoundError

            raise LocalEntryNotFoundError("not cached")
        e = RepositoryNotFoundError.__new__(RepositoryNotFoundError)  # its constructor wants an HTTP response
        Exception.__init__(e, "404 Client Error")
        raise e

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_found)
    with pytest.raises(ModelLoadError, match="ChakraTS-OS-999M is not a model yhatmodels can find.*list_models"):
        ChakraTS.from_pretrained("ChakraTS-OS-999M")


def test_the_command_line_prints_errors_without_a_traceback(tmp_path, capsys):
    assert main(["forecast", str(tmp_path / "missing.csv"), "--h", "7"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("yhatmodels: error:") and "missing.csv" in err and "Traceback" not in err
