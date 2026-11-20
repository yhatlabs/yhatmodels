import json

import numpy as np
import pytest

from yhatmodels import cli

pd = pytest.importorskip("pandas")


@pytest.fixture
def files(tmp_path):
    r = np.random.default_rng(0)
    t = np.arange(24 * 20)
    df = pd.concat([pd.DataFrame({"store": f"s{i}", "date": pd.date_range("2025-01-01", periods=len(t), freq="h"),
                                  "sales": 10 + np.sin(t / 24 * 2 * np.pi + i) + r.normal(0, 0.2, len(t)),
                                  "price": np.where(t % 50 < 10, 0.8, 1.0)}) for i in range(2)])  # fmt: skip
    df.to_csv(tmp_path / "sales.csv", index=False)
    end = df.date.max()
    fut = pd.concat([pd.DataFrame({"store": f"s{i}", "date": pd.date_range(end + pd.Timedelta(hours=1), periods=12, freq="h"), "price": 1.0})
                     for i in range(2)])  # fmt: skip
    fut.to_csv(tmp_path / "future.csv", index=False)
    return tmp_path


def test_forecast_evaluate_and_tune(files, tiny_model_dir, capsys):
    m = ["--model", str(tiny_model_dir), "--device", "cpu", "--id-col", "store", "--known", "price"]
    assert cli.main(["forecast", str(files / "sales.csv"), "--h", "12", "--level", "80", "--future", str(files / "future.csv"),
                     "-o", str(files / "fc.csv"), *m]) == 0  # fmt: skip
    fc = pd.read_csv(files / "fc.csv")
    assert list(fc.columns) == ["store", "date", "q10", "median", "q90"] and len(fc) == 24
    assert cli.main(["evaluate", str(files / "sales.csv"), "--h", "12", "--windows", "2", "-o", str(files / "scores.csv"), *m]) == 0
    assert "seasonal_naive" in capsys.readouterr().out and len(pd.read_csv(files / "scores.csv")) == 2 * 2 * 2
    assert (
        cli.main(["tune", str(files / "sales.csv"), "--h", "12", "--steps", "10", "--min-gain", "-1", "-o", str(files / "tuned"), *m]) == 0
    )
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") : out.rindex("}") + 1])["used"]
    assert cli.main(["forecast", str(files / "sales.csv"), "--h", "12", "--future", str(files / "future.csv"), "-o", str(files / "fc2.csv"),
                     *m[:1], str(files / "tuned"), *m[2:]]) == 0  # fmt: skip
    assert len(pd.read_csv(files / "fc2.csv")) == 24


def test_models_lists_the_catalog(capsys):
    assert cli.main(["models"]) == 0
    assert "ChakraTS-OS-10M" in capsys.readouterr().out
