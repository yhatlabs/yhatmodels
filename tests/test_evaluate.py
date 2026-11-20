import numpy as np
import pytest

from yhatmodels import ChakraTS
from yhatmodels.ts import evaluate as ev

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def model(tiny_model_dir):
    return ChakraTS.from_pretrained(tiny_model_dir, device="cpu")


def _frame(n_series=3, T=24 * 20, cov=False):
    r = np.random.default_rng(0)
    t = np.arange(T)
    parts = []
    for i in range(n_series):
        f = pd.DataFrame({"unique_id": f"s{i}", "ds": pd.date_range("2025-01-01", periods=T, freq="h"),
                          "y": 10 + 2 * np.sin(t / 24 * 2 * np.pi + i) + r.normal(0, 0.3, T)})  # fmt: skip
        if cov:
            f["price"] = np.where(t % 50 < 10, 0.8, 1.0)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def test_scores_windows_and_baselines(model):
    res = model.evaluate(_frame(), h=24, windows=3, baselines=("seasonal_naive", "naive"))
    assert list(res.summary.index) == ["tiny", "seasonal_naive", "naive"]
    assert len(res.scores) == 3 * 3 * 3  # models x series x windows
    assert {"skill_mase", "coverage@0.8"} <= set(res.summary.columns)
    assert res.summary.loc["seasonal_naive", "skill_mase"] == pytest.approx(0.0)
    cuts = res.scores[res.scores.model == "tiny"].groupby("window").cutoff.first()
    assert (cuts.diff().dropna() == pd.Timedelta(hours=24)).all()  # step defaults to h: back to back windows
    assert set(res.calibration.kind) == {"below", "interval"}


def test_no_window_reads_past_its_cutoff(model, monkeypatch):
    seen = []
    real = ChakraTS._forecast_series

    def spy(self, series, h, levels, pred, group):
        seen.extend(s.target.shape[-1] for s in series)
        return real(self, series, h, levels, pred, group)

    monkeypatch.setattr(ChakraTS, "_forecast_series", spy)
    res = model.evaluate(_frame(n_series=1, T=200), h=10, windows=2, step=5, baselines=())
    assert sorted(set(seen)) == [185, 190]  # T - h - (windows - 1 - w) * step
    assert len(res.scores) == 2


def test_known_covariates_are_cut_at_cutoff_plus_h(model, monkeypatch):
    lengths = []
    real = ChakraTS._forecast_series

    def spy(self, series, h, levels, pred, group):
        lengths.extend((s.target.shape[-1], s.future.shape[-1]) for s in series)
        return real(self, series, h, levels, pred, group)

    monkeypatch.setattr(ChakraTS, "_forecast_series", spy)
    model.evaluate(_frame(n_series=1, T=200, cov=True), h=10, windows=1, known=["price"], baselines=())
    assert lengths == [(190, 200)]


def test_arrays_and_compare(model):
    y = _frame(n_series=1).y.to_numpy()
    res = model.evaluate(y, h=24, windows=2, freq="h")
    assert set(res.summary.index) == {"tiny", "seasonal_naive"}
    df = _frame(n_series=1)
    mine = pd.DataFrame({"unique_id": "s0", "ds": df.ds.iloc[-24:].to_numpy(), "forecast": df.y.iloc[-48:-24].to_numpy()})
    res = model.evaluate(df, h=24, windows=1, compare={"mine": mine})
    assert "mine" in res.summary.index and np.isfinite(res.summary.loc["mine", "mase"])
    with pytest.raises(ValueError, match="compare= needs dated data"):
        model.evaluate(y, h=24, compare={"mine": mine})


def test_metric_definitions():
    y = np.array([1.0, 2.0, 3.0, np.nan])
    q = np.stack([y - 1, y, y + 1], axis=1)
    out = ev.score(y, q, (0.1, 0.5, 0.9), scale=2.0, metrics=("mae", "mase", "coverage@0.8", "coverage@80"))
    assert out["mae"] == 0 and out["mase"] == 0 and out["coverage@0.8"] == 1 and out["coverage@80"] == 1
    with pytest.raises(ValueError, match="needs the 0.025 and 0.975 quantiles"):
        ev.score(y, q, (0.1, 0.5, 0.9), 1.0, ("coverage@0.95",))
    assert ev.season_for("h") == 24 and ev.season_for("15min") == 96 and ev.season_for("D", ev.BASELINE_SEASONS) == 7


def test_too_short_is_refused(model):
    with pytest.raises(ValueError, match="no series is long enough"):
        model.evaluate(_frame(n_series=1, T=30), h=24, windows=3)
