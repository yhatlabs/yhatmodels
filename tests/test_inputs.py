import numpy as np
import pytest
import torch

from yhatmodels import ChakraTS
from yhatmodels.ts import forecaster as fc
from yhatmodels.ts.predictor import Series

LEVELS = (0.1, 0.5, 0.9)


@pytest.fixture(scope="module")
def model(tiny_model_dir):
    return ChakraTS.from_pretrained(tiny_model_dir, device="cpu")


def _series(n=120, seed=0):
    r = np.random.default_rng(seed)
    return (10 + np.sin(np.arange(n) / 5) + r.normal(0, 0.1, n)).astype(np.float32)


def test_any_one_dimensional_input_is_one_series(model):
    pd = pytest.importorskip("pandas")
    y = _series()
    expect = model.forecast(y, 8)
    for same in (pd.Series(y), y.tolist(), torch.tensor(y), tuple(y.tolist())):
        out = model.forecast(same, 8)
        assert out.shape == (8, 3) and np.array_equal(out, expect)


def test_a_cube_is_several_multivariate_series(model):
    r = np.random.default_rng(1)
    cube = r.normal(size=(4, 2, 90)).astype(np.float32)
    out = model.forecast(cube, 6)
    assert out.shape == (4, 2, 6, 3)
    assert np.array_equal(out, np.stack(model.forecast(list(cube), 6)))


def test_dataframe_arguments_are_refused_for_arrays(model):
    with pytest.raises(TypeError, match="for dated data"):
        model.forecast(_series(), 4, freq="D")


@pytest.mark.parametrize("horizon", [0, -3, 1.5, True, "12"])
def test_a_bad_horizon_is_refused(model, horizon):
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(_series(), horizon)


@pytest.mark.parametrize("levels", [[0.5, 1.2], [0.0, 0.5], [], [float("nan")], "median"])
def test_quantiles_outside_zero_one_are_refused(model, levels):
    with pytest.raises(ValueError, match="quantiles"):
        model.forecast(_series(), 4, quantiles=levels)
    with pytest.raises(ValueError, match="level takes"):
        model.forecast(_series(), 4, level=[100])


def test_series_with_nothing_observed_are_forecast_with_a_warning(model):
    # benchmarks hold such rows (a window cut before a series starts); they get a finite, uninformed forecast
    y = _series()
    with pytest.warns(UserWarning, match="no observed values .* e.g. series 1"):
        out = model.forecast([y, np.full(50, np.nan, np.float32)], 4)
    assert all(np.isfinite(q).all() for q in out)
    mv = np.stack([y, np.full_like(y, np.nan)])
    with pytest.warns(UserWarning, match=r"series 0 \(target row 1\)"):
        out = model.forecast([mv], 4)
    assert np.isfinite(out[0]).all() and out[0].shape == (2, 4, 3)
    with pytest.raises(ValueError, match="no series"):
        model.forecast([], 4)
    with pytest.raises(ValueError, match="cross_series"):
        model.forecast(y, 4, cross_series=-1)


def test_one_series_takes_its_covariates_as_a_dict(model):
    r = np.random.default_rng(2)
    y = _series(150)
    past, future = {"temp": r.normal(size=150)}, {"price": r.normal(size=160)}
    as_dict = model.forecast(y, 10, past=past, known=future)
    as_list = model.forecast([y], 10, past=[past], known=[future])[0]
    assert np.array_equal(as_dict, as_list)
    with pytest.raises(ValueError, match="series 0: known covariates have 150 values; they need history \\+ h = 160"):
        model.forecast(y, 10, known={"price": r.normal(size=150)})


def test_dataframes_check_their_columns_and_future_rows(model):
    pd = pytest.importorskip("pandas")
    ts = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.concat([pd.DataFrame({"item_id": k, "timestamp": ts, "target": _series(100, k), "price": 1.0}) for k in range(2)])
    out = model.forecast(df, 5, quantiles=(0.1, 0.9))
    assert list(out.columns) == ["item_id", "timestamp", "q10", "q90"]
    with pytest.raises(ValueError, match="data has no column 'sales'"):
        model.forecast(df, 5, target_col="sales")
    future = pd.DataFrame({"item_id": 0, "timestamp": pd.date_range("2026-04-11", periods=5, freq="D"), "price": 1.0})
    with pytest.raises(ValueError, match="future has no rows for series 1"):
        model.forecast(df, 5, known=["price"], future=future)
    with pytest.raises(ValueError, match="known covariates need their values over the next 5 steps"):
        model.forecast(df, 5, known=["price"])


def test_long_histories_fill_a_pass_with_fewer_rows():
    def lengths(chunks):
        return [len(c) for c in chunks]

    short = [Series(np.zeros((1, 500), np.float32)) for _ in range(3000)]
    assert lengths(fc._chunks(short)) == [2048, 952]  # the row budget alone
    long = [Series(np.zeros((1, 32768), np.float32)) for _ in range(3000)]
    assert max(lengths(fc._chunks(long, max_context=32768))) == 2048 * 8192 // 32768
    assert lengths(fc._chunks(long, max_context=8192)) == [2048, 952]  # the model reads at most 8192 steps of each


def test_the_device_is_cuda_then_the_apple_gpu_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert fc._default_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert fc._default_device() == "cpu"


def test_out_of_memory_splits_the_pass_and_other_errors_do_not(monkeypatch):
    calls = []

    class Fake:
        device = "mps"

        def predict(self, chunk, horizon, levels):
            calls.append(len(chunk))
            if len(chunk) > 2:
                raise RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)")
            return [np.zeros((1, horizon, len(levels)))] * len(chunk)

    monkeypatch.setattr(fc, "_release", lambda device: None)
    out = fc._fit(Fake(), [Series(np.zeros((1, 10), np.float32))] * 8, 3, LEVELS)
    assert len(out) == 8 and calls == [8, 4, 2, 2, 4, 2, 2]

    class Broken(Fake):
        def predict(self, chunk, horizon, levels):
            raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        fc._fit(Broken(), [Series(np.zeros((1, 10), np.float32))] * 4, 3, LEVELS)


def test_rows_after_the_last_target_are_the_future(model):
    pd = pytest.importorskip("pandas")
    ts = pd.date_range("2025-01-01", periods=120, freq="D")
    r = np.random.default_rng(5)
    df = pd.DataFrame({"unique_id": "a", "ds": ts, "y": 50 + r.normal(0, 5, 120), "price": r.normal(3, 0.2, 120)})
    df.loc[df.index[-15:], "y"] = np.nan  # the next 15 days: price known, sales not yet
    fc = model.forecast(df, 5, known=["price"])
    assert fc.ds.min() == ts[-16] + pd.Timedelta(days=1)  # straight after the last observed day
    split = model.forecast(df.iloc[:-15], 5, known=["price"], future=df.iloc[-15:][["unique_id", "ds", "price"]])
    assert fc.equals(split)  # the same as passing those rows as future
    assert len(model.forecast(df, 5)) == 5  # without covariates the empty tail is simply not history
    with pytest.raises(ValueError, match="give the future rows once"):
        model.forecast(df, 5, known=["price"], future=df.iloc[-15:])


def test_missing_timestamps_are_filled_and_duplicates_refused(model):
    pd = pytest.importorskip("pandas")
    ts = pd.date_range("2025-01-01", periods=100, freq="D")
    y = np.sin(np.arange(100) / 5) + 10
    full = pd.DataFrame({"ds": ts, "y": y})
    gappy = full.drop(index=range(40, 60))
    filled = full.assign(y=np.where((np.arange(100) >= 40) & (np.arange(100) < 60), np.nan, y))
    assert model.forecast(gappy, 3).equals(model.forecast(filled, 3))  # a missing day is a missing value
    with pytest.raises(ValueError, match="duplicate timestamps"):
        model.forecast(pd.concat([gappy, gappy.tail(1)]), 3)
    with pytest.raises(ValueError, match="not on a 'W-SUN' grid"):
        model.forecast(full, 3, freq="W-SUN")


def test_a_dated_pandas_series_keeps_its_dates(model):
    pd = pytest.importorskip("pandas")
    s = pd.Series(np.sin(np.arange(90) / 4) + 5, index=pd.date_range("2025-03-01", periods=90, freq="h"))
    fc = model.forecast(s, 4)
    assert list(fc.columns) == ["ds", "q10", "median", "q90"] and fc.ds.iloc[0] == s.index[-1] + pd.Timedelta(hours=1)


def test_research_settings_are_not_forecast_arguments(model):
    y = _series()
    with pytest.raises(TypeError, match="calendar, static: features of the hosted ChakraTS"):
        model.forecast(y, 4, static=["region"], calendar=True)
    with pytest.raises(TypeError, match="unexpected arguments \\['flip'\\]"):
        model.forecast(y, 4, flip=False)
    assert model.predictor("pro", flip=False).flip is False  # still reachable for research and replication
    assert model.default_preset == "fast"


def test_numbers_with_gaps_are_a_numeric_covariate():
    from yhatmodels.ts.covariates import is_categorical, rows

    assert not is_categorical([1.0, None, 2.0]) and is_categorical(["a", None, "b"])
    out = rows({"temp": [1.0, None, 2.0]}, np.zeros(3, np.float32))
    assert np.isnan(out[0, 1]) and out[0, 2] == 2.0
