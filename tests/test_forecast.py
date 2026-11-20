import numpy as np
import pytest

from yhatmodels import ChakraTS

LEVELS = (0.1, 0.5, 0.9)


@pytest.fixture(scope="module")
def model(tiny_model_dir):
    return ChakraTS.from_pretrained(tiny_model_dir, device="cpu")


def _monotone(q):
    return bool(np.all(np.diff(q, axis=-1) >= -1e-4))


def test_shapes_follow_the_input(model):
    r = np.random.default_rng(0)
    one = r.normal(size=100).astype(np.float32)
    assert model.forecast(one, 12).shape == (12, 3)  # default quantiles 0.1, 0.5, 0.9
    assert model.forecast(r.normal(size=(5, 80)), 7).shape == (5, 7, 3)
    out = model.forecast([r.normal(size=60), r.normal(size=(3, 90))], 10)
    assert [q.shape for q in out] == [(10, 3), (3, 10, 3)]  # each series as it came


@pytest.mark.parametrize("preset", ["fast", "pro"])
@pytest.mark.parametrize("cross_series", [False, 4])
def test_presets_and_groups_give_monotone_finite_quantiles(model, preset, cross_series):
    r = np.random.default_rng(1)
    series = [(10 + np.sin(np.arange(n) / 5) + r.normal(0, 0.1, n)).astype(np.float32) for n in (30, 64, 120, 200, 333)]
    out = model.forecast(series, 20, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95), preset=preset, cross_series=cross_series)
    for q in out:
        assert q.shape == (20, 5) and np.isfinite(q).all() and _monotone(q)


def test_grouping_keeps_each_series_in_its_place(model):
    r = np.random.default_rng(2)
    series = [np.full(50 + 10 * i, float(100 * i), np.float32) + r.normal(0, 0.01, 50 + 10 * i) for i in range(1, 7)]
    out = model.forecast(series, 5, quantiles=(0.5,), cross_series=3)
    medians = [float(q[0, 0]) for q in out]
    assert medians == sorted(medians)  # levels 100, 200, ... stay with their own series


def test_non_negative_integer_history_gives_non_negative_integer_forecasts(model):
    counts = np.random.default_rng(3).poisson(2, 300).astype(np.float32)
    q = model.forecast(counts, 16)
    assert (q >= 0).all() and np.array_equal(q, np.round(q))


def test_missing_values_and_covariates(model):
    r = np.random.default_rng(4)
    y = (5 + r.normal(0, 1, 200)).astype(np.float32)
    y[:20] = np.nan
    promo = np.array(["a", "b", None, "a"] * 55, dtype=object)[: 200 + 12]
    q = model.forecast([y], 12, past=[{"temp": r.normal(size=200)}], known=[{"price": r.normal(size=212), "promo": promo}])
    assert q[0].shape == (12, 3) and np.isfinite(q[0]).all()
    with pytest.raises(ValueError):
        model.forecast([y], 12, known=[{"price": r.normal(size=200)}])


def test_levels_all_and_central_intervals(model):
    y = np.sin(np.arange(120) / 6).astype(np.float32)
    assert model.forecast(y, 6, quantiles="all").shape == (6, len(model.net.cfg.levels))
    a = model.forecast(y, 6, level=[80])
    assert np.allclose(a, model.forecast(y, 6, quantiles=(0.1, 0.5, 0.9)))
    assert model.forecast(y, 6, quantiles=(0.9, 0.1, 0.5, 0.1)).shape == (6, 3)  # sorted, duplicates dropped
    with pytest.raises(ValueError, match="not both"):
        model.forecast(y, 6, quantiles=(0.5,), level=[80])


def test_dataframes_in_and_out_with_sdk_names(model):
    pd = pytest.importorskip("pandas")
    ts = pd.date_range("2026-01-01", periods=120, freq="h")
    df = pd.concat([pd.DataFrame({"unique_id": k, "ds": ts, "y": np.sin(np.arange(120) / 6) + k}) for k in range(3)])
    fc = model.forecast(df, 6)
    assert len(fc) == 18 and list(fc.columns) == ["unique_id", "ds", "q10", "median", "q90"]
    assert fc.groupby("unique_id").ds.min().eq(ts[-1] + pd.Timedelta(hours=1)).all()
    one = df[df.unique_id == 0].drop(columns="unique_id").rename(columns={"ds": "date", "y": "sales"})
    single = model.forecast(one, 4, level=[95])  # no id column: one series; the one numeric column is the target
    assert list(single.columns) == ["date", "q2.5", "median", "q97.5"] and len(single) == 4


def test_clip_and_snap_hold_at_any_level(model):
    """The model's own levels are clipped and rounded; levels off its grid must be too (interpolated, then clipped)."""
    from yhatmodels.ts.predictor import Series

    pred = model.predictor()
    native = np.array(model.net.cfg.levels)
    counts = np.array([[0, 1, 0, 2, 1, 0, 3, 1, 0, 1] * 10], np.float32)
    grid = np.tile(np.linspace(0, 4, len(native)).round(), (1, 3, 1)).astype(np.float32)  # q(0.01) = 0, the next one above 0
    grid[..., 1] = 1.0
    pred._native = lambda series, h: [grid.copy() for _ in series]
    pred.flip = False
    q = pred.predict([Series(counts)], 3, (0.001, 0.025, 0.5, 0.975))[0]
    assert (q >= 0).all() and np.array_equal(q, np.round(q))


def test_pro_reads_one_patch_size_past_its_horizon_cap(model):
    y = (10 + np.sin(np.arange(100) / 7)).astype(np.float32)
    capped = model.predictor("pro", multi_scale_max_h=10)
    plain = model.predictor("pro", multi_scale=False)
    from yhatmodels.ts.predictor import Series

    assert np.array_equal(capped.predict([Series(y)], 12)[0], plain.predict([Series(y)], 12)[0])  # past the cap: one size
    assert not np.array_equal(capped.predict([Series(y)], 8)[0], plain.predict([Series(y)], 8)[0])  # within it: every size
