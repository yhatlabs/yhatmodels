import numpy as np
import pytest

from yhatmodels import ChakraTS, plot_forecast

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")


def test_plot_draws_history_intervals_and_median_and_changes_nothing_global(tiny_model_dir):
    import matplotlib.pyplot as plt

    before = dict(mpl.rcParams)
    model = ChakraTS.from_pretrained(tiny_model_dir, device="cpu")
    y = (10 + np.sin(np.arange(300) / 6)).astype(np.float32)
    q = model.forecast(y, 24)
    ax = plot_forecast(y, q, title="demo")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["history", "80% interval", "median"]
    assert len(ax.lines[0].get_xdata()) == 200 and ax.lines[-1].get_xdata()[0] == 300
    assert dict(mpl.rcParams) == before
    nine = model.forecast(y, 24, quantiles="all")
    assert len(plot_forecast(y, nine, model.net.cfg.levels).collections) == 10  # 21 levels: 10 symmetric intervals
    with pytest.raises(ValueError, match="forecast must be"):
        plot_forecast(y, q[:, :2])
    plt.close("all")


def _frame(n_series=3, T=24 * 15):
    import pandas as pd

    t = np.arange(T)
    return pd.concat([pd.DataFrame({"unique_id": f"s{i}", "ds": pd.date_range("2025-01-01", periods=T, freq="h"),
                                    "y": 10 + 2 * np.sin(t / 24 * 2 * np.pi + i)}) for i in range(n_series)], ignore_index=True)  # fmt: skip


def test_dataframe_forecast_one_panel_per_series_after_its_history(tiny_model_dir):
    import matplotlib.pyplot as plt

    model = ChakraTS.from_pretrained(tiny_model_dir, device="cpu")
    df = _frame()
    fc = model.forecast(df, 24, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
    fig = plot_forecast(fc, history=df, max_series=2)
    assert [a.get_title() for a in fig.axes] == ["s0", "s1"]
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["history", "80% interval", "50% interval", "median"]
    assert fig.axes[0].lines[0].get_xdata()[-1] < fig.axes[0].lines[-1].get_xdata()[0]  # history ends before the forecast
    assert [a.get_title() for a in plot_forecast(fc, ids=["s2"]).axes] == ["s2"]
    one = model.forecast(df[df.unique_id == "s0"].set_index("ds")["y"], 12)  # a dated Series: no id column
    assert len(plot_forecast(one).axes) == 1
    _, ax = plt.subplots()
    assert plot_forecast(fc, ax=ax, max_series=1) is ax.figure
    plt.close("all")


def test_backtest_and_calibration_plots(tiny_model_dir):
    import matplotlib.pyplot as plt

    model = ChakraTS.from_pretrained(tiny_model_dir, device="cpu")
    res = model.evaluate(_frame(2), h=24, windows=3, baselines=["naive"])
    fig = res.plot()
    assert [a.get_title() for a in fig.axes] == ["s0 (tiny)", "s1 (tiny)"]
    a = fig.axes[0]
    bands = ["80% interval", "60% interval", "40% interval", "20% interval"]  # backtests keep every decile
    assert [t.get_text() for t in a.get_legend().get_texts()] == ["actual", *bands, "median"]  # once each, not per window
    assert sum(1 for ln in a.lines if ln.get_linestyle() == ":") == 3  # a cutoff line per window
    assert res.plot(model="naive", ids=["s1"]).axes[0].get_title() == "s1 (naive)"
    cal = res.plot_calibration().axes[0]
    assert [t.get_text() for t in cal.get_legend().get_texts()] == ["calibrated", "tiny", "naive"]
    plt.close("all")
