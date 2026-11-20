# Using yhatmodels

The reference behind the [README](../README.md): every input, option and output of `ChakraTS`.

## Inputs

```python
from yhatmodels import ChakraTS

model = ChakraTS.from_pretrained("ChakraTS-OS-10M")  # device="cuda" / "mps" / "cpu"; default: NVIDIA GPU, Apple GPU, CPU
q = model.forecast(data, h=24)
```

- **One series:** a numpy array, a torch tensor, a pandas Series or a list of numbers. NaN marks missing values.
- **Several series:** an array `(N, T)`, or a list of series of different lengths. Each forecast comes back in its
  series' shape.
- **Several targets:** a series given as `(V, T)` forecasts its V targets jointly and gives `(V, h, Q)`; an array
  `(N, V, T)` is N such series and gives `(N, V, h, Q)`.
- **Covariates:** per series, `past={"name": values}` (known up to the forecast origin, length T) and
  `known={"name": values}` (known over the history and the next h steps, length T + h); for several series, a list with
  one dict per series. Text values are treated as categories.

## DataFrames

A long-format DataFrame (one row per series and timestamp) in gives a DataFrame out:

```python
fc = model.forecast(df, h=24, known=["price"], future=future)  # future: id, time and price over the next 24 steps
```

- The id, time and target columns are found by name (`unique_id` / `id` / `item_id`, `ds` / `timestamp` / `date`,
  `y` / `target` / `value`) or given with `id_col`, `time_col` and `target_col`; `past` and `known` name covariate
  columns.
- The result has the id and time columns, then one column per quantile level (`q10`, `median`, `q90`, ...).
- Missing timestamps are filled in as missing values, on each series' frequency (inferred, or `freq=`); duplicate
  timestamps are refused.
- Rows after a series' last observed target (target empty, covariates filled in) are its future: the known covariates'
  values over the next `h` steps. Or pass them separately as `future`.
- A pandas Series with a DatetimeIndex is one dated series and comes back as a DataFrame.

## Options

- **Quantiles:** `quantiles=[0.05, 0.5, 0.95]` (any levels in (0, 1), returned in increasing order), `quantiles="all"`
  (the 21 levels the model produces, 0.01 to 0.99), or central intervals by width: `level=[80, 95]`.
- **Presets:** `preset="pro"` (the default) reads the series at every patch size the model knows and averages the
  forecasts, for horizons up to 400 steps; `preset="fast"` reads it at the model's own patch size only, several times
  faster on long histories.
- **Related series:** `cross_series=True` reads single-target series without covariates 16 at a time (or pass a group
  size), so the model can learn from their neighbours.
- **Devices:** NVIDIA GPUs run in bf16, as the model was trained. Large batches are split into passes sized for the
  device, and a pass that runs out of GPU memory is retried in halves.

## Backtests

`evaluate` cuts each series at several points of its own history, forecasts from each cut and scores against what
happened, next to simple baselines:

```python
res = model.evaluate(df, h=24, windows=3, baselines=["seasonal_naive", "naive"])
res.summary            # per model: mase, sql, wql, mae, rmse, coverage@0.8, and skill_<metric> vs seasonal_naive (> 0 beats it)
res.scores             # per series and window
res.forecasts          # each window's quantiles next to the actuals
res.calibration        # share of actuals below each quantile and inside each interval, vs nominal
res.plot()             # actuals, each window's forecast and its cutoff (pip install "yhatmodels[plot]")
res.plot_calibration() # share below each quantile against its level, one line per model
```

`compare={"current": my_forecasts}` scores your own forecasts (quantile columns or a `forecast` column) on the same
windows. Known covariates are read from the data over the history; nothing after a cutoff reaches that window's
forecast.

## Fine-tuning

`tune` adapts the model to your series (paydays, promotions, your own seasonality) with low-rank adapters on every linear
layer, the pretrained weights frozen. It holds out the last `h` steps of each series and keeps the tune only if it
beats the pretrained model there (by 2% by default); the tune is then retrained on the whole history and merged into a
copy of the weights, so a tuned model forecasts as fast as the pretrained one.

```python
tuned = model.tune(df, h=24, steps=300)  # seconds to a few minutes; the pretrained model is unchanged
tuned.report                             # holdout loss and WAPE, tuned vs pretrained; steps kept
tuned.save("my-tuned/")                  # weights, config and the model's license files
ChakraTS.from_pretrained("my-tuned/")
```

A tuned model stays under the license of the model it was tuned from; `save` writes that license and a notice stating
the modification.

## Plots

```python
from yhatmodels import plot_forecast

plot_forecast(history, q)      # arrays: history, the median and every interval; returns the matplotlib Axes
plot_forecast(fc, history=df)  # a forecast DataFrame: one panel per series (the first 4, or ids=[...]); the Figure
```

## Command line

```bash
yhatmodels forecast sales.csv --h 24 --level 80,95 --known price --future future.csv -o fc.csv
yhatmodels evaluate sales.csv --h 24 --windows 3
yhatmodels tune sales.csv --h 24 --steps 300 -o my-tuned/
yhatmodels forecast sales.csv --h 24 --model my-tuned/
yhatmodels models
yhatmodels accept-license ChakraTS-OS-10M
```

Files are CSV or Parquet in long format; `--id-col`, `--time-col`, `--target-col`, `--past`, `--known`, `--freq`,
`--preset` and `--cross-series` work as in Python.

## Limits

- Forecasts are quantiles; there are no sample paths.
- No static covariates (per-series attributes such as a store's size).
- The model reads at most the last 32,768 steps of a history.
- Horizons longer than 768 steps are forecast in blocks, each continuing from the median of the one before.
