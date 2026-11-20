# Changelog

## 0.1.0 (unreleased)

- First release: `ChakraTS` for time-series forecasting with ChakraTS-OS-10M.
- `forecast(data, h, ...)` with the YHat Labs SDK's arguments: arrays in, arrays out (univariate, multivariate, past and
  known covariates, categorical covariates), or a long-format pandas DataFrame in, a DataFrame out (`id_col`,
  `time_col`, `target_col`, `known` + `future`, `past`, `freq`; columns `median`, `q10`, ...). One series can be a numpy
  array, a torch tensor, a pandas Series or a plain list of numbers; `(N, V, T)` arrays are several multivariate series.
- `quantiles` (default 0.1, 0.5, 0.9; any levels in (0, 1), or "all") or `level` (central intervals in percent).
- Presets `fast` (the model's own patch size) and `pro` (every patch size the model reads, averaged); cross-series
  grouping with `cross_series`.
- Runs on NVIDIA GPUs (bf16), Apple GPUs and CPU, picked in that order; a pass that runs out of GPU memory is retried
  in halves, and passes are sized by rows and history length.
- Clear errors for bad horizons, quantile levels, empty series and covariates of the wrong length; a series with no
  observed value is forecast uninformed, with a warning.
- Plots (with `yhatmodels[plot]`), as the SDK draws them: `plot_forecast()` for arrays or a forecast DataFrame (one
  panel per series, after its history), `EvaluationResult.plot()` (each window's forecast against the actuals) and
  `EvaluationResult.plot_calibration()`.
- Models: `yhatmodels.list_models()`; architectures registered per task; strict model settings (a model made for a newer
  yhatmodels asks for an upgrade); sha256-checked and sharded weights.
- Model licenses: YHat Labs' models under a non-open license (ChakraTS-OS-10M) are accepted once per person with a
  yhatlabs.com account before the first download (the browser opens and hands a token back; pasted over SSH;
  `YHAT_MODELS_TOKEN` or an API key in CI); weights already downloaded are never checked. Other repositories under a
  non-open license are accepted once on the machine with `yhatmodels.accept_license()`.
- `evaluate()`: rolling-origin backtests with the SDK's windows, metrics (mase, sql, wql, mae, rmse, coverage@w),
  baselines (seasonal_naive, naive), skill scores, calibration and `compare=` for your own forecasts.
- `tune()`: fine-tuning on your own series (LoRA on every linear layer, holdout-checked against the pretrained model,
  merged into a copy of the weights), `report`, and `save()` to a loadable folder with the license files.
- The `yhatmodels` command line: `forecast`, `evaluate`, `tune`, `models`, `accept-license`.
- pandas is a dependency (DataFrames, backtests and the command line work on a plain install); `[plot]` adds matplotlib.
- A list of series gives each forecast in its series' own shape: `(h, Q)` for `(T,)`, `(V, h, Q)` for `(V, T)`.
- DataFrames: missing timestamps filled as missing values, duplicates refused, rows after the last observed target used
  as the future; a dated pandas Series keeps its dates.
- Levels off the model's grid keep the non-negative / integer guarantees; numbers with gaps are numeric covariates.
- Loading for every task: an architecture may serve several tasks (forecasting, tabular, anomaly); config.json may list
  extra files; `model.info` for model lists; presets and the default preset per model in config.json.
