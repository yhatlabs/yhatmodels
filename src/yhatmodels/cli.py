#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""`yhatmodels`: forecast, backtest and fine-tune from the command line, on a CSV or Parquet file in long format (one row
per series and timestamp). The commands and options follow the YHat Labs SDK's CLI.

    yhatmodels forecast sales.csv --h 24 --level 80,95 --known price --future future.csv -o fc.csv
    yhatmodels evaluate sales.csv --h 24 --windows 3
    yhatmodels tune sales.csv --h 24 --steps 300 -o my-tuned/
    yhatmodels forecast sales.csv --h 24 --model my-tuned/
    yhatmodels models
    yhatmodels accept-license ChakraTS-OS-10M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read(path: str):
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    if suffix == ".json":
        return pd.read_json(p)
    return pd.read_csv(p)


def _write(df, path: str | None) -> None:
    if path is None:
        df.to_csv(sys.stdout, index=False)
    elif path.lower().endswith((".parquet", ".pq")):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _list(s: str | None, kind=str):
    return None if s is None else [kind(x) for x in s.split(",") if x.strip()]


def _cols(p: argparse.ArgumentParser) -> None:
    p.add_argument("--known", help="covariate columns known over the next h steps too (comma-separated)")
    p.add_argument("--past", help="covariate columns known only up to now (comma-separated)")
    p.add_argument("--id-col"), p.add_argument("--time-col")
    p.add_argument("--target-col", help="the target column (several, comma-separated, are forecast jointly)")
    p.add_argument("--freq", help="pandas frequency alias (h, D, W-SUN, MS, 15min, ...); inferred when left out")


def _model_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="ChakraTS-OS-10M", help="a released model's name, a Hugging Face repo id or a local folder")
    p.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    p.add_argument("--preset", help="fast or pro (default: the model's own)")
    p.add_argument("--cross-series", nargs="?", type=int, const=16, default=None, metavar="N",
                   help="read single-target series without covariates N at a time (default N: 16)")  # fmt: skip


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="yhatmodels", description="Open YHat Labs models, run locally.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="forecast the next h steps of every series")
    f.add_argument("data"), f.add_argument("--h", type=int, required=True)
    f.add_argument("--quantiles", help="levels in (0, 1), comma-separated (default 0.1,0.5,0.9), or 'all'")
    f.add_argument("--level", help="central interval widths in percent, comma-separated (e.g. 80,95)")
    f.add_argument("--future", help="file with the --known columns' values over the next h steps")
    f.add_argument("-o", "--output", help="CSV or Parquet file (default: CSV on stdout)")
    _cols(f), _model_opts(f)
    e = sub.add_parser("evaluate", help="rolling backtest against baselines")
    e.add_argument("data"), e.add_argument("--h", type=int, required=True)
    e.add_argument("--windows", type=int, default=3), e.add_argument("--step", type=int)
    e.add_argument("--metrics", help="comma-separated: mase, sql, wql, mae, rmse, coverage@0.8 (default: all of them)")
    e.add_argument("--baselines", default="seasonal_naive", help="comma-separated: seasonal_naive, naive")
    e.add_argument("--season", type=int, help="seasonal period in steps (default: from the frequency)")
    e.add_argument("-o", "--output", help="file for the per-series, per-window scores")
    _cols(e), _model_opts(e)
    t = sub.add_parser("tune", help="fine-tune the model on your series and save the tuned model")
    t.add_argument("data"), t.add_argument("--h", type=int, required=True)
    t.add_argument("--steps", type=int, default=300), t.add_argument("--seed", type=int, default=0)
    t.add_argument("--season", type=int), t.add_argument("--min-gain", type=float, default=0.02)
    t.add_argument("-o", "--output", required=True, help="folder for the tuned model (load it with --model FOLDER)")
    _cols(t), _model_opts(t)
    sub.add_parser("models", help="the released models, their tasks and licences")
    a = sub.add_parser(
        "accept-license", help="accept a model's licence before its first download (YHat Labs models: with a yhatlabs.com account)"
    )
    a.add_argument("model")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import yhatmodels

    if args.cmd == "models":
        for m in yhatmodels.list_models():
            print("\t".join(str(m.get(k, "")) for k in ("name", "task", "license", "repo")))
        return 0
    if args.cmd == "accept-license":
        where = yhatmodels.accept_license(args.model)
        print("licence accepted" + (f"; recorded in {where}" if where else ""))
        return 0
    data = _read(args.data)
    model = yhatmodels.ChakraTS.from_pretrained(args.model, device=args.device)
    target = _list(args.target_col)
    cols = {"known": _list(args.known), "past": _list(args.past), "id_col": args.id_col, "time_col": args.time_col,
            "target_col": target[0] if target and len(target) == 1 else target, "freq": args.freq}  # fmt: skip
    common = {"preset": args.preset, "cross_series": args.cross_series, **cols}
    if args.cmd == "forecast":
        quantiles = "all" if args.quantiles == "all" else _list(args.quantiles, float)
        future = _read(args.future) if args.future else None
        fc = model.forecast(data, args.h, quantiles=quantiles, level=_list(args.level, float), future=future, **common)
        _write(fc, args.output)
        return 0
    if args.cmd == "evaluate":
        kw = {"metrics": _list(args.metrics)} if args.metrics else {}
        res = model.evaluate(data, args.h, windows=args.windows, step=args.step, baselines=_list(args.baselines), season=args.season,
                             **kw, **common)  # fmt: skip
        print(res.summary.to_string(float_format=lambda x: f"{x:.4f}"))
        for w in res.warnings:
            print(f"note: {w}", file=sys.stderr)
        if args.output:
            _write(res.scores, args.output)
        return 0
    tuned = model.tune(data, args.h, steps=args.steps, seed=args.seed, season=args.season, min_gain=args.min_gain, verbose=True, **common)
    print(json.dumps(tuned.report, indent=2))
    if tuned.report and tuned.report.get("used"):
        print(f"saved to {tuned.save(args.output)}; forecast with --model {args.output}")
    else:
        print("not saved: the tune did not beat the pretrained model on your holdout", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
