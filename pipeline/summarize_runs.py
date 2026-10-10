#!/usr/bin/env python3
"""Collect the per-run `eval.csv` files of the additional conditions and print seed-wise summaries.

Usage:  python summarize_runs.py --root results/conditions [--out results/conditions/summary]
Writes  all_rows.csv (one row per run x method), summary_mean_sd.csv (mean and SD over seeds),
        and paired_wins.csv (number of seeds in which Proposed is better than each comparison row).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"RMSE_all", "RMSE_P", "RMSE_Pc", "ctrl_recon_RMSE", "resp_RMSE_P", "resp_RMSE_Pc"}
MAIN_METRICS = ["RMSE_all", "RMSE_P", "RMSE_Pc", "ctrl_recon_RMSE", "resp_R2_P", "resp_slope_P",
                "resp_genewise_spatial_corr_P", "resp_spotwise_multigene_corr_P", "resp_RMSE_P",
                "ms_spotwise_change_corr", "ms_genewise_spatial_corr_P", "ms_spotwise_multigene_corr_P"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    files = sorted(a.root.glob("*/seed_*/*/eval.csv"))
    if not files:
        print("no eval.csv under", a.root)
        return
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["family"] = f.parts[-4]
        d["condition"] = f.parts[-2]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    out = a.out or (a.root / "summary")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_rows.csv", index=False)
    metrics = [m for m in MAIN_METRICS if m in df.columns]
    grouped = df.groupby(["family", "condition", "method"], sort=False)
    mean = grouped[metrics].mean()
    sd = grouped[metrics].std(ddof=1)
    n = grouped["seed"].nunique().rename("n_seeds")
    summary = mean.join(sd, rsuffix="_sd").join(n).reset_index()
    summary.to_csv(out / "summary_mean_sd.csv", index=False)

    wins = []
    for (family, condition), block in df.groupby(["family", "condition"], sort=False):
        wide = {m: block.pivot(index="seed", columns="method", values=m) for m in metrics}
        for other in [c for c in block["method"].unique() if c != "proposed"]:
            row = {"family": family, "condition": condition, "versus": other}
            for m in metrics:
                w = wide[m]
                if "proposed" not in w or other not in w:
                    continue
                pair = w[["proposed", other]].dropna()
                if pair.empty:
                    continue
                better = pair["proposed"] < pair[other] if m in LOWER_IS_BETTER else pair["proposed"] > pair[other]
                row[m] = f"{int(better.sum())}/{len(pair)}"
            wins.append(row)
    pd.DataFrame(wins).to_csv(out / "paired_wins.csv", index=False)

    if not a.quiet:
        pd.set_option("display.width", 320, "display.max_columns", 50, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
        show = ["RMSE_all", "RMSE_P", "ctrl_recon_RMSE", "resp_R2_P", "resp_slope_P", "resp_genewise_spatial_corr_P", "resp_spotwise_multigene_corr_P"]
        show = [m for m in show if m in summary.columns]
        for (family, condition), block in summary.groupby(["family", "condition"], sort=False):
            print(f"\n=== {family} / {condition}  (n_seeds={int(block['n_seeds'].max())})")
            print(block[["method"] + show].to_string(index=False))


if __name__ == "__main__":
    main()
