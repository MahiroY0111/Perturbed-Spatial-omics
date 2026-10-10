"""Export main simulation summaries as CSV tables."""
import csv
import pathlib
import statistics as st
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = ROOT / "results/conditions/summary/all_rows.csv"
OUT = ROOT / "results/tables"
CONDITIONS = [
    ("unpaired group", "capture", "main_cap_none"),
    ("heterogeneous response", "hetero", "hetero_main"),
    ("modality shift", "capture", "main_cap_orig"),
]
METHODS = {
    "提案法": "proposed",
    "SIMO+CellOT": "simo[norepeat]",
    "Identity": "identity",
    "Mean-ratio": "mean_ratio",
}
METRICS = [
    ("RMSE_all", -1),
    ("RMSE_P", -1),
    ("RMSE_Pc", -1),
    ("ms_spotwise_change_corr", 1),
    ("ms_sign_accuracy", 1),
    ("ms_genewise_spatial_corr_P", 1),
    ("ms_spotwise_multigene_corr_P", 1),
]

rows = list(csv.DictReader(ROWS.open()))
values = {}
for row in rows:
    for condition_label, family, condition in CONDITIONS:
        if row["family"] != family or row["condition"] != condition:
            continue
        if row["method"] in METHODS.values():
            for metric, _ in METRICS:
                if metric != "ms_sign_accuracy" and row.get(metric) not in (None, ""):
                    values.setdefault((condition_label, row["method"], metric), {})[
                        int(row["seed"])
                    ] = float(row[metric])
        if row["method"] == "proposed_ctrl_recon" and row.get("ctrl_recon_RMSE"):
            values.setdefault((condition_label, "proposed", "ctrl_recon_RMSE"), {})[
                int(row["seed"])
            ] = float(row["ctrl_recon_RMSE"])

sign_file = ROOT / "results/tables/sign_accuracy.csv"
for row in csv.DictReader(sign_file.open()):
    method = {"proposed": "proposed", "simo": "simo[norepeat]"}[row["method"]]
    values.setdefault((row["condition"], method, "ms_sign_accuracy"), {})[
        int(row["seed"])
    ] = float(row["sign_accuracy_log2fc_P"])

strict = "--strict" in sys.argv


def mean_stats(condition: str, method: str, metric: str):
    by_seed = values.get((condition, method, metric), {})
    if not by_seed:
        return None
    observations = list(by_seed.values())
    result = {
        "mean": st.mean(observations),
        "std": st.stdev(observations) if len(observations) > 1 else None,
        "n": len(observations),
        "seeds": sorted(by_seed),
    }
    return result


def write_csv(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def main() -> None:
    metric_records = []
    baseline_records = []
    control_records = []
    win_records = []

    for condition, _, _ in CONDITIONS:
        proposed_counts = {
            mean_stats(condition, "proposed", metric)["n"]
            for metric, _ in METRICS
            if mean_stats(condition, "proposed", metric) is not None
        }
        simo_counts = {
            mean_stats(condition, "simo[norepeat]", metric)["n"]
            for metric, _ in METRICS
            if mean_stats(condition, "simo[norepeat]", metric) is not None
        }
        if strict:
            assert proposed_counts == {10} and simo_counts == {10}, (
                condition,
                proposed_counts,
                simo_counts,
            )

        for method_label, method in METHODS.items():
            for metric, _ in METRICS:
                result = mean_stats(condition, method, metric)
                if result is None:
                    continue
                record = {
                    "condition": condition,
                    "method": method,
                    "method_label": method_label,
                    "metric": metric,
                    "mean": result["mean"],
                    "std": result["std"],
                    "n": result["n"],
                }
                metric_records.append(record)
                if metric == "RMSE_all":
                    baseline_records.append(record)

        control = mean_stats(condition, "proposed", "ctrl_recon_RMSE")
        prediction = mean_stats(condition, "proposed", "RMSE_all")
        if control is not None and prediction is not None:
            control_records.append(
                {
                    "condition": condition,
                    "control_reconstruction_mean": control["mean"],
                    "control_reconstruction_std": control["std"],
                    "control_reconstruction_n": control["n"],
                    "stimulated_prediction_mean": prediction["mean"],
                    "stimulated_prediction_std": prediction["std"],
                    "stimulated_prediction_n": prediction["n"],
                    "mean_difference": prediction["mean"] - control["mean"],
                }
            )

        for metric, direction in METRICS:
            proposed = values.get((condition, "proposed", metric), {})
            simo = values.get((condition, "simo[norepeat]", metric), {})
            common = sorted(set(proposed) & set(simo))
            if common:
                win_records.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "proposed_better": sum(
                            direction * (proposed[seed] - simo[seed]) > 0 for seed in common
                        ),
                        "n": len(common),
                    }
                )

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "simulation_table.csv", metric_records)
    write_csv(OUT / "baseline_comparison.csv", baseline_records)
    write_csv(OUT / "control_reconstruction.csv", control_records)
    write_csv(OUT / "paired_wins.csv", win_records)


if __name__ == "__main__":
    main()
