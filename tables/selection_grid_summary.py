"""Export hyperparameter-grid evaluation summaries as CSV."""
import argparse, pathlib
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
METRICS = [
    "RMSE_all",
    "RMSE_P",
    "RMSE_Pc",
    "ms_spotwise_change_corr",
    "ms_genewise_spatial_corr_P",
    "ms_spotwise_multigene_corr_P",
    "resp_slope_P",
    "resp_R2_P",
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=pathlib.Path, default=HERE.parent / "results/selection_grid")
    a = ap.parse_args()
    rows = []
    for f in sorted(a.root.glob("seed_*/*/sp*_g*/eval.csv")):
        d = pd.read_csv(f)
        d = d[d["method"] == "proposed"]
        if len(d) != 1:
            continue
        _, scenario, sp, g = str(d["tag"].iloc[0]).split(";")
        r = d.iloc[0].to_dict(); r.update(scenario=scenario, lambda_sp_gamma=float(sp), lambda_gamma=float(g), seed=int(d["seed"].iloc[0]))
        rows.append(r)
    if not rows:
        raise ValueError(f"No unique proposed-method rows found under {a.root}")
    cells = pd.DataFrame(rows)
    out = a.root / "summary"; out.mkdir(parents=True, exist_ok=True)
    keep = ["scenario", "seed", "lambda_sp_gamma", "lambda_gamma"] + METRICS
    cells[keep].sort_values(keep[:4]).to_csv(out / "grid_cells.csv", index=False)
    n = cells.groupby(["scenario", "lambda_sp_gamma", "lambda_gamma"]).size()
    summ = cells.groupby(["scenario", "lambda_sp_gamma", "lambda_gamma"])[METRICS].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ["n_seeds"] = n
    summ = summ.reset_index()
    summ.to_csv(out / "grid_summary.csv", index=False)


if __name__ == "__main__":
    main()
