#!/usr/bin/env python3
"""Accuracy of the estimated cell-type proportions (Proposed) and of the composition of SIMO-mapped cells.

For every finished run under --root (layout of scripts/run_condition.sh) that has a `diag` output, the proportion assigned to the
responding group (Group1) and to all groups is compared with the true spot composition.
Slots of the proposed model are matched to true groups through the type index stored in the trained model.

Usage: python composition_summary.py --root results/conditions [--out results/conditions/summary/composition.csv]
"""
from __future__ import annotations

import argparse
import pathlib

import anndata as ad
import numpy as np
import pandas as pd

GROUPS = [f"Group{i}" for i in range(1, 10)]


def stats(true: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    if est.std() == 0:
        return float("nan"), 0.0
    return float(np.corrcoef(true, est)[0, 1]), float((true * est).sum() / (true * true).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    rows = []
    for npz in sorted(a.root.glob("*/seed_*/*/model/diagnostics/*_composition_modelseed*.npz")):
        run = npz.parents[2]
        family, seed, condition = run.parts[-3], int(run.parts[-2].split("_")[1]), run.parts[-1]
        scen = run / "scenario"
        if not scen.exists():  # 'orig' conditions use the first author's scenario directory
            if condition == "main_cap_orig":
                scen = a.root.parent / f"simulation/data_seed_{seed}/unpaired_group_modality"
            elif condition == "complex_cap_orig":
                scen = a.root.parent / f"complex_scenario/seed_{seed}/unpaired_group_mapping_ambiguity_spatial_context"
            else:
                continue
        z = np.load(npz, allow_pickle=True)
        if "trained_cell_type_index" not in z.files:
            continue
        meta = pd.read_parquet(scen / "cell_metadata_ctrl.parquet").set_index("cell_id")
        cell_ids = [c.split("ctrl::")[-1] if "::" in c else c for c in z["ctrl_cell_id"].astype(str)]
        groups = meta.loc[cell_ids, "true_group"].astype(str).to_numpy()
        slot = z["trained_cell_type_index"].astype(int)
        p = z["p_known"].astype(np.float64)
        truth = pd.read_csv(scen / "true_cell_type_mixture.csv").set_index("spot_id").loc[z["spot_id"].astype(str), GROUPS].to_numpy()
        tab = pd.crosstab(slot, groups).reindex(index=range(p.shape[1]), columns=GROUPS, fill_value=0)
        major = tab.idxmax(axis=1).to_numpy()
        purity = float((tab.max(axis=1) / tab.sum(axis=1).clip(lower=1)).mean())
        est = np.stack([p[:, major == g].sum(axis=1) if (major == g).any() else np.zeros(len(p)) for g in GROUPS], axis=1)
        c1, s1 = stats(truth[:, 0], est[:, 0])
        call = float(np.nanmean([stats(truth[:, k], est[:, k])[0] for k in range(9)]))
        row = {"family": family, "condition": condition, "seed": seed, "n_types": int(p.shape[1]), "type_purity": purity,
               "target_group_has_own_type": bool((major == "Group1").any()),
               "proposed_G1_corr": c1, "proposed_G1_slope": s1, "proposed_all_groups_mean_corr": call,
               "proposed_background_mass": float(1.0 - p.sum(axis=1).mean())}
        for label in ("norepeat", "repeat"):
            mp = run / f"simo/map_{label}.h5ad"
            if mp.exists():
                m = ad.read_h5ad(mp)
                g = meta.loc[m.obs["Cell"].astype(str), "true_group"].astype(str).to_numpy()
                frac = pd.crosstab(m.obs["spot"].astype(str).to_numpy(), g, normalize="index").reindex(
                    index=z["spot_id"].astype(str), columns=GROUPS).fillna(0.0).to_numpy()
                c, s = stats(truth[:, 0], frac[:, 0])
                row[f"simo_{label}_G1_corr"], row[f"simo_{label}_G1_slope"] = c, s
                row[f"simo_{label}_all_groups_mean_corr"] = float(np.nanmean([stats(truth[:, k], frac[:, k])[0] for k in range(9)]))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = a.out or (a.root / "summary/composition.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    num = [c for c in df.columns if c not in ("family", "condition", "seed", "target_group_has_own_type")]
    print(df.groupby(["family", "condition"], sort=False)[num].mean().to_string())
    print(df.groupby(["family", "condition"], sort=False)["target_group_has_own_type"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
