#!/usr/bin/env python3
"""Reference rows and response-specific metrics from saved predictions (no training).

Works on the output layout of `run_condition.py`:
  predictions/<S>_proposed_control_raw_G8000_modelseedN.h5ad   Proposed, control reconstruction (no transport)
  predictions/<S>_proposed_stim_raw_G8000_modelseedN.h5ad      Proposed, stimulated prediction
  predictions/<S>_simo_cellot_normlog_G8000_modelseedN.h5ad    SIMO+CellOT, stimulated prediction
  predictions/<S>_simo_*_normlog_G8000_modelseedN__<label>.h5ad  (run_condition.py) variants per SIMO mapping

Rows (method):
  identity, mean_ratio            simple comparison methods
  proposed, simo                  the two methods as reported
  proposed_noT, simo_noT          the same pipelines with the transport replaced by the identity map
                                  (perturbation-blind; they isolate what reconstruction/denoising alone achieves)
  simo_cal, simo_noT_cal          SIMO+CellOT with a post hoc gene-wise scale fitted on control ST (run_condition.py)
  ref_denoised_ctrl, ref_oracle   true spot composition x scRNA-seq group means (yardsticks that use truth labels)

Metrics:
  RMSE_all / RMSE_P / RMSE_Pc     against the observed stimulated ST
  ms_*                            expression-change correlations as defined in the paper
                                  (the observed control ST is subtracted from both truth and prediction)
  ctrl_recon_RMSE                 RMSE between a method's no-transport output and the observed control ST
  resp_*                          the method's own predicted change (prediction minus its no-transport output)
                                  against the counterfactual change of the control spots, i.e. the change obtained
                                  by applying the generating perturbation to the very cells that make up each
                                  control spot.  No sampling noise is shared between truth and prediction.
      resp_R2_P                   1 - sum (pred - truth)^2 / sum truth^2 over perturbed genes
      resp_slope_P                least-squares slope through the origin of predicted on true change
      resp_genewise_spatial_corr_P, resp_spotwise_multigene_corr_P
      resp_RMSE_P, resp_RMSE_Pc
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from cellot.models.ae import AutoEncoder  # noqa: E402

TARGET_SUM = 1e4


def dense(x):
    return np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float64)


def normlog(raw):
    lib = raw.sum(axis=1, keepdims=True)
    return np.log1p(raw * (TARGET_SUM / np.where(lib > 0, lib, 1.0)))


def rowcorr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def nanmean(x):
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def evaluate(pred, truth, ctrl_obs, mask_p, base=None, d_cf=None):
    out = {}
    err = pred - truth
    out["RMSE_all"] = float(np.sqrt(np.mean(err**2)))
    out["RMSE_P"] = float(np.sqrt(np.mean(err[:, mask_p] ** 2)))
    out["RMSE_Pc"] = float(np.sqrt(np.mean(err[:, ~mask_p] ** 2)))
    d_true, d_pred = truth - ctrl_obs, pred - ctrl_obs
    out["ms_spotwise_change_corr"] = nanmean(rowcorr(d_true, d_pred))
    out["ms_genewise_spatial_corr_P"] = nanmean(rowcorr(d_true[:, mask_p].T, d_pred[:, mask_p].T))
    out["ms_spotwise_multigene_corr_P"] = nanmean(rowcorr(d_true[:, mask_p], d_pred[:, mask_p]))
    gm_true, gm_pred = d_true.mean(axis=0), d_pred.mean(axis=0)
    out["ms_sign_accuracy"] = float(np.mean(np.sign(gm_true) == np.sign(gm_pred)))
    if base is not None:
        out["ctrl_recon_RMSE"] = float(np.sqrt(np.mean((base - ctrl_obs) ** 2)))
        if d_cf is not None:
            dp = pred - base
            t, q = d_cf[:, mask_p], dp[:, mask_p]
            ss = float((t * t).sum())
            out["resp_R2_P"] = 1.0 - float(((q - t) ** 2).sum()) / ss
            out["resp_slope_P"] = float((q * t).sum()) / ss
            out["resp_genewise_spatial_corr_P"] = nanmean(rowcorr(t.T, q.T))
            out["resp_spotwise_multigene_corr_P"] = nanmean(rowcorr(t, q))
            out["resp_RMSE_P"] = float(np.sqrt(np.mean((q - t) ** 2)))
            out["resp_RMSE_Pc"] = float(np.sqrt(np.mean((dp[:, ~mask_p] - d_cf[:, ~mask_p]) ** 2)))
    return out


def counterfactual_change(D: pathlib.Path, genes, spots, kind: str):
    """Change obtained by perturbing the cells that make up each control spot (expected capture applied)."""
    cells_file = D / "st_cells_ctrl_raw.h5ad" if (D / "st_cells_ctrl_raw.h5ad").exists() else D / "sc_ctrl_raw.h5ad"
    map_file = D / "st_cells_mapping_ctrl.parquet" if (D / "st_cells_mapping_ctrl.parquet").exists() else D / "mapping_ctrl.parquet"
    meta_file = D / "st_cells_metadata_ctrl.parquet" if (D / "st_cells_metadata_ctrl.parquet").exists() else D / "cell_metadata_ctrl.parquet"
    cells = ad.read_h5ad(cells_file)
    assert (cells.var_names.astype(str) == genes).all()
    x = dense(cells.X)
    ids = cells.obs_names.astype(str)
    meta = pd.read_parquet(meta_file).set_index("cell_id").loc[ids]
    mapping = pd.read_parquet(map_file).set_index("cell_id").loc[ids]
    spot_code = pd.Index(spots).get_indexer(mapping["spot_id"].astype(str))
    assert (spot_code >= 0).all()
    pert = pd.read_csv(D / "perturbed_genes.csv")
    pidx = pert["gene_index_zero_based"].to_numpy(int)
    target = np.flatnonzero((meta["true_group"].astype(str) == "Group1").to_numpy())
    x_cf = x.copy()
    if kind == "group":
        x_cf[np.ix_(target, pidx)] = np.rint(x[np.ix_(target, pidx)] * pert["factor"].to_numpy(float)[None, :])
    elif kind == "complex":
        coef = pd.read_csv(D / "perturbation_response_coefficients.csv")
        assert (coef["gene_index_zero_based"].to_numpy(int) == pidx).all()
        base = pert["log2fc"].to_numpy(float)
        z = mapping["latent_state_z"].to_numpy(float)[target]
        u = mapping["microenvironment_u"].to_numpy(float)[target]
        signed_abs = np.abs(base)[None, :] + coef["theta_state"].to_numpy(float)[None, :] * z[:, None] \
            + coef["gamma_environment"].to_numpy(float)[None, :] * u[:, None]
        lfc = np.sign(base)[None, :] * np.clip(signed_abs, 0.05, 2.25)
        x_cf[np.ix_(target, pidx)] = np.rint(x[np.ix_(target, pidx)] * np.exp2(lfc))
    else:
        raise ValueError(kind)
    agg = np.zeros((len(spots), x.shape[1]))
    agg_cf = np.zeros_like(agg)
    np.add.at(agg, spot_code, x)
    np.add.at(agg_cf, spot_code, x_cf)
    cap = np.ones(len(genes))
    if (D / "capture_efficiency.csv").exists():
        cap = pd.read_csv(D / "capture_efficiency.csv").set_index("gene_id").loc[genes, "capture_efficiency"].to_numpy()
    return normlog(agg_cf * cap) - normlog(agg * cap)


def simo_no_transport(D, A, S, N, mapping_path, spot_names):
    """Decode the SIMO-mapped control cells without transport and aggregate per spot (standard layout only)."""
    bundle = torch.load(A / f"models/{S}_ae_G8000_modelseed{N}.pt", map_location="cpu")
    ctrl = ad.read_h5ad(D / "sc_ctrl_raw.h5ad")
    ctrl.X = dense(ctrl.X).astype(np.float32)
    sc.pp.normalize_total(ctrl, target_sum=TARGET_SUM)
    sc.pp.log1p(ctrl)
    model = AutoEncoder(ctrl.n_vars, hidden_units=[128, 128], latent_dim=50)
    model.load_state_dict(bundle["state"])
    model.eval()
    mapped = ad.read_h5ad(mapping_path)
    pos = pd.Index(ctrl.obs_names.astype(str)).get_indexer(mapped.obs["Cell"].astype(str))
    assert (pos >= 0).all()
    spot_code = pd.Index(spot_names).get_indexer(mapped.obs["spot"].astype(str))
    assert (spot_code >= 0).all()
    x = np.asarray(ctrl.X, dtype=np.float32)[pos]
    out = np.zeros((len(spot_names), ctrl.n_vars), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, x.shape[0], 512):
            lin = np.expm1(model.decode(model.encode(torch.as_tensor(x[i : i + 512]))).numpy()).clip(min=0)
            np.add.at(out, spot_code[i : i + len(lin)], lin)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", type=pathlib.Path, required=True)
    ap.add_argument("--arm-dir", type=pathlib.Path, required=True)
    ap.add_argument("--name", required=True, help="scenario label used in the prediction file names")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mapping", type=pathlib.Path, help="SIMO mapping (standard layout: used to rebuild simo_noT)")
    ap.add_argument("--truth-kind", choices=("group", "complex", "none"), default="group")
    ap.add_argument("--tag", default="", help="free text copied to every row (e.g. the condition of a sweep)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    S, N, D, A = a.name, a.seed, a.scenario_dir, a.arm_dir

    st_ctrl = ad.read_h5ad(D / "st_ctrl_raw.h5ad")
    st_stim = ad.read_h5ad(D / "st_stim_raw.h5ad")
    genes = st_ctrl.var_names.astype(str).to_numpy()
    spots = st_ctrl.obs_names.astype(str).to_numpy()
    mask_p = np.isin(genes, pd.read_csv(D / "perturbed_genes.csv")["gene_id"].astype(str))
    y_ctrl_raw, y_stim_raw = dense(st_ctrl.X), dense(st_stim.X)
    y_true, y_ctrl = normlog(y_stim_raw), normlog(y_ctrl_raw)

    def load(path):
        p = ad.read_h5ad(path)
        assert (p.var_names.astype(str) == genes).all() and (p.obs_names.astype(str) == spots).all(), path
        return dense(p.X)

    pred_dir = A / "predictions"
    nl, base_of = {"identity": y_ctrl}, {"identity": "identity", "mean_ratio": "identity"}
    sc_ctrl = ad.read_h5ad(D / "sc_ctrl_raw.h5ad")
    sc_stim = ad.read_h5ad(D / "sc_stim_raw.h5ad")
    xc, xs = dense(sc_ctrl.X), dense(sc_stim.X)
    mean_c, mean_s = xc.mean(axis=0), xs.mean(axis=0)
    ratio = np.divide(mean_s, mean_c, out=np.zeros_like(mean_s), where=mean_c > 0)
    nl["mean_ratio"] = normlog(y_ctrl_raw * ratio)
    nl["proposed"] = normlog(load(pred_dir / f"{S}_proposed_stim_raw_G8000_modelseed{N}.h5ad"))
    # T = identity through the same prediction path if available (run_condition diag stage); the control
    # reconstruction of the fitted model is kept as a separate row (it differs only by the pseudo-cell mass threshold).
    nl["proposed_ctrl_recon"] = normlog(load(pred_dir / f"{S}_proposed_control_raw_G8000_modelseed{N}.h5ad"))
    exact = pred_dir / f"{S}_proposed_noT_raw_G8000_modelseed{N}.h5ad"
    nl["proposed_noT"] = normlog(load(exact)) if exact.exists() else nl["proposed_ctrl_recon"]
    base_of.update({"proposed": "proposed_noT", "proposed_noT": "proposed_noT", "proposed_ctrl_recon": "proposed_ctrl_recon"})

    variants = sorted(pred_dir.glob(f"{S}_simo_cellot_normlog_G8000_modelseed{N}__*.h5ad"))
    if variants:  # run_condition.py layout
        for path in variants:
            label = path.stem.split("__", 1)[1]
            for kind, key in (("simo_cellot", "simo"), ("simo_noT", "simo_noT"), ("simo_cellot_cal", "simo_cal"), ("simo_noT_cal", "simo_noT_cal")):
                nl[f"{key}[{label}]"] = load(pred_dir / f"{S}_{kind}_normlog_G8000_modelseed{N}__{label}.h5ad")
            base_of.update({f"simo[{label}]": f"simo_noT[{label}]", f"simo_noT[{label}]": f"simo_noT[{label}]",
                            f"simo_cal[{label}]": f"simo_noT_cal[{label}]", f"simo_noT_cal[{label}]": f"simo_noT_cal[{label}]"})
    else:  # standard layout of the final pipeline
        nl["simo"] = load(pred_dir / f"{S}_simo_cellot_normlog_G8000_modelseed{N}.h5ad")
        if a.mapping is not None and a.mapping.exists():
            lin_noT = simo_no_transport(D, A, S, N, a.mapping, spots)
            nl["simo_noT"] = normlog(lin_noT)
            base_of.update({"simo": "simo_noT", "simo_noT": "simo_noT"})

    counts_file = D / "spot_group_counts.csv"
    if counts_file.exists():
        groups = [f"Group{i}" for i in range(1, 10)]
        counts = pd.read_csv(counts_file).set_index("spot_id").loc[spots, groups].to_numpy(float)
        meta_c = pd.read_parquet(D / "cell_metadata_ctrl.parquet").set_index("cell_id").loc[sc_ctrl.obs_names]
        meta_s = pd.read_parquet(D / "cell_metadata_stim.parquet").set_index("cell_id").loc[sc_stim.obs_names]
        m_ctrl = np.stack([xc[(meta_c["true_group"] == g).to_numpy()].mean(axis=0) for g in groups])
        m_stim = np.stack([xs[(meta_s["true_group"] == g).to_numpy()].mean(axis=0) for g in groups])
        cap = np.ones(len(genes))
        if (D / "capture_efficiency.csv").exists():
            cap = pd.read_csv(D / "capture_efficiency.csv").set_index("gene_id").loc[genes, "capture_efficiency"].to_numpy()
        nl["ref_denoised_ctrl"] = normlog((counts @ m_ctrl) * cap)
        nl["ref_oracle"] = normlog((counts @ m_stim) * cap)
        base_of.update({"ref_denoised_ctrl": "ref_denoised_ctrl", "ref_oracle": "ref_denoised_ctrl"})
    del xc, xs

    d_cf = None if a.truth_kind == "none" else counterfactual_change(D, genes, spots, a.truth_kind)
    rows = []
    for key, pred in nl.items():
        base = nl.get(base_of.get(key, ""))
        rows.append({"scenario": S, "tag": a.tag, "seed": N, "method": key, **evaluate(pred, y_true, y_ctrl, mask_p, base=base, d_cf=d_cf)})
    df = pd.DataFrame(rows)
    if d_cf is not None:
        df["truth_response_RMS_P"] = float(np.sqrt(np.mean(d_cf[:, mask_p] ** 2)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 320, "display.max_columns", 40, "display.float_format", "{:.4f}".format)
    print(df.drop(columns=["scenario", "seed", "tag"]).to_string(index=False))


if __name__ == "__main__":
    main()
