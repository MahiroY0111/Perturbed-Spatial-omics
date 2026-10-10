"""Decompose the predicted change of the proposed method by pseudo-cell type and read it biologically: which cell types
carry the change in each region, how this compares with the cell-type-wise change of the reference snRNA-seq, and how
it relates to the observed region-level change.

  python realdata/kidney/type_attribution.py --scenario results/kidney_iri/scenario_12h --model-dir model_mix1 \
      --regions results/kidney_iri/scenario_12h/regions_spots.csv --ctrl-section sham \
      --markers Havcr1 Lcn2 Spp1 Clu Krt8 Fgb Hspa1a Slc34a1 Slc12a1 --out results/kidney_iri/scenario_12h/attrib_mix1
"""
from __future__ import annotations

import argparse
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

TARGET_SUM = 1e4


def dense(x):
    return np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=pathlib.Path, required=True)
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--name", default="kidney")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--regions", type=pathlib.Path, required=True, help="regions_spots.csv written by the region evaluation")
    ap.add_argument("--ctrl-section", default="sham")
    ap.add_argument("--markers", nargs="+", required=True)
    ap.add_argument("--min-spots", type=int, default=30)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.scenario / a.model_dir / f"diagnostics/{a.name}_type_attribution_modelseed{a.seed}.npz", allow_pickle=True)
    cT, c0 = z["contrib_T"].astype(np.float32), z["contrib_noT"].astype(np.float32)   # [S, T, G]
    bg, genes, spots = z["bg"].astype(np.float32), z["genes"].astype(str), z["spot_id"].astype(str)
    S, T, G = cT.shape
    # ---- 細胞タイプのスロットに注釈を付ける（control の核の注釈の多数派）
    sc_c = ad.read_h5ad(a.scenario / "sc_ctrl_raw.h5ad")
    ann = sc_c.obs["celltype"].astype(str) if "celltype" in sc_c.obs else pd.Series("?", index=sc_c.obs_names)
    ann = ann.reindex(z["ctrl_cell_id"].astype(str)).fillna("?").to_numpy()
    tab = pd.crosstab(z["trained_cell_type_index"], ann)
    label = {t: f"{t}:{tab.loc[t].idxmax()}" if t in tab.index else f"{t}:?" for t in range(T)}
    # ---- 領域
    reg = pd.read_csv(a.regions, index_col=0)
    reg = reg[reg["section"] == a.ctrl_section].set_index("barcode")["cluster"].astype(str).reindex(spots)
    clusters = [c for c in sorted(reg.dropna().unique(), key=lambda s: int(s) if s.isdigit() else s) if (reg == c).sum() >= a.min_spots]
    # ---- スポット単位：予測の log1p 正規化変化（提案法の eval と同じ尺度）と，その細胞タイプ別の分解（線形尺度の差を総量で正規化）
    tot_T, tot_0 = cT.sum(axis=1) + bg, c0.sum(axis=1) + bg          # [S, G]
    gi = {g: int(np.flatnonzero(genes == g)[0]) for g in a.markers if g in set(genes)}
    rows = []
    for c in clusters:
        m = (reg == c).to_numpy()
        for g, i in gi.items():
            # 領域内で合計したカウント（CPM 相当に正規化）
            lib_T, lib_0 = tot_T[m].sum(axis=1).sum(), tot_0[m].sum(axis=1).sum()
            per_type_T = cT[m, :, i].sum(axis=0) / lib_T * 1e6
            per_type_0 = c0[m, :, i].sum(axis=0) / lib_0 * 1e6
            bg_T, bg_0 = bg[m, i].sum() / lib_T * 1e6, bg[m, i].sum() / lib_0 * 1e6
            total_T, total_0 = per_type_T.sum() + bg_T, per_type_0.sum() + bg_0
            delta = per_type_T - per_type_0
            top = np.argsort(-np.abs(delta))[:3]
            rows.append({"region": f"c{c}", "gene": g, "n_spots": int(m.sum()), "CPM_noT": round(float(total_0), 1), "CPM_T": round(float(total_T), 1),
                         "log2FC_pred": round(float(np.log2((total_T + 1) / (total_0 + 1))), 2),
                         **{f"top{k+1}": f"{label[int(t)]} ({delta[t]:+.1f})" for k, t in enumerate(top)},
                         "share_top1": round(float(np.abs(delta[top[0]]) / max(np.abs(delta).sum(), 1e-9)), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "per_region_marker_attribution.csv", index=False)
    # ---- 参照 snRNA-seq の細胞タイプ別の変化（同じタイプ割当て）
    sc_s = ad.read_h5ad(a.scenario / "sc_stim_raw.h5ad")
    rna = ad.concat({"ctrl": sc_c, "stim": sc_s}, index_unique="::", join="inner", merge="same")
    rna.X = dense(rna.X).astype(np.float32)
    raw = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=TARGET_SUM); sc.pp.log1p(rna)
    sc.tl.pca(rna, n_comps=50, svd_solver="arpack", random_state=0)
    pca = np.asarray(rna.obsm["X_pca"], float)[:, :30]
    is_ctrl = (rna.obs["condition"].astype(str) == "ctrl").to_numpy()
    cid = np.array([n.split("::")[0] for n in rna.obs_names])
    t_ctrl = pd.Series(z["trained_cell_type_index"], index=z["ctrl_cell_id"].astype(str)).reindex(cid[is_ctrl]).to_numpy()
    cent = np.stack([pca[is_ctrl][t_ctrl == t].mean(axis=0) for t in range(T)])
    t_stim = ((pca[~is_ctrl][:, None, :] - cent[None, :, :]) ** 2).sum(-1).argmin(axis=1)
    ref_rows = []
    for g, i in gi.items():
        for t in range(T):
            mc, ms = raw[is_ctrl][t_ctrl == t], raw[~is_ctrl][t_stim == t]
            if len(mc) < 20: continue
            cpm_c = mc[:, i].sum() / mc.sum() * 1e6
            cpm_s = ms[:, i].sum() / ms.sum() * 1e6 if len(ms) >= 20 else np.nan
            ref_rows.append({"gene": g, "type": label[t], "n_ctrl": int(len(mc)), "n_stim": int(len(ms)), "CPM_ctrl": round(float(cpm_c), 1), "CPM_stim": round(float(cpm_s), 1),
                             "log2FC_ref": round(float(np.log2((cpm_s + 1) / (cpm_c + 1))), 2) if np.isfinite(cpm_s) else np.nan})
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(a.out / "reference_type_changes.csv", index=False)
    # ---- 領域ごとの推定構成比（主要タイプ）
    comp = pd.DataFrame({label[t]: [float(z["p_known"][(reg == c).to_numpy(), t].mean()) for c in clusters] for t in range(T)}, index=[f"c{c}" for c in clusters])
    comp.round(3).to_csv(a.out / "region_composition.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    print("== region composition (mean p_known)"); print(comp.loc[:, comp.mean() > 0.02].round(2).to_string())
    print("== predicted change per region and marker, attributed to cell types"); print(df.to_string(index=False))
    print("== reference snRNA-seq: type-wise change of the markers (types with >=20 ctrl nuclei)")
    for g in gi:
        sub = ref[ref.gene == g].dropna().sort_values("log2FC_ref", ascending=False)
        print(g, "|", ", ".join(f"{r.type} {r.log2FC_ref:+.1f} ({r.CPM_ctrl:.0f}->{r.CPM_stim:.0f})" for r in sub.head(5).itertuples()))


if __name__ == "__main__":
    main()
