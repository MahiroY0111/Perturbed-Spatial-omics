"""Compare predicted and observed changes by tissue region in the kidney check (one section per time point). Regions are
defined by joint clustering of the four sections (Harmony, Leiden) and are used only for the evaluation.
Observed change: region mean of the injured section minus that of the sham section (total count 1e4, log1p).
Predicted change: region mean on the sham section of (prediction - the method's prediction without transport).

  python realdata/kidney/region_eval.py --root results/kidney_iri --data-root data/kidney_iri --scenario scenario_12h \
      --stim-section hour12 --seed 1 --out results/kidney_iri/scenario_12h/eval.csv
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_scenario import read_visium  # noqa: E402

TARGET_SUM = 1e4
MARKERS = ["Havcr1", "Lcn2", "Spp1", "Hspa1a", "Fgb", "Clu", "Krt8", "Slc34a1", "Kcnj1", "Slc12a1", "Umod", "Aqp2"]


def dense(x):
    return np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float64)


def normlog(raw):
    lib = raw.sum(axis=1, keepdims=True)
    return np.log1p(raw * (TARGET_SUM / np.where(lib > 0, lib, 1.0)))


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--data-root", type=pathlib.Path, default=None, help="directory with visium/<section>/ (default: --root)")
    ap.add_argument("--stim-section", default="hour12")
    ap.add_argument("--sections", nargs="*", default=["sham", "hour4", "hour12", "day2"])
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--name", default="kidney")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-dir", default="model", help="model output directory inside the scenario (model, model_<tag>)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    D = a.root / a.scenario
    S, N = a.name, a.seed
    st = ad.read_h5ad(D / "st_ctrl_raw.h5ad")
    genes, spots = st.var_names.astype(str).to_numpy(), st.obs_names.astype(str).to_numpy()
    y_raw = dense(st.X)
    y_ctrl = normlog(y_raw)

    # ---- 領域：4 切片の共同クラスタリング
    parts = []
    for sec in a.sections:
        x = read_visium((a.data_root or a.root) / "visium" / sec)[:, genes].copy()
        x.obs["section"] = sec
        x.obs["barcode"] = x.obs_names
        x.obs_names = [f"{sec}:{b}" for b in x.obs_names]
        parts.append(x)
    al = ad.concat(parts, join="inner")
    sc.pp.normalize_total(al, target_sum=TARGET_SUM)
    sc.pp.log1p(al)
    normlog_all = al.X.copy()
    sc.pp.highly_variable_genes(al, n_top_genes=2000, batch_key="section")
    work = al[:, al.var["highly_variable"]].copy()
    sc.pp.scale(work, max_value=10)
    sc.tl.pca(work, n_comps=30, random_state=0)
    sc.external.pp.harmony_integrate(work, key="section", random_state=0)
    sc.pp.neighbors(work, use_rep="X_pca_harmony", n_neighbors=15, random_state=0)
    sc.tl.leiden(work, resolution=a.resolution, random_state=0, flavor="leidenalg")
    al.obs["cluster"] = work.obs["leiden"].astype(str).to_numpy()
    clusters = sorted(al.obs["cluster"].unique(), key=int)
    sc.tl.rank_genes_groups(al, "cluster", method="t-test_overestim_var", n_genes=6)
    marker_table = pd.DataFrame(al.uns["rank_genes_groups"]["names"]).T
    means = {}
    for sec in a.sections:
        in_s = (al.obs["section"] == sec).to_numpy()
        means[sec] = np.stack([np.asarray(normlog_all[in_s & (al.obs["cluster"] == c).to_numpy()].mean(axis=0)).ravel()
                               if (in_s & (al.obs["cluster"] == c).to_numpy()).sum() >= 10 else np.full(len(genes), np.nan) for c in clusters])
    counts = pd.DataFrame({sec: [int(((al.obs["section"] == sec) & (al.obs["cluster"] == c)).sum()) for c in clusters] for sec in a.sections}, index=[f"c{c}" for c in clusters])
    print(counts.T.to_string()); print(marker_table.to_string())
    sham_lab = al.obs[al.obs["section"] == "sham"].set_index("barcode")["cluster"].astype(str)
    spot_cluster = sham_lab.loc[spots].to_numpy()
    valid = np.array([(spot_cluster == c).sum() >= 30 and np.isfinite(means["sham"][k]).all() and np.isfinite(means[a.stim_section][k]).all() for k, c in enumerate(clusters)])
    use = np.flatnonzero(valid)
    d_obs = (means[a.stim_section] - means["sham"])[use]

    # ---- 予測
    P = D / a.model_dir / "predictions"
    def load(path):
        p = ad.read_h5ad(path)
        assert (p.var_names.astype(str) == genes).all() and (p.obs_names.astype(str) == spots).all(), path
        return dense(p.X)
    sc_c, sc_s = ad.read_h5ad(D / "sc_ctrl_raw.h5ad"), ad.read_h5ad(D / "sc_stim_raw.h5ad")
    mean_c, mean_s = dense(sc_c.X).mean(axis=0), dense(sc_s.X).mean(axis=0)
    ratio = np.divide(mean_s, mean_c, out=np.ones_like(mean_s), where=mean_c > 0)
    sn_lfc = np.log1p(mean_s / mean_s.sum() * TARGET_SUM) - np.log1p(mean_c / mean_c.sum() * TARGET_SUM)
    changes = {"mean_ratio": normlog(y_raw * ratio) - y_ctrl}
    def one(pattern):
        hits = sorted(P.glob(pattern)); assert len(hits) == 1, (pattern, hits); return hits[0]
    exact = sorted(P.glob(f"{S}_proposed_noT_raw_G*_modelseed{N}.h5ad"))
    base = normlog(load(exact[0] if exact else one(f"{S}_proposed_control_raw_G*_modelseed{N}.h5ad")))
    changes["proposed"] = normlog(load(one(f"{S}_proposed_stim_raw_G*_modelseed{N}.h5ad"))) - base
    for path in sorted(P.glob(f"{S}_simo_cellot_normlog_G*_modelseed{N}__*.h5ad")):
        label = path.stem.split("__", 1)[1]
        changes[f"simo[{label}]"] = load(path) - load(one(f"{S}_simo_noT_normlog_G*_modelseed{N}__{label}.h5ad"))
    # ---- 追加の基準：推定構成比 × 細胞タイプ別の参照変化（提案法の第1段階だけを使う），および遺伝子別の大きさ補正
    comp_path = D / a.model_dir / f"diagnostics/{S}_composition_modelseed{N}.npz"
    if comp_path.exists():
        z = np.load(comp_path, allow_pickle=True)
        assert (z["spot_id"].astype(str) == spots).all()
        p_known = np.asarray(z["p_known"], float)
        p_known = p_known / p_known.sum(axis=1, keepdims=True)
        ctrl_type = pd.Series(z["trained_cell_type_index"], index=z["ctrl_cell_id"].astype(str))
        # stimulated 核を control の Leiden タイプへ最近傍重心で割り当てる（run_condition と同じ手順）
        rna = ad.concat({"ctrl": sc_c, "stim": sc_s}, index_unique="::", join="inner", merge="same")
        rna.X = dense(rna.X).astype(np.float32)
        sc.pp.normalize_total(rna, target_sum=TARGET_SUM); sc.pp.log1p(rna)
        sc.tl.pca(rna, n_comps=50, svd_solver="arpack", random_state=0)
        pca = np.asarray(rna.obsm["X_pca"], float)[:, :30]
        is_ctrl = (rna.obs["condition"].astype(str) == "ctrl").to_numpy()
        cid = np.array([n.split("::")[0] for n in rna.obs_names])
        t_ctrl = ctrl_type.reindex(cid[is_ctrl]).to_numpy()
        T = p_known.shape[1]
        cent = np.stack([pca[is_ctrl][t_ctrl == t].mean(axis=0) for t in range(T)])
        d2 = ((pca[~is_ctrl][:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        t_stim = d2.argmin(axis=1)
        Xc_lin, Xs_lin = np.expm1(np.asarray(rna.X[is_ctrl].todense() if sp.issparse(rna.X) else rna.X[is_ctrl])), np.expm1(np.asarray(rna.X[~is_ctrl].todense() if sp.issparse(rna.X) else rna.X[~is_ctrl]))
        lfc_t = np.zeros((T, len(genes)))
        for t in range(T):
            mc, ms = Xc_lin[t_ctrl == t].mean(axis=0), (Xs_lin[t_stim == t].mean(axis=0) if (t_stim == t).sum() >= 20 else Xc_lin[t_ctrl == t].mean(axis=0))
            lfc_t[t] = np.log1p(ms) - np.log1p(mc)
        changes["composition_x_typeLFC"] = p_known @ lfc_t
        print("stim nuclei per control type:", np.bincount(t_stim, minlength=T).tolist())
    def calibrate(delta):  # 遺伝子ごとに擬似バルクの変化の大きさを参照に合わせる（空間パターンだけを残す）
        m = delta.mean(axis=0)
        f = np.where(np.abs(m) > 1e-3, sn_lfc / np.where(np.abs(m) > 1e-3, m, 1.0), 0.0)
        f = np.clip(f, 0.0, 50.0)  # 符号が逆の遺伝子は 0 に
        return delta * f[None, :]
    changes["proposed_calibrated"] = calibrate(changes["proposed"])
    for k in [k for k in list(changes) if k.startswith("simo[")]:
        changes[k.replace("simo[", "simo_calibrated[")] = calibrate(changes[k])
    region = {k: np.stack([v[spot_cluster == clusters[c]].mean(axis=0) for c in use]) for k, v in changes.items()}
    region["snRNA_pseudobulk_uniform"] = np.tile(sn_lfc, (len(use), 1))

    # ---- 指標
    expressed = (y_ctrl.mean(axis=0) > 0.05) | (means[a.stim_section][use].mean(axis=0) > 0.05)
    strong = (np.abs(d_obs) > 0.5) & expressed[None, :]          # 観測された変化が大きい（領域, 遺伝子）
    resp_genes = np.flatnonzero((np.abs(d_obs) > 0.5).any(axis=0) & expressed)
    spread = d_obs.max(axis=0) - d_obs.min(axis=0)
    hetero = np.flatnonzero(((np.abs(d_obs) > 0.5).any(axis=0)) & expressed & (spread > 0.5))  # 領域差のある遺伝子
    # 領域差が他の時点の切片でも再現する遺伝子（切片間のばらつきではなく応答の空間依存とみなせるもの）
    others = [sec for sec in a.sections if sec not in ("sham", a.stim_section) and np.isfinite(means[sec][use]).all()]
    rep = np.zeros(len(genes), dtype=bool)
    for sec in others:
        d_o = (means[sec] - means["sham"])[use]
        for g in hetero:
            if corr(d_o[:, g], d_obs[:, g]) > 0.5:
                rep[g] = True
    hetero_rep = np.flatnonzero(rep)
    print(f"region-dependent genes: {len(hetero)}; reproducible in another section ({others}): {len(hetero_rep)}")
    gi = {g: int(np.flatnonzero(genes == g)[0]) for g in MARKERS if g in set(genes)}
    rows = []
    for k, pr in region.items():
        rows.append({"method": k,
                     "corr_all_pairs": corr(pr[:, expressed], d_obs[:, expressed]),
                     "corr_strong_pairs": corr(pr[strong], d_obs[strong]),
                     "slope_strong_pairs": float((pr[strong] * d_obs[strong]).sum() / (d_obs[strong] ** 2).sum()),
                     "sign_agreement_strong": float(np.mean(np.sign(pr[strong]) == np.sign(d_obs[strong]))),
                     "corr_responsive_genes_all_regions": corr(pr[:, resp_genes], d_obs[:, resp_genes]),
                     "within_gene_across_region_corr": float(np.nanmean([corr(pr[:, g], d_obs[:, g]) for g in hetero])) if len(hetero) else float("nan"),
                     "within_gene_region_corr_reproducible": float(np.nanmean([corr(pr[:, g], d_obs[:, g]) for g in hetero_rep])) if len(hetero_rep) else float("nan"),
                     "n_reproducible_genes": int(len(hetero_rep)),
                     "marker_within_gene_across_region_corr": float(np.nanmean([corr(pr[:, g], d_obs[:, g]) for g in gi.values()])),
                     "n_regions": int(len(use)), "n_strong_pairs": int(strong.sum()), "n_responsive_genes": int(len(resp_genes)), "n_region_dependent_genes": int(len(hetero))})
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print(df.to_string(index=False))
    idx = [f"c{clusters[c]}" for c in use]
    for k in ["observed"] + list(region):
        mat = d_obs if k == "observed" else region[k]
        print(f"\n{k}: change per region")
        print(pd.DataFrame(mat[:, list(gi.values())], index=idx, columns=list(gi)).T.round(2).to_string())
    al.obs[["section", "barcode", "cluster"]].to_csv(a.out.with_name("regions_spots.csv"))
    np.savez_compressed(a.out.with_suffix(".npz"), observed=d_obs, clusters=np.asarray(idx), genes=genes, **{f"pred_{k}": v for k, v in region.items()})


if __name__ == "__main__":
    main()
