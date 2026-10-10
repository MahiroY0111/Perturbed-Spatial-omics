"""Collect the numbers of the kidney check: Havcr1 induction by region (observed and predicted), its attribution to
proximal tubule S1/S2/S3 pseudo-cells per unit composition, the rank correlation across regions, the four anatomical
zones defined by known markers on each section, the gene-category RMSE and, with --figure, the figure
(results/kidney_iri/figures/).

  python realdata/kidney/summarize.py --ws results/kidney_iri --data-root data/kidney_iri [--figure]
"""
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ORDER = ["c3", "c0", "c4", "c5", "c7", "c6", "c1", "c2"]   # 領域の順序（観測された Havcr1 誘導の大きい順）
ZONES = {"cortex_S1S2": ["Slc5a2", "Slc34a1", "Slc13a3", "Cyp4b1"], "OSOM_S3": ["Slc7a13", "Cyp7b1", "Acsm3", "Slc22a7"],
         "ISOM_TAL": ["Slc12a1", "Umod", "Kcnj1"], "inner_medulla": ["Aqp2", "Slc14a2", "Aqp1", "Fxyd4"]}
RUNS = [("model_mix1", 1, "mix1"), ("model", 1, "frozen_s1"), ("model", 2, "frozen_s2")]


def dense(x):
    import scipy.sparse as sp
    return np.asarray(x.toarray() if sp.issparse(x) else x, float)


def normlog(X):
    return np.log1p(X / X.sum(1, keepdims=True) * 1e4)


def load(ws):
    import anndata as ad
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from make_scenario import read_visium
    D = ws / "scenario_12h"
    st = ad.read_h5ad(D / "st_ctrl_raw.h5ad"); genes = st.var_names.astype(str).to_numpy(); spots = st.obs_names.astype(str).to_numpy()
    h12 = read_visium(DATA_ROOT / "visium/hour12")[:, genes].copy(); sh = read_visium(DATA_ROOT / "visium/sham")[:, genes].copy()
    return D, st, genes, spots, h12, sh


def havcr1(ws, D, st, genes, spots, h12, sh):
    import anndata as ad
    gi = int(np.flatnonzero(genes == "Havcr1")[0])
    reg = pd.read_csv(D / "regions_spots.csv", index_col=0); rs = reg[reg.section == "sham"].set_index("barcode")["cluster"].astype(str).reindex(spots).to_numpy()
    ev = np.load(D / "eval.npz", allow_pickle=True); cl = list(ev["clusters"].astype(str)); oi = int(np.flatnonzero(ev["genes"].astype(str) == "Havcr1")[0])
    order = [c for c in ORDER if c in cl]; rows = {}; dpred_mix1 = None
    sc_c = ad.read_h5ad(D / "sc_ctrl_raw.h5ad")
    for mdir, N, label in RUNS:
        P = D / mdir / "predictions"
        noT = normlog(dense(ad.read_h5ad(P / f"kidney_proposed_noT_raw_G8000_modelseed{N}.h5ad").X)); stim = normlog(dense(ad.read_h5ad(P / f"kidney_proposed_stim_raw_G8000_modelseed{N}.h5ad").X))
        d = stim[:, gi] - noT[:, gi]; rows[label] = {c: float(np.mean(d[rs == c[1:]])) for c in order}
        if label == "mix1": dpred_mix1 = d
        z = np.load(D / mdir / f"diagnostics/kidney_type_attribution_modelseed{N}.npz", allow_pickle=True)
        cT, c0, p, bg = z["contrib_T"].astype(np.float32), z["contrib_noT"].astype(np.float32), z["p_known"], z["bg"].astype(np.float32)
        libT = (cT.sum(1) + bg).sum(1); lib0 = (c0.sum(1) + bg).sum(1)
        ann = sc_c.obs["celltype"].astype(str).reindex(z["ctrl_cell_id"].astype(str)).to_numpy()
        tab = pd.crosstab(z["trained_cell_type_index"], ann); lab = {t: tab.loc[t].idxmax() for t in tab.index}
        i = int(np.flatnonzero(z["genes"].astype(str) == "Havcr1")[0])
        rows[label + "_log2fc"], rows[label + "_permass"], rows[label + "_p"] = {}, {}, {}
        for c in order:
            m = rs == c[1:]
            rows[label + "_log2fc"][c] = float(np.log2((cT[m, :, i].sum() + bg[m, i].sum()) / libT[m].sum() * 1e6 + 1) - np.log2((c0[m, :, i].sum() + bg[m, i].sum()) / lib0[m].sum() * 1e6 + 1))
            pm_, per = {}, {}
            for name in ["PTS1", "PTS2", "PTS3"]:
                ts = [t for t in lab if lab[t] == name]
                if not ts: continue
                t = max(ts, key=lambda t: p[:, t].mean()); pm = p[m, t]; ok = pm > 0.02
                per[name] = float((((cT[m, t, i][ok] / libT[m][ok]) - (c0[m, t, i][ok] / lib0[m][ok])) * 1e6 / pm[ok]).mean()) if ok.sum() >= 10 else None
                pm_[name] = float(pm.mean())
            rows[label + "_permass"][c] = per; rows[label + "_p"][c] = pm_
    rows["observed"] = {c: float(ev["observed"][cl.index(c), oi]) for c in order}
    rows["n_spots"] = {c: int((rs == c[1:]).sum()) for c in order}
    rows["spearman"] = {k: float(spearmanr([rows[k + "_log2fc"][c] for c in order], [rows["observed"][c] for c in order])[0]) for _, _, k in RUNS}
    return rows, order, dpred_mix1, gi


def anatomical_zones(ws, D, st, genes, h12):
    """既知マーカーだけで各切片に 4 帯を定め（摂動後のデータを帯の定義に使わない），Havcr1 の観測・予測変化を帯平均で比べる。"""
    import anndata as ad
    y0, y12 = normlog(dense(st.X)), normlog(dense(h12.X)); gset = set(genes)
    def assign(y):
        sc = {k: y[:, [int(np.flatnonzero(genes == g)[0]) for g in v if g in gset]].mean(1) for k, v in ZONES.items()}
        M = np.column_stack([(sc[k] - sc[k].mean()) / sc[k].std() for k in ZONES]); return np.array(list(ZONES))[M.argmax(1)]
    z0, z12 = assign(y0), assign(y12); gi = int(np.flatnonzero(genes == "Havcr1")[0])
    d_obs = np.stack([y12[z12 == k].mean(0) - y0[z0 == k].mean(0) for k in ZONES])
    out = {"zones": list(ZONES), "n_spots_sham": [int((z0 == k).sum()) for k in ZONES], "n_spots_12h": [int((z12 == k).sum()) for k in ZONES],
           "observed": [round(float(v), 2) for v in d_obs[:, gi]],
           "note": "zones from a priori markers on each section (cortex: Slc5a2/Slc34a1/Slc13a3/Cyp4b1; OSOM: Slc7a13/Cyp7b1/Acsm3/Slc22a7; ISOM: Slc12a1/Umod/Kcnj1; inner medulla: Aqp2/Slc14a2/Aqp1/Fxyd4)"}
    for mdir, N, label in RUNS:
        P = D / mdir / "predictions"
        noT = normlog(dense(ad.read_h5ad(P / f"kidney_proposed_noT_raw_G8000_modelseed{N}.h5ad").X)); stim = normlog(dense(ad.read_h5ad(P / f"kidney_proposed_stim_raw_G8000_modelseed{N}.h5ad").X))
        pz = np.stack([(stim - noT)[z0 == k].mean(0) for k in ZONES])
        out["pred_" + label] = [round(float(v), 2) for v in pz[:, gi]]; out["spearman_" + label] = float(spearmanr(pz[:, gi], d_obs[:, gi])[0])
    out["spearman"] = min(out[f"spearman_{l}"] for _, _, l in RUNS)
    return out


def category_rmse(D, thr=0.3):
    """遺伝子種類別 RMSE（realdata/liver/summarize.py の定義を 8 領域に適用；標準設定 seed 1 の eval.npz）。"""
    import anndata as ad
    z = np.load(D / "eval.npz", allow_pickle=True); genes = np.asarray(z["genes"]).astype(str); d_obs = z["observed"]
    Xst = dense(ad.read_h5ad(D / "st_ctrl_raw.h5ad").X); Xc = dense(ad.read_h5ad(D / "sc_ctrl_raw.h5ad").X); Xs = dense(ad.read_h5ad(D / "sc_stim_raw.h5ad").X)
    cpm = lambda X: X.sum(0) / X.sum() * 1e6; y = normlog(Xst)
    expressed = (y.mean(0) > 0.05) | (np.abs(d_obs).max(0) > 0.5)
    modality = np.log2((cpm(Xst) + 1) / (cpm(Xc) + 1)); ref_change = np.log2((cpm(Xs) + 1) / (cpm(Xc) + 1))
    sel = expressed & (np.abs(d_obs).max(0) > thr)
    meths = {"proposed": "pred_proposed", "uniform": "pred_snRNA_pseudobulk_uniform", "mean_ratio": "pred_mean_ratio", "simo[norepeat]": "pred_simo[norepeat]"}
    err = {m: np.sqrt(((z[k] - d_obs) ** 2).mean(0)) for m, k in meths.items()}
    rows = []
    for cname, cm in [("all", np.ones_like(sel)), ("capture_low", modality < -1), ("decreasing", ref_change < 0), ("increasing", ref_change > 0)]:
        s = sel & cm; rows.append({"category": cname, "n_genes": int(s.sum()), **{m: float(err[m][s].mean()) for m in meths}})
    return pd.DataFrame(rows)


def figure(ws, st, genes, h12, sh, rows, order, dpred, gi):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xy = np.asarray(st.obsm["spatial"], float); h12x = np.asarray(h12.obsm["spatial"], float)
    obs12 = normlog(dense(h12.X))[:, gi]; obs0 = normlog(dense(sh.X))[:, gi]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4)); s = 6
    a = ax[0]; sc_ = a.scatter(xy[:, 0], -xy[:, 1], c=obs0, s=s, cmap="viridis", vmin=0, vmax=3); a.set_title("(a) sham section: observed Havcr1"); plt.colorbar(sc_, ax=a, fraction=0.04)
    a = ax[1]; sc_ = a.scatter(h12x[:, 0], -h12x[:, 1], c=obs12, s=s, cmap="viridis", vmin=0, vmax=3); a.set_title("(b) 12 h section: observed Havcr1"); plt.colorbar(sc_, ax=a, fraction=0.04)
    a = ax[2]; sc_ = a.scatter(xy[:, 0], -xy[:, 1], c=dpred, s=s, cmap="magma", vmin=0, vmax=np.percentile(dpred, 99)); a.set_title("(c) predicted change on the sham section"); plt.colorbar(sc_, ax=a, fraction=0.04)
    for a in ax[:3]: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a = ax[3]; x = np.arange(len(order)); ob = [rows["observed"][c] for c in order]; pr = [rows["mix1"][c] for c in order]
    a.bar(x - 0.2, ob, 0.4, label="observed (12 h $-$ sham)"); a.bar(x + 0.2, np.array(pr) * max(ob) / max(pr), 0.4, label="predicted (rescaled)")
    a.set_xticks(x); a.set_xticklabels([f"{c}\nS3 {rows['mix1_p'][c]['PTS3']:.2f}" for c in order], fontsize=8); a.set_ylabel(r"$\Delta$ Havcr1 (log1p, per $10^4$)"); a.legend(fontsize=8, loc="upper right"); a.set_title("(d) regions of the sham section")
    plt.tight_layout(); plt.savefig(ws / "figures/kidney_havcr1.png", dpi=150); print("figure saved", ws / "figures/kidney_havcr1.png")


DATA_ROOT = pathlib.Path("data/kidney_iri")


def main():
    global DATA_ROOT
    ap = argparse.ArgumentParser(); ap.add_argument("--ws", type=pathlib.Path, default=pathlib.Path("results/kidney_iri")); ap.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data/kidney_iri")); ap.add_argument("--figure", action="store_true"); a = ap.parse_args()
    DATA_ROOT = a.data_root
    D = a.ws / "scenario_12h"
    if not (D / "st_ctrl_raw.h5ad").exists() or not (D / "model/predictions").exists():
        ev = np.load(D / "eval.npz", allow_pickle=True); cl = list(ev["clusters"].astype(str)); oi = int(np.flatnonzero(ev["genes"].astype(str) == "Havcr1")[0])
        print("model outputs / scenario files not found: only the observed Havcr1 change by region is recomputed from eval.npz:",
              {c: round(float(ev["observed"][cl.index(c), oi]), 3) for c in ORDER if c in cl}, "; shipped values are in figures/havcr1_numbers.json")
        return
    D, st, genes, spots, h12, sh = load(a.ws)
    rows, order, dpred, gi = havcr1(a.ws, D, st, genes, spots, h12, sh)
    rows["anatomical_zones"] = anatomical_zones(a.ws, D, st, genes, h12)
    (a.ws / "figures").mkdir(exist_ok=True)
    json.dump(rows, open(a.ws / "figures/havcr1_numbers.json", "w"), indent=1)
    print("spearman (8 regions):", {k: round(v, 3) for k, v in rows["spearman"].items()}); print("anatomical zones:", {k: rows["anatomical_zones"][k] for k in ("observed", "pred_frozen_s1", "pred_frozen_s2", "pred_mix1", "spearman")})
    df = category_rmse(D); df.to_csv(a.ws / "figures/kidney_category_rmse.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format); print(df.to_string(index=False))
    if a.figure:
        figure(a.ws, st, genes, h12, sh, rows, order, dpred, gi)


if __name__ == "__main__":
    main()
