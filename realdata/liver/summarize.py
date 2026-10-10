"""Collect the numbers of the liver check from the zone evaluations: correlations and sign agreement of every method
(standard settings, lambda_mix_gene = 1.0, clustering seeds 2 and 3, the other healthy section), the injury-related
genes, the gene-category RMSE and, with --figure, the figure (results/liver_apap/figures/).

  python realdata/liver/summarize.py --ws results/liver_apap --data-root data/liver_apap [--figure]
"""
import argparse, json, pathlib
import numpy as np, pandas as pd

INJURY = ["Anxa2", "Hmox1", "Krt8", "Vim", "Lgals3", "Tnfrsf12a", "Hspa1b", "Cd44"]
ZONE = {"PC": 0, "mid": 1, "PP": 2}   # liver_zone_eval.py の行順（門脈周囲スコアの 3 分位：低い方から PC, mid, PP）


def table(csv):
    df = pd.read_csv(csv).set_index("method")
    cols = ["corr_all_pairs", "corr_strong_pairs", "sign_agreement_strong", "contrast_corr_zone_dep", "contrast_sign_agreement", "injury_contrast_corr", "n_zone_dep_genes", "n_strong_pairs"]
    out = {}
    for m, r in df.iterrows():
        out[m] = {c: (None if pd.isna(r[c]) else float(r[c])) for c in cols}
    return out


def injury(npz):
    z = np.load(npz, allow_pickle=True); genes = list(z["genes"].astype(str))
    out = {}
    for g in INJURY:
        i = genes.index(g)
        out[g] = {"obs_PC": float(z["d_obs"][ZONE["PC"], i]), "obs_PP": float(z["d_obs"][ZONE["PP"], i]),
                  "pred_PC": float(z["pred_proposed"][ZONE["PC"], i]), "pred_PP": float(z["pred_proposed"][ZONE["PP"], i]),
                  "simo_PC": float(z["pred_simo[norepeat]"][ZONE["PC"], i]), "simo_PP": float(z["pred_simo[norepeat]"][ZONE["PP"], i]),
                  "mr_PC": float(z["pred_mean_ratio"][ZONE["PC"], i]), "mr_PP": float(z["pred_mean_ratio"][ZONE["PP"], i])}
    return out


def category_rmse(npz, scen, thr=0.3):
    """遺伝子種類別の誤差。遺伝子ごとに 3 帯にわたる RMSE を求め，種類ごとに遺伝子平均する。
    対象遺伝子：ST で発現がある（健常切片の log1p 正規化発現の平均 > 0.05，または帯の観測変化の最大絶対値 > 0.5）かつ 帯の観測変化の最大絶対値 > 0.3。
    capture_low  = 健常の ST と snRNA-seq の CPM の比 log2((CPM_ST+1)/(CPM_sn+1)) が −1 未満（ST での捕捉が snRNA-seq より大きく低い）
    decreasing / increasing = 参照 snRNA-seq の障害後／健常の CPM 比 log2((CPM_stim+1)/(CPM_ctrl+1)) の符号（障害で減少／増加する遺伝子）"""
    import anndata as ad, scipy.sparse as sp
    dense = lambda x: np.asarray(x.toarray() if sp.issparse(x) else x, float)
    z = np.load(npz, allow_pickle=True); genes = np.asarray(z["genes"]).astype(str); d_obs = z["d_obs"]
    Xst = dense(ad.read_h5ad(scen / "st_ctrl_raw.h5ad").X); Xc = dense(ad.read_h5ad(scen / "sc_ctrl_raw.h5ad").X); Xs = dense(ad.read_h5ad(scen / "sc_stim_raw.h5ad").X)
    cpm = lambda X: X.sum(0) / X.sum() * 1e6
    y = np.log1p(Xst / Xst.sum(1, keepdims=True) * 1e4)
    expressed = (y.mean(0) > 0.05) | (np.abs(d_obs).max(0) > 0.5)
    modality = np.log2((cpm(Xst) + 1) / (cpm(Xc) + 1)); ref_change = np.log2((cpm(Xs) + 1) / (cpm(Xc) + 1))
    sel = expressed & (np.abs(d_obs).max(0) > thr)
    meths = ["proposed", "uniform", "mean_ratio", "simo[norepeat]", "simo_cal[norepeat]"]
    err = {m: np.sqrt(((z[f"pred_{m}"] - d_obs) ** 2).mean(0)) for m in meths}   # 遺伝子ごと（3 帯）の RMSE
    rows = []
    for cname, cm in [("all", np.ones_like(sel)), ("capture_low", modality < -1), ("decreasing", ref_change < 0), ("increasing", ref_change > 0)]:
        s = sel & cm; rows.append({"category": cname, "n_genes": int(s.sum()), **{m: float(err[m][s].mean()) for m in meths}})
    return pd.DataFrame(rows)


def figure(ws, scen):
    """肝の検証の図：(a) 健常切片の門脈周囲スコア，(b) 24 h 切片の障害遺伝子スコア（観測），(c) 健常切片に対する予測変化（λ_mix=1.0），(d) 帯間差（原稿では a–c を使用）"""
    import anndata as ad, scipy.sparse as sp, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dense = lambda x: np.asarray(x.toarray() if sp.issparse(x) else x, float)
    normlog = lambda X: np.log1p(X / X.sum(1, keepdims=True) * 1e4)
    st = ad.read_h5ad(scen / "st_ctrl_raw.h5ad"); genes = st.var_names.astype(str).to_numpy(); xy = np.asarray(st.obsm["spatial"], float); y0 = normlog(dense(st.X))
    gset = set(genes); ppi = [int(np.flatnonzero(genes == g)[0]) for g in ["Cyp2f2", "Hal", "Sds", "Pck1", "Hsd17b13", "Aldh1b1"] if g in gset]
    x = ad.read_h5ad(DATA_ROOT / "sections/pod24A.h5ad"); x.var_names_make_unique(); low = {g.lower(): g for g in x.var_names.astype(str)}
    x = x[:, [low[g.lower()] for g in genes]].copy(); y24, xy24 = normlog(dense(x.X)), np.asarray(x.obsm["spatial"], float)
    zone = np.load(scen / "zone_eval_model_mix1.npz", allow_pickle=True); d_obs = zone["d_obs"]
    P = scen / "model_mix1/predictions"
    noT = normlog(dense(ad.read_h5ad(P / "liver_proposed_noT_raw_G8000_modelseed1.h5ad").X)); stim = normlog(dense(ad.read_h5ad(P / "liver_proposed_stim_raw_G8000_modelseed1.h5ad").X))
    gi = [int(np.flatnonzero(genes == g)[0]) for g in INJURY]
    score24 = y24[:, gi].mean(1); dpred = (stim - noT)[:, gi].mean(1)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4)); s = 9
    a = ax[0]; sc_ = a.scatter(xy[:, 0], -xy[:, 1], c=y0[:, ppi].mean(1), s=s, cmap="coolwarm"); a.set_title("(a) healthy section: periportal score\n(zones = tertiles; red = periportal)"); plt.colorbar(sc_, ax=a, fraction=0.04)
    a = ax[1]; sc_ = a.scatter(xy24[:, 0], -xy24[:, 1], c=score24, s=s, cmap="viridis", vmin=np.percentile(score24, 1), vmax=np.percentile(score24, 99)); a.set_title("(b) 24 h section: injury-gene score (observed)"); plt.colorbar(sc_, ax=a, fraction=0.04)
    a = ax[2]; sc_ = a.scatter(xy[:, 0], -xy[:, 1], c=dpred, s=s, cmap="magma", vmin=np.percentile(dpred, 1), vmax=np.percentile(dpred, 99)); a.set_title("(c) predicted change of the injury-gene score\non the healthy section"); plt.colorbar(sc_, ax=a, fraction=0.04)
    for a in ax[:3]: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a = ax[3]; xx = np.arange(len(INJURY)); pc, pp = ZONE["PC"], ZONE["PP"]
    a.bar(xx - 0.3, d_obs[pc, gi] - d_obs[pp, gi], 0.28, label="observed: PC $-$ PP"); a.bar(xx, zone["pred_proposed"][pc, gi] - zone["pred_proposed"][pp, gi], 0.28, label="proposed: PC $-$ PP")
    a.bar(xx + 0.3, zone["pred_mean_ratio"][pc, gi] - zone["pred_mean_ratio"][pp, gi], 0.28, label="Mean-ratio: PC $-$ PP")
    a.axhline(0, color="k", lw=0.5); a.set_xticks(xx); a.set_xticklabels(INJURY, rotation=45, fontsize=8); a.set_ylabel(r"zone contrast of $\Delta$ (log1p)"); a.legend(fontsize=7); a.set_title("(d) pericentral $-$ periportal contrast")
    plt.tight_layout(); plt.savefig(ws / "figures/liver_apap.png", dpi=150); print("figure saved", ws / "figures/liver_apap.png")


DATA_ROOT = pathlib.Path("data/liver_apap")


def main():
    global DATA_ROOT
    ap = argparse.ArgumentParser(); ap.add_argument("--ws", type=pathlib.Path, default=pathlib.Path("results/liver_apap")); ap.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data/liver_apap")); ap.add_argument("--figure", action="store_true", help="図（figures/liver_apap.png）も描く"); a = ap.parse_args()
    DATA_ROOT = a.data_root
    S = a.ws / "scenario"; H = a.ws / "scenario_hea03"
    z = np.load(S / "zone_eval_model.npz", allow_pickle=True)
    out = {"zone_sizes": [int(x) for x in np.bincount(z["zones_ctrl"])],
           "model": table(S / "zone_eval_model.csv"), "model_injury": injury(S / "zone_eval_model.npz"),
           "model_mix1": table(S / "zone_eval_model_mix1.csv"), "model_mix1_injury": injury(S / "zone_eval_model_mix1.npz"),
           "replicates_corr_all": {}}
    for k, f in [("seed2", S / "zone_eval_model_seed2.csv"), ("seed3", S / "zone_eval_model_seed3.csv"), ("hea03", H / "zone_eval_model.csv"), ("hea03_mix1", H / "zone_eval_model_mix1.csv")]:
        if f.exists():
            out["replicates_corr_all"][k] = {m: v["corr_all_pairs"] for m, v in table(f).items() if m in ("mean_ratio", "proposed", "simo[norepeat]", "simo_cal[norepeat]", "uniform")}
    (a.ws / "figures").mkdir(exist_ok=True)
    json.dump(out, open(a.ws / "figures/liver_numbers.json", "w"), indent=1)
    print("written", a.ws / "figures/liver_numbers.json")
    if (S / "st_ctrl_raw.h5ad").exists():
        df = category_rmse(S / "zone_eval_model.npz", S)
        df.to_csv(a.ws / "figures/liver_category_rmse.csv", index=False)
        pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format); print(df.to_string(index=False)); print("written liver_category_rmse.csv")
        if a.figure:
            figure(a.ws, S)
    else:
        print("scenario files (st_ctrl_raw.h5ad etc.) not found: category RMSE and figure skipped; shipped values are in figures/")


if __name__ == "__main__":
    main()
