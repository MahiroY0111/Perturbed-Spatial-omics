"""Compare predicted and observed changes by liver lobule zone. In every section the spots are split into three zones
by the tertiles of a periportal score (mean log1p expression of Cyp2f2, Hal, Sds, Pck1, Hsd17b13 and Aldh1b1).
Observed change: zone mean of each injured section minus the zone mean of the healthy section. Predicted change: zone
mean of (prediction - the method's prediction without transport) on the healthy section. Writes the metrics of every
method (.csv) and the zone-level changes (.npz).

  python realdata/liver/zone_eval.py --scenario results/liver_apap/scenario --model-dir model --seed 1 \
      --sections-dir data/liver_apap/sections --ctrl-section hea02 --stim-sections pod24A pod24B \
      --out results/liver_apap/scenario/zone_eval_model.csv
"""
from __future__ import annotations

import argparse
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

TARGET_SUM = 1e4
PP = ["Cyp2f2", "Hal", "Sds", "Pck1", "Hsd17b13", "Aldh1b1"]
PC = ["Cyp2e1", "Glul", "Cyp1a2", "Slc1a2", "Lect2", "Oat"]
INJURY = ["Hmox1", "Mt1", "Mt2", "Anxa2", "Cd44", "Vim", "Tnfrsf12a", "Hspa1a", "Hspa1b", "Lcn2", "Saa1", "Saa2", "Fos", "Jun", "Atf3", "Gadd45b", "Cdkn1a", "Krt8", "Krt18", "Lgals3", "S100a6", "Spp1"]


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


def zones(y, genes, n=3):
    gi = [int(np.flatnonzero(genes == g)[0]) for g in PP if g in set(genes)]
    score = y[:, gi].mean(axis=1)
    q = np.quantile(score, np.linspace(0, 1, n + 1)[1:-1])
    z = np.digitize(score, q)  # 0 = lowest PP score = pericentral/necrotic ... n-1 = periportal
    return z, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=pathlib.Path, required=True)
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--name", default="liver")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sections-dir", type=pathlib.Path, required=True)
    ap.add_argument("--ctrl-section", required=True)
    ap.add_argument("--stim-sections", nargs="+", required=True)
    ap.add_argument("--n-zones", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    D = a.scenario; S, N = a.name, a.seed
    st = ad.read_h5ad(D / "st_ctrl_raw.h5ad")
    genes, spots = st.var_names.astype(str).to_numpy(), st.obs_names.astype(str).to_numpy()
    y_ctrl = normlog(dense(st.X))
    z_ctrl, _ = zones(y_ctrl, genes, a.n_zones)

    def load_section(name):
        x = ad.read_h5ad(a.sections_dir / f"{name}.h5ad"); x.var_names_make_unique()
        lower = {g.lower(): g for g in x.var_names.astype(str)}
        x = x[:, [lower[g.lower()] for g in genes]].copy()
        return normlog(dense(x.X))
    obs_means = {}
    for sec in a.stim_sections:
        y = load_section(sec); z, _ = zones(y, genes, a.n_zones)
        obs_means[sec] = np.stack([y[z == k].mean(axis=0) for k in range(a.n_zones)])
    ctrl_means = np.stack([y_ctrl[z_ctrl == k].mean(axis=0) for k in range(a.n_zones)])
    d_obs = {sec: m - ctrl_means for sec, m in obs_means.items()}
    d_obs_mean = np.mean([d_obs[s] for s in a.stim_sections], axis=0)

    P = D / a.model_dir / "predictions"
    def load(path):
        p = ad.read_h5ad(path); assert (p.var_names.astype(str) == genes).all() and (p.obs_names.astype(str) == spots).all(); return dense(p.X)
    sc_c, sc_s = ad.read_h5ad(D / "sc_ctrl_raw.h5ad"), ad.read_h5ad(D / "sc_stim_raw.h5ad")
    mean_c, mean_s = dense(sc_c.X).mean(axis=0), dense(sc_s.X).mean(axis=0)
    ratio = np.divide(mean_s, mean_c, out=np.ones_like(mean_s), where=mean_c > 0)
    sn_lfc = np.log1p(mean_s / mean_s.sum() * TARGET_SUM) - np.log1p(mean_c / mean_c.sum() * TARGET_SUM)
    y_raw = dense(st.X)
    changes = {"mean_ratio": normlog(y_raw * ratio) - y_ctrl}
    noT = sorted(P.glob(f"{S}_proposed_noT_raw_G*_modelseed{N}.h5ad"))
    base = normlog(load(noT[0] if noT else sorted(P.glob(f"{S}_proposed_control_raw_G*_modelseed{N}.h5ad"))[0]))
    changes["proposed"] = normlog(load(sorted(P.glob(f"{S}_proposed_stim_raw_G*_modelseed{N}.h5ad"))[0])) - base
    for path in sorted(P.glob(f"{S}_simo_cellot_normlog_G*_modelseed{N}__*.h5ad")):
        label = path.stem.split("__", 1)[1]
        changes[f"simo[{label}]"] = load(path) - load(sorted(P.glob(f"{S}_simo_noT_normlog_G*_modelseed{N}__{label}.h5ad"))[0])
    for path in sorted(P.glob(f"{S}_simo_cellot_cal_normlog_G*_modelseed{N}__*.h5ad")):  # 事後の遺伝子別スケール補正つき
        label = path.stem.split("__", 1)[1]
        changes[f"simo_cal[{label}]"] = load(path) - load(sorted(P.glob(f"{S}_simo_noT_cal_normlog_G*_modelseed{N}__{label}.h5ad"))[0])
    pred = {k: np.stack([v[z_ctrl == kz].mean(axis=0) for kz in range(a.n_zones)]) for k, v in changes.items()}
    pred["uniform"] = np.tile(sn_lfc, (a.n_zones, 1))

    expressed = (y_ctrl.mean(axis=0) > 0.05) | (np.mean([obs_means[s] for s in a.stim_sections], axis=0).mean(axis=0) > 0.05)
    strong = (np.abs(d_obs_mean) > 0.5) & expressed[None, :]
    contrast_obs = d_obs_mean[0] - d_obs_mean[-1]              # PC − PP
    zone_dep = np.flatnonzero(expressed & (np.abs(d_obs_mean).max(axis=0) > 0.5) & (np.abs(contrast_obs) > 0.5))
    # reproducibility of the zone contrast between the two 24 h sections
    rep = np.flatnonzero([(np.sign(d_obs[a.stim_sections[0]][0, g] - d_obs[a.stim_sections[0]][-1, g]) == np.sign(d_obs[a.stim_sections[1]][0, g] - d_obs[a.stim_sections[1]][-1, g])) for g in range(len(genes))]) if len(a.stim_sections) > 1 else np.arange(len(genes))
    zone_dep_rep = np.intersect1d(zone_dep, rep)
    inj = [g for g in INJURY if g in set(genes)]
    gi = [int(np.flatnonzero(genes == g)[0]) for g in inj]
    rows = []
    for k, pr in pred.items():
        cp = pr[0] - pr[-1]
        rows.append({"method": k,
                     "corr_all_pairs": corr(pr[:, expressed], d_obs_mean[:, expressed]),
                     "corr_strong_pairs": corr(pr[strong], d_obs_mean[strong]),
                     "sign_agreement_strong": float(np.mean(np.sign(pr[strong]) == np.sign(d_obs_mean[strong]))),
                     "contrast_corr_zone_dep": corr(cp[zone_dep], contrast_obs[zone_dep]),
                     "contrast_corr_zone_dep_reproducible": corr(cp[zone_dep_rep], contrast_obs[zone_dep_rep]),
                     "contrast_sign_agreement": float(np.mean(np.sign(cp[zone_dep_rep]) == np.sign(contrast_obs[zone_dep_rep]))) if len(zone_dep_rep) else float("nan"),
                     "injury_contrast_corr": corr(cp[gi], contrast_obs[gi]),
                     "n_zone_dep_genes": int(len(zone_dep)), "n_zone_dep_reproducible": int(len(zone_dep_rep)), "n_strong_pairs": int(strong.sum())})
    df = pd.DataFrame(rows); a.out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print("zone sizes (ctrl):", np.bincount(z_ctrl).tolist(), "| zone 0 = pericentral/necrotic, zone", a.n_zones - 1, "= periportal")
    print(df.to_string(index=False))
    tab = pd.DataFrame({"obs_PC": d_obs_mean[0, gi], "obs_mid": d_obs_mean[1, gi] if a.n_zones == 3 else np.nan, "obs_PP": d_obs_mean[-1, gi],
                        **{f"{k}_PC": pred[k][0, gi] for k in ["proposed", "mean_ratio"] + [m for m in pred if m.startswith("simo[norepeat")]},
                        **{f"{k}_PP": pred[k][-1, gi] for k in ["proposed", "mean_ratio"] + [m for m in pred if m.startswith("simo[norepeat")]}}, index=inj)
    print("\ninjury genes: observed and predicted change per zone (log1p)"); print(tab.round(2).to_string())
    np.savez_compressed(a.out.with_suffix(".npz"), d_obs=d_obs_mean, genes=genes, zones_ctrl=z_ctrl, **{f"pred_{k}": v for k, v in pred.items()})


if __name__ == "__main__":
    main()
