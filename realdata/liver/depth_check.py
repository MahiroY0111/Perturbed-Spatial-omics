"""Relation between the predicted spatial pattern and spot depth (total counts) in the liver check, and the zone contrast
of the injury-related genes within depth tertiles.

  python realdata/liver/depth_check.py --ws results/liver_apap
"""
import argparse, pathlib
import numpy as np

INJURY = ["Anxa2", "Hmox1", "Krt8", "Vim", "Lgals3", "Tnfrsf12a", "Hspa1b", "Cd44"]
PP_MARKERS = ["Cyp2f2", "Hal", "Sds", "Pck1", "Hsd17b13", "Aldh1b1"]


def main():
    import anndata as ad, scipy.sparse as sp
    ap = argparse.ArgumentParser(); ap.add_argument("--ws", type=pathlib.Path, default=pathlib.Path("results/liver_apap")); a = ap.parse_args()
    D = a.ws / "scenario"
    dense = lambda x: np.asarray(x.toarray() if sp.issparse(x) else x, float); normlog = lambda X: np.log1p(X / X.sum(1, keepdims=True) * 1e4)
    st = ad.read_h5ad(D / "st_ctrl_raw.h5ad"); genes = st.var_names.astype(str).to_numpy(); X = dense(st.X); y0 = normlog(X); lib = X.sum(1)
    gset = set(genes); pp = y0[:, [int(np.flatnonzero(genes == g)[0]) for g in PP_MARKERS if g in gset]].mean(1)
    gi = [int(np.flatnonzero(genes == g)[0]) for g in INJURY]
    for tag in ["model", "model_mix1"]:
        P = D / tag / "predictions"
        noT = normlog(dense(ad.read_h5ad(P / "liver_proposed_noT_raw_G8000_modelseed1.h5ad").X)); stim = normlog(dense(ad.read_h5ad(P / "liver_proposed_stim_raw_G8000_modelseed1.h5ad").X))
        d = (stim - noT)[:, gi]; s = d.mean(1)
        print(f"== {tag}: corr(predicted injury score, log total count) = {np.corrcoef(s, np.log(lib))[0, 1]:+.3f}")
        lq = np.quantile(lib, [1 / 3, 2 / 3])
        for k, (lo, hi) in enumerate([(0, lq[0]), (lq[0], lq[1]), (lq[1], np.inf)]):
            m = (lib >= lo) & (lib < hi); q = np.quantile(pp[m], [1 / 3, 2 / 3]); pcz = m & (pp < q[0]); ppz = m & (pp >= q[1])
            c = d[pcz].mean(0) - d[ppz].mean(0)
            print(f"   depth tertile {k} (n={m.sum()}, median count {int(np.median(lib[m]))}): PC-PP contrast positive for {int((c > 0).sum())}/8 injury genes (mean {c.mean():+.3f})")
        Xd = np.column_stack([np.ones_like(lib), np.log(lib)]); r1 = s - Xd @ np.linalg.lstsq(Xd, s, rcond=None)[0]; r2 = pp - Xd @ np.linalg.lstsq(Xd, pp, rcond=None)[0]
        print(f"   partial corr(predicted injury score, periportal score | log total count) = {np.corrcoef(r1, r2)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
