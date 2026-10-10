"""Build the inputs of the liver injury check from GEO: reference snRNA-seq (GSE223558; healthy nuclei as control,
the chosen time points after acetaminophen overdose as stimulated, up to 20,000 nuclei each), one healthy Visium
section (GSE223560) as control ST, and every Visium section as .h5ad under <root>/sections for the evaluation.
The inputs use the same file layout as the simulated conditions.

  python realdata/liver/make_scenario.py --root data/liver_apap --stim 018h 024h 030h --min-umi 2000 \
      --min-genes 500 --ctrl-section hea02 --out results/liver_apap/scenario
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

N_GENES, N_CELLS = 8000, 20000
SN = {"hea001": "GSM6962654", "hea002": "GSM6962655", "003h": "GSM6962656", "006h": "GSM6962657", "012h": "GSM6962658", "018h": "GSM6962659",
      "024h": "GSM6962661", "030h": "GSM6962662", "036h": "GSM6962663", "042h": "GSM6962664", "048h": "GSM6962665", "072h": "GSM6962666"}
ST = {"hea02": "GSM6963514", "hea03": "GSM6963515", "hea04": "GSM6963517", "pod24A": "GSM6963518", "pod24B": "GSM6963519",
      "pod36A": "GSM6963520", "pod36B": "GSM6963521", "pod48A": "GSM6963523", "pod48B": "GSM6963524", "pod48C": "GSM6963525", "pod48D": "GSM6963526"}


def read_mtx(folder: pathlib.Path, gsm: str) -> ad.AnnData:
    m = next(folder.glob(f"{gsm}_*matrix.mtx.gz")); stem = m.name[: -len("matrix.mtx.gz")]
    X = scipy.io.mmread(m).T.tocsr()
    bc = pd.read_csv(folder / f"{stem}barcodes.tsv.gz", header=None)[0].astype(str)
    ft = pd.read_csv(folder / f"{stem}features.tsv.gz", header=None, sep="\t")
    names = ft[1].astype(str) if ft.shape[1] > 1 else ft[0].astype(str)
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=bc.to_numpy()), var=pd.DataFrame(index=names.to_numpy()))
    a.var_names_make_unique()
    return a, stem


def read_visium(folder: pathlib.Path, gsm: str) -> ad.AnnData:
    a, stem = read_mtx(folder, gsm)
    pos_file = next(folder.glob(f"{gsm}_*tissue_positions_list.csv.gz"))  # the image-derived files can carry a slightly different stem
    pos = pd.read_csv(pos_file, header=None,
                      names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]).set_index("barcode")
    pos = pos.reindex(a.obs_names)
    a = a[pos["in_tissue"].fillna(0).to_numpy() == 1].copy(); pos = pos.loc[a.obs_names]
    a.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    a.obs["array_row"], a.obs["array_col"] = pos["array_row"].to_numpy(), pos["array_col"].to_numpy()
    return a


def write_counts(path, x, obs_names, genes, obs: dict, spatial=None) -> None:
    a = ad.AnnData(X=np.asarray(x.toarray() if sp.issparse(x) else x).astype(np.uint32),
                   obs=pd.DataFrame(obs, index=pd.Index(np.asarray(obs_names, dtype=str), name="id")),
                   var=pd.DataFrame(index=pd.Index(np.asarray(genes, dtype=str), name="gene_id")))
    a.uns["matrix_type"] = "raw_integer_counts"
    if spatial is not None:
        a.obsm["spatial"] = np.asarray(spatial, dtype=np.float64)
    a.write_h5ad(path, compression="gzip", compression_opts=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--stim", nargs="+", default=["024h"], help="time points pooled as stimulated")
    ap.add_argument("--min-umi", type=int, default=500)
    ap.add_argument("--min-genes", type=int, default=300)
    ap.add_argument("--ctrl-section", default="hea02")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if (a.out / "scenario_config.json").exists():
        print("already done:", a.out); return
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    dl = a.root / "download"
    # ---- 切片を h5ad に（評価用；初回だけ）
    secdir = a.root / "sections"; secdir.mkdir(exist_ok=True)
    for name, gsm in ST.items():
        if not (secdir / f"{name}.h5ad").exists():
            x = read_visium(dl / "st", gsm); x.write_h5ad(secdir / f"{name}.h5ad"); print("section", name, x.shape, "median UMI", int(np.median(np.asarray(x.X.sum(1)).ravel())))
    # ---- 参照核
    ctrl = ad.concat([read_mtx(dl / "sn", SN["hea001"])[0], read_mtx(dl / "sn", SN["hea002"])[0]], join="inner", index_unique="-")
    stim = ad.concat([read_mtx(dl / "sn", SN[t])[0] for t in a.stim], join="inner", index_unique="-") if len(a.stim) > 1 else read_mtx(dl / "sn", SN[a.stim[0]])[0]
    ctrl.obs["condition"], stim.obs["condition"] = "ctrl", "stim"
    rna = ad.concat([ctrl, stim], join="inner", index_unique=None)
    tot = np.asarray(rna.X.sum(axis=1)).ravel(); ng = np.asarray((rna.X > 0).sum(axis=1)).ravel()
    rna.var["mt"] = rna.var_names.str.lower().str.startswith("mt-")
    mt = np.asarray(rna[:, rna.var["mt"]].X.sum(axis=1)).ravel() / np.maximum(tot, 1)
    n_before = int(rna.n_obs)
    rna = rna[(tot >= a.min_umi) & (ng >= a.min_genes) & (mt <= 0.05)].copy()
    st = ad.read_h5ad(secdir / f"{a.ctrl_section}.h5ad")
    # ---- 遺伝子
    shared = [g for g in rna.var_names if g in set(st.var_names) and not g.lower().startswith("mt-")]
    rna, st = rna[:, shared].copy(), st[:, shared].copy()
    st_total = np.asarray(st.X.sum(axis=0)).ravel(); rna_detect = np.asarray((rna.X > 0).sum(axis=0)).ravel()
    expressed = (st_total >= 0.05 * st.n_obs) & (rna_detect >= 0.005 * rna.n_obs)
    rna, st = rna[:, expressed].copy(), st[:, expressed].copy()
    work = rna.copy()
    sc.pp.highly_variable_genes(work, n_top_genes=min(N_GENES, rna.n_vars), flavor="seurat_v3", batch_key="condition")
    genes = np.sort(work.var_names[work.var["highly_variable"]].to_numpy())
    rna, st = rna[:, genes].copy(), st[:, genes].copy()
    picked = {}
    for cond in ("ctrl", "stim"):
        idx = np.flatnonzero((rna.obs["condition"] == cond).to_numpy())
        picked[cond] = np.sort(rng.choice(idx, size=min(N_CELLS, len(idx)), replace=False))
        sub = rna[picked[cond]]
        write_counts(a.out / f"sc_{cond}_raw.h5ad", sub.X, [f"{cond}_{b}" for b in sub.obs_names], genes, {"condition": [cond] * sub.n_obs})
        pd.DataFrame({"cell_id": [f"{cond}_{b}" for b in sub.obs_names]}).to_csv(a.out / f"cell_types_{cond}.csv", index=False)
    write_counts(a.out / "st_ctrl_raw.h5ad", st.X, st.obs_names, genes, {"condition": ["ctrl"] * st.n_obs}, spatial=np.asarray(st.obsm["spatial"], float))
    config = {"dataset": "GSE223558 snRNA-seq + GSE223560 Visium (Matchett 2024), mouse APAP liver injury", "stim": a.stim, "min_umi": a.min_umi, "min_genes": a.min_genes, "ctrl_section": a.ctrl_section,
              "n_genes": int(len(genes)), "n_spots_ctrl": int(st.n_obs), "n_ctrl_nuclei": int(len(picked["ctrl"])), "n_stim_nuclei": int(len(picked["stim"])),
              "nuclei_before_qc": n_before, "nuclei_after_qc": {c: int((rna.obs["condition"] == c).sum()) for c in ("ctrl", "stim")}, "seed": int(a.seed), "median_umi_per_spot_selected_genes": float(np.median(np.asarray(st.X.sum(axis=1)).ravel())),
              "zonation_markers_included": {g: bool(g in set(genes)) for g in ["Cyp2e1", "Glul", "Oat", "Cyp1a2", "Cyp2f2", "Hal", "Sds", "Pck1", "Hmox1", "Anxa2", "Cd44", "Vim", "Tnfrsf12a", "Gadd45a", "Mki67", "Top2a"]}}
    (a.out / "scenario_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
