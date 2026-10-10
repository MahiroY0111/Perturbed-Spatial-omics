"""Build the inputs of the kidney injury check from GEO: reference snRNA-seq (GSE139107; control nuclei and one time
point after ischemia-reperfusion, up to 20,000 nuclei each) and the sham Visium section (GSE269622) as control ST; the
section of the same time point is used only for the evaluation. Genes are selected from the reference and the sham
section only. The inputs use the same file layout as the simulated conditions.

  python realdata/kidney/make_scenario.py --root data/kidney_iri --stim 12hours --out results/kidney_iri/scenario_12h
"""
from __future__ import annotations

import argparse
import json
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

N_GENES, N_CELLS = 8000, 20000


def read_dge(path: pathlib.Path) -> ad.AnnData:
    """GEO の DGE テキスト（行=遺伝子，列=核）を AnnData（核 x 遺伝子）にする。"""
    df = pd.read_csv(path, sep="\t", index_col=0)
    a = ad.AnnData(X=sp.csr_matrix(df.to_numpy(dtype=np.float32).T), obs=pd.DataFrame(index=df.columns.astype(str)),
                   var=pd.DataFrame(index=df.index.astype(str)))
    a.var_names_make_unique()
    return a


def read_visium(folder: pathlib.Path) -> ad.AnnData:
    h5 = next(folder.rglob("filtered_feature_bc_matrix.h5"))
    a = sc.read_10x_h5(h5)
    a.var_names_make_unique()
    pos_file = next(folder.rglob("tissue_positions*.csv"))
    pos = pd.read_csv(pos_file)
    if "barcode" not in pos.columns:  # Space Ranger 1.x: ヘッダなし
        pos = pd.read_csv(pos_file, header=None, names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"])
    pos = pos.set_index("barcode").loc[a.obs_names]
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
    ap.add_argument("--stim", default="12hours", help="snRNA-seq group used as stimulated (4hours, 12hours, 2days)")
    ap.add_argument("--stim-section", default="hour12", help="Visium section of the same time point (evaluation only)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if (a.out / "scenario_config.json").exists():
        print("already done:", a.out)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    dl = a.root / "download"
    meta = pd.read_csv(dl / "GSE139107_MouseIRI.metadata.txt.gz", sep="\t", index_col=0)
    ctrl = read_dge(dl / "GSE139107_MouseIRI_control.dge.txt.gz")
    stim = read_dge(dl / f"GSE139107_MouseIRI_{a.stim}.dge.txt.gz")
    for x, cond in ((ctrl, "ctrl"), (stim, "stim")):
        x.obs["condition"] = cond
        x.obs["celltype"] = meta.reindex(x.obs_names)["celltype"].astype(str).to_numpy()
        x.obs["group"] = meta.reindex(x.obs_names)["Group"].astype(str).to_numpy()
    rna = ad.concat([ctrl, stim], join="inner", index_unique=None)
    # 核の QC は全遺伝子の上で緩めに行う（公開データは既に QC 済み。パイプライン側では --skip-qc で再適用しない）
    total_all = np.asarray(rna.X.sum(axis=1)).ravel(); genes_all = np.asarray((rna.X > 0).sum(axis=1)).ravel()
    n_before_qc = int(rna.n_obs)
    rna = rna[(total_all >= 500) & (genes_all >= 300)].copy()
    st = read_visium(a.root / "visium" / "sham")
    st_stim = read_visium(a.root / "visium" / a.stim_section)

    # ---- 遺伝子：両方で測られている遺伝子のうち，参照と sham 切片で発現のあるもの．参照（両条件）の高変動遺伝子を 8,000
    shared = [g for g in rna.var_names if g in set(st.var_names) and g in set(st_stim.var_names) and not g.lower().startswith("mt-")]
    rna, st, st_stim = rna[:, shared].copy(), st[:, shared].copy(), st_stim[:, shared].copy()
    st_total = np.asarray(st.X.sum(axis=0)).ravel()
    rna_detect = np.asarray((rna.X > 0).sum(axis=0)).ravel()
    expressed = (st_total >= 0.05 * st.n_obs) & (rna_detect >= 0.005 * rna.n_obs)
    rna, st, st_stim = rna[:, expressed].copy(), st[:, expressed].copy(), st_stim[:, expressed].copy()
    n_hvg = min(N_GENES, rna.n_vars)
    work = rna.copy()
    sc.pp.highly_variable_genes(work, n_top_genes=n_hvg, flavor="seurat_v3", batch_key="condition")
    genes = np.sort(work.var_names[work.var["highly_variable"]].to_numpy())
    rna, st, st_stim = rna[:, genes].copy(), st[:, genes].copy(), st_stim[:, genes].copy()

    # ---- 核：各条件から最大 20,000 核（QC は上で済ませている）
    total = np.asarray(rna.X.sum(axis=1)).ravel()
    ok = np.ones(rna.n_obs, dtype=bool)
    picked = {}
    for cond in ("ctrl", "stim"):
        idx = np.flatnonzero(ok & (rna.obs["condition"] == cond).to_numpy())
        picked[cond] = np.sort(rng.choice(idx, size=min(N_CELLS, len(idx)), replace=False))
    for cond in ("ctrl", "stim"):
        sub = rna[picked[cond]]
        write_counts(a.out / f"sc_{cond}_raw.h5ad", sub.X, [f"{cond}_{b}" for b in sub.obs_names], genes,
                     {"condition": [cond] * sub.n_obs, "celltype": sub.obs["celltype"].to_numpy(), "group": sub.obs["group"].to_numpy()})
        pd.DataFrame({"cell_id": [f"{cond}_{b}" for b in sub.obs_names], "celltype": sub.obs["celltype"].to_numpy()}).to_csv(a.out / f"cell_types_{cond}.csv", index=False)
    coords = np.asarray(st.obsm["spatial"], dtype=np.float64)
    write_counts(a.out / "st_ctrl_raw.h5ad", st.X, st.obs_names, genes, {"condition": ["ctrl"] * st.n_obs}, spatial=coords)
    write_counts(a.out / "st_stim_observed_raw.h5ad", st_stim.X, st_stim.obs_names, genes, {"condition": ["stim"] * st_stim.n_obs},
                 spatial=np.asarray(st_stim.obsm["spatial"], dtype=np.float64))
    config = {
        "dataset": "GSE139107 snRNA-seq (Kirita 2020) + GSE269622 Visium (Humphreys lab 2025), mouse bilateral IRI, male",
        "stim_group": a.stim, "stim_section": a.stim_section, "n_genes": int(len(genes)), "n_spots_ctrl": int(st.n_obs), "n_spots_stim_observed": int(st_stim.n_obs),
        "n_ctrl_nuclei": int(len(picked["ctrl"])), "n_stim_nuclei": int(len(picked["stim"])),
        "nuclei_before_qc": n_before_qc, "nuclei_after_qc_all_genes": {c: int((rna.obs["condition"] == c).sum()) for c in ("ctrl", "stim")},
        "qc": "total UMIs >= 500 and detected genes >= 300 on all shared genes; pipeline QC not re-applied (--skip-qc)",
        "shared_expressed_genes": int(expressed.sum()), "post_perturbation_visium_used": False, "seed": int(a.seed),
        "median_umi_per_spot_selected_genes": float(np.median(np.asarray(st.X.sum(axis=1)).ravel())),
        "median_umi_per_nucleus_selected_genes": float(np.median(total[np.concatenate(list(picked.values()))])),
        "injury_markers_included": {g: bool(g in set(genes)) for g in ["Havcr1", "Lcn2", "Krt20", "Vcam1", "Spp1", "Hspa1a", "Fgb", "Clu", "Timp1", "Krt8", "Slc34a1", "Kcnj1", "Slc12a1"]},
    }
    (a.out / "scenario_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
