#!/usr/bin/env python3
"""SIMO mapping of the reference cells to the spots of one condition.  Run in the SIMO environment.

The procedure and every argument are those of
`pipeline/simo_base.py` (preprocessing,
Leiden resolutions, marker selection, `alignment_1_batch`, `assign_coord_1`).  Differences:

  * the number of reference cells is not fixed at 20,000;
  * incomplete spot coverage is recorded instead of raising;
  * from the same transport plan, two assignments are written:
      <prefix>_norepeat.h5ad   no_repeated_cells=True  (setting of the main analysis)
      <prefix>_repeat.h5ad     no_repeated_cells=False (a reference cell may be placed in several spots)
    The second is the natural setting when the reference is not the set of cells in the tissue.

Ground-truth labels are never read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simo_base as base  # noqa: E402
from simo import alignment_1_batch, assign_coord_1, find_marker, process_anndata  # noqa: E402


def write_mapping(table: pd.DataFrame, path: Path, extra: dict) -> dict:
    obs = pd.DataFrame({
        "Cell": table["cell"].astype(str).to_numpy(),
        "spot": table["spot"].astype(str).to_numpy(),
        "spot_type": table["spot_type"].astype(str).to_numpy(),
    }, index=pd.Index([f"m{i}" for i in range(len(table))], name="mapping_row"))
    out = ad.AnnData(X=sp.csr_matrix((len(obs), 1), dtype=np.float32), obs=obs,
                     var=pd.DataFrame(index=pd.Index(["__mapping_only__"], name="feature")))
    out.obsm["spatial"] = table[["Cell_xcoord", "Cell_ycoord"]].to_numpy(dtype=np.float32)
    out.uns["mapping_only_file"] = True
    out.uns["truth_labels_used"] = False
    out.write_h5ad(path)
    per_spot = obs["spot"].value_counts()
    info = {
        "output": str(path),
        "mapped_rows": int(len(obs)),
        "distinct_cells": int(obs["Cell"].nunique()),
        "covered_spots": int(per_spot.size),
        "cells_per_spot_min": int(per_spot.min()),
        "cells_per_spot_median": float(per_spot.median()),
        "cells_per_spot_max": int(per_spot.max()),
        **extra,
    }
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--model-seed", type=int, required=True)
    p.add_argument("--skip-qc", action="store_true", help="do not re-apply the per-cell QC (real data already QC'd on all genes)")
    a = p.parse_args()
    scenario_dir = a.scenario_dir.resolve()
    prefix = a.output_prefix.resolve()
    done = prefix.with_name(prefix.name + ".json")
    if done.exists():
        print("already done:", done)
        return
    t0 = time.time()
    base.set_seed(int(a.model_seed))
    rna = base.read_v2_input_without_new_anndata_encoding(scenario_dir / "sc_ctrl_raw.h5ad", require_spatial=False)
    st = base.read_v2_input_without_new_anndata_encoding(scenario_dir / "st_ctrl_raw.h5ad", require_spatial=True)
    n_genes = rna.n_vars
    if not rna.var_names.equals(st.var_names):
        raise AssertionError("sc/ST gene order mismatch")

    rna.layers["counts"] = rna.X.copy()
    rna.var["mt"] = rna.var_names.astype(str).str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    cell_mask = (rna.obs["total_counts"] >= 1000) & (rna.obs["n_genes_by_counts"] >= 500)
    if int(rna.var["mt"].sum()) > 0:
        cell_mask &= rna.obs["pct_counts_mt"] <= 5.0
    if a.skip_qc:
        cell_mask[:] = True
    rna = rna[cell_mask].copy()
    rna.X = rna.layers["counts"].copy()
    st.layers["counts"] = st.X.copy()
    st.obsm["spatial"] = np.asarray(st.obsm["spatial"], dtype=np.float32)

    st_processed = process_anndata(st, neighbors=True, umap=True, n_comps=100)
    sc.tl.leiden(st_processed, resolution=0.95, key_added="leiden", random_state=int(a.model_seed))
    rna_processed = process_anndata(rna, neighbors=True, umap=True, n_comps=100)
    sc.tl.leiden(rna_processed, resolution=0.2, key_added="leiden", random_state=int(a.model_seed))
    common = st_processed.var_names.intersection(rna_processed.var_names, sort=False)
    if len(common) != n_genes:
        raise AssertionError("SIMO preprocessing changed the gene universe")
    st_processed = st_processed[:, common].copy()
    rna_processed = rna_processed[:, common].copy()
    gene_list = find_marker(st_processed, rna_processed, gene_selection_method="deg", deg_num=100,
                            marker1_by="leiden", marker2_by="leiden")
    if not gene_list:
        raise RuntimeError("SIMO marker-gene selection returned an empty list")
    st_processed.obsm["reduction"] = st_processed.obsm["X_pca"]
    rna_processed.obsm["reduction"] = rna_processed.obsm["X_pca"]
    st_processed.obs["type"] = st_processed.obs["leiden"].astype(str)
    rna_processed.obs["type"] = rna_processed.obs["leiden"].astype(str)

    transport = alignment_1_batch(
        st_processed[:, gene_list], rna_processed[:, gene_list], alpha=0.1, numItermax=200, norm="l2", k=10,
        graph_mode="connectivity", aware_st=True, aware_sc=True, aware_st_label="leiden", aware_sc_label="leiden",
        aware_power_st=2, aware_power_sc=2, batch_size=5,
    )
    t_align = time.time() - t0
    shared = {
        "scenario_dir": str(scenario_dir), "model_seed": int(a.model_seed), "simo_omics_version": version("simo-omics"),
        "n_spots": int(st.n_obs), "sc_cells_after_qc": int(rna.n_obs), "marker_gene_count": int(len(gene_list)),
        "sc_leiden_clusters": int(rna_processed.obs["leiden"].nunique()),
        "st_leiden_clusters": int(st_processed.obs["leiden"].nunique()),
        "alpha": 0.1, "numItermax": 200, "k": 10, "top_num": 10, "truth_labels_used": False,
    }
    report = {"shared": shared, "alignment_seconds": t_align}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for label, no_repeat in (("norepeat", True), ("repeat", False)):
        try:
            table = assign_coord_1(
                st_processed[:, gene_list], rna_processed[:, gene_list], transport, non_zero_probabilities=True,
                no_repeated_cells=no_repeat, top_num=10, expected_num=None, pos_random=False, layer="data",
                weighted_average=False,
            )
            if table.empty:
                raise RuntimeError("empty assignment")
            report[label] = write_mapping(table, prefix.with_name(f"{prefix.name}_{label}.h5ad"), {"no_repeated_cells": no_repeat})
        except Exception as error:  # the other variant is still useful
            report[label] = {"failed": repr(error), "no_repeated_cells": no_repeat}
    report["total_seconds"] = time.time() - t0
    done.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
