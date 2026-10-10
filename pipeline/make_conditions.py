#!/usr/bin/env python3
"""Derive the additional simulation conditions from an already generated data seed.

No new Splatter run is needed: every variant reuses the seed-specific pool and the
spatial design of the main analysis, and changes exactly one generating factor.

  capture   Change only the ST-side gene-wise capture efficiency of a scenario that
            stores the pre-capture ST (`st_*_latent_raw.h5ad`): the main modality-shift
            scenario or the complex spatial scenario.  `--capture none` removes the
            modality shift; `--capture beta:A:B` draws kappa_g ~ Beta(A, B);
            `--capture const:Q` uses the same capture probability Q for all genes (ST depth only).
            scRNA-seq, spatial design, perturbation and cell placement are unchanged.

  refshift  Replace only the scRNA-seq reference of `unpaired_group` by cells that are
            NOT contained in the ST (drawn from the unused part of the pool), optionally
            with a different abundance of the responding group.  ST (control and
            stimulated truth) is unchanged.

Outputs follow the layout of the main scenarios so that the same model code and the
same evaluation can be applied.  Large unchanged files are hard-linked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation"))

from scenario_generator import (  # noqa: E402
    apply_perturbation,
    create_perturbation_truth,
    read_pool_rows,
    write_raw_h5ad,
)

GROUPS = [f"Group{i}" for i in range(1, 10)]


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def dense_u32(x) -> np.ndarray:
    return np.asarray(x.toarray() if sp.issparse(x) else x).astype(np.uint32)


def cmd_capture(a: argparse.Namespace) -> None:
    src, out = a.source.resolve(), a.out.resolve()
    if (out / "variant_config.json").exists():
        print("already done:", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    has_latent = (src / "st_ctrl_latent_raw.h5ad").exists()
    if not has_latent and (src / "capture_efficiency.csv").exists():
        raise RuntimeError("source already has a capture step but no pre-capture ST")
    lat_c = ad.read_h5ad(src / ("st_ctrl_latent_raw.h5ad" if has_latent else "st_ctrl_raw.h5ad"))
    lat_s = ad.read_h5ad(src / ("st_stim_latent_raw.h5ad" if has_latent else "st_stim_raw.h5ad"))
    genes = lat_c.var_names.astype(str).to_numpy()
    spots = lat_c.obs_names.astype(str).to_numpy()
    coords = np.asarray(lat_c.obsm["spatial"], dtype=np.float64)
    xc, xs = dense_u32(lat_c.X), dense_u32(lat_s.X)
    config = {"variant": "capture", "source": str(src), "capture": a.capture, "seed": int(a.seed)}
    if a.capture == "none":
        st_c, st_s = xc, xs
    elif a.capture.startswith("const:"):
        # the same capture probability for every gene: only the sequencing depth of the ST changes
        q = float(a.capture.split(":")[1])
        ss = [int(a.seed), 9002, int(round(q * 1e6))]
        kappa = np.full(len(genes), q)
        st_c = np.random.default_rng(np.random.SeedSequence(ss + [10])).binomial(xc, q).astype(np.uint32)
        st_s = np.random.default_rng(np.random.SeedSequence(ss + [20])).binomial(xs, q).astype(np.uint32)
        pd.DataFrame({"gene_id": genes, "capture_efficiency": kappa}).to_csv(out / "capture_efficiency.csv", index=False)
        config.update({"constant_capture": q, "median_umis_per_spot_ctrl": float(np.median(st_c.sum(axis=1))),
                       "median_umis_per_spot_before": float(np.median(xc.sum(axis=1)))})
    elif a.capture.startswith("betaconst:"):
        # gene-wise Beta(A, B) capture multiplied by a common factor Q: modality shift at a lower ST depth
        _kind, alpha, beta, q = a.capture.split(":")
        alpha, beta, q = float(alpha), float(beta), float(q)
        ss = [int(a.seed), 9003, int(round(alpha * 1000)), int(round(beta * 1000)), int(round(q * 1e6))]
        kappa = np.random.default_rng(np.random.SeedSequence(ss + [1])).beta(alpha, beta, size=len(genes)) * q
        st_c = np.random.default_rng(np.random.SeedSequence(ss + [10])).binomial(xc, kappa[None, :]).astype(np.uint32)
        st_s = np.random.default_rng(np.random.SeedSequence(ss + [20])).binomial(xs, kappa[None, :]).astype(np.uint32)
        pd.DataFrame({"gene_id": genes, "capture_efficiency": kappa}).to_csv(out / "capture_efficiency.csv", index=False)
        config.update({"beta_a": alpha, "beta_b": beta, "common_factor": q, "log_kappa_sd": float(np.log(kappa).std()),
                       "median_umis_per_spot_ctrl": float(np.median(st_c.sum(axis=1)))})
    else:
        kind, alpha, beta = a.capture.split(":")
        if kind != "beta":
            raise ValueError(a.capture)
        alpha, beta = float(alpha), float(beta)
        ss = [int(a.seed), 9001, int(round(alpha * 1000)), int(round(beta * 1000))]
        kappa = np.random.default_rng(np.random.SeedSequence(ss + [1])).beta(alpha, beta, size=len(genes))
        st_c = np.random.default_rng(np.random.SeedSequence(ss + [10])).binomial(xc, kappa[None, :]).astype(np.uint32)
        st_s = np.random.default_rng(np.random.SeedSequence(ss + [20])).binomial(xs, kappa[None, :]).astype(np.uint32)
        pd.DataFrame({"gene_id": genes, "capture_efficiency": kappa}).to_csv(out / "capture_efficiency.csv", index=False)
        config.update({"beta_a": alpha, "beta_b": beta, "kappa_mean": float(kappa.mean()), "kappa_sd": float(kappa.std()),
                       "log_kappa_sd": float(np.log(kappa).std())})
    write_raw_h5ad(out / "st_ctrl_raw.h5ad", st_c, spots, genes, {"condition": ["ctrl"] * len(spots)}, 4, spatial=coords)
    write_raw_h5ad(out / "st_stim_raw.h5ad", st_s, spots, genes, {"condition": ["stim"] * len(spots)}, 4, spatial=coords)
    for name in os.listdir(src):
        if name in {"st_ctrl_raw.h5ad", "st_stim_raw.h5ad", "capture_efficiency.csv", "checksums.sha256",
                    "identity_mean_ratio_diagnostic.json", "generation_audit.json", "figures", "variant_config.json"}:
            continue
        if (src / name).is_file():
            link(src / name, out / name)
    if not has_latent:  # keep the pre-capture ST next to the new one, as the modality-shift scenario does
        link(src / "st_ctrl_raw.h5ad", out / "st_ctrl_latent_raw.h5ad")
        link(src / "st_stim_raw.h5ad", out / "st_stim_latent_raw.h5ad")
        if (src / "variant_config.json").exists():
            config["source_variant"] = json.loads((src / "variant_config.json").read_text())
    (out / "variant_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


def cmd_refshift(a: argparse.Namespace) -> None:
    root, out = a.data_root.resolve(), a.out.resolve()
    if (out / "variant_config.json").exists():
        print("already done:", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    base = root / "unpaired_group"
    gen_config = json.loads((base / "generation_config.json").read_text())
    pool_meta = pd.read_csv(root / "common/splatter_pool_cell_metadata.csv")
    used = set(pd.read_parquet(base / "cell_metadata_ctrl.parquet")["pool_cell_id"].astype(str))
    used |= set(pd.read_parquet(base / "cell_metadata_stim.parquet")["pool_cell_id"].astype(str))
    free = ~pool_meta["pool_cell_id"].astype(str).isin(used).to_numpy()
    rng = np.random.default_rng(np.random.SeedSequence([int(gen_config["data_seed"]), 9101]))
    target = gen_config["perturbation"]["target_group"]
    ctrl_idx, stim_idx, counts = [], [], {}
    group_of = pool_meta["true_group"].astype(str).to_numpy()
    # the number of unused pool cells per group differs between data seeds; use the largest balanced size available
    n_base = min(int(a.ref_per_group), min(int((free & (group_of == g)).sum()) for g in GROUPS) // 2)
    for group in GROUPS:
        available = rng.permutation(np.flatnonzero(free & (pool_meta["true_group"].astype(str) == group).to_numpy()))
        n = n_base
        if group == target:
            n = max(int(round(n * float(a.target_factor))), 1)
        # the same leading cells are kept for every factor, so smaller references are nested in larger ones
        half = len(available) // 2
        ctrl_idx.extend(available[:n].tolist())
        stim_idx.extend(available[half : half + n].tolist())
        counts[group] = n
    ctrl_idx = np.sort(np.asarray(ctrl_idx, dtype=np.int64))
    stim_idx = np.sort(np.asarray(stim_idx, dtype=np.int64))
    assert np.intersect1d(ctrl_idx, stim_idx).size == 0
    with h5py.File(root / "common/splatter_pool_counts.h5", "r") as handle:
        ctrl = read_pool_rows(handle["X"], ctrl_idx)
        stim_base = read_pool_rows(handle["X"], stim_idx)

    def meta(idx: np.ndarray, condition: str) -> pd.DataFrame:
        sel = pool_meta.iloc[idx].reset_index(drop=True)
        return pd.DataFrame({
            "cell_id": [f"ref_{condition}_cell_{i:05d}" for i in range(1, len(sel) + 1)],
            "pool_cell_id": sel["pool_cell_id"].astype(str).to_numpy(),
            "true_group": sel["true_group"].astype(str).to_numpy(),
            "condition": condition,
            "family": "reference_shift",
        })

    ctrl_meta, stim_meta = meta(ctrl_idx, "ctrl"), meta(stim_idx, "stim")
    truth = create_perturbation_truth(gen_config)
    perturbed = pd.read_csv(base / "perturbed_genes.csv")
    if not np.array_equal(perturbed["gene_index_zero_based"].to_numpy(), truth.gene_indices) or not np.allclose(
        perturbed["factor"].to_numpy(), truth.factors
    ):
        raise AssertionError("Reconstructed perturbation truth differs from the stored truth")
    treated = (stim_meta["true_group"] == target).to_numpy()
    stim = apply_perturbation(stim_base, treated, truth, eta=None)
    ctrl_meta["treated"] = False
    stim_meta["treated"] = treated
    genes = pd.read_csv(root / "common/gene_metadata.csv")["gene_id"].astype(str).to_numpy()
    write_raw_h5ad(out / "sc_ctrl_raw.h5ad", ctrl, ctrl_meta["cell_id"], genes,
                   {"condition": ["ctrl"] * len(ctrl_meta), "family": ["reference_shift"] * len(ctrl_meta)}, 4)
    write_raw_h5ad(out / "sc_stim_raw.h5ad", stim, stim_meta["cell_id"], genes,
                   {"condition": ["stim"] * len(stim_meta), "family": ["reference_shift"] * len(stim_meta)}, 4)
    ctrl_meta.to_parquet(out / "cell_metadata_ctrl.parquet", index=False)
    stim_meta.to_parquet(out / "cell_metadata_stim.parquet", index=False)
    for name in ("st_ctrl_raw.h5ad", "st_stim_raw.h5ad", "perturbed_genes.csv", "spot_group_counts.csv",
                 "true_cell_type_mixture.csv", "spot_metadata.parquet", "generation_config.json"):
        link(base / name, out / name)
    # the cells that physically make up the ST are those of the base scenario; kept for the counterfactual truth
    link(base / "sc_ctrl_raw.h5ad", out / "st_cells_ctrl_raw.h5ad")
    link(base / "mapping_ctrl.parquet", out / "st_cells_mapping_ctrl.parquet")
    link(base / "cell_metadata_ctrl.parquet", out / "st_cells_metadata_ctrl.parquet")
    config = {"variant": "refshift", "data_root": str(root), "target_group": target, "target_factor": float(a.target_factor),
              "reference_cells_per_group": counts, "n_ref_ctrl": int(len(ctrl_meta)), "n_ref_stim": int(len(stim_meta)),
              "reference_overlaps_st_cells": False,
              "st_target_fraction": 1.0 / len(GROUPS), "reference_target_fraction": counts[target] / sum(counts.values())}
    (out / "variant_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


def cmd_refctrl(a: argparse.Namespace) -> None:
    """Independent control reference of larger size: all unused pool cells form the control reference;
    the stimulated reference is that of the main analysis.  Optionally the responding group is made rarer
    in both reference conditions (same factor), while the ST is unchanged."""
    root, out = a.data_root.resolve(), a.out.resolve()
    if (out / "variant_config.json").exists():
        print("already done:", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    base = root / "unpaired_group"
    gen_config = json.loads((base / "generation_config.json").read_text())
    target = gen_config["perturbation"]["target_group"]
    pool_meta = pd.read_csv(root / "common/splatter_pool_cell_metadata.csv")
    used = set(pd.read_parquet(base / "cell_metadata_ctrl.parquet")["pool_cell_id"].astype(str))
    used |= set(pd.read_parquet(base / "cell_metadata_stim.parquet")["pool_cell_id"].astype(str))
    free = ~pool_meta["pool_cell_id"].astype(str).isin(used).to_numpy()
    if getattr(a, "from_st_cells", False):
        # control experiment: a reference of the same size drawn from the cells that DO make up the control ST
        st_cells = set(pd.read_parquet(base / "cell_metadata_ctrl.parquet")["pool_cell_id"].astype(str))
        free = pool_meta["pool_cell_id"].astype(str).isin(st_cells).to_numpy()
    rng = np.random.default_rng(np.random.SeedSequence([int(gen_config["data_seed"]), 9201]))
    ctrl_idx, counts = [], {}
    group_of = pool_meta["true_group"].astype(str).to_numpy()
    n_base = min(int(a.ref_per_group), min(int((free & (group_of == g)).sum()) for g in GROUPS))
    for group in GROUPS:
        available = rng.permutation(np.flatnonzero(free & (pool_meta["true_group"].astype(str) == group).to_numpy()))
        n = n_base
        if group == target:
            n = max(int(round(n * float(a.target_factor))), 1)
        ctrl_idx.extend(available[:n].tolist())
        counts[group] = n
    ctrl_idx = np.sort(np.asarray(ctrl_idx, dtype=np.int64))
    with h5py.File(root / "common/splatter_pool_counts.h5", "r") as handle:
        ctrl = read_pool_rows(handle["X"], ctrl_idx)
    sel = pool_meta.iloc[ctrl_idx].reset_index(drop=True)
    ctrl_meta = pd.DataFrame({
        "cell_id": [f"ref_ctrl_cell_{i:05d}" for i in range(1, len(sel) + 1)],
        "pool_cell_id": sel["pool_cell_id"].astype(str).to_numpy(),
        "true_group": sel["true_group"].astype(str).to_numpy(),
        "condition": "ctrl", "family": "reference_shift", "treated": False,
    })
    genes = pd.read_csv(root / "common/gene_metadata.csv")["gene_id"].astype(str).to_numpy()
    if not (out / "sc_ctrl_raw.h5ad").exists():  # deterministic, so a file left by an interrupted run is identical
        write_raw_h5ad(out / "sc_ctrl_raw.h5ad", ctrl, ctrl_meta["cell_id"], genes,
                       {"condition": ["ctrl"] * len(ctrl_meta), "family": ["reference_shift"] * len(ctrl_meta)}, 4)
    ctrl_meta.to_parquet(out / "cell_metadata_ctrl.parquet", index=False)

    stim_meta = pd.read_parquet(base / "cell_metadata_stim.parquet")
    if float(a.target_factor) >= 1.0:
        link(base / "sc_stim_raw.h5ad", out / "sc_stim_raw.h5ad")
        n_target_stim = int((stim_meta["true_group"] == target).sum())
    else:
        stim = ad.read_h5ad(base / "sc_stim_raw.h5ad")
        stim_meta = stim_meta.set_index("cell_id").loc[stim.obs_names.astype(str)].rename_axis("cell_id").reset_index()
        is_target = (stim_meta["true_group"] == target).to_numpy()
        target_rows = rng.permutation(np.flatnonzero(is_target))
        n_target_stim = max(int(round(len(target_rows) * float(a.target_factor))), 1)
        keep = np.sort(np.concatenate([np.flatnonzero(~is_target), target_rows[:n_target_stim]]))
        stim_meta = stim_meta.iloc[keep].reset_index(drop=True)
        write_raw_h5ad(out / "sc_stim_raw.h5ad", dense_u32(stim.X)[keep], stim_meta["cell_id"], genes,
                       {"condition": ["stim"] * len(keep), "family": ["reference_shift"] * len(keep)}, 4)
    stim_meta.to_parquet(out / "cell_metadata_stim.parquet", index=False)
    for name in ("st_ctrl_raw.h5ad", "st_stim_raw.h5ad", "perturbed_genes.csv", "spot_group_counts.csv",
                 "true_cell_type_mixture.csv", "spot_metadata.parquet", "generation_config.json"):
        link(base / name, out / name)
    link(base / "sc_ctrl_raw.h5ad", out / "st_cells_ctrl_raw.h5ad")
    link(base / "mapping_ctrl.parquet", out / "st_cells_mapping_ctrl.parquet")
    link(base / "cell_metadata_ctrl.parquet", out / "st_cells_metadata_ctrl.parquet")
    config = {"variant": "refctrl", "data_root": str(root), "target_group": target, "target_factor": float(a.target_factor),
              "control_reference_cells_per_group": counts, "n_ref_ctrl": int(len(ctrl_meta)), "n_ref_stim": int(len(stim_meta)),
              "n_target_ref_stim": n_target_stim, "control_reference_overlaps_st_cells": bool(getattr(a, "from_st_cells", False)),
              "stimulated_reference": "same cells as the main analysis" + ("" if float(a.target_factor) >= 1.0 else " (responding group subsampled)"),
              "st_target_fraction": 1.0 / len(GROUPS), "control_reference_target_fraction": counts[target] / sum(counts.values())}
    (out / "variant_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--source", type=Path, required=True, help="scenario directory containing st_*_latent_raw.h5ad")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--capture", required=True, help="'none', 'beta:A:B' or 'const:Q'")
    p.add_argument("--seed", type=int, required=True, help="data seed (enters the capture RNG stream)")
    p.set_defaults(func=cmd_capture)
    p = sub.add_parser("refshift")
    p.add_argument("--data-root", type=Path, required=True, help="results/simulation/data_seed_N")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ref-per-group", type=int, default=500)
    p.add_argument("--target-factor", type=float, default=1.0,
                   help="abundance of the responding group in the reference relative to the other groups")
    p.set_defaults(func=cmd_refshift)
    p = sub.add_parser("refctrl")
    p.add_argument("--data-root", type=Path, required=True, help="results/simulation/data_seed_N")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ref-per-group", type=int, default=1000)
    p.add_argument("--target-factor", type=float, default=1.0)
    p.add_argument("--from-st-cells", action="store_true",
                   help="draw the control reference from the cells that make up the control ST (same size; control for 'independence')")
    p.set_defaults(func=cmd_refctrl)
    a = parser.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
