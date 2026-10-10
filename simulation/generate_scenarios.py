#!/usr/bin/env python3
"""Generate and audit M2 simulation v2 scenarios.

The public CLI remains the frozen data-seed-1 entry point.  The reusable
``generate_from_args`` function is also used by the independently audited
seed-general final-simulation driver.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scenario_generator import (
    PerturbationTruth,
    aggregate_counts,
    apply_perturbation,
    assign_cells_to_spots,
    balanced_group_targets,
    baseline_diagnostics,
    checksums_for_directory,
    constant_multiplier_audit,
    copy_frozen_file,
    create_perturbation_truth,
    create_spatial_design,
    git_commit,
    independent_aggregation_audit,
    load_json,
    package_versions,
    read_pool_rows,
    require_parquet_engine,
    rng_for,
    sha256,
    stratified_disjoint_split,
    treated_mask,
    write_json,
    write_raw_h5ad,
)


SCENARIO_SPECS = {
    "legacy_all": {"family": "legacy", "mode": "all"},
    "legacy_fraction": {"family": "legacy", "mode": "fraction"},
    "legacy_group": {"family": "legacy", "mode": "group"},
    "unpaired_all": {"family": "unpaired", "mode": "all"},
    "unpaired_fraction": {"family": "unpaired", "mode": "fraction"},
    "unpaired_group": {"family": "unpaired", "mode": "group"},
    "unpaired_group_heterogeneous": {
        "family": "unpaired",
        "mode": "group",
        "heterogeneous": True,
    },
    "unpaired_group_modality": {
        "family": "unpaired",
        "mode": "group",
        "modality": True,
    },
}


def make_cell_metadata(
    pool_meta: pd.DataFrame,
    pool_indices: np.ndarray,
    prefix: str,
    condition: str,
    family: str,
) -> pd.DataFrame:
    selected = pool_meta.iloc[pool_indices].reset_index(drop=True)
    n = len(selected)
    return pd.DataFrame(
        {
            "cell_id": np.asarray([f"{prefix}_{i:05d}" for i in range(1, n + 1)]),
            "pool_cell_id": selected["pool_cell_id"].astype(str).to_numpy(),
            "true_group": selected["true_group"].astype(str).to_numpy(),
            "condition": condition,
            "family": family,
        }
    )


def perturbation_frame(truth: PerturbationTruth, gene_ids: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_id": gene_ids[truth.gene_indices],
            "gene_index_zero_based": truth.gene_indices,
            "direction": truth.directions,
            "log2fc": truth.log2fc,
            "factor": truth.factors,
        }
    )


def generate_from_args(
    args: argparse.Namespace,
    *,
    expected_data_seed: int | None = 1,
    scenario_specs: dict[str, dict[str, object]] | None = None,
) -> None:
    require_parquet_engine()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    project_root = args.project_root.resolve()
    config = load_json(config_path)
    if expected_data_seed is not None and config["data_seed"] != expected_data_seed:
        raise AssertionError(
            f"This entry point is frozen to data_seed={expected_data_seed}."
        )
    selected_specs = SCENARIO_SPECS if scenario_specs is None else scenario_specs
    if config["scenarios"] != list(selected_specs):
        raise AssertionError("Config scenario order/specification mismatch.")

    common = output_root / "common"
    pool_path = common / "splatter_pool_counts.h5"
    pool_meta_path = common / "splatter_pool_cell_metadata.csv"
    gene_meta_path = common / "gene_metadata.csv"
    for required in (pool_path, pool_meta_path, gene_meta_path, common / "sessionInfo.txt"):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Run simulation/create_pool.R first."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    frozen_config = output_root / "generation_config.json"
    copy_frozen_file(config_path, frozen_config)
    source_dir = Path(__file__).resolve().parent
    source_files = [
        source_dir / "create_pool.R",
        source_dir / "scenario_generator.py",
        Path(__file__).resolve(),
    ]
    provenance = {
        "git_commit": git_commit(project_root),
        "config_sha256": sha256(config_path),
        "source_sha256": {p.name: sha256(p) for p in source_files},
        "python_environment": package_versions(),
        "formal_input": str(pool_path),
        "legacy_simulation_input_used": False,
    }
    write_json(output_root / "generation_provenance.json", provenance)

    d = config["dimensions"]
    pool_meta = pd.read_csv(pool_meta_path)
    gene_ids = pd.read_csv(gene_meta_path)["gene_id"].astype(str).to_numpy()
    if pool_meta["pool_cell_id"].duplicated().any() or len(pool_meta) != d["splatter_pool_cells"]:
        raise AssertionError("Invalid Splatter pool cell metadata.")
    if len(gene_ids) != d["n_genes"] or len(np.unique(gene_ids)) != len(gene_ids):
        raise AssertionError("Invalid gene IDs.")
    groups = [f"Group{i}" for i in range(1, d["n_groups"] + 1)]
    if set(pool_meta["true_group"]) != set(groups):
        raise AssertionError("Splatter true groups do not match Group1..Group9.")

    targets = balanced_group_targets(int(d["n_ctrl_cells"]), groups)
    ctrl_pool_idx, stim_pool_idx = stratified_disjoint_split(
        pool_meta, targets, config["seeds"]["seed_split"]
    )
    with h5py.File(pool_path, "r") as pool_h5:
        dataset = pool_h5["X"]
        expected_shape = (d["splatter_pool_cells"], d["n_genes"])
        if dataset.shape != expected_shape:
            raise AssertionError(
                f"Pool HDF5 orientation/shape is {dataset.shape}, expected {expected_shape}."
            )
        ctrl_counts = read_pool_rows(dataset, ctrl_pool_idx)
        stim_baseline_counts = read_pool_rows(dataset, stim_pool_idx)

    legacy_ctrl_meta = make_cell_metadata(
        pool_meta, ctrl_pool_idx, "legacy_cell", "ctrl", "legacy"
    )
    legacy_stim_meta = legacy_ctrl_meta.copy()
    legacy_stim_meta["condition"] = "stim"
    unpaired_ctrl_meta = make_cell_metadata(
        pool_meta, ctrl_pool_idx, "unpaired_ctrl_cell", "ctrl", "unpaired"
    )
    unpaired_stim_meta = make_cell_metadata(
        pool_meta, stim_pool_idx, "unpaired_stim_cell", "stim", "unpaired"
    )
    if set(unpaired_ctrl_meta["cell_id"]) & set(unpaired_stim_meta["cell_id"]):
        raise AssertionError("Unpaired final cell IDs overlap.")

    design = create_spatial_design(config, targets)
    legacy_mapping = assign_cells_to_spots(
        legacy_ctrl_meta["cell_id"].to_numpy(),
        legacy_ctrl_meta["true_group"].to_numpy(),
        design,
        config["seeds"]["seed_spatial"],
        10,
    )
    unpaired_ctrl_mapping = assign_cells_to_spots(
        unpaired_ctrl_meta["cell_id"].to_numpy(),
        unpaired_ctrl_meta["true_group"].to_numpy(),
        design,
        config["seeds"]["seed_spatial"],
        20,
    )
    unpaired_stim_mapping = assign_cells_to_spots(
        unpaired_stim_meta["cell_id"].to_numpy(),
        unpaired_stim_meta["true_group"].to_numpy(),
        design,
        config["seeds"]["seed_spatial"],
        30,
    )

    truth = create_perturbation_truth(config)
    perturb_df = perturbation_frame(truth, gene_ids)
    if len(perturb_df) != 800 or (perturb_df["direction"] == "up").sum() != 400:
        raise AssertionError("Perturbation gene truth dimensions are invalid.")

    capture_rng = rng_for(config["seeds"]["seed_modality"], 1)
    modality_cfg = config["modality"]
    capture_efficiency = capture_rng.beta(
        modality_cfg["capture_efficiency_beta_a"],
        modality_cfg["capture_efficiency_beta_b"],
        size=len(gene_ids),
    )

    audit_rows: list[dict[str, object]] = []
    canonical_order = list(SCENARIO_SPECS)
    for scenario, spec in selected_specs.items():
        # Preserve the seed-1 scenario-specific RNG stream even when the final
        # analysis requests only a subset of scenarios.
        scenario_index = canonical_order.index(scenario) + 1
        scenario_dir = output_root / scenario
        completed = scenario_dir / "generation_audit.json"
        if completed.exists() and args.resume:
            print(f"[{scenario}] complete; resume mode skips it")
            audit = json.loads(completed.read_text(encoding="utf-8"))
            audit_rows.append(audit["audit_table_row"])
            continue
        if scenario_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite incomplete/existing scenario: {scenario_dir}"
            )
        scenario_dir.mkdir(parents=True)
        print(f"[{scenario}] generating")

        family = spec["family"]
        mode = spec["mode"]
        if family == "legacy":
            sc_ctrl = ctrl_counts
            sc_stim_baseline = ctrl_counts
            ctrl_meta = legacy_ctrl_meta.copy()
            stim_meta = legacy_stim_meta.copy()
            ctrl_mapping = legacy_mapping.copy()
            stim_mapping = legacy_mapping.copy()
        else:
            sc_ctrl = ctrl_counts
            sc_stim_baseline = stim_baseline_counts
            ctrl_meta = unpaired_ctrl_meta.copy()
            stim_meta = unpaired_stim_meta.copy()
            ctrl_mapping = unpaired_ctrl_mapping.copy()
            stim_mapping = unpaired_stim_mapping.copy()

        treated = treated_mask(
            mode,
            stim_meta["true_group"].to_numpy(),
            config,
            stream=100 + scenario_index,
        )
        eta = None
        if spec.get("heterogeneous", False):
            response_rng = rng_for(config["seeds"]["seed_response"], scenario_index)
            eta = response_rng.normal(
                0.0,
                config["perturbation"]["heterogeneous_eta_sd"],
                size=int(treated.sum()),
            )
        sc_stim = apply_perturbation(sc_stim_baseline, treated, truth, eta=eta)
        st_ctrl_latent = aggregate_counts(sc_ctrl, ctrl_mapping, design)
        st_stim_latent = aggregate_counts(sc_stim, stim_mapping, design)

        modality_shift = bool(spec.get("modality", False))
        if modality_shift:
            rng_ctrl = rng_for(config["seeds"]["seed_modality"], 10)
            rng_stim = rng_for(config["seeds"]["seed_modality"], 20)
            st_ctrl = rng_ctrl.binomial(
                st_ctrl_latent, capture_efficiency[None, :]
            ).astype(np.uint32)
            st_stim = rng_stim.binomial(
                st_stim_latent, capture_efficiency[None, :]
            ).astype(np.uint32)
        else:
            st_ctrl = st_ctrl_latent
            st_stim = st_stim_latent

        ctrl_audit = independent_aggregation_audit(
            sc_ctrl, ctrl_mapping, st_ctrl_latent, design
        )
        stim_audit = independent_aggregation_audit(
            sc_stim, stim_mapping, st_stim_latent, design
        )
        if not ctrl_audit["exact"] or not stim_audit["exact"]:
            raise AssertionError(f"ID-based aggregation failed for {scenario}.")

        ctrl_counts_by_group = (
            ctrl_mapping.groupby(["spot_id", "true_group"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(index=design.spot_ids, columns=groups, fill_value=0)
        )
        stim_counts_by_group = (
            stim_mapping.groupby(["spot_id", "true_group"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(index=design.spot_ids, columns=groups, fill_value=0)
        )
        same_composition = bool(
            np.array_equal(ctrl_counts_by_group.to_numpy(), stim_counts_by_group.to_numpy())
        )
        if family == "unpaired" and not same_composition:
            raise AssertionError("Unpaired ctrl/stim spot-group compositions differ.")

        ctrl_meta["treated"] = False
        stim_meta["treated"] = treated
        treated_df = stim_meta.loc[treated, ["cell_id", "true_group"]].copy()
        if eta is not None:
            treated_df["eta"] = eta

        ctrl_meta.to_parquet(scenario_dir / "cell_metadata_ctrl.parquet", index=False)
        stim_meta.to_parquet(scenario_dir / "cell_metadata_stim.parquet", index=False)
        ctrl_mapping.to_parquet(scenario_dir / "mapping_ctrl.parquet", index=False)
        stim_mapping.to_parquet(scenario_dir / "mapping_stim.parquet", index=False)
        perturb_df.to_csv(scenario_dir / "perturbed_genes.csv", index=False)
        treated_df.to_csv(scenario_dir / "perturbed_cells.csv", index=False)
        if eta is not None:
            treated_df[["cell_id", "eta"]].to_csv(
                scenario_dir / "heterogeneous_eta.csv", index=False
            )
        if modality_shift:
            pd.DataFrame(
                {"gene_id": gene_ids, "capture_efficiency": capture_efficiency}
            ).to_csv(scenario_dir / "capture_efficiency.csv", index=False)

        group_count_output = ctrl_counts_by_group.copy()
        group_count_output.index.name = "spot_id"
        group_count_output.reset_index().to_csv(
            scenario_dir / "spot_group_counts.csv", index=False
        )
        true_mix = group_count_output / int(d["cells_per_spot"])
        true_mix.reset_index().to_csv(
            scenario_dir / "true_cell_type_mixture.csv", index=False
        )
        spot_meta = pd.DataFrame(
            {
                "spot_id": design.spot_ids,
                "x": design.coordinates[:, 0],
                "y": design.coordinates[:, 1],
                "ctrl_library_size_raw": st_ctrl.sum(axis=1, dtype=np.uint64),
                "stim_library_size_raw": st_stim.sum(axis=1, dtype=np.uint64),
                "ctrl_library_size_latent_raw": st_ctrl_latent.sum(
                    axis=1, dtype=np.uint64
                ),
                "stim_library_size_latent_raw": st_stim_latent.sum(
                    axis=1, dtype=np.uint64
                ),
            }
        )
        for group in groups:
            spot_meta[f"p_true_{group}"] = true_mix[group].to_numpy()
        spot_meta.to_parquet(scenario_dir / "spot_metadata.parquet", index=False)

        compression_level = config["storage"]["h5ad_compression_level"]
        write_raw_h5ad(
            scenario_dir / "sc_ctrl_raw.h5ad",
            sc_ctrl,
            ctrl_meta["cell_id"],
            gene_ids,
            {"condition": ["ctrl"] * len(ctrl_meta), "family": [family] * len(ctrl_meta)},
            compression_level,
        )
        write_raw_h5ad(
            scenario_dir / "sc_stim_raw.h5ad",
            sc_stim,
            stim_meta["cell_id"],
            gene_ids,
            {"condition": ["stim"] * len(stim_meta), "family": [family] * len(stim_meta)},
            compression_level,
        )
        write_raw_h5ad(
            scenario_dir / "st_ctrl_raw.h5ad",
            st_ctrl,
            design.spot_ids,
            gene_ids,
            {"condition": ["ctrl"] * len(design.spot_ids)},
            compression_level,
            spatial=design.coordinates,
        )
        write_raw_h5ad(
            scenario_dir / "st_stim_raw.h5ad",
            st_stim,
            design.spot_ids,
            gene_ids,
            {"condition": ["stim"] * len(design.spot_ids)},
            compression_level,
            spatial=design.coordinates,
        )
        if modality_shift:
            write_raw_h5ad(
                scenario_dir / "st_ctrl_latent_raw.h5ad",
                st_ctrl_latent,
                design.spot_ids,
                gene_ids,
                {"condition": ["ctrl"] * len(design.spot_ids)},
                compression_level,
                spatial=design.coordinates,
            )
            write_raw_h5ad(
                scenario_dir / "st_stim_latent_raw.h5ad",
                st_stim_latent,
                design.spot_ids,
                gene_ids,
                {"condition": ["stim"] * len(design.spot_ids)},
                compression_level,
                spatial=design.coordinates,
            )

        diagnostic = baseline_diagnostics(
            sc_ctrl, sc_stim, st_ctrl, st_stim, truth.gene_indices
        )
        write_json(scenario_dir / "identity_mean_ratio_diagnostic.json", diagnostic)
        if scenario == "legacy_all":
            write_json(
                scenario_dir / "legacy_all_constant_multiplier_audit.json",
                constant_multiplier_audit(
                    st_ctrl, st_stim, truth, int(d["cells_per_spot"])
                ),
            )

        ctrl_stim_disjoint = bool(
            set(ctrl_meta["cell_id"]).isdisjoint(set(stim_meta["cell_id"]))
        )
        table_row = {
            "scenario": scenario,
            "G": int(d["n_genes"]),
            "N_ctrl": len(ctrl_meta),
            "N_stim": len(stim_meta),
            "S": int(d["n_spots"]),
            "cells_per_spot": int(d["cells_per_spot"]),
            "K": int(d["n_groups"]),
            "splatter_true_groups": len(groups),
            "ctrl_stim_disjoint": ctrl_stim_disjoint,
            "same_spot_group_composition": same_composition,
            "aggregation_exact_ctrl": ctrl_audit["exact"],
            "aggregation_exact_stim": stim_audit["exact"],
            "perturbed_gene_count": len(truth.gene_indices),
            "up_gene_count": int((truth.directions == "up").sum()),
            "down_gene_count": int((truth.directions == "down").sum()),
            "modality_shift": modality_shift,
            "heterogeneous_response": bool(spec.get("heterogeneous", False)),
            "max_aggregation_error_ctrl": ctrl_audit["max_absolute_difference"],
            "max_aggregation_error_stim": stim_audit["max_absolute_difference"],
        }
        audit = {
            "audit_table_row": table_row,
            "legacy_simulation_input_used": False,
            "raw_counts_integer": True,
            "gene_order_identical_sc_st_ctrl_stim": True,
            "duplicate_cell_ids_ctrl": int(ctrl_meta["cell_id"].duplicated().sum()),
            "duplicate_cell_ids_stim": int(stim_meta["cell_id"].duplicated().sum()),
            "duplicate_spot_ids": int(pd.Index(design.spot_ids).duplicated().sum()),
            "duplicate_gene_ids": int(pd.Index(gene_ids).duplicated().sum()),
            "ctrl_aggregation": ctrl_audit,
            "stim_aggregation": stim_audit,
            "modality_aggregation_note": (
                "Exact aggregation is asserted against latent ST; observed ST includes "
                "the pre-specified binomial capture step."
                if modality_shift
                else "Observed ST is the exact ID-based aggregation."
            ),
            "target_group_from_config": config["perturbation"]["target_group"],
            "treated_cell_count": int(treated.sum()),
            "source_pool_ctrl_stim_overlap": int(
                np.intersect1d(ctrl_pool_idx, stim_pool_idx).size
            ),
        }
        copy_frozen_file(frozen_config, scenario_dir / "generation_config.json")
        copy_frozen_file(common / "sessionInfo.txt", scenario_dir / "sessionInfo.txt")
        write_json(scenario_dir / "generation_audit.json", audit)
        checksums_for_directory(scenario_dir)
        audit_rows.append(table_row)

    design.group_centers.to_csv(common / "group_centers.csv", index=False)
    probability_df = pd.DataFrame(
        design.group_probabilities, columns=design.group_names
    )
    probability_df.insert(0, "spot_id", design.spot_ids)
    probability_df.to_csv(common / "spot_group_generation_probabilities.csv", index=False)
    pd.DataFrame(
        {
            "spot_id": design.spot_ids,
            "x": design.coordinates[:, 0],
            "y": design.coordinates[:, 1],
        }
    ).to_csv(common / "spot_coordinates.csv", index=False)
    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(output_root / "simulation_v2_generation_audit.csv", index=False)
    if len(audit_frame) != len(selected_specs):
        raise AssertionError(
            f"Audit table must contain {len(selected_specs)} scenarios."
        )
    checksums_for_directory(common)
    print(
        f"{len(selected_specs)} data_seed={config['data_seed']} scenarios generated "
        "and audited. No model was trained."
    )
