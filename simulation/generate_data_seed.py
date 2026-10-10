#!/usr/bin/env python3
"""Generate one independently seeded final M2 data repetition.

This is deliberately separate from the historical seed-1 CLI.  All fixed
generation parameters are copied from the frozen seed-1 template; only the
seven declared RNG streams and the data-seed identifier vary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import pandas as pd

from generate_scenarios import SCENARIO_SPECS, generate_from_args


FINAL_SCENARIOS = (
    "unpaired_group",
    "unpaired_group_heterogeneous",
    "unpaired_group_modality",
)
STREAM_OFFSETS = {
    "seed_splatter": 101,
    "seed_split": 201,
    "seed_spatial": 301,
    "seed_perturb_gene": 401,
    "seed_perturb_cell": 501,
    "seed_response": 601,
    "seed_modality": 701,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values produced by audits to JSON scalars."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def fixed_projection(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"data_seed", "seeds", "scenarios"}
    }


def build_config(template: dict[str, Any], data_seed: int) -> dict[str, Any]:
    config = json.loads(json.dumps(template))
    config["data_seed"] = int(data_seed)
    config["seeds"] = {
        key: 1000 * int(data_seed) + offset
        for key, offset in STREAM_OFFSETS.items()
    }
    config["scenarios"] = list(FINAL_SCENARIOS)
    return config


def assert_scenario(path: Path, scenario: str, data_seed: int) -> dict[str, Any]:
    audit = read_json(path / "generation_audit.json")
    row = audit["audit_table_row"]
    config = read_json(path / "generation_config.json")
    expected_modality = scenario == "unpaired_group_modality"
    expected_heterogeneous = scenario == "unpaired_group_heterogeneous"
    perturbed_cells = pd.read_csv(path / "perturbed_cells.csv")
    mixture = pd.read_csv(path / "true_cell_type_mixture.csv")
    assertions = {
        "generation_audit_passed": bool(
            row["aggregation_exact_ctrl"]
            and row["aggregation_exact_stim"]
            and audit["raw_counts_integer"]
            and audit["gene_order_identical_sc_st_ctrl_stim"]
        ),
        "scenario_matches": row["scenario"] == scenario,
        "data_seed_matches": int(config["data_seed"]) == data_seed,
        "G_is_8000": int(row["G"]) == 8000,
        "S_is_2500": int(row["S"]) == 2500,
        "ctrl_cells_is_20000": int(row["N_ctrl"]) == 20000,
        "stim_cells_is_20000": int(row["N_stim"]) == 20000,
        "cell_id_overlap_zero": bool(row["ctrl_stim_disjoint"]),
        "source_pool_overlap_zero": int(audit["source_pool_ctrl_stim_overlap"]) == 0,
        "modality_flag_matches": bool(row["modality_shift"]) == expected_modality,
        "heterogeneity_flag_matches": bool(row["heterogeneous_response"]) == expected_heterogeneous,
        "perturbed_gene_count_is_800": int(row["perturbed_gene_count"]) == 800,
        "treated_cell_truth_nonempty": len(perturbed_cells) > 0,
        "true_mixture_has_2500_spots": len(mixture) == 2500,
        "target_group_matches_config": audit["target_group_from_config"] == config["perturbation"]["target_group"],
    }
    if expected_heterogeneous:
        eta = pd.read_csv(path / "heterogeneous_eta.csv")
        assertions.update(
            {
                "heterogeneous_eta_matches_treated_cells": len(eta) == len(perturbed_cells),
                "heterogeneous_eta_cell_ids_unique": not eta["cell_id"].duplicated().any(),
                "heterogeneous_eta_has_variation": float(eta["eta"].std(ddof=1)) > 0,
            }
        )
    if expected_modality:
        capture = pd.read_csv(path / "capture_efficiency.csv")
        assertions.update(
            {
                "capture_truth_has_8000_genes": len(capture) == 8000,
                "capture_gene_ids_unique": not capture["gene_id"].duplicated().any(),
                "capture_efficiency_strictly_between_zero_and_one": capture["capture_efficiency"].between(0, 1, inclusive="neither").all(),
                "latent_ctrl_ST_present": (path / "st_ctrl_latent_raw.h5ad").exists(),
                "latent_stim_ST_present": (path / "st_stim_latent_raw.h5ad").exists(),
            }
        )
    failed = [key for key, value in assertions.items() if not value]
    if failed:
        raise AssertionError(f"{scenario} generation audit failed: {failed}")
    for name in ("sc_ctrl_raw.h5ad", "sc_stim_raw.h5ad"):
        value = ad.read_h5ad(path / name, backed="r")
        try:
            if value.shape != (20000, 8000):
                raise AssertionError(f"{scenario}/{name} has shape {value.shape}")
        finally:
            value.file.close()
    for name in ("st_ctrl_raw.h5ad", "st_stim_raw.h5ad"):
        value = ad.read_h5ad(path / name, backed="r")
        try:
            if value.shape != (2500, 8000):
                raise AssertionError(f"{scenario}/{name} has shape {value.shape}")
        finally:
            value.file.close()
    return assertions


def write_manifest(output_root: Path, project_root: Path) -> Path:
    manifest = output_root / "generated_data_manifest.sha256"
    files = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path != manifest
    )
    lines = [
        f"{sha256(path)}  {path.resolve().relative_to(project_root.resolve())}"
        for path in files
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    template_path = args.template_config.resolve()
    template = read_json(template_path)
    config = build_config(template, args.data_seed)
    if fixed_projection(config) != fixed_projection(template):
        raise AssertionError("A fixed generation parameter differs from seed 1")

    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "generation_config_input.json"
    encoded = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("Existing generation config differs; refusing overwrite")
    if not config_path.exists():
        config_path.write_text(encoded, encoding="utf-8")

    common = output_root / "common"
    pool = common / "splatter_pool_counts.h5"
    if not pool.exists():
        subprocess.run(
            [
                str(args.rscript),
                str(Path(__file__).with_name("create_pool.R")),
                str(config_path),
                str(common),
            ],
            cwd=project_root,
            check=True,
        )
    elif not args.resume:
        raise FileExistsError(f"Splatter pool already exists: {pool}")

    selected = {name: SCENARIO_SPECS[name] for name in FINAL_SCENARIOS}
    generate_from_args(
        SimpleNamespace(
            config=config_path,
            output_root=output_root,
            project_root=project_root,
            resume=bool(args.resume),
        ),
        expected_data_seed=args.data_seed,
        scenario_specs=selected,
    )

    scenario_assertions = {
        scenario: assert_scenario(output_root / scenario, scenario, args.data_seed)
        for scenario in FINAL_SCENARIOS
    }
    # Within one repetition these equalities are part of the frozen design.
    group = output_root / "unpaired_group"
    hetero = output_root / "unpaired_group_heterogeneous"
    modality = output_root / "unpaired_group_modality"
    cross_scenario = {
        "group_vs_heterogeneous_sc_ctrl_identical": sha256(group / "sc_ctrl_raw.h5ad") == sha256(hetero / "sc_ctrl_raw.h5ad"),
        "group_vs_heterogeneous_st_ctrl_identical": sha256(group / "st_ctrl_raw.h5ad") == sha256(hetero / "st_ctrl_raw.h5ad"),
        "group_vs_modality_sc_ctrl_identical": sha256(group / "sc_ctrl_raw.h5ad") == sha256(modality / "sc_ctrl_raw.h5ad"),
        "group_vs_modality_sc_stim_identical": sha256(group / "sc_stim_raw.h5ad") == sha256(modality / "sc_stim_raw.h5ad"),
        "group_vs_modality_latent_st_ctrl_identical": sha256(group / "st_ctrl_raw.h5ad") == sha256(modality / "st_ctrl_latent_raw.h5ad"),
        "group_vs_modality_observed_st_ctrl_differs": sha256(group / "st_ctrl_raw.h5ad") != sha256(modality / "st_ctrl_raw.h5ad"),
    }
    if not all(cross_scenario.values()):
        raise AssertionError(f"Cross-scenario generation invariant failed: {cross_scenario}")

    seed1_root = project_root / "results/simulation/data_seed_1"
    seed1_comparison: dict[str, Any] = {
        "performed": args.data_seed != 1 and seed1_root.exists(),
        "fixed_parameter_projection_identical": fixed_projection(config) == fixed_projection(template),
    }
    if seed1_comparison["performed"]:
        seed1_comparison.update(
            {
                "splatter_pool_differs": sha256(output_root / "common/splatter_pool_counts.h5") != sha256(seed1_root / "common/splatter_pool_counts.h5"),
                "perturbed_gene_truth_differs": sha256(group / "perturbed_genes.csv") != sha256(seed1_root / "unpaired_group/perturbed_genes.csv"),
                "sc_ctrl_differs": sha256(group / "sc_ctrl_raw.h5ad") != sha256(seed1_root / "unpaired_group/sc_ctrl_raw.h5ad"),
                "st_ctrl_differs": sha256(group / "st_ctrl_raw.h5ad") != sha256(seed1_root / "unpaired_group/st_ctrl_raw.h5ad"),
            }
        )
        if not all(value for key, value in seed1_comparison.items() if key not in {"performed"}):
            raise AssertionError(f"Seed-1/seed-{args.data_seed} audit failed: {seed1_comparison}")

    report = {
        "data_seed": args.data_seed,
        "seed_policy": {
            key: {"offset": offset, "value": config["seeds"][key]}
            for key, offset in STREAM_OFFSETS.items()
        },
        "template_config": str(template_path),
        "template_config_sha256": sha256(template_path),
        "fixed_parameter_projection_identical_to_seed1": True,
        "scenario_assertions": scenario_assertions,
        "cross_scenario_assertions": cross_scenario,
        "seed1_randomness_comparison": seed1_comparison,
        "generator_source": str(Path(__file__).resolve()),
        "generator_source_sha256": sha256(Path(__file__).resolve()),
    }
    audit_path = output_root / "final_generation_seed_audit.json"
    audit_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest(output_root, project_root)
    print(
        json.dumps(
            {**report, "manifest": str(manifest)},
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
