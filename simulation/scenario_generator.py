"""Core generation and audit utilities for M2 simulation v2.

This module never reads the legacy simulation. Its only expression input is the
new Splatter pool produced by create_pool.R.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp


UINT32_MAX = np.iinfo(np.uint32).max


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        return "unavailable"


def package_versions() -> dict[str, str]:
    import anndata
    import h5py as h5py_module

    try:
        import pyarrow

        pyarrow_version = pyarrow.__version__
    except ImportError:
        pyarrow_version = "missing"
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "anndata": anndata.__version__,
        "h5py": h5py_module.__version__,
        "pyarrow": pyarrow_version,
    }


def require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for the mandated Parquet metadata files. "
            "Install it in the execution environment before formal generation."
        ) from exc


def rng_for(base_seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(stream)]))


def read_pool_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("Pool indices must be non-empty, unique, and sorted.")
    out = np.empty((len(indices), dataset.shape[1]), dtype=np.uint32)
    for start in range(0, len(indices), 256):
        stop = min(start + 256, len(indices))
        block = np.asarray(dataset[indices[start:stop], :])
        if np.any(block < 0) or np.any(block != np.floor(block)):
            raise AssertionError("Splatter pool contains invalid raw counts.")
        if block.max(initial=0) > UINT32_MAX:
            raise OverflowError("Splatter count exceeds uint32.")
        out[start:stop] = block.astype(np.uint32, copy=False)
    return out


def balanced_group_targets(n_cells: int, groups: list[str]) -> dict[str, int]:
    base, remainder = divmod(n_cells, len(groups))
    return {group: base + int(i < remainder) for i, group in enumerate(groups)}


def stratified_disjoint_split(
    pool_meta: pd.DataFrame,
    targets: dict[str, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng_for(seed, 1)
    ctrl: list[int] = []
    stim: list[int] = []
    for group, target in targets.items():
        available = pool_meta.index[pool_meta["true_group"] == group].to_numpy()
        if len(available) < 2 * target:
            raise AssertionError(
                f"Pool has too few {group} cells: {len(available)} < {2 * target}."
            )
        selected = rng.permutation(available)[: 2 * target]
        ctrl.extend(selected[:target].tolist())
        stim.extend(selected[target:].tolist())
    ctrl_idx = np.sort(np.asarray(ctrl, dtype=np.int64))
    stim_idx = np.sort(np.asarray(stim, dtype=np.int64))
    if np.intersect1d(ctrl_idx, stim_idx).size:
        raise AssertionError("Unpaired pool selections overlap.")
    return ctrl_idx, stim_idx


@dataclass(frozen=True)
class SpatialDesign:
    spot_ids: np.ndarray
    coordinates: np.ndarray
    group_names: list[str]
    group_centers: pd.DataFrame
    group_probabilities: np.ndarray
    slot_groups: np.ndarray
    spot_group_counts: np.ndarray


def create_spatial_design(
    config: dict[str, Any], targets: dict[str, int]
) -> SpatialDesign:
    d = config["dimensions"]
    s = config["spatial"]
    n_spots = int(d["n_spots"])
    cells_per_spot = int(d["cells_per_spot"])
    groups = list(targets)
    if int(s["grid_rows"]) * int(s["grid_cols"]) != n_spots:
        raise AssertionError("Grid dimensions do not equal n_spots.")

    rng = rng_for(config["seeds"]["seed_spatial"], 1)
    grid = np.asarray(
        [
            (x * float(s["spot_spacing"]), y * float(s["spot_spacing"]))
            for y in range(int(s["grid_rows"]))
            for x in range(int(s["grid_cols"]))
        ],
        dtype=np.float64,
    )
    if s["shuffle_coordinate_assignment"]:
        grid = grid[rng.permutation(n_spots)]
    spot_ids = np.asarray([f"spot_{i:04d}" for i in range(1, n_spots + 1)])

    center_indices = rng.choice(n_spots, size=len(groups), replace=False)
    centers_xy = grid[center_indices]
    centers = pd.DataFrame(
        {
            "true_group": groups,
            "center_spot_id": spot_ids[center_indices],
            "center_x": centers_xy[:, 0],
            "center_y": centers_xy[:, 1],
        }
    )
    diff = grid[:, None, :] - centers_xy[None, :, :]
    sq_distance = np.square(diff).sum(axis=2)
    sigma = float(s["rbf_sigma_gen"])
    alpha = float(s["alpha_mix"])
    weights = np.exp(-sq_distance / (2.0 * sigma * sigma))
    weights = (1.0 - alpha) * weights + alpha
    probabilities = weights / weights.sum(axis=1, keepdims=True)

    remaining = np.asarray([targets[g] for g in groups], dtype=np.int64)
    slot_groups = np.empty((n_spots, cells_per_spot), dtype=np.int16)
    for spot in range(n_spots):
        for slot in range(cells_per_spot):
            available = remaining > 0
            p = probabilities[spot].copy()
            p[~available] = 0.0
            if s["remaining_count_reweight"]:
                p *= remaining / remaining.sum()
            if p.sum() <= 0:
                p = available.astype(np.float64)
            p /= p.sum()
            chosen = int(rng.choice(len(groups), p=p))
            slot_groups[spot, slot] = chosen
            remaining[chosen] -= 1
    if np.any(remaining != 0):
        raise AssertionError(f"Unallocated group cells remain: {remaining.tolist()}")
    counts = np.stack(
        [(slot_groups == k).sum(axis=1) for k in range(len(groups))], axis=1
    ).astype(np.int16)
    if not np.all(counts.sum(axis=1) == cells_per_spot):
        raise AssertionError("A spot does not contain exactly cells_per_spot slots.")
    if not np.array_equal(counts.sum(axis=0), np.asarray(list(targets.values()))):
        raise AssertionError("Spot-group margins do not match sampled group targets.")
    return SpatialDesign(
        spot_ids=spot_ids,
        coordinates=grid,
        group_names=groups,
        group_centers=centers,
        group_probabilities=probabilities,
        slot_groups=slot_groups,
        spot_group_counts=counts,
    )


def assign_cells_to_spots(
    cell_ids: np.ndarray,
    true_groups: np.ndarray,
    design: SpatialDesign,
    seed: int,
    stream: int,
) -> pd.DataFrame:
    rng = rng_for(seed, stream)
    spot_for_cell = np.empty(len(cell_ids), dtype=object)
    slot_for_cell = np.empty(len(cell_ids), dtype=np.int16)
    for group_index, group in enumerate(design.group_names):
        cells = np.flatnonzero(true_groups == group)
        positions = np.argwhere(design.slot_groups == group_index)
        if len(cells) != len(positions):
            raise AssertionError(f"Group margin mismatch for {group}.")
        cells = rng.permutation(cells)
        positions = positions[rng.permutation(len(positions))]
        for cell_index, (spot_index, slot_index) in zip(cells, positions, strict=True):
            spot_for_cell[cell_index] = design.spot_ids[spot_index]
            slot_for_cell[cell_index] = slot_index + 1
    mapping = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "spot_id": spot_for_cell,
            "true_group": true_groups,
            "slot_within_spot": slot_for_cell,
        }
    )
    _assert_mapping(mapping, cell_ids, design)
    return mapping


def _assert_mapping(
    mapping: pd.DataFrame, cell_ids: np.ndarray, design: SpatialDesign
) -> None:
    if len(mapping) != len(cell_ids):
        raise AssertionError("Mapping row count differs from cell count.")
    if mapping["cell_id"].duplicated().any():
        raise AssertionError("A cell appears multiple times in mapping.")
    if set(mapping["cell_id"]) != set(cell_ids):
        raise AssertionError("Mapping cell IDs differ from expression cell IDs.")
    counts = mapping.groupby("spot_id", observed=True).size().reindex(design.spot_ids)
    if counts.isna().any() or not np.all(counts.to_numpy() == design.slot_groups.shape[1]):
        raise AssertionError("Every spot must contain exactly eight mapped cells.")


@dataclass(frozen=True)
class PerturbationTruth:
    gene_indices: np.ndarray
    log2fc: np.ndarray
    factors: np.ndarray
    directions: np.ndarray


def create_perturbation_truth(config: dict[str, Any]) -> PerturbationTruth:
    d = config["dimensions"]
    p = config["perturbation"]
    rng = rng_for(config["seeds"]["seed_perturb_gene"], 1)
    gene_indices = np.sort(
        rng.choice(
            int(d["n_genes"]), size=int(p["n_perturbed_genes"]), replace=False
        )
    )
    abs_fc = np.maximum(
        rng.normal(
            float(p["abs_log2fc_mean"]),
            float(p["abs_log2fc_sd"]),
            size=len(gene_indices),
        ),
        0.0,
    )
    directions = np.full(len(gene_indices), "down", dtype=object)
    up_positions = rng.choice(
        len(gene_indices), size=int(p["n_up_genes"]), replace=False
    )
    directions[up_positions] = "up"
    signs = np.where(directions == "up", 1.0, -1.0)
    log2fc = signs * abs_fc
    return PerturbationTruth(
        gene_indices=gene_indices,
        log2fc=log2fc,
        factors=np.exp2(log2fc),
        directions=directions,
    )


def treated_mask(
    mode: str,
    true_groups: np.ndarray,
    config: dict[str, Any],
    stream: int,
) -> np.ndarray:
    p = config["perturbation"]
    if mode == "all":
        return np.ones(len(true_groups), dtype=bool)
    if mode == "fraction":
        rng = rng_for(config["seeds"]["seed_perturb_cell"], stream)
        mask = np.zeros(len(true_groups), dtype=bool)
        n_treated = round(len(mask) * float(p["fraction_treated"]))
        mask[rng.choice(len(mask), size=n_treated, replace=False)] = True
        return mask
    if mode == "group":
        return true_groups == p["target_group"]
    raise ValueError(f"Unknown perturbation mode: {mode}")


def apply_perturbation(
    baseline: np.ndarray,
    treated: np.ndarray,
    truth: PerturbationTruth,
    eta: np.ndarray | None = None,
) -> np.ndarray:
    out = baseline.copy()
    treated_indices = np.flatnonzero(treated)
    if eta is not None and len(eta) != len(treated_indices):
        raise ValueError("eta must have one value per treated cell.")
    for start in range(0, len(truth.gene_indices), 64):
        stop = min(start + 64, len(truth.gene_indices))
        genes = truth.gene_indices[start:stop]
        source = baseline[np.ix_(treated_indices, genes)].astype(np.float64)
        if eta is None:
            factors = truth.factors[start:stop][None, :]
        else:
            factors = np.exp2(eta[:, None] + truth.log2fc[start:stop][None, :])
        modified = np.rint(source * factors)
        if modified.max(initial=0) > UINT32_MAX:
            raise OverflowError("Perturbed count exceeds uint32.")
        out[np.ix_(treated_indices, genes)] = modified.astype(np.uint32)
    return out


def mapping_spot_indices(mapping: pd.DataFrame, design: SpatialDesign) -> np.ndarray:
    lookup = {spot_id: i for i, spot_id in enumerate(design.spot_ids)}
    result = mapping["spot_id"].map(lookup)
    if result.isna().any():
        raise AssertionError("Unknown spot ID in mapping.")
    return result.to_numpy(dtype=np.int64)


def aggregate_counts(
    counts: np.ndarray, mapping: pd.DataFrame, design: SpatialDesign
) -> np.ndarray:
    spot_indices = mapping_spot_indices(mapping, design)
    output = np.zeros((len(design.spot_ids), counts.shape[1]), dtype=np.uint64)
    for start in range(0, len(design.spot_ids), 64):
        stop = min(start + 64, len(design.spot_ids))
        for spot_index in range(start, stop):
            rows = np.flatnonzero(spot_indices == spot_index)
            if len(rows) != design.slot_groups.shape[1]:
                raise AssertionError("Spot aggregation did not find exactly eight cells.")
            output[spot_index] = counts[rows].sum(axis=0, dtype=np.uint64)
    if output.max(initial=0) > UINT32_MAX:
        raise OverflowError("Aggregated spot count exceeds uint32.")
    return output.astype(np.uint32)


def independent_aggregation_audit(
    counts: np.ndarray,
    mapping: pd.DataFrame,
    observed_st: np.ndarray,
    design: SpatialDesign,
) -> dict[str, Any]:
    spot_indices = mapping_spot_indices(mapping, design)
    membership = sp.csr_matrix(
        (
            np.ones(len(counts), dtype=np.uint8),
            (np.arange(len(counts), dtype=np.int64), spot_indices),
        ),
        shape=(len(counts), len(design.spot_ids)),
    )
    expected = membership.T @ counts
    expected = np.asarray(expected, dtype=np.uint32)
    difference = expected.astype(np.int64) - observed_st.astype(np.int64)
    return {
        "difference_nnz": int(np.count_nonzero(difference)),
        "max_absolute_difference": int(np.abs(difference).max(initial=0)),
        "exact": bool(np.count_nonzero(difference) == 0),
    }


def write_raw_h5ad(
    path: Path,
    counts: np.ndarray,
    obs_names: Iterable[str],
    gene_ids: np.ndarray,
    obs_columns: dict[str, Iterable[Any]],
    compression_level: int,
    spatial: np.ndarray | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    if counts.dtype != np.uint32:
        raise AssertionError("H5AD raw count matrix must be uint32 integers.")
    obs_index = pd.Index(np.asarray(list(obs_names), dtype=str), name="id")
    if obs_index.has_duplicates:
        raise AssertionError("Duplicate observation IDs.")
    obs = pd.DataFrame(obs_columns, index=obs_index)
    var_index = pd.Index(np.asarray(gene_ids, dtype=str), name="gene_id")
    if var_index.has_duplicates:
        raise AssertionError("Duplicate gene IDs.")
    var = pd.DataFrame(index=var_index)
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.uns["matrix_type"] = "raw_integer_counts"
    adata.uns["n_genes_formal_evaluation"] = int(len(gene_ids))
    if spatial is not None:
        adata.obsm["spatial"] = np.asarray(spatial, dtype=np.float64)
    adata.write_h5ad(
        path,
        compression="gzip",
        compression_opts=int(compression_level),
    )


def baseline_diagnostics(
    sc_ctrl: np.ndarray,
    sc_stim: np.ndarray,
    st_ctrl: np.ndarray,
    st_stim: np.ndarray,
    perturbed_indices: np.ndarray,
) -> dict[str, Any]:
    mean_ctrl = sc_ctrl.mean(axis=0, dtype=np.float64)
    mean_stim = sc_stim.mean(axis=0, dtype=np.float64)
    both_zero = (mean_ctrl == 0) & (mean_stim == 0)
    invalid = (mean_ctrl == 0) & (mean_stim > 0)
    ratio = np.ones_like(mean_ctrl)
    valid = mean_ctrl > 0
    ratio[valid] = mean_stim[valid] / mean_ctrl[valid]
    ratio[invalid] = np.nan

    perturbed = np.zeros(st_ctrl.shape[1], dtype=bool)
    perturbed[perturbed_indices] = True

    def rmse(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
        delta = prediction[:, mask].astype(np.float64) - truth[:, mask].astype(np.float64)
        return float(np.sqrt(np.mean(np.square(delta))))

    identity = st_ctrl
    ratio_prediction = st_ctrl.astype(np.float64) * ratio[None, :]
    energy_delta = st_stim.astype(np.float64) - st_ctrl.astype(np.float64)
    result = {
        "identity_rmse_all": rmse(identity, st_stim, np.ones_like(perturbed)),
        "identity_rmse_perturbed": rmse(identity, st_stim, perturbed),
        "identity_rmse_nonperturbed": rmse(identity, st_stim, ~perturbed),
        "perturbation_energy_all": float(np.mean(np.square(energy_delta))),
        "ctrl_mean_zero_genes": int((mean_ctrl == 0).sum()),
        "both_mean_zero_genes": int(both_zero.sum()),
        "stim_positive_ctrl_zero_genes": int(invalid.sum()),
    }
    if invalid.any():
        result.update(
            {
                "mean_ratio_rmse_all": None,
                "mean_ratio_rmse_perturbed": None,
                "mean_ratio_rmse_nonperturbed": None,
            }
        )
    else:
        result.update(
            {
                "mean_ratio_rmse_all": rmse(
                    ratio_prediction, st_stim, np.ones_like(perturbed)
                ),
                "mean_ratio_rmse_perturbed": rmse(
                    ratio_prediction, st_stim, perturbed
                ),
                "mean_ratio_rmse_nonperturbed": rmse(
                    ratio_prediction, st_stim, ~perturbed
                ),
            }
        )
    return result


def constant_multiplier_audit(
    st_ctrl: np.ndarray,
    st_stim: np.ndarray,
    truth: PerturbationTruth,
    cells_per_spot: int,
) -> dict[str, Any]:
    factors = np.ones(st_ctrl.shape[1], dtype=np.float64)
    factors[truth.gene_indices] = truth.factors
    error = st_stim.astype(np.float64) - st_ctrl.astype(np.float64) * factors[None, :]
    nonperturbed = np.ones(st_ctrl.shape[1], dtype=bool)
    nonperturbed[truth.gene_indices] = False
    bound = cells_per_spot / 2.0
    perturbed_abs = np.abs(error[:, truth.gene_indices])
    return {
        "formula": "E = Y_stim - f_g * Y_ctrl",
        "max_absolute_error": float(np.abs(error).max(initial=0)),
        "mean_absolute_error": float(np.abs(error).mean()),
        "nonperturbed_exact_zero_fraction": float(
            np.mean(error[:, nonperturbed] == 0)
        ),
        "rounding_bound": bound,
        "rounding_bound_holds": bool(np.all(perturbed_abs <= bound + 1e-9)),
        "max_perturbed_absolute_error": float(perturbed_abs.max(initial=0)),
    }


def checksums_for_directory(directory: Path) -> None:
    checksum_path = directory / "checksums.sha256"
    lines = []
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        if path == checksum_path:
            continue
        lines.append(f"{sha256(path)}  {path.name}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_frozen_file(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Frozen file differs: {destination}")
        return
    shutil.copy2(source, destination)
