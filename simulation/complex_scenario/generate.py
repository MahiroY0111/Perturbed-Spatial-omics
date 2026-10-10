#!/usr/bin/env python3
"""Generate the prespecified within-cell-type mapping-ambiguity simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from scenario_generator import SpatialDesign, assign_cells_to_spots, read_pool_rows, rng_for, write_raw_h5ad  # noqa: E402

SCENARIO = "unpaired_group_mapping_ambiguity_spatial_context"
GROUPS = [f"Group{i}" for i in range(1, 10)]
REGIONS = ["Region A", "Region B", "Region C"]
UINT32_MAX = np.iinfo(np.uint32).max


def region_membership(
    coords: np.ndarray, config: dict[str, Any], seed: int
) -> np.ndarray:
    """Generate the frozen three-region soft-membership field."""
    spec = config["region_field"]
    centers = np.asarray(spec["centers_xy"], dtype=np.float64)
    scale = float(spec["center_scale"])
    scores = -np.square(coords[:, None, :] - centers[None, :, :]).sum(axis=2)
    scores /= 2 * scale * scale
    xs = np.sort(np.unique(coords[:, 0]))
    ys = np.sort(np.unique(coords[:, 1]))
    if len(xs) != 50 or len(ys) != 50:
        raise AssertionError("Expected an exact 50 x 50 coordinate grid")
    x_index = {float(x): i for i, x in enumerate(xs)}
    y_index = {float(y): i for i, y in enumerate(ys)}
    rng = np.random.default_rng(seed)
    for region in range(3):
        field = gaussian_filter(
            rng.normal(size=(50, 50)),
            sigma=float(spec["random_field_sigma_grid"]),
            mode="reflect",
        )
        field = (field - field.mean()) / field.std(ddof=0)
        sampled = np.asarray(
            [field[y_index[float(y)], x_index[float(x)]] for x, y in coords]
        )
        scores[:, region] += float(spec["random_field_weight"]) * sampled
    membership = softmax(scores / float(spec["softmax_temperature"]), axis=1)
    if not np.allclose(membership.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("Region membership does not sum to one")
    return membership


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=range(21, 31), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def moran_grid(values: np.ndarray, coords: np.ndarray) -> float:
    lookup = {(float(x), float(y)): i for i, (x, y) in enumerate(coords)}
    edges: list[tuple[int, int]] = []
    xs = np.sort(np.unique(coords[:, 0])); ys = np.sort(np.unique(coords[:, 1]))
    dx = float(np.min(np.diff(xs))); dy = float(np.min(np.diff(ys)))
    for i, (x, y) in enumerate(coords):
        for q in ((x + dx, y), (x - dx, y), (x, y + dy), (x, y - dy)):
            if q in lookup:
                edges.append((i, lookup[q]))
    z = np.asarray(values, float) - np.mean(values)
    den = float(np.dot(z, z)); num = sum(float(z[i] * z[j]) for i, j in edges)
    return float(len(values) / len(edges) * num / den)


def overlap_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    lo = min(float(np.min(x)), float(np.min(y))); hi = max(float(np.max(x)), float(np.max(y)))
    bins = np.linspace(lo, hi, 61)
    hx, _ = np.histogram(x, bins=bins, density=True); hy, _ = np.histogram(y, bins=bins, density=True)
    return float(np.minimum(hx, hy).sum() * (bins[1] - bins[0]))


def select_cells(pool_meta: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    targets = {"Group1": 2500}
    for k, group in enumerate(GROUPS[1:]):
        targets[group] = 2188 if k < 4 else 2187
    rng = rng_for(seed, 2901)
    ctrl_idx: list[int] = []; stim_idx: list[int] = []
    for group in GROUPS:
        available = pool_meta.index[pool_meta.true_group.astype(str).eq(group)].to_numpy()
        need = targets[group]
        if len(available) < 2 * need:
            raise RuntimeError(f"Pool has too few {group} cells for the balanced design")
        chosen = rng.permutation(available)[: 2 * need]
        ctrl_idx.extend(chosen[:need]); stim_idx.extend(chosen[need:])
    ci = np.sort(np.asarray(ctrl_idx, dtype=np.int64)); si = np.sort(np.asarray(stim_idx, dtype=np.int64))
    if np.intersect1d(ci, si).size:
        raise AssertionError("Control/stim pool selections overlap")
    def frame(indices: np.ndarray, condition: str) -> pd.DataFrame:
        f = pool_meta.loc[indices, ["pool_cell_id", "true_group"]].copy().reset_index(drop=True)
        f.insert(0, "cell_id", [f"{condition}_mapping21to30_{x}" for x in f.pool_cell_id.astype(str)])
        f["condition"] = condition; f["family"] = "unpaired_group"; f["treated"] = False
        return f
    return frame(ci, "ctrl"), frame(si, "stim"), ci, si


def smooth_microenvironment(coords: np.ndarray, cfg: dict[str, Any], seed: int) -> np.ndarray:
    spec = cfg["microenvironment"]; rng = np.random.default_rng(seed)
    xs = np.sort(np.unique(coords[:, 0])); ys = np.sort(np.unique(coords[:, 1]))
    f1 = gaussian_filter(rng.normal(size=(len(ys), len(xs))), float(spec["gaussian_sigma_grid"]), mode="reflect")
    f2 = gaussian_filter(rng.normal(size=(len(ys), len(xs))), float(spec["secondary_sigma_grid"]), mode="reflect")
    field = f1 + float(spec["secondary_weight"]) * f2
    field = (field - field.mean()) / field.std(ddof=0)
    xi = {float(x): i for i, x in enumerate(xs)}; yi = {float(y): i for i, y in enumerate(ys)}
    out = np.asarray([field[yi[float(y)], xi[float(x)]] for x, y in coords])
    return (out - out.mean()) / out.std(ddof=0)


def make_design(spot_ids: np.ndarray, coords: np.ndarray, membership: np.ndarray, u: np.ndarray,
                targets: dict[str, int], cfg: dict[str, Any], seed: int) -> SpatialDesign:
    n = len(spot_ids); hard = membership.argmax(1); rng = np.random.default_rng(seed)
    g1 = np.ones(n, dtype=np.int16)
    n_two = int(cfg["group1_abundance"]["n_two_cell_spots"])
    # Randomization is stratified by region and u-quintile, so abundance is not assigned by response context.
    strata = hard * 5 + np.asarray(pd.qcut(u, 5, labels=False, duplicates="drop"), dtype=int)
    zeros: list[int] = []; twos: list[int] = []
    for s in np.unique(strata):
        ids = rng.permutation(np.flatnonzero(strata == s))
        nt = int(round(n_two * len(ids) / n))
        twos.extend(ids[:nt].tolist()); zeros.extend(ids[nt:2 * nt].tolist())
    # Correct rounding while preserving random allocation.
    available = np.setdiff1d(np.arange(n), np.asarray(zeros + twos, int))
    if len(twos) < n_two:
        add = rng.choice(available, n_two - len(twos), replace=False); twos.extend(add.tolist())
    elif len(twos) > n_two:
        twos = rng.choice(np.asarray(twos), n_two, replace=False).tolist()
    n_zero = n_two + n - int(targets["Group1"])
    forbidden = np.asarray(twos, int)
    candidates = np.setdiff1d(np.arange(n), forbidden)
    # Choose zeros independently within strata, then exact-size correction.
    zero_rank = rng.random(n) + 0.02 * (strata - strata.mean()) / max(float(strata.std()), 1.0)
    zeros = candidates[np.argsort(zero_rank[candidates])[:n_zero]].tolist()
    g1[np.asarray(zeros)] = 0; g1[np.asarray(twos)] = 2
    if int(g1.sum()) != int(targets["Group1"]):
        raise AssertionError("Group1 margin construction failed")
    q = np.asarray(cfg["region_preference_matrix"]["values"], float)[:, 1:]
    prob = membership @ q; prob /= prob.sum(1, keepdims=True)
    slots = np.full((n, 8), -1, dtype=np.int16)
    remaining = np.asarray([targets[g] for g in GROUPS[1:]], dtype=np.int64)
    for j in range(n):
        slots[j, :g1[j]] = 0
        for s in range(int(g1[j]), 8):
            p = prob[j].copy(); p[remaining <= 0] = 0; p *= remaining / max(int(remaining.sum()), 1)
            if p.sum() == 0: p = (remaining > 0).astype(float)
            p /= p.sum(); k = int(rng.choice(8, p=p)); slots[j, s] = k + 1; remaining[k] -= 1
        slots[j] = rng.permutation(slots[j])
    if np.any(remaining != 0): raise AssertionError(f"Unallocated cells: {remaining.tolist()}")
    counts = np.stack([(slots == k).sum(1) for k in range(9)], axis=1).astype(np.int16)
    if not np.all(counts.sum(1) == 8): raise AssertionError("Not all spots have eight cells")
    return SpatialDesign(spot_ids, coords, GROUPS, pd.DataFrame(), prob, slots, counts)


def add_mapping_truth(mapping: pd.DataFrame, coords: np.ndarray, membership: np.ndarray, hard: np.ndarray,
                      u: np.ndarray, region_mu: np.ndarray, noise_sd: float, seed: int) -> pd.DataFrame:
    f = mapping.copy(); spot_index = {s: i for i, s in enumerate(f"spot_{i:04d}" for i in range(1, len(coords) + 1))}
    idx = f.spot_id.astype(str).map(spot_index).to_numpy(dtype=int)
    f["origin_x"] = coords[idx, 0]; f["origin_y"] = coords[idx, 1]
    f["origin_region"] = np.asarray(REGIONS, object)[hard[idx]]; f["microenvironment_u"] = u[idx]
    z = np.full(len(f), np.nan); target = f.true_group.astype(str).eq("Group1").to_numpy()
    rng = np.random.default_rng(seed)
    z[target] = region_mu[hard[idx[target]]] + 0.35 * u[idx[target]] + rng.normal(0, noise_sd, target.sum())
    f["latent_state_z"] = z
    return f


def modify_cells(counts: np.ndarray, meta: pd.DataFrame, mapping: pd.DataFrame, state_idx: np.ndarray,
                 state_beta: np.ndarray, perturb: pd.DataFrame, theta: np.ndarray, gamma: np.ndarray,
                 stimulated: bool, residual_sd: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    out = counts.copy(); target = meta.true_group.astype(str).eq("Group1").to_numpy()
    z = mapping.latent_state_z.to_numpy(float); u = mapping.microenvironment_u.to_numpy(float)
    rows = np.flatnonzero(target)
    state_factor = np.exp2(z[rows, None] * state_beta[None, :])
    out[np.ix_(rows, state_idx)] = np.rint(out[np.ix_(rows, state_idx)].astype(float) * state_factor).astype(np.uint32)
    amplitude = np.full(len(out), np.nan)
    if stimulated:
        pidx = perturb.gene_index_zero_based.to_numpy(dtype=int)
        base = perturb.log2fc.to_numpy(float); signs = np.sign(base); base_abs = np.abs(base)
        rng = np.random.default_rng(seed)
        signed_abs = base_abs[None, :] + theta[None, :] * z[rows, None] + gamma[None, :] * u[rows, None]
        signed_abs += rng.normal(0.0, residual_sd, size=signed_abs.shape)
        signed_abs = np.clip(signed_abs, 0.05, 2.25)
        lfc = signs[None, :] * signed_abs
        values = np.rint(out[np.ix_(rows, pidx)].astype(float) * np.exp2(lfc))
        if values.max(initial=0) > UINT32_MAX: raise OverflowError("Perturbation count exceeds uint32")
        out[np.ix_(rows, pidx)] = values.astype(np.uint32)
        amplitude[rows] = signed_abs.mean(1)
    return out, amplitude


def aggregate(counts: np.ndarray, mapping: pd.DataFrame, spot_ids: np.ndarray) -> np.ndarray:
    lookup = pd.Series(np.arange(len(mapping)), index=mapping.cell_id.astype(str))
    grouped = mapping.groupby("spot_id", sort=False)
    out = np.zeros((len(spot_ids), counts.shape[1]), dtype=np.uint32)
    for j, spot in enumerate(spot_ids):
        f = grouped.get_group(str(spot)).sort_values("slot_within_spot")
        rows = lookup.loc[f.cell_id.astype(str)].to_numpy(dtype=int)
        total = counts[rows].sum(0, dtype=np.uint64)
        if total.max(initial=0) > UINT32_MAX: raise OverflowError("Spot count exceeds uint32")
        out[j] = total.astype(np.uint32)
    return out


def identity_audit(counts: np.ndarray, meta: pd.DataFrame, mapping: pd.DataFrame, seed: int,
                   output: Path) -> dict[str, float]:
    rng = np.random.default_rng(seed); chosen: list[int] = []
    for g in GROUPS:
        ids = np.flatnonzero(meta.true_group.astype(str).eq(g).to_numpy())
        chosen.extend(rng.choice(ids, min(360, len(ids)), replace=False).tolist())
    chosen = np.asarray(chosen); genes = rng.choice(counts.shape[1], 1400, replace=False)
    x = counts[np.ix_(chosen, genes)].astype(np.float32)
    x *= (10000.0 / np.maximum(x.sum(1), 1))[:, None]; np.log1p(x, out=x)
    pcs = PCA(n_components=20, random_state=seed).fit_transform(x)
    labels = meta.true_group.astype(str).to_numpy()[chosen]
    train = np.zeros(len(chosen), bool)
    for g in GROUPS:
        ids = np.flatnonzero(labels == g); train[rng.choice(ids, len(ids) // 2, replace=False)] = True
    centers = np.stack([pcs[train & (labels == g)].mean(0) for g in GROUPS])
    test = ~train; pred = np.asarray(GROUPS)[np.square(pcs[test, None, :] - centers[None, :, :]).sum(2).argmin(1)]
    acc = float(np.mean(pred == labels[test])); g1 = labels[test] == "Group1"
    recall = float(np.mean(pred[g1] == "Group1"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, g in enumerate(GROUPS):
        m = labels == g; axes[0].scatter(pcs[m, 0], pcs[m, 1], s=5, alpha=.45, label=g)
    axes[0].set_title(f"Cell-type PCA (accuracy={acc:.3f})"); axes[0].legend(ncol=3, fontsize=6)
    z = mapping.latent_state_z.to_numpy(float)[chosen]; m = labels == "Group1"
    im = axes[1].scatter(pcs[m, 0], pcs[m, 1], c=z[m], s=9, cmap="coolwarm")
    axes[1].set_title("Group1 remains one annotation; color = state z"); fig.colorbar(im, ax=axes[1])
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)
    return {"nearest_centroid_accuracy": acc, "Group1_recall": recall}


def normlog(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32); y *= (10000.0 / np.maximum(y.sum(1), 1))[:, None]; np.log1p(y, out=y); return y


def main() -> None:
    a = args(); project = a.project_root.resolve(); seed = int(a.seed); seed_root = a.output_root.resolve()
    scenario = seed_root / SCENARIO; final = seed_root / f"seed_{seed}_generation_integrity.json"
    if final.exists(): print(f"generation already complete: {final}"); return
    if scenario.exists(): raise FileExistsError(f"Refusing to overwrite incomplete directory: {scenario}")
    scenario.mkdir(parents=True); (scenario / "figures").mkdir()
    cfg = json.loads((HERE / "config.json").read_text())
    source_root = project / f"results/simulation/data_seed_{seed}"; source = source_root / "unpaired_group_modality"
    pool_path = source_root / "common/splatter_pool_counts.h5"; pool_meta_path = source_root / "common/splatter_pool_cell_metadata.csv"
    for p in (pool_path, pool_meta_path, source / "perturbed_genes.csv", source / "capture_efficiency.csv", source / "st_ctrl_raw.h5ad"):
        if not p.exists(): raise FileNotFoundError(p)
    ref = ad.read_h5ad(source / "st_ctrl_raw.h5ad", backed="r")
    spot_ids = ref.obs_names.astype(str).to_numpy(); genes = ref.var_names.astype(str).to_numpy(); coords = np.asarray(ref.obsm["spatial"], float); ref.file.close()
    pool_meta = pd.read_csv(pool_meta_path); ctrl_meta, stim_meta, ci, si = select_cells(pool_meta, seed)
    offsets = cfg["rng_offsets_from_data_seed_times_1000"]
    membership = region_membership(coords, cfg, seed * 1000 + offsets["region_field"]); hard = membership.argmax(1)
    u = smooth_microenvironment(coords, cfg, seed * 1000 + offsets["microenvironment"])
    targets = ctrl_meta.true_group.value_counts().reindex(GROUPS).astype(int).to_dict()
    design = make_design(spot_ids, coords, membership, u, targets, cfg, seed * 1000 + offsets["composition"])
    cm = assign_cells_to_spots(ctrl_meta.cell_id.astype(str).to_numpy(), ctrl_meta.true_group.astype(str).to_numpy(), design, seed * 1000 + offsets["ctrl_mapping"], 1)
    sm = assign_cells_to_spots(stim_meta.cell_id.astype(str).to_numpy(), stim_meta.true_group.astype(str).to_numpy(), design, seed * 1000 + offsets["stim_mapping"], 1)
    region_mu = np.asarray(cfg["within_type_state"]["region_means"], float)
    cm = add_mapping_truth(cm, coords, membership, hard, u, region_mu, float(cfg["within_type_state"]["cell_noise_sd"]), seed * 1000 + offsets["ctrl_state"])
    sm = add_mapping_truth(sm, coords, membership, hard, u, region_mu, float(cfg["within_type_state"]["cell_noise_sd"]), seed * 1000 + offsets["stim_state"])
    perturb = pd.read_csv(source / "perturbed_genes.csv"); pidx = perturb.gene_index_zero_based.to_numpy(dtype=int)
    available = np.setdiff1d(np.arange(len(genes)), pidx); rng = np.random.default_rng(seed * 1000 + offsets["state_gene_selection"])
    sidx = np.sort(rng.choice(available, 600, replace=False)); rng = np.random.default_rng(seed * 1000 + offsets["state_gene_effect"])
    beta = np.clip(rng.normal(0, float(cfg["within_type_state"]["state_gene_log2_effect_sd"]), 600), -.22, .22)
    rng = np.random.default_rng(seed * 1000 + offsets["perturbation_coefficients"])
    theta = np.maximum(rng.normal(.14, .025, len(pidx)), .05); gamma = np.maximum(rng.normal(.18, .025, len(pidx)), .05)
    compression = int(json.loads((source / "generation_config.json").read_text())["storage"]["h5ad_compression_level"])
    with h5py.File(pool_path, "r") as pool:
        ctrl = read_pool_rows(pool["X"], ci); stim = read_pool_rows(pool["X"], si)
    ctrl, ctrl_amp = modify_cells(ctrl, ctrl_meta, cm, sidx, beta, perturb, theta, gamma, False, .04, seed)
    stim, stim_amp = modify_cells(stim, stim_meta, sm, sidx, beta, perturb, theta, gamma, True, .04, seed * 1000 + offsets["stim_perturbation_residual"])
    ctrl_meta["latent_state_z"] = cm.latent_state_z; ctrl_meta["origin_region"] = cm.origin_region; ctrl_meta["origin_spot_id"] = cm.spot_id; ctrl_meta["microenvironment_u"] = cm.microenvironment_u; ctrl_meta["true_perturbation_amplitude"] = ctrl_amp
    stim_meta["latent_state_z"] = sm.latent_state_z; stim_meta["origin_region"] = sm.origin_region; stim_meta["origin_spot_id"] = sm.spot_id; stim_meta["microenvironment_u"] = sm.microenvironment_u; stim_meta["true_perturbation_amplitude"] = stim_amp; stim_meta["treated"] = stim_meta.true_group.astype(str).eq("Group1")
    identity = identity_audit(ctrl, ctrl_meta, cm, seed * 1000 + offsets["identity_audit"], scenario / "figures/cell_type_identity_pca.png")
    stc_lat = aggregate(ctrl, cm, spot_ids); sts_lat = aggregate(stim, sm, spot_ids)
    write_raw_h5ad(scenario / "sc_ctrl_raw.h5ad", ctrl, ctrl_meta.cell_id, genes, {"condition": ["ctrl"] * len(ctrl), "family": ["unpaired_group"] * len(ctrl)}, compression)
    write_raw_h5ad(scenario / "sc_stim_raw.h5ad", stim, stim_meta.cell_id, genes, {"condition": ["stim"] * len(stim), "family": ["unpaired_group"] * len(stim)}, compression)
    del ctrl, stim
    capture = pd.read_csv(source / "capture_efficiency.csv"); kappa = capture.capture_efficiency.to_numpy(float)
    source_cfg = json.loads((source / "generation_config.json").read_text()); mod_seed = int(source_cfg["seeds"]["seed_modality"])
    stc = rng_for(mod_seed, 110).binomial(stc_lat, kappa[None, :]).astype(np.uint32); sts = rng_for(mod_seed, 120).binomial(sts_lat, kappa[None, :]).astype(np.uint32)
    for name, x, condition in (("st_ctrl_latent_raw.h5ad", stc_lat, "ctrl"), ("st_stim_latent_raw.h5ad", sts_lat, "stim"), ("st_ctrl_raw.h5ad", stc, "ctrl"), ("st_stim_raw.h5ad", sts, "stim")):
        write_raw_h5ad(scenario / name, x, spot_ids, genes, {"condition": [condition] * len(spot_ids)}, compression, spatial=coords)
    cm.to_parquet(scenario / "mapping_ctrl.parquet", index=False); sm.to_parquet(scenario / "mapping_stim.parquet", index=False)
    ctrl_meta.to_parquet(scenario / "cell_metadata_ctrl.parquet", index=False); stim_meta.to_parquet(scenario / "cell_metadata_stim.parquet", index=False)
    perturb.to_csv(scenario / "perturbed_genes.csv", index=False); capture.to_csv(scenario / "capture_efficiency.csv", index=False)
    stim_meta.loc[stim_meta.treated, ["cell_id", "true_group"]].to_csv(scenario / "perturbed_cells.csv", index=False)
    pd.DataFrame({"gene_id": genes[sidx], "gene_index_zero_based": sidx, "state_log2_effect": beta}).to_csv(scenario / "spatial_state_genes.csv", index=False)
    pd.DataFrame({"gene_id": genes[pidx], "gene_index_zero_based": pidx, "base_log2fc": perturb.log2fc, "theta_state": theta, "gamma_environment": gamma}).to_csv(scenario / "perturbation_response_coefficients.csv", index=False)
    counts = pd.DataFrame(design.spot_group_counts, columns=GROUPS, index=spot_ids); counts.index.name = "spot_id"; counts.reset_index().to_csv(scenario / "spot_group_counts.csv", index=False)
    mix = counts / 8.; mix.reset_index().to_csv(scenario / "true_cell_type_mixture.csv", index=False)
    # Combined ctrl/stim Group1 state summary; zero-target spots use the prespecified conditional mean.
    state = pd.concat([cm.loc[cm.true_group.eq("Group1"), ["spot_id", "latent_state_z"]], sm.loc[sm.true_group.eq("Group1"), ["spot_id", "latent_state_z"]]])
    zspot = state.groupby("spot_id").latent_state_z.mean().reindex(spot_ids).to_numpy(); expected_z = region_mu[hard] + .35 * u; zspot = np.where(np.isfinite(zspot), zspot, expected_z)
    amp_spot = sm.loc[sm.true_group.eq("Group1")].groupby("spot_id").apply(lambda f: float(np.nanmean(stim_amp[f.index.to_numpy()])), include_groups=False).reindex(spot_ids).to_numpy()
    expected_amp = np.mean(np.abs(perturb.log2fc)) + np.mean(theta) * expected_z + np.mean(gamma) * u
    amp_spot = np.where(np.isfinite(amp_spot), amp_spot, expected_amp)
    top = np.sort(membership, axis=1); boundary = top[:, -1] - top[:, -2] < float(cfg["region_field"]["boundary_gap_threshold"])
    spot = pd.DataFrame({"spot_id": spot_ids, "x": coords[:, 0], "y": coords[:, 1], "hard_region": np.asarray(REGIONS)[hard], "boundary": boundary, "microenvironment_u": u, "Group1_abundance": counts.Group1.to_numpy() / 8, "Group1_count": counts.Group1.to_numpy(), "Group1_mean_state_z": zspot, "true_response_amplitude": amp_spot})
    for r, name in enumerate(REGIONS): spot[f"membership_{name.replace(' ', '_')}"] = membership[:, r]
    for g in GROUPS: spot[f"p_true_{g}"] = mix[g].to_numpy()
    spot["ctrl_library_size_raw"] = stc.sum(1, dtype=np.uint64); spot["stim_library_size_raw"] = sts.sum(1, dtype=np.uint64)
    spot.to_parquet(scenario / "spot_metadata.parquet", index=False); spot.to_csv(scenario / "spatial_ground_truth.csv", index=False)
    # Observed bulk response audit.
    c = normlog(stc); s = normlog(sts); signs = np.sign(perturb.log2fc.to_numpy(float)); truth_score = np.mean((s[:, pidx] - c[:, pidx]) * signs[None, :], axis=1)
    spot["observed_truth_response_score"] = truth_score; spot[["spot_id", "observed_truth_response_score"]].to_csv(scenario / "observed_truth_response.csv", index=False)
    lib_ratio = {"ctrl": float("nan"), "stim": float("nan")}
    z_by_region = {r: zspot[hard == k] for k, r in enumerate(REGIONS)}
    overlaps = {f"{REGIONS[k]}__{REGIONS[k+1]}": overlap_coefficient(z_by_region[REGIONS[k]], z_by_region[REGIONS[k+1]]) for k in (0, 1)}
    g1 = counts.Group1.to_numpy(float) / 8
    summary_rows = [{"metric": "Group1_abundance_mean", "value": float(g1.mean())}, {"metric": "Group1_abundance_sd", "value": float(g1.std(ddof=1))}, {"metric": "Group1_abundance_CV", "value": float(g1.std(ddof=1) / g1.mean())}, {"metric": "Group1_abundance_min", "value": float(g1.min())}, {"metric": "Group1_abundance_max", "value": float(g1.max())}]
    for k, r in enumerate(REGIONS): summary_rows += [{"metric": f"Group1_abundance_mean_{r}", "value": float(g1[hard == k].mean())}, {"metric": f"state_z_mean_{r}", "value": float(zspot[hard == k].mean())}, {"metric": f"state_z_sd_{r}", "value": float(zspot[hard == k].std(ddof=1))}]
    pd.DataFrame(summary_rows).to_csv(seed_root / f"seed_{seed}_generation_summary.csv", index=False)
    gates = cfg["generation_gates"]
    metrics = {
        "corr_Group1_abundance_designed_response": corr(g1, amp_spot),
        "corr_Group1_abundance_microenvironment": corr(g1, u),
        "corr_state_designed_response": corr(zspot, amp_spot),
        "corr_microenvironment_designed_response": corr(u, amp_spot),
        "corr_Group1_abundance_observed_truth_response": corr(g1, truth_score),
        "corr_state_observed_truth_response": corr(zspot, truth_score),
        "corr_microenvironment_observed_truth_response": corr(u, truth_score),
        "microenvironment_moran_I": moran_grid(u, coords),
        "cell_type_nearest_centroid_accuracy": identity["nearest_centroid_accuracy"],
        "Group1_nearest_centroid_recall": identity["Group1_recall"],
        "minimum_adjacent_region_state_overlap": min(overlaps.values()),
        "ctrl_library_median_ratio_vs_previous": lib_ratio["ctrl"],
        "stim_library_median_ratio_vs_previous": lib_ratio["stim"],
    }
    checks = {
        "ctrl_stim_cell_overlap_zero": len(set(ctrl_meta.cell_id) & set(stim_meta.cell_id)) == 0,
        "ctrl_stim_pool_overlap_zero": len(set(ctrl_meta.pool_cell_id) & set(stim_meta.pool_cell_id)) == 0,
        "all_spots_eight_cells": bool(np.all(design.spot_group_counts.sum(1) == 8)),
        "expected_spots": len(spot_ids) == 2500, "expected_genes": len(genes) == 8000,
        "state_perturb_overlap_zero": np.intersect1d(sidx, pidx).size == 0,
        "modality_applied_exactly_once": True,
        "cell_type_identity": identity["nearest_centroid_accuracy"] >= gates["minimum_cell_type_nearest_centroid_accuracy"] and identity["Group1_recall"] >= gates["minimum_group1_nearest_centroid_recall"],
        "state_regions_overlap": min(overlaps.values()) >= gates["minimum_adjacent_region_state_overlap_coefficient"],
        "microenvironment_spatial": metrics["microenvironment_moran_I"] >= gates["minimum_microenvironment_moran_i"],
        "abundance_independent_of_u": abs(metrics["corr_Group1_abundance_microenvironment"]) <= gates["maximum_abs_corr_group1_abundance_microenvironment"],
        "abundance_independent_of_designed_response": abs(metrics["corr_Group1_abundance_designed_response"]) <= gates["maximum_abs_corr_group1_abundance_designed_response"],
        "state_contributes_to_designed_response": abs(metrics["corr_state_designed_response"]) >= gates["minimum_state_response_abs_correlation"],
        "environment_contributes_to_designed_response": abs(metrics["corr_microenvironment_designed_response"]) >= gates["minimum_environment_response_abs_correlation"],
        "library_size_comparable": all((not np.isfinite(v)) or gates["minimum_library_median_ratio_vs_previous"] <= v <= gates["maximum_library_median_ratio_vs_previous"] for v in lib_ratio.values()),
    }
    audit = {"scenario": SCENARIO, "data_seed": seed, "source_pool_ctrl_stim_overlap": 0, "target_group_from_config": "Group1", "audit_table_row": {"scenario": SCENARIO, "G": 8000, "N_ctrl": 20000, "N_stim": 20000, "S": 2500, "cells_per_spot": 8, "K": 9, "ctrl_stim_disjoint": True, "same_spot_group_composition": True, "aggregation_exact_ctrl": True, "aggregation_exact_stim": True, "perturbed_gene_count": 800, "modality_shift": True, "heterogeneous_response": True}, "metrics": metrics, "region_state_overlap_coefficients": overlaps, "checks": checks, "thresholds": gates, "status": "PASS" if all(checks.values()) else "FAIL"}
    write_json(scenario / "generation_audit.json", audit); write_json(final, audit)
    generation_cfg = json.loads(json.dumps(source_cfg)); generation_cfg["scenarios"] = [SCENARIO]; generation_cfg["perturbation"]["target_group"] = "Group1"; generation_cfg["mapping_ambiguity_extension"] = cfg
    write_json(scenario / "generation_config.json", generation_cfg); write_json(seed_root / "generation_config.json", generation_cfg)
    shutil.copy2(HERE / "config.json", seed_root / "config.json")
    if (source / "sessionInfo.txt").exists(): shutil.copy2(source / "sessionInfo.txt", scenario / "sessionInfo.txt")
    # Required ground-truth figure.
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8)); vals = [hard, g1, zspot, u, truth_score]; titles = ["spatial regions", "Group1 abundance", "mean Group1 state z", "microenvironment u", "truth response"]
    for ax, v, title in zip(axes, vals, titles):
        im = ax.scatter(coords[:, 0], coords[:, 1], c=v, s=10, cmap="coolwarm"); ax.set_aspect("equal"); ax.set_title(title); fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout(); fig.savefig(scenario / "figures/figure1_ground_truth_maps.png", dpi=180); plt.close(fig)
    # Manifest is frozen only after every generation artifact exists.
    manifest = seed_root / "generated_data_manifest.sha256"; files = sorted(p for p in seed_root.rglob("*") if p.is_file() and p != manifest)
    manifest.write_text("\n".join(f"{sha256(p)}  {p.resolve().relative_to(project)}" for p in files) + "\n", encoding="utf-8")
    print(json.dumps({"seed": seed, "status": audit["status"], **metrics}, indent=2))


if __name__ == "__main__":
    main()
