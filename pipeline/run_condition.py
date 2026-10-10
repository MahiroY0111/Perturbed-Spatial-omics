#!/usr/bin/env python3
"""Runner for one condition (simulated or public data).

Training and prediction are done by the functions of the final pipeline
(`core` = the core used by the seed 21--30 runner): same
preprocessing, same control-only Leiden cell types, same AE, STMix with the frozen
canonical configuration and shuffled loader, same CellOT, same prediction code.
What is left out are the provenance gates that tie the final runner to the frozen
20,000-cell inputs, and the built-in evaluation (use `reference_rows.py`).

Additions, all on the comparison-method side and all written as separate files:
  * several SIMO mappings can be evaluated with the same trained models
    (`--mappings label=path ...`; the first one is written under the standard name);
  * SIMO+CellOT accepts mappings in which a reference cell is placed in several spots;
    spots without any mapped cell are filled from their nearest covered spots (count recorded);
  * for every mapping, the prediction without transport (T = identity) and a
    prediction with a post hoc gene-wise scale calibrated on control ST are saved.

Stages:  stmix (AE + STMix + control reconstruction)  ->  full (CellOT + predictions)  ->  diag
AE / CellOT checkpoints placed under the standard names are reused, as in the modality-shift arm
of the main analysis (valid only when the scRNA-seq input is identical).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent

import standard_runner as canonical  # noqa: E402
import core  # noqa: E402
import epoch_shuffle as shuffle_mod  # noqa: E402
from epoch_shuffle import install_shuffle_true_patch  # noqa: E402

CONFIG_PATH = HERE / "standard_settings.json"
model_helpers = core.model_helpers
TARGET_SUM = 1e4

MAPPINGS: dict[str, Path] = {}
OUT_PRED: Path | None = None
NAME = ""
SEED = 0
SIMO_INFO: dict[str, Any] = {}


class IdentityTransport:
    """Stand-in for the CellOT network g with T(z) = z."""

    def eval(self):
        return self

    def transport(self, x):
        return x


def restore_rng_state_device_safe(state: dict[str, Any]) -> None:
    """Same as model_helpers._restore_rng_state; the saved generator states are moved to the CPU first
    (a checkpoint loaded with map_location=cuda carries them on the GPU, which torch.set_rng_state rejects)."""
    import random

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def prepare_inputs_without_qc(scenario_dir: Path) -> dict[str, Any]:
    """core.prepare_inputs without the per-cell QC filter (for real data whose QC was done on all genes before gene selection).
    Everything else (normalization, PCA, control-only Leiden, nearest-centroid labels for stim cells, ST preparation) is the same."""
    import scanpy as sc

    ctrl = ad.read_h5ad(scenario_dir / "sc_ctrl_raw.h5ad")
    stim = ad.read_h5ad(scenario_dir / "sc_stim_raw.h5ad")
    st_ctrl = ad.read_h5ad(scenario_dir / "st_ctrl_raw.h5ad")
    if not ctrl.var_names.equals(stim.var_names) or not ctrl.var_names.equals(st_ctrl.var_names):
        raise AssertionError("Model input gene order mismatch")
    for condition, value in (("ctrl", ctrl), ("stim", stim)):
        value.layers["raw_counts"] = value.X.copy()
        value.obs["condition"] = condition
        value.obs["cell_id"] = value.obs_names.astype(str)
    rna = ad.concat({"ctrl": ctrl, "stim": stim}, index_unique="::", join="inner", merge="same")
    rna.layers["raw_counts"] = rna.X.copy()
    rna.var["mt"] = rna.var_names.astype(str).str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    rna.layers["raw_counts"] = rna.layers["raw_counts"].copy()
    sc.pp.normalize_total(rna, target_sum=core.TARGET_SUM)
    sc.pp.log1p(rna)
    sc.tl.pca(rna, n_comps=50, svd_solver="arpack", random_state=core.MODEL_SEED)
    ctrl_mask = rna.obs["condition"].to_numpy() == "ctrl"
    ctrl_cluster = rna[ctrl_mask].copy()
    ctrl_cluster.obsm["X_pca"] = np.asarray(rna.obsm["X_pca"])[ctrl_mask]
    sc.pp.neighbors(ctrl_cluster, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(ctrl_cluster, resolution=0.5, key_added="control_leiden", random_state=core.MODEL_SEED, flavor="leidenalg")
    control_labels = ctrl_cluster.obs["control_leiden"].astype(str).to_numpy()
    categories = pd.Index(sorted(pd.unique(control_labels)))
    control_pca = np.asarray(ctrl_cluster.obsm["X_pca"], dtype=np.float64)[:, :30]
    centroids = np.stack([control_pca[control_labels == category].mean(axis=0) for category in categories])
    all_pca = np.asarray(rna.obsm["X_pca"], dtype=np.float64)[:, :30]
    assigned = np.empty(rna.n_obs, dtype=object)
    for start in range(0, rna.n_obs, 1024):
        stop = min(start + 1024, rna.n_obs)
        distance = np.square(all_pca[start:stop, None, :] - centroids[None, :, :]).sum(axis=2)
        assigned[start:stop] = categories[np.argmin(distance, axis=1)]
    assigned[ctrl_mask] = control_labels
    rna.obs["cell_type"] = pd.Categorical(assigned, categories=categories)
    rna.obs["cell_type_source"] = np.where(ctrl_mask, "control_leiden", "nearest_control_PCA_centroid")
    st_ctrl.layers["raw_counts"] = st_ctrl.X.copy()
    st_ctrl.obs["lib_raw"] = np.asarray(st_ctrl.X.sum(axis=1)).ravel()
    sc.pp.normalize_total(st_ctrl, target_sum=core.TARGET_SUM)
    sc.pp.log1p(st_ctrl)
    st_ctrl.obsm["X_spatial"] = np.asarray(st_ctrl.obsm["spatial"], dtype=np.float32)
    return {"rna": rna, "rna_ctrl": rna[ctrl_mask].copy(), "st_ctrl": st_ctrl,
            "preprocessing": {"scenario": core.SCENARIO, "model_seed": core.MODEL_SEED, "sc_cells_before_qc": int(ctrl.n_obs + stim.n_obs),
                              "sc_cells_after_qc": int(rna.n_obs), "G": int(rna.n_vars), "target_sum": core.TARGET_SUM, "transform": "log1p",
                              "gene_filtering": "none", "cell_type_truth_used": False, "control_type_source": "control-only Leiden resolution 0.5",
                              "stim_type_source": "nearest control-cluster PCA centroid", "per_cell_qc_in_pipeline": False}}


def normlog_from_linear(linear: np.ndarray) -> np.ndarray:
    total = linear.sum(axis=1, keepdims=True)
    return np.log1p(linear * (TARGET_SUM / np.where(total > 0, total, 1.0))).astype(np.float32)


def write_pred(matrix: np.ndarray, st_ctrl: ad.AnnData, kind: str, label: str) -> Path:
    path = OUT_PRED / f"{NAME}_{kind}_normlog_G8000_modelseed{SEED}__{label}.h5ad"
    out = ad.AnnData(X=matrix.astype(np.float32), obs=st_ctrl.obs.copy(), var=st_ctrl.var.copy())
    out.var_names = st_ctrl.var_names.copy()
    out.uns["prediction_scale"] = "library-normalized target_sum=10000 then log1p"
    out.write_h5ad(path)
    return path


def simo_predictions(*, mapped_path: Path, rna_ctrl: ad.AnnData, st_ctrl: ad.AnnData, g, vae_model, device) -> ad.AnnData:
    """Replacement for model_helpers.predict_simo_normalized_log1p (duplicates and partial coverage allowed)."""
    primary = None
    ctrl_by_id = pd.Series(np.arange(rna_ctrl.n_obs), index=rna_ctrl.obs["cell_id"].astype(str))
    latent_all = np.asarray(rna_ctrl.obsm["Z_latent"], dtype=np.float32)
    spot_index = pd.Index(st_ctrl.obs_names.astype(str))
    coords = np.asarray(st_ctrl.obsm["X_spatial"], dtype=np.float64)
    raw_ctrl = st_ctrl.layers["raw_counts"]
    raw_ctrl = np.asarray(raw_ctrl.toarray() if hasattr(raw_ctrl, "toarray") else raw_ctrl, dtype=np.float64)
    library = raw_ctrl.sum(axis=1, keepdims=True)
    if not MAPPINGS:
        placeholder = ad.AnnData(X=np.zeros((st_ctrl.n_obs, rna_ctrl.n_vars), dtype=np.float32), obs=st_ctrl.obs.copy(), var=rna_ctrl.var.copy())
        placeholder.var_names = rna_ctrl.var_names.copy()
        placeholder.uns["simo_skipped"] = True
        return placeholder
    for label, path in MAPPINGS.items():
        mapped = ad.read_h5ad(path)
        ids = mapped.obs["Cell"].astype(str)
        if not ids.isin(ctrl_by_id.index).all():
            raise RuntimeError(f"[{label}] SIMO cells missing from the modeling set")
        rows = ctrl_by_id.loc[ids].to_numpy(dtype=np.int64)
        spot_code = spot_index.get_indexer(mapped.obs["spot"].astype(str))
        if (spot_code < 0).any():
            raise ValueError(f"[{label}] unknown spot id in SIMO mapping")
        source = latent_all[rows]
        transported = model_helpers._transport_in_chunks(g, source, device=device)
        linear = {}
        vae_model.eval()
        with torch.no_grad():
            # spot x mapped-row incidence matrix; summing with it is the same aggregation as np.add.at, only faster
            incidence = sp.csc_matrix(
                (np.ones(len(spot_code), dtype=np.float32), (spot_code, np.arange(len(spot_code)))),
                shape=(st_ctrl.n_obs, len(spot_code)))
            for key, latent in (("noT", source), ("T", transported)):
                acc = np.zeros((st_ctrl.n_obs, rna_ctrl.n_vars), dtype=np.float32)
                for start in range(0, len(latent), 512):
                    z = torch.as_tensor(latent[start : start + 512], dtype=torch.float32, device=device)
                    cell_linear = np.expm1(vae_model.decode(z).cpu().numpy()).clip(min=0)
                    acc += incidence[:, start : start + len(cell_linear)] @ cell_linear
                linear[key] = acc
        covered = np.zeros(st_ctrl.n_obs, dtype=bool)
        covered[np.unique(spot_code)] = True
        n_uncovered = int((~covered).sum())
        if n_uncovered:
            # frequency profile of an uncovered spot := mean frequency profile of its 6 nearest covered spots
            cov_idx = np.flatnonzero(covered)
            for key in linear:
                freq = linear[key][cov_idx] / linear[key][cov_idx].sum(axis=1, keepdims=True)
                for j in np.flatnonzero(~covered):
                    d2 = np.square(coords[cov_idx] - coords[j]).sum(axis=1)
                    linear[key][j] = freq[np.argsort(d2)[:6]].mean(axis=0)
        normlog = {key: normlog_from_linear(value) for key, value in linear.items()}
        # post hoc gene-wise scale, estimated on control only (observed control ST vs. the no-transport reconstruction)
        freq_noT = linear["noT"] / linear["noT"].sum(axis=1, keepdims=True)
        expected_ctrl = (library * freq_noT).sum(axis=0)
        scale = np.divide(raw_ctrl.sum(axis=0), expected_ctrl, out=np.ones_like(expected_ctrl), where=expected_ctrl > 0)
        for key in ("noT", "T"):
            normlog[f"{key}_cal"] = normlog_from_linear(linear[key] * scale[None, :])
        for key, kind in (("T", "simo_cellot"), ("noT", "simo_noT"), ("T_cal", "simo_cellot_cal"), ("noT_cal", "simo_noT_cal")):
            write_pred(normlog[key], st_ctrl, kind, label)
        per_spot = np.bincount(spot_code, minlength=st_ctrl.n_obs)
        SIMO_INFO[label] = {
            "mapping": str(path), "mapped_rows": int(len(rows)), "distinct_cells": int(len(np.unique(rows))),
            "uncovered_spots_filled_from_neighbours": n_uncovered,
            "cells_per_spot_mean": float(per_spot.mean()), "cells_per_spot_min": int(per_spot.min()),
            "cells_per_spot_max": int(per_spot.max()),
            "calibration_scale_log_sd": float(np.log(scale[scale > 0]).std()),
        }
        if primary is None:
            primary = ad.AnnData(X=normlog["T"], obs=st_ctrl.obs.copy(), var=rna_ctrl.var.copy())
            primary.var_names = rna_ctrl.var_names.copy()
            primary.uns["prediction_scale"] = "library-normalized target_sum=10000 then log1p"
            primary.uns["simo_mapping_label"] = label
    return primary


def main() -> None:
    global OUT_PRED, NAME, SEED
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--name", required=True, help="label used in output file names")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", choices=("stmix", "full", "diag", "attrib"), required=True)
    p.add_argument("--mappings", nargs="*", default=[], help="label=path of SIMO mapping files (full stage)")
    p.add_argument("--ae-epochs", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--train-spots-fraction", type=float, default=1.0,
                   help="train STMix on this random fraction of spots only; the control reconstruction is still written for all spots")
    p.add_argument("--holdout-seed", type=int, default=0)
    p.add_argument("--skip-qc", action="store_true", help="do not re-apply the per-cell QC of the pipeline (real data already QC'd on all genes)")
    p.add_argument("--override", nargs="*", default=[],
                   help="STMix config overrides key=value (recorded in the run_info file)")
    p.add_argument("--ablation", choices=("none", "s_off", "delta_mu_off"), default="none",
                   help="fix the ST gene scale (s=1) or the type shift (delta_mu=0), as in the paper's ablation runner")
    p.add_argument("--auto-dims", action="store_true",
                   help="take the gene and spot counts from the ST input instead of the 8000 / 2500 of the simulation")
    a = p.parse_args()
    NAME, SEED = a.name, int(a.seed)
    out_root = a.output_root.resolve()
    for key in ("models", "predictions", "diagnostics"):
        (out_root / key).mkdir(parents=True, exist_ok=True)
    OUT_PRED = out_root / "predictions"

    core.SCENARIO = NAME
    core.MODEL_SEED = SEED
    if a.auto_dims:
        st_shape = ad.read_h5ad(a.scenario_dir.resolve() / "st_ctrl_raw.h5ad", backed="r").shape
        core.EXPECTED_SPOTS, core.EXPECTED_G = int(st_shape[0]), int(st_shape[1])
        shuffle_mod.EXPECTED_SPOTS = int(st_shape[0])
        model_helpers.MODEL_GENE_COUNT = int(st_shape[1])
        print(f"[run] dimensions taken from the ST input: {st_shape[0]} spots x {st_shape[1]} genes", flush=True)
    core.output_paths = lambda root: canonical.output_paths_for(root, NAME, SEED)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    canonical.CONFIG_PATH = CONFIG_PATH
    installed = canonical.install_final_config(core, config)
    if installed["changed"]:
        raise RuntimeError(f"Canonical settings changed: {installed['changed']}")
    install_shuffle_true_patch(core)
    overrides = {}
    for item in a.override:
        key, _, value = item.partition("=")
        overrides[key] = int(value) if value.lstrip("-").isdigit() else float(value)
    if overrides:
        # the loss-history check compares the trained weights with config["loss_weights"]; keep them consistent
        for key, value in overrides.items():
            if key in config["loss_weights"]:
                config["loss_weights"][key] = float(value)
        frozen_default = core.default_stmix_config

        def overridden_default():
            cfg = frozen_default()
            unknown = set(overrides) - set(cfg)
            if unknown:
                raise KeyError(f"unknown STMix config keys: {sorted(unknown)}")
            cfg.update(overrides)
            return cfg

        core.default_stmix_config = overridden_default
        print(f"[run] STMix config overrides: {overrides}", flush=True)
    canonical.install_full_loss_history(core, config)
    if a.ablation != "none":
        original_init = core.initialize_stmix

        def initialize_with_ablation(prepared, device, vae_model):
            model, sc_ref, loader = original_init(prepared, device, vae_model)
            with torch.no_grad():
                (model.st_log_scale if a.ablation == "s_off" else model.delta_mu).zero_()
            (model.st_log_scale if a.ablation == "s_off" else model.delta_mu).requires_grad_(False)
            return model, sc_ref, loader

        core.initialize_stmix = initialize_with_ablation
        print(f"[run] ablation: {a.ablation}", flush=True)
    model_helpers.predict_simo_normalized_log1p = simo_predictions
    model_helpers._restore_rng_state = restore_rng_state_device_safe

    outputs = core.output_paths(out_root)
    if a.skip_qc:
        core.prepare_inputs = prepare_inputs_without_qc
    prepared = core.prepare_inputs(a.scenario_dir.resolve())
    use_cuda = a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    counts = prepared["rna"].obs["condition"].astype(str).value_counts().to_dict()
    cell_types = prepared["rna_ctrl"].obs["cell_type"].astype(str).value_counts().to_dict()
    print(f"[run] {NAME} seed={SEED} device={device} torch={torch.__version__} cells={counts} control Leiden types={len(cell_types)}", flush=True)

    if a.stage == "stmix":
        if outputs["control"].exists():
            print("stmix stage already complete")
            return
        prepared_train = prepared
        if a.train_spots_fraction < 1.0:
            # held-out spots: the amortized encoder still reconstructs them, but they do not enter the STMix objective
            st = prepared["st_ctrl"]
            keep = np.random.default_rng(a.holdout_seed).random(st.n_obs) < a.train_spots_fraction
            prepared_train = dict(prepared)
            prepared_train["st_ctrl"] = st[keep].copy()
            shuffle_mod.EXPECTED_SPOTS = int(keep.sum())
            pd.DataFrame({"spot_id": st.obs_names.astype(str), "train": keep}).to_csv(
                out_root / f"diagnostics/{NAME}_holdout_spots_modelseed{SEED}.csv", index=False)
            print(f"[run] STMix trained on {int(keep.sum())} of {st.n_obs} spots (holdout seed {a.holdout_seed})", flush=True)
        _vae, stmix, _sc_ref, _history, _ae_history = core.run_stmix_stage(prepared_train, outputs, device, a.ae_epochs, allow_training=True)
        core.save_stmix_diagnostics(prepared, outputs, device, stmix)
        return

    if a.stage == "full":
        if outputs["proposed"].exists():
            print("full stage already complete")
            return
        for item in a.mappings:
            label, path = item.split("=", 1)
            if not Path(path).exists():
                print(f"[run] mapping {label} is missing and is skipped: {path}")
                continue
            MAPPINGS[label] = Path(path).resolve()
        if not MAPPINGS:
            # Proposed method only: the core still asks for a mapping file and a SIMO prediction;
            # an all-zero placeholder is written under the standard name and flagged.
            print("[run] no SIMO mapping: SIMO+CellOT is skipped", flush=True)
            first = a.scenario_dir.resolve() / "st_ctrl_raw.h5ad"
        else:
            first = next(iter(MAPPINGS.values()))
        core.run_full_stage(prepared, outputs, first, device, a.ae_epochs)
        info = {"scenario_dir": str(a.scenario_dir.resolve()), "name": NAME, "seed": SEED, "cells": counts,
                "control_leiden_type_sizes": cell_types, "simo": SIMO_INFO,
                "device": str(device), "torch_version": torch.__version__,
                "torch_seed_note": "core.set_seed() is called without argument, as in the final pipeline",
                "stmix_config_overrides": overrides, "ablation": a.ablation}
        (out_root / f"diagnostics/{NAME}_run_info_modelseed{SEED}.json").write_text(json.dumps(info, indent=2) + "\n")
        return

    if a.stage == "attrib":
        # Per-cell-type contribution to each spot's prediction (count scale, before library normalisation is undone):
        # contribution[j, t, :] = p_{j,t} * s * expm1(decoder(T(z_{j,t}) + r_j)) * lib_j / kappa, for the transport T and
        # for the identity map.  Written as float16 [S, T, G] arrays for the biological reading of the predicted change.
        target = out_root / f"diagnostics/{NAME}_type_attribution_modelseed{SEED}.npz"
        if target.exists():
            print("attrib stage already complete")
            return
        _vae, stmix, sc_ref, _history, _ae_history = core.run_stmix_stage(prepared, outputs, device, a.ae_epochs, allow_training=False)
        z_pseudo, r_spot, p_known = core.get_st_pseudo_latents(prepared["st_ctrl"], stmix, device=device, batch_size=256)
        st_pseudo = core.build_pseudo_adata_from_latents(
            prepared["st_ctrl"], z_pseudo, r_spot, p_known, cell_type_categories=pd.Index(sc_ref["cell_type_categories"]),
            use_residual=False, mass_threshold=core.MASS_THRESHOLD)
        st_pseudo.obs["condition"] = "st_ctrl_pseudo"
        adata_latent, latent_st = core.prepare_cellot_anndata(prepared["rna"], st_pseudo, latent_key="Z_latent", condition_col="condition")
        cellot_config = core.default_cellot_config()
        core.canonical_source_audit()
        cellot_config, _iterator, f, g, _opt_f, _opt_g = core.init_cellot(cellot_config, adata_latent, device=device)
        bundle = core.torch_load(outputs["cellot"], device)
        f.load_state_dict(bundle["f_state"]); g.load_state_dict(bundle["g_state"])
        latent = latent_st[latent_st.obs["condition"] == "st_ctrl_pseudo"].copy()
        source = np.asarray(latent.X, dtype=np.float32)
        residual = np.asarray(latent.obsm["R_spot"], dtype=np.float32)
        spot_idx = latent.obs["spot_index"].to_numpy(dtype=np.int64)
        type_idx = latent.obs["type_index"].to_numpy(dtype=np.int64)
        S, T, G = prepared["st_ctrl"].n_obs, stmix.T_known, stmix.G
        raw = prepared["st_ctrl"].layers["raw_counts"]
        lib = np.asarray(raw.sum(axis=1)).ravel().astype(np.float32)
        scale = np.clip(lib / float(stmix.target_sum), 1e-3, 1e3)
        st_scale = torch.exp(stmix.st_log_scale).detach().cpu().numpy().ravel()
        p_known = np.asarray(p_known, dtype=np.float32)
        out = {}
        stmix.eval()
        for label, G_ in (("T", g), ("noT", IdentityTransport())):
            moved = model_helpers._transport_in_chunks(G_, source, device=device, batch_size=512) + residual
            contrib = np.zeros((S, T, G), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, len(moved), 512):
                    stop = min(start + 512, len(moved))
                    z = torch.as_tensor(moved[start:stop], dtype=torch.float32, device=device)
                    mu = torch.expm1(stmix.decoder(z).clamp(-10.0, 10.0)).clamp(0.0, 1e6).cpu().numpy()
                    si, ti = spot_idx[start:stop], type_idx[start:stop]
                    contrib[si, ti, :] = (p_known[si, ti][:, None] * mu) * st_scale[None, :] * scale[si][:, None]
            out[label] = contrib.astype(np.float16)
        mask = np.zeros((S, T), dtype=bool); mask[spot_idx, type_idx] = True
        with torch.no_grad():
            bg = torch.expm1(stmix.decoder(stmix.mu_bg_latent).clamp(-10.0, 10.0)).clamp(0.0, 1e6).cpu().numpy().ravel()
        p_bg = np.clip(1.0 - (p_known * mask).sum(axis=1), 0.0, 1.0)
        bg_contrib = (p_bg[:, None] * bg[None, :]) * st_scale[None, :] * scale[:, None]
        np.savez_compressed(target, contrib_T=out["T"], contrib_noT=out["noT"], bg=bg_contrib.astype(np.float16),
                            p_known=p_known, selected=mask, spot_id=prepared["st_ctrl"].obs_names.astype(str).to_numpy(),
                            genes=prepared["st_ctrl"].var_names.astype(str).to_numpy(),
                            cell_type_categories=np.asarray(list(map(str, sc_ref["cell_type_categories"]))),
                            trained_cell_type_index=stmix.cell_type_index_sc.detach().cpu().numpy(),
                            ctrl_cell_id=prepared["rna_ctrl"].obs["cell_id"].astype(str).to_numpy())
        print("[run] type attribution written:", target, flush=True)
        return

    if a.stage == "diag":
        target = out_root / f"diagnostics/{NAME}_composition_modelseed{SEED}.npz"
        if target.exists():
            print("diag stage already complete")
            return
        _vae, stmix, sc_ref, _history, _ae_history = core.run_stmix_stage(prepared, outputs, device, a.ae_epochs, allow_training=False)
        z_pseudo, r_spot, p_known = core.get_st_pseudo_latents(prepared["st_ctrl"], stmix, device=device, batch_size=256)
        ctrl = prepared["rna_ctrl"]
        # Proposed prediction with the transport replaced by the identity map, through exactly the same
        # code path as the stimulated prediction (same pseudo-cell selection, residual, background, gene scale).
        st_pseudo = core.build_pseudo_adata_from_latents(
            prepared["st_ctrl"], z_pseudo, r_spot, p_known, cell_type_categories=pd.Index(sc_ref["cell_type_categories"]),
            use_residual=False, mass_threshold=core.MASS_THRESHOLD)
        st_pseudo.obs["condition"] = "st_ctrl_pseudo"
        _latent, latent_st = core.prepare_cellot_anndata(prepared["rna"], st_pseudo, latent_key="Z_latent", condition_col="condition")
        no_transport = model_helpers.predict_proposed_raw_full_gene(
            adata_latent_st=latent_st, condition_value="st_ctrl_pseudo", g=IdentityTransport(), st_mix_model=stmix,
            st_ctrl=prepared["st_ctrl"], p_known=p_known, device=device)
        no_transport.write_h5ad(OUT_PRED / f"{NAME}_proposed_noT_raw_G8000_modelseed{SEED}.h5ad")
        np.savez_compressed(
            target,
            p_known=np.asarray(p_known, dtype=np.float32),
            cell_type_categories=np.asarray(list(map(str, sc_ref["cell_type_categories"]))),
            ctrl_cell_id=ctrl.obs["cell_id"].astype(str).to_numpy(),
            ctrl_cell_type=ctrl.obs["cell_type"].astype(str).to_numpy(),
            # type index of every control cell as stored in the trained model (the slot order of p_known);
            # Leiden numbering recomputed in another process can differ when cluster sizes are tied
            trained_cell_type_index=stmix.cell_type_index_sc.detach().cpu().numpy(),
            spot_id=prepared["st_ctrl"].obs_names.astype(str).to_numpy(),
            st_log_scale=stmix.st_log_scale.detach().cpu().numpy().ravel(),
        )
        return


if __name__ == "__main__":
    main()
