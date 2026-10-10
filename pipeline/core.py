#!/usr/bin/env python3
"""Core steps of the method for one run: input preparation, training of the autoencoder, the spot model
and CellOT, and the predictions of the proposed method and of SIMO+CellOT.  pipeline/run_condition.py
drives these functions; the fixed checks in canonical_source_audit() guard the standard settings."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
for path in (SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import method_helpers as model_helpers  # noqa: E402
from perturbed_spatial_omics.transport import (  # noqa: E402
    default_cellot_config,
    init_cellot,
    prepare_cellot_anndata,
    train_cellot,
)
from perturbed_spatial_omics.stmix import (  # noqa: E402
    build_pseudo_adata_from_latents,
    default_stmix_config,
    get_st_pseudo_latents,
    init_stmix_model,
    make_st_dataloader,
    train_stmix_model,
)
from cellot.models.ae import AutoEncoder  # noqa: E402


SCENARIO = "legacy_all"
MODEL_SEED = 1
EXPECTED_G = 8000
EXPECTED_P = 800
EXPECTED_SC_PER_CONDITION = 20000
EXPECTED_SPOTS = 2500
TARGET_SUM = 1e4
MASS_THRESHOLD = 1e-2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int = MODEL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "ae": output_root / "models/legacy_all_ae_G8000_modelseed1.pt",
        "stmix": output_root / "models/legacy_all_stmix_G8000_modelseed1.pt",
        "cellot": output_root / "models/legacy_all_cellot_G8000_modelseed1.pt",
        "control": output_root
        / "predictions/legacy_all_proposed_control_raw_G8000_modelseed1.h5ad",
        "proposed": output_root
        / "predictions/legacy_all_proposed_stim_raw_G8000_modelseed1.h5ad",
        "simo": output_root
        / "predictions/legacy_all_simo_cellot_normlog_G8000_modelseed1.h5ad",
        "batch_diagnostic": output_root
        / "diagnostics/legacy_all_residual_batch_64_vs_256.json",
        "preflight": output_root
        / "diagnostics/legacy_all_preflight_post_parameter_audit.json",
        "training_metadata": output_root
        / "diagnostics/legacy_all_training_metadata.json",
        "metrics": output_root / "evaluation/legacy_all_four_method_metrics.csv",
        "evaluation_assertions": output_root
        / "evaluation/legacy_all_evaluation_assertions.json",
    }


def canonical_source_audit() -> dict[str, Any]:
    stmix_source = Path(inspect.getsourcefile(default_stmix_config))
    cellot_source = Path(inspect.getsourcefile(default_cellot_config))
    for path in (stmix_source, cellot_source):
        if not path.exists():
            raise FileNotFoundError(path)
    stmix_source_text = stmix_source.read_text(encoding="utf-8")
    if "sigma = float(np.median(neigh_dist) + 1e-8)" not in stmix_source_text:
        raise AssertionError("Canonical spatial RBF sigma rule changed")
    stmix_config = default_stmix_config()
    cellot_config = default_cellot_config()
    expected_stmix = {
        "lambda_pseudo_proto": 0.0,
        "lambda_ot": 20.0,
        "lambda_res": 0.2,
        "lambda_gamma": 1.0,
        "lambda_proto_sc": 5.0,
        "lambda_sp_gamma": 0.1,
        "sp_k": 10,
        "ot_eps": 0.5,
        "ot_max_iter": 5,
    }
    for key, expected in expected_stmix.items():
        if not np.isclose(float(stmix_config[key]), expected):
            raise AssertionError(
                f"Canonical STMix {key}={stmix_config[key]} != {expected}"
            )
    expected_centering = {
        "residual_centering_scope": "global_spot",
        "residual_center_refresh": "each_epoch",
        "residual_center_batch_size": 256,
    }
    for key, expected in expected_centering.items():
        if stmix_config.get(key) != expected:
            raise AssertionError(
                f"Canonical STMix {key}={stmix_config.get(key)!r} != {expected!r}"
            )
    if stmix_config.get("lmix_mode") != "frequency":
        raise AssertionError("Canonical STMix must use frequency L_mix")
    if stmix_config.get("require_frequency_lmix") is not True:
        raise AssertionError("Canonical STMix must fail closed on non-frequency L_mix")
    required_frequency_fragments = (
        "def _frequency_mix_prediction(",
        "R_bg_frequency = torch.softmax(self.R_bg_raw, dim=-1)",
        "X_mix_pred = torch.log1p(self.target_sum * q_mix.clamp_min(0.0))",
    )
    for fragment in required_frequency_fragments:
        if fragment not in stmix_source_text:
            raise AssertionError(f"Canonical frequency L_mix fragment missing: {fragment}")
    if not np.isclose(float(cellot_config.model.g.fnorm_penalty), 1.0):
        raise AssertionError("Canonical CellOT g penalty must equal 1")
    if int(cellot_config.training.n_inner_iters) != 10:
        raise AssertionError("Canonical CellOT must use 10 g inner updates")
    return {
        "stmix_source": str(stmix_source),
        "stmix_sha256": sha256(stmix_source),
        "residual_centering_scope": "global_spot",
        "residual_center_refresh": "each_epoch",
        "lmix_mode": "frequency",
        "cellot_source": str(cellot_source),
        "cellot_sha256": sha256(cellot_source),
        "stmix_config": stmix_config,
        "cellot_g_fnorm_penalty": float(cellot_config.model.g.fnorm_penalty),
        "cellot_n_iters": int(cellot_config.training.n_iters),
        "cellot_n_inner_iters": int(cellot_config.training.n_inner_iters),
        "spatial_rbf_sigma_rule": "median of all kNN distances",
        "canonical_checks_passed": True,
    }


def prepare_inputs(scenario_dir: Path) -> dict[str, Any]:
    ctrl = ad.read_h5ad(scenario_dir / "sc_ctrl_raw.h5ad")
    stim = ad.read_h5ad(scenario_dir / "sc_stim_raw.h5ad")
    st_ctrl = ad.read_h5ad(scenario_dir / "st_ctrl_raw.h5ad")
    if not ctrl.var_names.equals(stim.var_names) or not ctrl.var_names.equals(
        st_ctrl.var_names
    ):
        raise AssertionError("Model input gene order mismatch")

    for condition, value in (("ctrl", ctrl), ("stim", stim)):
        value.layers["raw_counts"] = value.X.copy()
        value.obs["condition"] = condition
        value.obs["cell_id"] = value.obs_names.astype(str)
    rna = ad.concat(
        {"ctrl": ctrl, "stim": stim},
        index_unique="::",
        join="inner",
        merge="same",
    )
    rna.layers["raw_counts"] = rna.X.copy()
    rna.var["mt"] = rna.var_names.astype(str).str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    cell_mask = (rna.obs["total_counts"] >= 1000) & (
        rna.obs["n_genes_by_counts"] >= 500
    )
    if int(rna.var["mt"].sum()) > 0:
        cell_mask &= rna.obs["pct_counts_mt"] <= 5.0
    rna = rna[cell_mask].copy()
    if rna.n_vars != EXPECTED_G:
        raise AssertionError("Cell QC must not alter G=8000")
    rna.layers["raw_counts"] = rna.layers["raw_counts"].copy()
    sc.pp.normalize_total(rna, target_sum=TARGET_SUM)
    sc.pp.log1p(rna)

    # No Splatter true group is used. Control-only Leiden defines known types;
    # stim cells receive the nearest control-cluster PCA centroid for CellOT.
    sc.tl.pca(rna, n_comps=50, svd_solver="arpack", random_state=MODEL_SEED)
    ctrl_mask = rna.obs["condition"].to_numpy() == "ctrl"
    ctrl_cluster = rna[ctrl_mask].copy()
    ctrl_cluster.obsm["X_pca"] = np.asarray(rna.obsm["X_pca"])[ctrl_mask]
    sc.pp.neighbors(ctrl_cluster, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(
        ctrl_cluster,
        resolution=0.5,
        key_added="control_leiden",
        random_state=MODEL_SEED,
        flavor="leidenalg",
    )
    control_labels = ctrl_cluster.obs["control_leiden"].astype(str).to_numpy()
    categories = pd.Index(sorted(pd.unique(control_labels)))
    control_pca = np.asarray(ctrl_cluster.obsm["X_pca"], dtype=np.float64)[:, :30]
    centroids = np.stack(
        [control_pca[control_labels == category].mean(axis=0) for category in categories]
    )
    all_pca = np.asarray(rna.obsm["X_pca"], dtype=np.float64)[:, :30]
    assigned = np.empty(rna.n_obs, dtype=object)
    for start in range(0, rna.n_obs, 1024):
        stop = min(start + 1024, rna.n_obs)
        distance = np.square(
            all_pca[start:stop, None, :] - centroids[None, :, :]
        ).sum(axis=2)
        assigned[start:stop] = categories[np.argmin(distance, axis=1)]
    # Preserve exact control labels rather than reclassifying the training set.
    assigned[ctrl_mask] = control_labels
    rna.obs["cell_type"] = pd.Categorical(assigned, categories=categories)
    rna.obs["cell_type_source"] = np.where(
        ctrl_mask, "control_leiden", "nearest_control_PCA_centroid"
    )

    st_ctrl.layers["raw_counts"] = st_ctrl.X.copy()
    st_ctrl.obs["lib_raw"] = np.asarray(st_ctrl.X.sum(axis=1)).ravel()
    sc.pp.normalize_total(st_ctrl, target_sum=TARGET_SUM)
    sc.pp.log1p(st_ctrl)
    st_ctrl.obsm["X_spatial"] = np.asarray(
        st_ctrl.obsm["spatial"], dtype=np.float32
    )
    if st_ctrl.shape != (EXPECTED_SPOTS, EXPECTED_G):
        raise AssertionError("Prepared control ST shape mismatch")

    return {
        "rna": rna,
        "rna_ctrl": rna[ctrl_mask].copy(),
        "st_ctrl": st_ctrl,
        "preprocessing": {
            "scenario": SCENARIO,
            "model_seed": MODEL_SEED,
            "sc_cells_before_qc": int(ctrl.n_obs + stim.n_obs),
            "sc_cells_after_qc": int(rna.n_obs),
            "G": int(rna.n_vars),
            "target_sum": TARGET_SUM,
            "transform": "log1p",
            "gene_filtering": "none",
            "cell_type_truth_used": False,
            "control_type_source": "control-only Leiden resolution 0.5",
            "stim_type_source": "nearest control-cluster PCA centroid",
        },
    }


def initialize_stmix(prepared: dict[str, Any], device: torch.device, vae_model):
    model, sc_ref = init_stmix_model(
        prepared["rna_ctrl"],
        vae_model,
        cell_type_col="cell_type",
        device=device,
        target_sum=TARGET_SUM,
    )
    _dataset, loader = make_st_dataloader(
        prepared["st_ctrl"],
        target_sum=TARGET_SUM,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        spatial_key="X_spatial",
    )
    return model, sc_ref, loader


def run_stmix_stage(
    prepared: dict[str, Any],
    outputs: dict[str, Path],
    device: torch.device,
    ae_epochs: int,
    allow_training: bool,
):
    rna = prepared["rna"]
    set_seed()
    if outputs["ae"].exists():
        bundle = torch_load(outputs["ae"], device)
        if bundle["genes"] != list(map(str, rna.var_names)):
            raise RuntimeError("Saved AE gene order mismatch")
        vae_model = AutoEncoder(
            rna.n_vars, hidden_units=[128, 128], latent_dim=50
        ).to(device)
        vae_model.load_state_dict(bundle["state"])
        model_helpers._encode_full_adata(vae_model, rna, device=device, batch_size=256)
        model_helpers._restore_rng_state(bundle["rng_after_stage"])
        ae_history = bundle["history"]
    elif allow_training:
        vae_model, ae_history = model_helpers.train_autoencoder_full_gene(
            rna,
            latent_dim=50,
            hidden_units=(128, 128),
            n_epochs=int(ae_epochs),
            batch_size=256,
            learning_rate=1e-3,
            device=device,
            obsm_key="Z_latent",
        )
        torch.save(
            {
                "scenario": SCENARIO,
                "model_seed": MODEL_SEED,
                "genes": list(map(str, rna.var_names)),
                "state": vae_model.state_dict(),
                "history": ae_history,
                "rng_after_stage": model_helpers._rng_state(),
            },
            outputs["ae"],
        )
    else:
        raise FileNotFoundError("Run the gated STMix stage before the full stage")

    prepared["rna_ctrl"] = rna[rna.obs["condition"] == "ctrl"].copy()
    stmix_model, sc_ref, st_loader = initialize_stmix(prepared, device, vae_model)
    stmix_config = default_stmix_config()
    canonical_source_audit()
    if outputs["stmix"].exists():
        bundle = torch_load(outputs["stmix"], device)
        if bundle["config"] != stmix_config or bundle["genes"] != list(
            map(str, rna.var_names)
        ):
            raise RuntimeError("Saved STMix stage is incompatible")
        stmix_model.load_state_dict(bundle["state"])
        model_helpers._restore_rng_state(bundle["rng_after_stage"])
        stmix_history = bundle["history"]
    elif allow_training:
        stmix_history = train_stmix_model(
            stmix_model,
            st_loader,
            sc_ref["z_sc_by_type"],
            stmix_config,
            device=device,
        )
        torch.save(
            {
                "scenario": SCENARIO,
                "model_seed": MODEL_SEED,
                "genes": list(map(str, rna.var_names)),
                "config": stmix_config,
                "state": stmix_model.state_dict(),
                "history": stmix_history,
                "rng_after_stage": model_helpers._rng_state(),
            },
            outputs["stmix"],
        )
    else:
        raise FileNotFoundError("Run the gated STMix stage before the full stage")

    return vae_model, stmix_model, sc_ref, stmix_history, ae_history


def save_stmix_diagnostics(
    prepared: dict[str, Any],
    outputs: dict[str, Path],
    device: torch.device,
    stmix_model,
) -> None:
    if outputs["control"].exists() or outputs["batch_diagnostic"].exists():
        raise FileExistsError("Refusing to overwrite existing STMix diagnostic outputs")
    control_raw = model_helpers.reconstruct_control_raw(
        stmix_model, prepared["st_ctrl"], device=device, batch_size=256
    )
    control = ad.AnnData(
        X=control_raw,
        obs=prepared["st_ctrl"].obs.copy(),
        var=prepared["st_ctrl"].var.copy(),
    )
    control.var_names = prepared["st_ctrl"].var_names.copy()
    control.layers["nb_mu_pred"] = control_raw
    control.uns["prediction_scale"] = "NB expected raw counts"
    control.uns["G"] = EXPECTED_G
    control.uns["residual_centering_scope"] = "global_spot"
    control.uns["residual_center_refresh"] = "recomputed_for_inference"
    control.write_h5ad(outputs["control"])
    diagnostic = model_helpers.residual_batch_diagnostic(
        stmix_model,
        prepared["st_ctrl"],
        device=device,
        batch_a=64,
        batch_b=256,
    )
    outputs["batch_diagnostic"].write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_full_stage(
    prepared: dict[str, Any],
    outputs: dict[str, Path],
    simo_mapping: Path,
    device: torch.device,
    ae_epochs: int,
) -> None:
    if not simo_mapping.exists():
        raise FileNotFoundError(f"SIMO mapping is missing: {simo_mapping}")
    if outputs["proposed"].exists() or outputs["simo"].exists():
        raise FileExistsError("Refusing to overwrite existing final predictions")
    vae_model, stmix_model, sc_ref, stmix_history, _ae_history = run_stmix_stage(
        prepared, outputs, device, ae_epochs, allow_training=False
    )
    z_pseudo, r_spot, p_known = get_st_pseudo_latents(
        prepared["st_ctrl"], stmix_model, device=device, batch_size=256
    )
    cell_types = pd.Index(sc_ref["cell_type_categories"])
    st_pseudo = build_pseudo_adata_from_latents(
        prepared["st_ctrl"],
        z_pseudo,
        r_spot,
        p_known,
        cell_type_categories=cell_types,
        use_residual=False,
        mass_threshold=MASS_THRESHOLD,
    )
    st_pseudo.obs["condition"] = "st_ctrl_pseudo"
    adata_latent, adata_latent_st = prepare_cellot_anndata(
        prepared["rna"],
        st_pseudo,
        latent_key="Z_latent",
        condition_col="condition",
    )
    cellot_config = default_cellot_config()
    canonical_source_audit()
    cellot_config, iterator, f, g, opt_f, opt_g = init_cellot(
        cellot_config, adata_latent, device=device
    )
    if outputs["cellot"].exists():
        bundle = torch_load(outputs["cellot"], device)
        if bundle["config"] != cellot_config.toDict():
            raise RuntimeError("Saved CellOT config mismatch")
        f.load_state_dict(bundle["f_state"])
        g.load_state_dict(bundle["g_state"])
        cellot_history = bundle["history"]
        min_mmd = float(bundle["min_mmd"])
    else:
        cellot_history, min_mmd = train_cellot(
            f, g, iterator, opt_f, opt_g, cellot_config, device=device
        )
        torch.save(
            {
                "scenario": SCENARIO,
                "model_seed": MODEL_SEED,
                "config": cellot_config.toDict(),
                "f_state": f.state_dict(),
                "g_state": g.state_dict(),
                "history": cellot_history,
                "min_mmd": min_mmd,
            },
            outputs["cellot"],
        )

    proposed = model_helpers.predict_proposed_raw_full_gene(
        adata_latent_st=adata_latent_st,
        condition_value="st_ctrl_pseudo",
        g=g,
        st_mix_model=stmix_model,
        st_ctrl=prepared["st_ctrl"],
        p_known=p_known,
        device=device,
    )
    proposed.uns["scenario"] = SCENARIO
    proposed.uns["model_seed"] = MODEL_SEED
    proposed.uns["G"] = EXPECTED_G
    proposed.write_h5ad(outputs["proposed"])

    simo = model_helpers.predict_simo_normalized_log1p(
        mapped_path=simo_mapping,
        rna_ctrl=prepared["rna_ctrl"],
        st_ctrl=prepared["st_ctrl"],
        g=g,
        vae_model=vae_model,
        device=device,
    )
    simo.uns["scenario"] = SCENARIO
    simo.uns["model_seed"] = MODEL_SEED
    simo.uns["G"] = EXPECTED_G
    simo.write_h5ad(outputs["simo"])

    metadata = {
        **prepared["preprocessing"],
        "canonical_sources": canonical_source_audit(),
        "stmix_drop_last": False,
        "stmix_shuffle": False,
        "stmix_batch_size": 64,
        "residual_centering_scope": "global_spot",
        "residual_center_refresh_training": "each_epoch",
        "residual_center_recomputed_for_inference": True,
        "stmix_config": default_stmix_config(),
        "cellot_config": cellot_config.toDict(),
        "cellot_min_mmd": min_mmd,
        "residual_free_cellot_input": True,
        "residual_added_after_transport": True,
        "truth_labels_used_for_model": False,
        "other_scenarios_run": False,
        "stmix_final_within_run_objective": float(stmix_history["total"][-1]),
        "cellot_final_convexity_penalty": float(
            cellot_history["convexity_penalty"][-1]
        ),
    }
    outputs["training_metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
