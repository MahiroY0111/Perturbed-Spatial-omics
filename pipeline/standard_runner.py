#!/usr/bin/env python3
"""Standard settings of the method: installs the fixed loss weights of standard_settings.json into the core
module, keeps the full loss history and names the output files of a run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch



HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CONFIG_PATH = HERE / "standard_settings.json"
ALLOWED_SCENARIOS = (
    "unpaired_group",
    "unpaired_group_heterogeneous",
    "unpaired_group_modality",
)
ALLOWED_MODEL_SEEDS = (1, 2, 3)
LOSS_HISTORY_TO_CONFIG = {
    "L_nb": "lambda_nb",
    "L_mix_gene": "lambda_mix_gene",
    "L_ot": "lambda_ot",
    "L_pbg": "lambda_pbg",
    "L_res": "lambda_res",
    "L_gamma": "lambda_gamma",
    "L_proto_sc": "lambda_proto_sc",
    "L_pseudo_proto": "lambda_pseudo_proto",
    "L_sp_gamma": "lambda_sp_gamma",
}


def output_paths_for(
    output_root: Path, scenario: str, model_seed: int
) -> dict[str, Path]:
    suffix = f"modelseed{model_seed}"
    return {
        "ae": output_root / f"models/{scenario}_ae_G8000_{suffix}.pt",
        "stmix": output_root / f"models/{scenario}_stmix_G8000_{suffix}.pt",
        "cellot": output_root / f"models/{scenario}_cellot_G8000_{suffix}.pt",
        "control": output_root
        / f"predictions/{scenario}_proposed_control_raw_G8000_{suffix}.h5ad",
        "proposed": output_root
        / f"predictions/{scenario}_proposed_stim_raw_G8000_{suffix}.h5ad",
        "simo": output_root
        / f"predictions/{scenario}_simo_cellot_normlog_G8000_{suffix}.h5ad",
        "batch_diagnostic": output_root
        / f"diagnostics/{scenario}_residual_batch_64_vs_256_{suffix}.json",
        "preflight": output_root
        / f"diagnostics/{scenario}_preflight_post_parameter_audit_{suffix}.json",
        "training_metadata": output_root
        / f"diagnostics/{scenario}_training_metadata_{suffix}.json",
        "stmix_diagnostics": output_root
        / f"diagnostics/{scenario}_stmix_diagnostics_{suffix}.json",
        "metrics": output_root
        / f"evaluation/{scenario}_four_method_metrics_{suffix}.csv",
        "evaluation_assertions": output_root
        / f"evaluation/{scenario}_evaluation_assertions_{suffix}.json",
        "scale_audit": output_root
        / f"evaluation/{scenario}_evaluation_scale_audit_{suffix}.json",
        "spatial_metrics": output_root
        / f"evaluation/{scenario}_spatial_metrics_{suffix}.csv",
        "spatial_maps": output_root
        / f"evaluation/{scenario}_spatial_difference_maps_{suffix}.csv.gz",
        "spatial_plot": output_root
        / f"evaluation/{scenario}_spatial_difference_maps_{suffix}.png",
    }


def install_final_config(core, frozen: dict[str, Any]) -> dict[str, Any]:
    original_default = core.default_stmix_config
    original_audit = core.canonical_source_audit
    baseline = original_default()
    if frozen.get("lmix_mode") != "frequency":
        raise RuntimeError("Frozen final config must declare lmix_mode='frequency'")
    if baseline.get("lmix_mode") != "frequency":
        raise RuntimeError("Canonical source does not use frequency L_mix")
    if baseline.get("require_frequency_lmix") is not True:
        raise RuntimeError("Canonical source does not fail closed on L_mix mode")
    weights = {key: float(value) for key, value in frozen["loss_weights"].items()}
    if set(weights) != set(LOSS_HISTORY_TO_CONFIG.values()):
        raise RuntimeError("Frozen loss-weight keys are incomplete")

    expected_canonical = {
        "lambda_nb": 5.0,
        "lambda_mix_gene": 0.2,
        "lambda_ot": 20.0,
        "lambda_pbg": 1.0,
        "lambda_res": 0.2,
        "lambda_gamma": 1.0,
        "lambda_proto_sc": 5.0,
        "lambda_pseudo_proto": 0.0,
        "lambda_sp_gamma": 0.1,
    }
    for key, value in expected_canonical.items():
        if not np.isclose(float(baseline[key]), value):
            raise AssertionError(f"Corrected canonical {key} unexpectedly changed")
        if not np.isclose(weights[key], value):
            raise AssertionError(f"Final canonical must preserve {key}")

    configured = dict(baseline)
    configured.update(weights)
    changed = sorted(key for key in baseline if baseline[key] != configured[key])
    expected_changed: list[str] = []
    if changed != expected_changed:
        raise AssertionError(f"Unexpected final canonical changes: {changed}")

    def final_default() -> dict[str, Any]:
        return dict(configured)

    def final_source_audit() -> dict[str, Any]:
        active = core.default_stmix_config
        core.default_stmix_config = original_default
        try:
            audit = original_audit()
        finally:
            core.default_stmix_config = active
        audit["final_simulation_canonical_override"] = {
            "config_file": str(CONFIG_PATH),
            "config_sha256": core.sha256(CONFIG_PATH),
            "loss_weights": weights,
            "changed_from_corrected_source_defaults": changed,
            "no_further_lambda_search": True,
        }
        return audit

    core.default_stmix_config = final_default
    core.canonical_source_audit = final_source_audit
    return {"baseline": baseline, "configured": configured, "changed": changed}


def install_full_loss_history(core, frozen: dict[str, Any]) -> None:
    trained = core.train_stmix_model
    expected_weights = frozen["loss_weights"]

    def train_with_full_history(
        st_mix_model,
        st_loader,
        z_sc_by_type,
        config,
        device=None,
        clip_grad=1.0,
        log_every=20,
    ):
        for key, value in expected_weights.items():
            if not np.isclose(float(config[key]), float(value)):
                raise AssertionError(f"Frozen weight mismatch: {key}")
        history = trained(
            st_mix_model,
            st_loader,
            z_sc_by_type,
            config,
            device=device,
            clip_grad=clip_grad,
            log_every=log_every,
        )
        n_epochs = int(config["n_epochs"])
        for loss_key, lambda_key in LOSS_HISTORY_TO_CONFIG.items():
            if loss_key not in history or len(history[loss_key]) != n_epochs:
                raise RuntimeError(f"Incomplete raw history: {loss_key}")
            raw_key = f"raw_{loss_key}"
            weighted_key = f"weighted_{loss_key}"
            history[raw_key] = [float(value) for value in history[loss_key]]
            weight = float(config[lambda_key])
            history[weighted_key] = [
                weight * value for value in history[raw_key]
            ]
        history["final_simulation_canonical"] = {
            "config_file": str(CONFIG_PATH),
            "loss_weights": expected_weights,
            "no_further_lambda_search": True,
        }
        return history

    core.train_stmix_model = train_with_full_history
