"""Training and prediction helpers of the method: full-gene autoencoder training, control
reconstruction, transport of latent pseudo-cells in chunks, decoding of the proposed method's
predictions and of the SIMO+CellOT comparison."""

from __future__ import annotations

import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from perturbed_spatial_omics.transport import (  # noqa: E402
    default_cellot_config,
    init_cellot,
    prepare_cellot_anndata,
    train_cellot,
)
from perturbed_spatial_omics.autoencoder import sparse_collate_fn  # noqa: E402
from perturbed_spatial_omics.stmix import (  # noqa: E402
    STDataset,
    build_pseudo_adata_from_latents,
    compute_global_residual_center,
    default_stmix_config,
    get_st_pseudo_latents,
    init_stmix_model,
    make_st_dataloader,
    train_stmix_model,
)
from normalization import normalize_log1p  # noqa: E402
from cellot.models.ae import AutoEncoder  # noqa: E402
from cellot.data.cell import AnnDataDataset  # noqa: E402


TRAINING_SEED = 1
TARGET_SUM = 1e4
MASS_THRESHOLD = 1e-2
MODEL_GENE_COUNT = 8000


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def _encode_full_adata(
    vae_model,
    rna: ad.AnnData,
    device: torch.device,
    batch_size: int = 256,
) -> None:
    """Encode without densifying the complete 40,000 x 8,000 matrix."""
    pieces = []
    for start in range(0, rna.n_obs, batch_size):
        X = rna.X[start : start + batch_size]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        latent = vae_model.encode(torch.as_tensor(X, dtype=torch.float32, device=device))
        pieces.append(latent.cpu().numpy())
    rna.obsm["Z_latent"] = np.concatenate(pieces, axis=0)


def train_autoencoder_full_gene(
    rna: ad.AnnData,
    *,
    latent_dim: int,
    hidden_units: tuple[int, ...],
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    obsm_key: str,
):
    """Canonical AE training with memory-safe batched latent extraction.

    The optimization steps match ``perturbed_spatial_omics.autoencoder.train_cellot_autoencoder``;
    only the final eval-mode encoding is chunked so G=8,000 does not require a
    single dense 40,000 x 8,000 tensor.
    """
    model = AutoEncoder(
        rna.n_vars,
        hidden_units=list(hidden_units),
        latent_dim=latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        AnnDataDataset(rna),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=sparse_collate_fn,
    )
    history: list[float] = []
    model.train()
    for epoch in range(n_epochs):
        total = 0.0
        for batch in tqdm(loader, desc=f"AE epoch {epoch + 1}/{n_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        history.append(total / len(loader))
        print(f"AE epoch {epoch + 1}/{n_epochs}, average loss={history[-1]:.6f}")
    model.eval()
    _encode_full_adata(model, rna, device=device, batch_size=batch_size)
    if rna.obsm["Z_latent"].shape != (rna.n_obs, latent_dim):
        raise AssertionError("Batched AE latent extraction returned an invalid shape")
    if obsm_key != "Z_latent":
        rna.obsm[obsm_key] = rna.obsm.pop("Z_latent")
    return model, history


@torch.no_grad()
def reconstruct_control_raw(
    st_mix_model,
    st_ctrl: ad.AnnData,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Collect the actual NB mean produced by STMix for control spots."""
    st_mix_model.eval()
    dataset = STDataset(st_ctrl, target_sum=TARGET_SUM, spatial_key="X_spatial")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    residual_center = compute_global_residual_center(
        st_mix_model,
        dataset,
        device=device,
        batch_size=batch_size,
    )
    values = []
    for x_norm, x_raw, log_lib, _spot_idx, _coords in loader:
        out = st_mix_model(
            x_norm.to(device),
            x_raw.to(device),
            log_lib.to(device),
            residual_center=residual_center,
        )
        values.append(out["mu_pred"].detach().cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def residual_batch_diagnostic(
    st_mix_model,
    st_ctrl: ad.AnnData,
    device: torch.device,
    batch_a: int = 64,
    batch_b: int = 256,
) -> dict[str, Any]:
    """Verify that global centering is invariant to inference chunk size."""
    pred_a = reconstruct_control_raw(st_mix_model, st_ctrl, device=device, batch_size=batch_a)
    pred_b = reconstruct_control_raw(st_mix_model, st_ctrl, device=device, batch_size=batch_b)
    difference = pred_a.astype(np.float64) - pred_b.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference**2)))
    denominator = float(np.sqrt(np.mean(pred_a.astype(np.float64) ** 2)))
    if np.std(pred_a) == 0 or np.std(pred_b) == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(pred_a.ravel(), pred_b.ravel())[0, 1])
    return {
        "same_checkpoint": True,
        "residual_centering_scope": "global_spot",
        "global_center_recomputed_for_each_complete_inference_pass": True,
        "batch_a": int(batch_a),
        "batch_b": int(batch_b),
        "prediction_RMSE": rmse,
        "prediction_relative_RMSE": rmse / (denominator + 1e-12),
        "prediction_global_correlation": correlation,
        "prediction_max_absolute_difference": float(np.max(np.abs(difference))),
        "interpretation_required_before_CellOT": True,
    }


def _transport_in_chunks(g, latent: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    latent = np.asarray(latent)
    expected_rows = int(latent.shape[0])
    if not np.isfinite(latent).all():
        raise FloatingPointError("CellOT input latent contains NaN/Inf")
    pieces = []
    g.eval()
    for start in range(0, len(latent), batch_size):
        source = torch.as_tensor(
            latent[start : start + batch_size], dtype=torch.float32, device=device
        ).requires_grad_(True)
        with torch.enable_grad():
            transported = g.transport(source)
        piece = transported.detach().cpu().numpy()
        if not np.isfinite(piece).all():
            raise FloatingPointError("CellOT transported latent contains NaN/Inf")
        pieces.append(piece)
    result = np.concatenate(pieces, axis=0)
    if int(result.shape[0]) != expected_rows:
        raise RuntimeError(
            "CellOT changed latent cardinality: "
            f"input={expected_rows}, output={result.shape[0]}"
        )
    return result


def predict_proposed_raw_full_gene(
    *,
    adata_latent_st: ad.AnnData,
    condition_value: str,
    g,
    st_mix_model,
    st_ctrl: ad.AnnData,
    p_known: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> ad.AnnData:
    """Memory-safe equivalent of transport + STMix NB conversion.

    The canonical converter materializes an S x T x G tensor.  At G=8,000
    that is unnecessary: each selected pseudo-cell contribution can be added
    directly into its spot row.  The transport, decoder, weights, background,
    ST scale, and library-size conversion are unchanged.
    """
    latent = adata_latent_st[adata_latent_st.obs["condition"] == condition_value].copy()
    source = np.asarray(latent.X, dtype=np.float32)
    residual = np.asarray(latent.obsm["R_spot"], dtype=np.float32)
    transported = _transport_in_chunks(g, source, device=device, batch_size=batch_size)
    transported += residual

    spot_idx = latent.obs["spot_index"].to_numpy(dtype=np.int64)
    type_idx = latent.obs["type_index"].to_numpy(dtype=np.int64)
    S, T, G = st_ctrl.n_obs, st_mix_model.T_known, st_mix_model.G
    p_known = np.asarray(p_known, dtype=np.float32)
    if p_known.shape != (S, T) or G != MODEL_GENE_COUNT:
        raise ValueError("Full-gene Proposed aggregation shape mismatch")
    pair_code = spot_idx * T + type_idx
    if len(np.unique(pair_code)) != len(pair_code):
        raise AssertionError("Pseudo-cell table contains duplicate spot/type pairs")

    mask = np.zeros((S, T), dtype=bool)
    mask[spot_idx, type_idx] = True
    selected_mass = (p_known * mask).sum(axis=1, dtype=np.float64).astype(np.float32)
    p_bg = np.clip(1.0 - selected_mass, 0.0, 1.0)
    spot_mix = torch.zeros((S, G), dtype=torch.float32, device=device)
    st_mix_model.eval()
    with torch.no_grad():
        for start in range(0, len(transported), batch_size):
            stop = min(start + batch_size, len(transported))
            z = torch.as_tensor(transported[start:stop], dtype=torch.float32, device=device)
            mu = torch.expm1(st_mix_model.decoder(z).clamp(-10.0, 10.0)).clamp(0.0, 1e6)
            si_np = spot_idx[start:stop]
            ti_np = type_idx[start:stop]
            weight = torch.as_tensor(
                p_known[si_np, ti_np], dtype=torch.float32, device=device
            ).unsqueeze(1)
            si = torch.as_tensor(si_np, dtype=torch.long, device=device)
            spot_mix.index_add_(0, si, weight * mu)

        bg = torch.expm1(
            st_mix_model.decoder(st_mix_model.mu_bg_latent).clamp(-10.0, 10.0)
        ).clamp(0.0, 1e6)
        spot_mix += torch.as_tensor(p_bg, device=device).unsqueeze(1) * bg
        spot_mix *= torch.exp(st_mix_model.st_log_scale)

        raw = st_ctrl.layers["raw_counts"]
        raw_totals = np.asarray(raw.sum(axis=1)).ravel().astype(np.float32)
        library_scale = torch.as_tensor(
            raw_totals / float(st_mix_model.target_sum), dtype=torch.float32, device=device
        ).clamp(1e-3, 1e3)
        prediction = (spot_mix * library_scale.unsqueeze(1)).clamp(1e-8, 1e6)
        prediction = prediction.cpu().numpy().astype(np.float32)

    result = ad.AnnData(X=prediction, obs=st_ctrl.obs.copy(), var=st_ctrl.var.copy())
    result.var_names = st_ctrl.var_names.copy()
    result.layers["nb_mu_pred"] = prediction
    result.uns["prediction_scale"] = "NB expected raw counts"
    result.uns["aggregation_implementation"] = (
        "memory-safe direct spot index_add; algebraically equivalent to SxTxG canonical converter"
    )
    result.uns["residual_free_cellot_input"] = True
    result.uns["residual_added_after_transport"] = True
    return result


def predict_simo_normalized_log1p(
    *,
    mapped_path: Path,
    rna_ctrl: ad.AnnData,
    st_ctrl: ad.AnnData,
    g,
    vae_model,
    device: torch.device,
) -> ad.AnnData:
    """Apply CellOT to SIMO-mapped cells and aggregate on normalized scale.

    `obs['spot']` is the SIMO assignment.  The ground-truth `spot_name` column
    is deliberately never used for aggregation.
    """
    mapped = ad.read_h5ad(mapped_path)
    if mapped.obs["Cell"].astype(str).duplicated().any():
        raise AssertionError("SIMO mapping contains duplicate cells")

    ctrl_by_id = pd.Series(np.arange(rna_ctrl.n_obs), index=rna_ctrl.obs["cell_id"].astype(str))
    mapped_ids = mapped.obs["Cell"].astype(str)
    missing = mapped_ids[~mapped_ids.isin(ctrl_by_id.index)]
    if len(missing):
        raise RuntimeError(
            f"{len(missing)} SIMO cells were removed by the modeling QC; refusing a silent subset."
        )
    ctrl_rows = ctrl_by_id.loc[mapped_ids].to_numpy(dtype=np.int64)
    ctrl_latent = np.asarray(rna_ctrl.obsm["Z_latent"], dtype=np.float32)[ctrl_rows]
    transported = _transport_in_chunks(g, ctrl_latent, device=device)

    spot_index = pd.Index(st_ctrl.obs_names.astype(str))
    assigned = mapped.obs["spot"].astype(str)
    if not assigned.isin(spot_index).all():
        raise ValueError("SIMO obs['spot'] contains IDs absent from ST")
    if set(assigned) != set(spot_index):
        raise ValueError("SIMO mapping does not cover every corrected ST spot")
    spot_code = spot_index.get_indexer(assigned)
    spot_linear = np.zeros((st_ctrl.n_obs, rna_ctrl.n_vars), dtype=np.float32)
    vae_model.eval()
    with torch.no_grad():
        for start in range(0, len(transported), 512):
            z = torch.as_tensor(transported[start : start + 512], dtype=torch.float32, device=device)
            cell_linear = np.expm1(vae_model.decode(z).cpu().numpy()).clip(min=0)
            stop = start + len(cell_linear)
            np.add.at(spot_linear, spot_code[start:stop], cell_linear)

    spot_log = normalize_log1p(spot_linear, target_sum=TARGET_SUM).astype(np.float32)
    result = ad.AnnData(
        X=spot_log,
        obs=st_ctrl.obs.copy(),
        var=rna_ctrl.var.copy(),
    )
    result.var_names = rna_ctrl.var_names.copy()
    result.obsm["spatial"] = np.asarray(st_ctrl.obsm["X_spatial"])
    result.uns["prediction_scale"] = "library-normalized target_sum=10000 then log1p"
    result.uns["raw_counts_available"] = False
    result.uns["simo_assignment_column"] = "spot"
    result.uns["ground_truth_assignment_column_not_used"] = "spot_name"
    return result


__all__ = [
    "MASS_THRESHOLD",
    "MODEL_GENE_COUNT",
    "TARGET_SUM",
    "TRAINING_SEED",
    "predict_proposed_raw_full_gene",
    "predict_simo_normalized_log1p",
    "reconstruct_control_raw",
    "residual_batch_diagnostic",
    "train_autoencoder_full_gene",
]
