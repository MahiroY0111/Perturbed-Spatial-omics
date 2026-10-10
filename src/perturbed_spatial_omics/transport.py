"""CellOT training and ST transport without experimental auxiliary losses."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def default_cellot_config() -> Any:
    """Return a DotMap config matching the notebook defaults."""
    from dotmap import DotMap

    config_dict = {
        "model": {
            "name": "cellot",
            "hidden_units": [64, 64, 64, 64],
            "f": {"kernel_init_fxn": {"name": "normal", "std": 0.02}},
            "g": {
                "kernel_init_fxn": {"name": "normal", "std": 0.02},
                "fnorm_penalty": 1,
            },
        },
        "kernel_init_fxn": {
            "b": 0.1,
            "name": "uniform",
        },
        "optim": {
            "optimizer": "Adam",
            "lr": 1e-4,
            "beta1": 0.5,
            "beta2": 0.9,
        },
        "training": {
            "n_iters": 4000,
            "n_inner_iters": 10,
            "logs_freq": 100,
            "eval_freq": 100,
            "cache_freq": 200,
        },
        "data": {
            "type": "cell_type",
            "source": "ctrl",
            "target": "stim",
            "condition": "condition",
        },
        "dataloader": {
            "batch_size": 256,
            "shuffle": True,
        },
    }
    return DotMap(config_dict)


def prepare_cellot_anndata(
    adata_rna,
    adata_st,
    latent_key: str = "Z_latent",
    condition_col: str = "condition",
    transport_map: dict[str, str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Build latent AnnData objects for CellOT training and ST transport."""
    import anndata as ad
    from sklearn.model_selection import train_test_split

    adata_latent = ad.AnnData(adata_rna.obsm[latent_key], adata_rna.obs.copy())
    adata_latent.var_names = [f"latent_{i+1}" for i in range(adata_latent.n_vars)]

    adata_latent_st = ad.AnnData(adata_st.obsm[latent_key], adata_st.obs.copy())
    adata_latent_st.var_names = [f"latent_{i+1}" for i in range(adata_latent_st.n_vars)]
    if "R_spot" in adata_st.obsm:
        adata_latent_st.obsm["R_spot"] = adata_st.obsm["R_spot"].copy()

    if transport_map is None:
        transport_map = {"ctrl": "source", "stim": "target"}

    adata_latent.obs["transport"] = (
        adata_latent.obs[condition_col].map(transport_map).astype("category")
    )

    train_idx, _test_idx = train_test_split(
        adata_latent.obs.index, test_size=test_size, random_state=random_state
    )
    adata_latent.obs["split"] = "test"
    adata_latent.obs.loc[train_idx, "split"] = "train"
    adata_latent.obs["split"] = adata_latent.obs["split"].astype("category")

    return adata_latent, adata_latent_st


def init_cellot(config, adata_latent, device: torch.device | None = None):
    """Initialize CellOT model, optimizers, and iterators."""
    from dotmap import DotMap
    from cellot.data.cell import load_cell_data
    from cellot.data.utils import cast_loader_to_iterator
    from cellot.models.cellot import load_cellot_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(config, dict):
        config = DotMap(config)

    loader = load_cell_data(config, data=adata_latent, return_as="loader")
    iterator = cast_loader_to_iterator(loader, cycle_all=True)

    (f, g), opts = load_cellot_model(config, input_dim=adata_latent.n_vars)
    f.to(device)
    g.to(device)

    return config, iterator, f, g, opts.f, opts.g


def train_cellot(
    f,
    g,
    iterator,
    opt_f,
    opt_g,
    config,
    device: torch.device | None = None,
):
    """Train CellOT with the original alternating neural OT objective."""
    from tqdm.auto import trange
    from cellot import losses as cellot_losses
    from cellot.models.cellot import compute_loss_f, compute_loss_g, compute_w2_distance

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_loss(*args):
        for arg in args:
            if torch.isnan(arg):
                raise ValueError("NaN detected in loss")

    n_iters = int(config.training.n_iters)
    step = 0
    minmmd = np.inf

    def evaluate():
        iterator_test_target = iterator.test.target
        iterator_test_source = iterator.test.source

        target = next(iterator_test_target).to(device)
        source = next(iterator_test_source).to(device).requires_grad_(True)
        transport = g.transport(source).detach()

        with torch.no_grad():
            gl = compute_loss_g(f, g, source, transport).mean()
            fl = compute_loss_f(f, g, source, target, transport).mean()
            dist = compute_w2_distance(f, g, source, target, transport)
            mmd = cellot_losses.compute_scalar_mmd(
                target.detach().cpu().numpy(),
                transport.detach().cpu().numpy(),
            )

        check_loss(gl, fl, dist)
        print(
            f"  Eval - G Loss: {gl.item():.4f}, F Loss: {fl.item():.4f}, "
            f"W2 Distance: {dist.item():.4f}, MMD: {mmd:.6f}"
        )
        return mmd

    loss_history = {
        "step": [],
        "loss_f": [],
        "loss_g_core": [],
        "loss_g_total": [],
        "convexity_penalty": [],
        "mmd": [],
        "loss_diff": [],
    }

    f.train()
    g.train()

    iterator_train_target = iterator.train.target
    iterator_train_source = iterator.train.source

    ticker = trange(step, n_iters, initial=step, total=n_iters, desc="Training CellOT")

    for step in ticker:
        target = next(iterator_train_target).to(device)

        g_losses_core = []
        g_losses_total = []
        penalties = []

        for _ in range(int(config.training.n_inner_iters)):
            source = next(iterator_train_source).to(device).requires_grad_(True)
            opt_g.zero_grad()

            g_loss_core = compute_loss_g(f, g, source).mean()

            penalty = 0.0
            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                penalty = g.penalize_w()

            g_loss_total = g_loss_core + penalty

            g_loss_total.backward()
            opt_g.step()

            g_losses_core.append(g_loss_core.item())
            g_losses_total.append(g_loss_total.item())
            penalties.append(penalty.item() if isinstance(penalty, torch.Tensor) else float(penalty))

        source = next(iterator_train_source).to(device).requires_grad_(True)
        opt_f.zero_grad()
        fl = compute_loss_f(f, g, source, target).mean()
        fl.backward()
        opt_f.step()
        check_loss(g_loss_total, fl)
        f.clamp_w()

        avg_g_core = float(np.mean(g_losses_core))
        avg_g_total = float(np.mean(g_losses_total))
        avg_penalty = float(np.mean(penalties))
        f_loss = fl.item()

        loss_history["step"].append(step)
        loss_history["loss_f"].append(f_loss)
        loss_history["loss_g_core"].append(avg_g_core)
        loss_history["loss_g_total"].append(avg_g_total)
        loss_history["convexity_penalty"].append(avg_penalty)

        loss_diff = abs(f_loss) - abs(avg_g_core)
        loss_history["loss_diff"].append(loss_diff)

        if step % int(config.training.logs_freq) == 0:
            postfix_dict = {
                "loss_f": f"{f_loss:.4f}",
                "loss_g_core": f"{avg_g_core:.4f}",
                "loss_g_total": f"{avg_g_total:.4f}",
                "penalty": f"{avg_penalty:.4f}",
            }
            ticker.set_postfix(postfix_dict)

        if step % int(config.training.eval_freq) == 0:
            f.eval()
            g.eval()
            mmd = evaluate()
            f.train()
            g.train()

            loss_history["mmd"].append({"step": step, "value": mmd})
            if mmd < minmmd:
                minmmd = mmd
                print(f"  New best MMD: {minmmd:.6f}")

    ticker.close()
    print("--- CellOT training complete ---")
    print(f"Best MMD achieved: {minmmd:.6f}")
    return loss_history, minmmd


def plot_cellot_loss_history(loss_history: dict[str, list[Any]]) -> None:
    """Plot the CellOT loss curves."""
    import matplotlib.pyplot as plt

    plt.style.use("default")
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("CellOT Training Loss History", fontsize=16, y=1.01)

    ax1 = axes[0, 0]
    ax1.plot(loss_history["step"], loss_history["loss_f"], "b-", alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss f")
    ax1.set_title("Potential Function Loss (f)")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(loss_history["step"], loss_history["loss_g_core"], "g-", alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss g")
    ax2.set_title("Transport Map Loss (g)")
    ax2.grid(True, alpha=0.3)
    if min(loss_history["loss_g_core"]) < 0:
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.5)

    ax3 = axes[1, 0]
    ax3.plot(
        loss_history["step"],
        loss_history["convexity_penalty"],
        "r-",
        alpha=0.7,
        linewidth=1.5,
    )
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Penalty")
    ax3.set_title("Negative-weight Penalty for g")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    ax4 = axes[1, 1]
    if loss_history["mmd"]:
        mmd_steps = [item["step"] for item in loss_history["mmd"]]
        mmd_values = [item["value"] for item in loss_history["mmd"]]
        ax4.plot(mmd_steps, mmd_values, "mo-", alpha=0.7, linewidth=1.5, markersize=5)
        ax4.set_xlabel("Step")
        ax4.set_ylabel("MMD")
        ax4.set_title("Maximum Mean Discrepancy (Lower is Better)")
        ax4.grid(True, alpha=0.3)

        min_idx = int(np.argmin(mmd_values))
        ax4.plot(
            mmd_steps[min_idx],
            mmd_values[min_idx],
            "r*",
            markersize=15,
            label=f"Best MMD: {mmd_values[min_idx]:.6f}",
        )
        ax4.legend()

    ax5 = axes[2, 0]
    ax5.plot(
        loss_history["step"],
        [abs(l) for l in loss_history["loss_f"]],
        "b-",
        alpha=0.7,
        linewidth=1.5,
        label="|Loss f|",
    )
    ax5.plot(
        loss_history["step"],
        [abs(l) for l in loss_history["loss_g_core"]],
        "g-",
        alpha=0.7,
        linewidth=1.5,
        label="|Loss g|",
    )
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Absolute Loss Value")
    ax5.set_title("Absolute Values of Losses (Convergence Check)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale("log")

    ax6 = axes[2, 1]
    ax6.plot(loss_history["step"], loss_history["loss_diff"], "purple", alpha=0.7, linewidth=1.5)
    ax6.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax6.set_xlabel("Step")
    ax6.set_ylabel("|Loss f| - |Loss g|")
    ax6.set_title("Difference in Absolute Losses (→0 means convergence)")
    ax6.grid(True, alpha=0.3)
    if len(loss_history["loss_diff"]) > 100:
        last_20_percent = loss_history["loss_diff"][int(len(loss_history["loss_diff"]) * 0.8) :]
        y_range = max(abs(min(last_20_percent)), abs(max(last_20_percent))) * 1.2
        ax6.set_ylim(-y_range, y_range)

    plt.tight_layout()
    plt.show()


def summarize_cellot_convergence(loss_history: dict[str, list[Any]]) -> None:
    """Print convergence stats for CellOT."""
    print("\n=== Convergence Analysis ===")
    print(f"Initial |f| - |g|: {loss_history['loss_diff'][0]:.6f}")
    print(f"Final |f| - |g|: {loss_history['loss_diff'][-1]:.6f}")

    last_10_percent_idx = int(len(loss_history["loss_diff"]) * 0.9)
    recent_diffs = loss_history["loss_diff"][last_10_percent_idx:]
    print("\nLast 10% of training:")
    print(f"  Mean |f| - |g|: {np.mean(recent_diffs):.6f}")
    print(f"  Std |f| - |g|: {np.std(recent_diffs):.6f}")
    print(f"  Min |f| - |g|: {min(recent_diffs):.6f}")
    print(f"  Max |f| - |g|: {max(recent_diffs):.6f}")

    convergence_rate = abs(loss_history["loss_diff"][-1] / loss_history["loss_diff"][0])
    print(f"\nConvergence rate: {convergence_rate:.4f} (closer to 0 is better)")

    print("\n=== Loss g Statistics ===")
    print(f"Min Loss g: {min(loss_history['loss_g_core']):.6f}")
    print(f"Max Loss g: {max(loss_history['loss_g_core']):.6f}")
    print(f"Mean Loss g: {np.mean(loss_history['loss_g_core']):.6f}")
    print(f"Final Loss g: {loss_history['loss_g_core'][-1]:.6f}")
    negative_ratio = sum(1 for l in loss_history["loss_g_core"] if l < 0) / len(
        loss_history["loss_g_core"]
    )
    print(f"Percentage of negative Loss g values: {negative_ratio*100:.2f}%")


def transport_and_decode(
    adata_latent_st,
    adata_hvg,
    g,
    vae_model,
    device: torch.device | None = None,
    condition_value: str = "st_ctrl_pseudo",
):
    """Transport ST pseudo latents and decode to gene space."""
    import anndata as ad

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g.eval()
    vae_model.eval()

    adata_st_latent = adata_latent_st[adata_latent_st.obs["condition"] == condition_value].copy()
    source_st_tensor = torch.tensor(
        adata_st_latent.X,
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)

    resid_tensor = torch.tensor(
        adata_st_latent.obsm["R_spot"],
        dtype=torch.float32,
        device=device,
    )

    with torch.enable_grad():
        transported_pseudo = g.transport(source_st_tensor)
        transported_latent = transported_pseudo + resid_tensor

    with torch.no_grad():
        decoded_X = vae_model.decode(transported_latent).cpu().numpy()

    transported_adata_st_decoded = ad.AnnData(
        decoded_X,
        obs=adata_st_latent.obs.copy(),
    )
    transported_adata_st_decoded.var_names = adata_hvg.var_names

    return transported_adata_st_decoded, transported_latent, transported_pseudo, adata_st_latent


def convert_transported_to_nb_counts(
    transported_adata_st_decoded,
    transported_latent,
    st_mix_model,
    st_ctrl_hvg,
    p_known_all,
    device: torch.device | None = None,
    use_bg: bool = True,
    renorm_if_no_bg: bool = True,
    subset_to_pseudo_spots_only: bool = True,
):
    """Convert transported pseudo latents into NB-count scale using STMix."""
    import anndata as ad

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spot_idx = transported_adata_st_decoded.obs["spot_index"].to_numpy().astype(np.int64)
    type_idx = transported_adata_st_decoded.obs["type_index"].to_numpy().astype(np.int64)

    S = st_ctrl_hvg.n_obs
    T_known = st_mix_model.T_known
    G = st_mix_model.G

    if not torch.is_tensor(transported_latent):
        transported_latent = torch.as_tensor(transported_latent, dtype=torch.float32, device=device)
    else:
        transported_latent = transported_latent.to(device)

    p_known_all_torch = torch.as_tensor(p_known_all, dtype=torch.float32, device=device)
    if p_known_all_torch.shape != (S, T_known):
        raise ValueError("p_known_all shape mismatch")

    mask = torch.zeros((S, T_known), dtype=torch.bool, device=device)
    si = torch.as_tensor(spot_idx, device=device, dtype=torch.long)
    ti = torch.as_tensor(type_idx, device=device, dtype=torch.long)
    mask[si, ti] = True

    p_known_masked = p_known_all_torch * mask.float()
    known_sum = p_known_masked.sum(dim=1, keepdim=True)

    if use_bg:
        p_bg = (1.0 - known_sum).clamp(min=0.0, max=1.0)
    else:
        p_bg = None
        if renorm_if_no_bg:
            p_known_masked = p_known_masked / known_sum.clamp_min(1e-8)

    with torch.no_grad():
        y_sel = st_mix_model.decoder(transported_latent).clamp(min=-10.0, max=10.0)
        mu_sel = torch.expm1(y_sel).clamp(min=0.0, max=1e6)

        mu_base_pseudo = torch.zeros((S, T_known, G), device=device)
        mu_base_pseudo[si, ti, :] = mu_sel

        if use_bg:
            y_bg = st_mix_model.decoder(st_mix_model.mu_bg_latent).clamp(min=-10.0, max=10.0)
            mu_base_bg = torch.expm1(y_bg).clamp(min=0.0, max=1e6)
            mu_base_mix = (p_known_masked.unsqueeze(-1) * mu_base_pseudo).sum(dim=1) + p_bg * mu_base_bg
        else:
            mu_base_mix = (p_known_masked.unsqueeze(-1) * mu_base_pseudo).sum(dim=1)

        st_scale = torch.exp(st_mix_model.st_log_scale)
        mu_st = mu_base_mix * st_scale

        RAW_full = st_ctrl_hvg.layers["raw_counts"]
        if hasattr(RAW_full, "toarray"):
            RAW_full = RAW_full.toarray()
        lib_arr = np.asarray(RAW_full.sum(axis=1)).astype(np.float32)
        lib_tensor = torch.as_tensor(lib_arr, dtype=torch.float32, device=device).view(-1, 1)

        kappa = float(st_mix_model.target_sum)
        scale = (lib_tensor / kappa).clamp(min=1e-3, max=1e3)

        mu_nb_counts_all = (mu_st * scale).clamp(min=1e-8, max=1e6)

    if subset_to_pseudo_spots_only:
        spot_keep = np.unique(spot_idx)
    else:
        spot_keep = np.arange(S)

    spot_keep_t = torch.as_tensor(spot_keep, device=device, dtype=torch.long)
    transported_nb_counts_t = mu_nb_counts_all[spot_keep_t, :]
    transported_nb_counts = transported_nb_counts_t.detach().cpu().numpy().astype(np.float32)

    transported_adata_st_nb = ad.AnnData(
        X=transported_nb_counts,
        obs=st_ctrl_hvg.obs.iloc[spot_keep].copy(),
        var=transported_adata_st_decoded.var.copy(),
    )
    transported_adata_st_nb.var_names = transported_adata_st_decoded.var_names.copy()
    transported_adata_st_nb.layers["nb_mu_pred"] = transported_nb_counts

    return transported_adata_st_nb
