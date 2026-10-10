# coding_0107_stmix_spatial.py
# ============================================================
# STMix (DestVI-style pseudo-cell + residual) with OPTIONAL
# gated spatial regularization on gamma_{j,t}.
#
# Drop-in replacement for your existing coding_0107_stmix.py.
# You can import the same symbols as before.
#
# Key additions:
#   - STDataset returns (x_norm, x_raw, log_lib, spot_idx, coords)
#   - kNN graph is built once from coords (whole dataset)
#   - L_sp_gamma is computed inside each batch using in-batch edges:
#       L = sum_{(j,k) in E_batch} w_jk * sum_t a_jk,t ||gamma_jt - gamma_kt||^2
#       a_jk,t = sqrt(p_jt p_kt) or min(p_jt, p_kt), using p_known.detach()
#
# IMPORTANT:
#   shuffle=False is beneficial only when dataset ordering is spatially local.
#   Otherwise, use a spatially aware sampler or another strategy ensuring that
#   kNN-connected spots are jointly sampled.
# ============================================================

import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Utils
# ============================================================

def inv_softplus(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse of softplus used for initializing NB dispersion parameters."""
    return torch.log(torch.exp(y) - 1.0 + eps)


def negative_binomial_nll(
    x_raw: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative binomial NLL.

    x_raw, mu: [B, G]
    theta: [G] (gene-wise dispersion) or [B, G]

    Returns mean negative log-likelihood (scalar).
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0).expand_as(mu)

    theta = theta.clamp(min=0.1)
    mu = mu.clamp(min=eps)

    log_theta_mu = torch.log(theta + mu + eps)

    lgamma_theta_x = torch.lgamma(theta + x_raw + eps)
    lgamma_theta = torch.lgamma(theta + eps)
    lgamma_x_plus_1 = torch.lgamma(x_raw + 1.0 + eps)

    log_likelihood = (
        lgamma_theta_x
        - lgamma_theta
        - lgamma_x_plus_1
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x_raw * (torch.log(mu + eps) - log_theta_mu)
    )
    return -log_likelihood.mean()


try:  # ResBlock で利用
    from torch.nn.utils.parametrizations import spectral_norm  # type: ignore
except Exception:  # pragma: no cover - fallback for older PyTorch
    from torch.nn.utils import spectral_norm  # type: ignore


class ResBlock(nn.Module):
    def __init__(self, d: int, res_scale: float = 0.25, use_sn: bool = False) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        lin1 = nn.Linear(d, d, bias=False)
        lin2 = nn.Linear(d, d, bias=True)
        if use_sn:
            lin1 = spectral_norm(lin1, n_power_iterations=1)
            lin2 = spectral_norm(lin2, n_power_iterations=1)
        self.fc1, self.fc2 = lin1, lin2
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.1)
        self.res_scale = float(res_scale)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + self.res_scale * h


class SafeCondFiLM(nn.Module):
    """Safe FiLM conditioned on log-library size (1D)."""

    def __init__(self, cond_dim: int, d: int, eta: float = 0.1, use_layernorm: bool = True) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * d),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.eta = float(eta)
        self.logit_gate = nn.Parameter(torch.tensor(-2.0))
        self.use_ln = use_layernorm
        self.ln = nn.LayerNorm(d, elementwise_affine=False) if use_layernorm else nn.Identity()

        self.register_buffer("c_mean", torch.zeros(1, 1))
        self.register_buffer("c_std", torch.ones(1, 1))
        self.register_buffer("g0", torch.zeros(1, 2 * d))

    @torch.no_grad()
    def set_condition_stats(self, mean: float, std: float) -> None:
        self.c_mean.copy_(torch.tensor([[mean]], dtype=torch.float32, device=self.c_mean.device))
        self.c_std.copy_(torch.tensor([[max(std, 1e-6)]], dtype=torch.float32, device=self.c_std.device))

    @torch.no_grad()
    def calibrate_g0(self, device: Optional[torch.device] = None) -> None:
        z = torch.zeros(1, 1, device=device if device is not None else self.c_mean.device)
        self.g0.copy_(self.mlp(z))

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """h: [B, d], c: [B, 1]."""
        cz = (c - self.c_mean) / self.c_std
        g = self.mlp(cz) - self.g0
        g_gamma, g_beta = g.chunk(2, dim=-1)
        gate = torch.sigmoid(self.logit_gate)
        gamma = 1.0 + self.eta * gate * torch.tanh(g_gamma)
        beta = self.eta * gate * torch.tanh(g_beta)
        x = self.ln(h)
        gamma = gamma.clamp(0.5, 1.5)
        return gamma * x + beta


# ============================================================
# Spatial graph helpers
# ============================================================

def build_knn_graph(coords_np: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build kNN graph from coords.

    coords_np: [S, 2] or [S, d_coord]
    return:
      neigh_idx: [S, k] (self excluded)
      neigh_w  : [S, k] gaussian weights from distances
    """
    from sklearn.neighbors import NearestNeighbors

    if coords_np.ndim != 2:
        raise ValueError("coords_np must be 2D array [S, d_coord]")
    S = coords_np.shape[0]
    if S <= k:
        raise ValueError(f"Too few spots (S={S}) for kNN (k={k}). Reduce k.")

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords_np)
    dists, idx = nbrs.kneighbors(coords_np)  # [S, k+1] includes self

    neigh_idx = idx[:, 1:].astype(np.int64)      # [S, k]
    neigh_dist = dists[:, 1:].astype(np.float64) # [S, k]

    # Robust distance scale (median of all neighbor distances)
    sigma = float(np.median(neigh_dist) + 1e-8)
    neigh_w = np.exp(-(neigh_dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)

    return neigh_idx, neigh_w


def make_batch_edges(
    idx_batch: torch.Tensor,
    neigh_idx: np.ndarray,
    neigh_w: np.ndarray,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert global kNN edges to in-batch edges (local indices).

    idx_batch: [B] global spot indices
    neigh_idx, neigh_w: [S, k] (numpy, CPU)
    return:
      src_local [E], dst_local [E], w [E] on same device as idx_batch
    """
    device = idx_batch.device
    idx_cpu = idx_batch.detach().cpu().numpy().astype(np.int64)

    # global -> local position in this batch
    pos = {int(g): i for i, g in enumerate(idx_cpu)}

    src_list, dst_list, w_list = [], [], []
    for g in idx_cpu:
        i = pos[int(g)]
        nbs = neigh_idx[int(g)]
        ws = neigh_w[int(g)]
        for nb, ww in zip(nbs, ws):
            j = pos.get(int(nb), None)
            if j is None:
                continue
            src_list.append(i)
            dst_list.append(j)
            w_list.append(float(ww))

    if len(src_list) == 0:
        return None

    src = torch.tensor(src_list, dtype=torch.long, device=device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    w = torch.tensor(w_list, dtype=torch.float32, device=device)
    return src, dst, w


def spatial_gamma_loss(
    gamma: torch.Tensor,      # [B, T, D]
    p_known: torch.Tensor,    # [B, T]
    src: torch.Tensor,        # [E]
    dst: torch.Tensor,        # [E]
    w: torch.Tensor,          # [E]
    gate: str = "sqrt",       # "sqrt" or "min"
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Gated spatial smoothness on gamma.

    L = sum_e w_e * sum_t a_e,t * ||gamma_src - gamma_dst||^2 / sum_e w_e

    a_e,t uses p_known.detach() to avoid "escaping" by shrinking p.
    """
    if gamma.dim() != 3:
        raise ValueError("gamma must be [B, T, D]")
    if p_known.dim() != 2:
        raise ValueError("p_known must be [B, T]")
    if gamma.size(0) != p_known.size(0) or gamma.size(1) != p_known.size(1):
        raise ValueError("gamma and p_known must match on [B, T] dims")

    p = p_known.detach()

    p_src = p[src]  # [E, T]
    p_dst = p[dst]  # [E, T]

    if gate == "sqrt":
        a = torch.sqrt((p_src * p_dst).clamp_min(0.0) + eps)  # [E, T]
    elif gate == "min":
        a = torch.minimum(p_src, p_dst)  # [E, T]
    else:
        raise ValueError("gate must be 'sqrt' or 'min'")

    diff = gamma[src] - gamma[dst]   # [E, T, D]
    sq = diff.pow(2).sum(dim=-1)     # [E, T]
    per_edge = (a * sq).sum(dim=-1)  # [E]

    return (w * per_edge).sum() / (w.sum() + eps)


# ============================================================
# Dataset
# ============================================================

class STDataset(Dataset):
    """ST dataset wrapper.
    Provides:
      - x_norm: normalized expression [S, G]
      - x_raw : raw counts          [S, G]
      - log_lib: log library size   [S, 1]
      - coords: spatial coords      [S, 2] (or [S, d])
    """

    def __init__(self, adata, target_sum: float = 1e4, spatial_key: str = "X_spatial") -> None:
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        RAW = (
            adata.layers["raw_counts"].toarray()
            if hasattr(adata.layers["raw_counts"], "toarray")
            else adata.layers["raw_counts"]
        )

        self.x_norm = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.x_raw = torch.as_tensor(np.asarray(RAW, dtype=np.float32))

        if "lib_raw" in getattr(adata, "obs", {}).columns:
            lib = adata.obs["lib_raw"].values.astype(np.float32)
        else:
            lib = np.array(RAW.sum(axis=1)).astype(np.float32)

        self.log_lib = torch.log(torch.as_tensor(lib).clamp_min(1.0)).unsqueeze(1)
        self.target_sum = float(target_sum)

        # spatial coords
        if hasattr(adata, "obsm") and (spatial_key in adata.obsm_keys()):
            coords = adata.obsm[spatial_key]
        else:
            raise ValueError(f"adata.obsm['{spatial_key}'] が見つかりません")
        self.coords = torch.as_tensor(np.asarray(coords, dtype=np.float32))

        if self.coords.shape[0] != self.x_norm.shape[0]:
            raise ValueError("coords and X must have the same number of rows/spots")

    def __len__(self) -> int:  # type: ignore[override]
        return self.x_norm.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        # return global index and coords for spatial regularization
        return self.x_norm[idx], self.x_raw[idx], self.log_lib[idx], idx, self.coords[idx]


# ============================================================
# Model
# ============================================================

class STMixEncoder(nn.Module):
    """Encoder for ST spots.

    Inputs x_norm, log_lib and outputs:
      - p_j      (cell-type mix, T+1 dim: last is unknown/bg)
      - logits   (mix logits)
      - r_j      (latent residual, D dim)
      - gamma_jt (state latent, [T_known, D])
    """

    def __init__(
        self,
        input_dim: int,
        n_types_plus_bg: int,
        latent_dim: int,
        hidden_units: list[int] | tuple[int, ...] = (128, 128),
        res_scale: float = 0.25,
        use_sn: bool = True,
        film_eta: float = 0.1,
    ) -> None:
        super().__init__()
        d = int(hidden_units[0])
        self.fc_in = nn.Linear(input_dim, d)

        self.blocks = nn.ModuleList(
            [ResBlock(d, res_scale=res_scale, use_sn=use_sn) for _ in range(len(hidden_units))]
        )
        self.film = SafeCondFiLM(cond_dim=1, d=d, eta=film_eta, use_layernorm=True)

        self.n_types_plus_bg = n_types_plus_bg
        self.T_known = n_types_plus_bg - 1  # last is bg/unknown

        # cell-type mix head (T+1 dim)
        self.fc_mix = nn.Linear(d, n_types_plus_bg)

        # pseudo-cell state head gamma_{j,t} (T_known×D)
        self.latent_dim = latent_dim
        self.fc_state = nn.Linear(d, self.T_known * latent_dim)

        # latent residual head (D)
        self.fc_res = nn.Linear(d, latent_dim)
        nn.init.zeros_(self.fc_res.weight)
        nn.init.zeros_(self.fc_res.bias)

    def forward(self, x: torch.Tensor, log_lib: torch.Tensor):
        """x: [B, G], log_lib: [B, 1]."""
        h = self.fc_in(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.film(h, log_lib)

        logits = self.fc_mix(h)
        p = F.softmax(logits, dim=-1)

        B = x.size(0)
        state_flat = self.fc_state(h)  # [B, T_known*D]
        gamma = state_flat.view(B, self.T_known, self.latent_dim)

        r = self.fc_res(h)  # [B, D]
        return p, logits, r, gamma


class STMixModel(nn.Module):
    """DestVI-style pseudo-cell + residual ST mix model."""

    def __init__(
        self,
        encoder: STMixEncoder,
        R_gene: torch.Tensor,          # [T, G], legacy normalized-log profiles
        mu_sc_types: torch.Tensor,     # [T, D]
        z_sc_all: torch.Tensor,        # [N_sc, D]
        cell_type_index_sc: torch.Tensor,  # [N_sc]
        decoder,
        target_sum: float = 1e4,
        R_gene_frequency: torch.Tensor | None = None,  # [T, G], row sums = 1
        lmix_mode: str = "frequency",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_sum = float(target_sum)
        self.decoder = decoder
        if lmix_mode not in {"frequency", "legacy_log"}:
            raise ValueError("lmix_mode must be 'frequency' or 'legacy_log'")
        self.lmix_mode = str(lmix_mode)

        T, G = R_gene.shape
        D = mu_sc_types.shape[1]
        assert mu_sc_types.shape[0] == T

        self.T_known = T
        self.G = G
        self.D = D

        # The normalized-log basis is kept for the legacy_log mode; the standard loss uses per-cell frequency profiles.
        self.register_buffer("R_gene_known", R_gene.clone())  # [T, G], legacy
        if R_gene_frequency is None:
            if self.lmix_mode == "frequency":
                raise ValueError(
                    "Canonical frequency L_mix requires R_gene_frequency"
                )
            R_gene_frequency = torch.empty(
                (0, G), dtype=R_gene.dtype, device=R_gene.device
            )
        if self.lmix_mode == "frequency":
            if tuple(R_gene_frequency.shape) != (T, G):
                raise ValueError("R_gene_frequency must have shape [T, G]")
            if not torch.isfinite(R_gene_frequency).all():
                raise ValueError("R_gene_frequency contains NaN/Inf")
            if (R_gene_frequency < 0).any():
                raise ValueError("R_gene_frequency must be non-negative")
            row_sums = R_gene_frequency.sum(dim=1)
            if not torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5
            ):
                raise ValueError("Each R_gene_frequency row must sum to one")
        # Reconstructed deterministically from raw scRNA counts at model
        # initialization, so it need not invalidate otherwise reusable
        # checkpoints such as the AE/CellOT artifacts.
        self.register_buffer(
            "R_gene_frequency_known", R_gene_frequency.clone(), persistent=False
        )

        # prototype (sc side mu_t + delta_mu_t)
        self.register_buffer("mu_sc_base", mu_sc_types.clone())  # [T, D]
        self.delta_mu = nn.Parameter(torch.zeros_like(mu_sc_types))

        # sc global latent & type index
        self.register_buffer("z_sc_all", z_sc_all.detach().clone())
        self.register_buffer("cell_type_index_sc", cell_type_index_sc.detach().clone().long())

        # sc global covariance / trace (for CORAL / trace floor)
        with torch.no_grad():
            zc = self.z_sc_all - self.z_sc_all.mean(dim=0, keepdim=True)
            cov_sc = (zc.T @ zc) / (self.z_sc_all.size(0) - 1)
            trace_sc = torch.trace(cov_sc)
        self.register_buffer("cov_sc_global", cov_sc)
        self.register_buffer("trace_sc_global", trace_sc)

        # bg prototype (latent & gene basis)
        self.mu_bg_latent = nn.Parameter(torch.zeros(1, D))
        self.R_bg_raw = nn.Parameter(torch.zeros(1, G))

        # NB dispersion
        theta_init = torch.full((G,), 1.0, dtype=torch.float32)
        self.theta_raw = nn.Parameter(inv_softplus(theta_init))

        # ST-side gene-wise scale (exp(st_log_scale) ~ 1.0 initially)
        self.st_log_scale = nn.Parameter(torch.zeros(1, G))

    # ---- prototype helpers ----
    def _get_mu_proto(self) -> torch.Tensor:
        return self.mu_sc_base + self.delta_mu  # [T, D]

    def _compose_bases(self):
        """Return legacy normalized-log bases and latent prototypes."""
        R_bg = F.softplus(self.R_bg_raw)  # [1, G] >= 0
        R_all = torch.cat([self.R_gene_known, R_bg], dim=0)  # [T+1, G]

        mu_proto = self._get_mu_proto()  # [T, D]
        mu_sc_all = torch.cat([mu_proto, self.mu_bg_latent], dim=0)  # [T+1, D]
        return R_all, mu_sc_all

    def _frequency_mix_prediction(
        self, p_full: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix on the linear gene-frequency scale, then apply log1p once."""
        if self.R_gene_frequency_known.shape != (self.T_known, self.G):
            raise RuntimeError("Canonical frequency profiles are unavailable")
        R_bg_frequency = torch.softmax(self.R_bg_raw, dim=-1)
        R_frequency_all = torch.cat(
            [self.R_gene_frequency_known, R_bg_frequency], dim=0
        )
        q_mix = p_full @ R_frequency_all
        X_mix_pred = torch.log1p(self.target_sum * q_mix.clamp_min(0.0))
        return X_mix_pred, q_mix, R_bg_frequency

    def compute_L_dmu(self) -> torch.Tensor:
        return (self.delta_mu ** 2).mean()

    def compute_L_proto_sc(self, batch_size: int = 1024) -> torch.Tensor:
        z_all = self.z_sc_all
        idx_all = self.cell_type_index_sc
        N = z_all.size(0)
        if N == 0:
            return self.delta_mu.new_tensor(0.0)

        b = min(batch_size, N)
        idx_sample = torch.randint(0, N, (b,), device=z_all.device)
        z = z_all[idx_sample]  # [b, D]
        t = idx_all[idx_sample].long()
        mu_proto = self._get_mu_proto()
        mu_sel = mu_proto[t]
        return ((z - mu_sel) ** 2).mean()

    def compute_L_pseudo_proto(self, z_pseudo: torch.Tensor, p_known: torch.Tensor) -> torch.Tensor:
        """L_pseudo_proto = E_j Σ_t p_{j,t} || z_pseudo(j,t) - μ_t ||^2."""
        mu_proto = self._get_mu_proto()  # [T, D]
        diff = z_pseudo - mu_proto.unsqueeze(0)  # [B, T, D]
        return (p_known.unsqueeze(-1) * diff.pow(2)).sum(dim=(1, 2)).mean()

    def forward(
        self,
        x_norm: torch.Tensor,
        x_raw: torch.Tensor,
        log_lib: torch.Tensor,
        residual_center: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        x_norm: [B, G], x_raw: [B, G], log_lib: [B, 1]
        """
        # Encoder: p, residual r, state gamma
        p_full, logits, r, gamma = self.encoder(x_norm, log_lib)
        B, Tp1 = p_full.shape
        T_known = self.T_known
        assert Tp1 == T_known + 1

        p_known = p_full[:, :T_known]              # [B, T]
        p_bg = p_full[:, -1].unsqueeze(1)          # [B, 1]

        # DestVI-style pseudo-cell latent
        mu_proto = self._get_mu_proto()            # [T, D]
        assert gamma.shape[1] == T_known
        assert gamma.shape[2] == self.D

        # z_{j,t} = mu_t + gamma_{j,t}
        z_pseudo = mu_proto.unsqueeze(0) + gamma   # [B, T, D]

        # barycenter over known types
        z_mix_known = (p_known.unsqueeze(-1) * z_pseudo).sum(dim=1)  # [B, D]

        # bg latent part
        z_mix_bg = p_bg * self.mu_bg_latent        # [B, D]

        # spot latent barycenter
        z_mix = z_mix_known + z_mix_bg             # [B, D]

        # Residual centering must use one mean over all spatial spots.  The
        # caller computes that dataset-level statistic once and supplies the
        # same tensor to every minibatch.  Failing closed here prevents an
        # accidental return to minibatch-dependent centering.
        if residual_center is None:
            raise RuntimeError(
                "STMixModel.forward requires the global all-spot residual center. "
                "Use compute_global_residual_center() before batched forward calls."
            )
        residual_center = residual_center.to(device=r.device, dtype=r.dtype)
        if residual_center.ndim == 1:
            residual_center = residual_center.unsqueeze(0)
        if residual_center.shape != (1, self.D):
            raise ValueError(
                f"residual_center shape={tuple(residual_center.shape)}, "
                f"expected (1, {self.D})"
            )
        r_centered = r - residual_center

        # geometry latent for ST
        z_st = z_mix + r_centered                  # [B, D]

        # partial losses
        L_res = (r_centered ** 2).mean()
        L_dmu = self.compute_L_dmu()
        L_gamma = (gamma ** 2).mean()

        # NB part: pseudo + residual through decoder, mix in expression space
        z_pseudo_resid = z_pseudo + r_centered.unsqueeze(1)          # [B, T, D]
        G = self.G

        z_flat = z_pseudo_resid.reshape(B * T_known, self.D)
        y_flat = self.decoder(z_flat)                               # [B*T, G] (log-scale)
        y_flat = y_flat.clamp(min=-10.0, max=10.0)
        mu_base_flat = torch.expm1(y_flat).clamp(min=0.0, max=1e6)
        mu_base_pseudo = mu_base_flat.view(B, T_known, G)           # [B, T, G]

        # bg latent decode
        y_bg = self.decoder(self.mu_bg_latent)                      # [1, G]
        y_bg = y_bg.clamp(min=-10.0, max=10.0)
        mu_base_bg = torch.expm1(y_bg).clamp(min=0.0, max=1e6)      # [1, G]

        # expression-space p-mix
        mu_base_mix = (p_known.unsqueeze(-1) * mu_base_pseudo).sum(dim=1) + p_bg * mu_base_bg

        # library size & ST gene-wise scale
        library_size = x_raw.sum(dim=1, keepdim=True)
        kappa = self.target_sum
        scale = (library_size / kappa).clamp(min=1e-3, max=1e3)

        st_scale = torch.exp(self.st_log_scale)                      # [1, G]
        mu_st = mu_base_mix * st_scale                               # [B, G]
        mu_pred = (mu_st * scale).clamp(min=1e-8, max=1e6)

        theta = (F.softplus(self.theta_raw) + 1e-8).clamp(max=1e3)

        # Auxiliary mixture reconstruction.  The standard path
        # performs the convex combination on the linear frequency scale and
        # applies target-sum scaling/log1p exactly once after mixing.
        if self.lmix_mode == "frequency":
            X_mix_pred, q_mix_pred, R_bg_frequency = self._frequency_mix_prediction(
                p_full
            )
        else:
            R_all, _ = self._compose_bases()
            X_mix_pred = p_full @ R_all
            q_mix_pred = X_mix_pred.new_empty((B, 0))
            R_bg_frequency = X_mix_pred.new_empty((1, 0))

        return {
            "mu_pred": mu_pred,
            "theta": theta,
            "p": p_full,
            "p_known": p_known,
            "p_bg": p_bg.squeeze(1),
            "logits": logits,
            "X_mix_pred": X_mix_pred,
            "q_mix_pred": q_mix_pred,
            "R_bg_frequency": R_bg_frequency,
            "lmix_mode": self.lmix_mode,
            "z_mix": z_mix,
            "z_st": z_st,
            "z_pseudo": z_pseudo,
            "r": r_centered,
            "gamma": gamma,                 # <-- for spatial regularization
            "L_res": L_res,
            "L_dmu": L_dmu,
            "L_gamma": L_gamma,
        }


# ============================================================
# Other losses (unchanged)
# ============================================================

def coral_cov_only(z_s: torch.Tensor, z_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Covariance-only CORAL loss between source and target latents."""
    zs = z_s - z_s.mean(dim=0, keepdim=True)
    cs = (zs.T @ zs) / (zs.size(0) - 1 + eps)

    zt = z_t - z_t.mean(dim=0, keepdim=True)
    ct = (zt.T @ zt) / (zt.size(0) - 1 + eps)

    return (cs - ct).pow(2).mean()


def trace_floor_loss(
    z_st: torch.Tensor,
    trace_sc: torch.Tensor,
    ratio_target: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Keep ST trace above ratio_target * trace_sc (soft floor)."""
    zc = z_st - z_st.mean(dim=0, keepdim=True)
    cov_st = (zc.T @ zc) / (zc.size(0) - 1 + eps)
    trace_st = torch.trace(cov_st)
    diff = ratio_target * trace_sc - trace_st
    return F.relu(diff) ** 2


def sinkhorn_ot_loss_typewise(
    z_sc_by_type,
    z_pseudo: torch.Tensor,
    p_detach_known: torch.Tensor,
    eps: float = 0.5,
    n_iter: int = 30,
    mass_threshold: float = 1e-4,
    sc_chunk_size: int | None = None,
) -> torch.Tensor:
    """Type-wise entropic OT loss (DestVI-style) on z_pseudo."""
    B, T_known = p_detach_known.shape
    assert z_pseudo.shape[0] == B and z_pseudo.shape[1] == T_known, "z_pseudo shape mismatch"

    total_cost = z_pseudo.new_tensor(0.0)
    total_mass = z_pseudo.new_tensor(0.0)

    for t in range(T_known):
        z_s_full = z_sc_by_type[t]  # [N_t, D]
        Ns = z_s_full.size(0)
        if Ns == 0:
            continue

        # ST side pseudo-cell for type t
        z_t = z_pseudo[:, t, :]  # [B, D]

        # ST mass
        w_full = p_detach_known[:, t].clamp(min=0.0)  # [B]
        mass_t = w_full.sum()
        if mass_t < mass_threshold:
            continue

        b = w_full / mass_t  # [B]

        if sc_chunk_size is None or Ns <= sc_chunk_size:
            a = torch.full((Ns,), 1.0 / Ns, device=z_s_full.device)  # [Ns]

            C = torch.cdist(z_s_full, z_t, p=2.0)  # [Ns, B]
            C = C / C.mean()
            C = C ** 2
            K = torch.exp(-C / eps)

            u = torch.ones_like(a)
            v = torch.ones_like(b)

            for _ in range(n_iter):
                Kv = K @ v + 1e-8
                u = a / Kv
                KTu = K.t() @ u + 1e-8
                v = b / KTu

            P = u.unsqueeze(1) * K * v.unsqueeze(0)  # [Ns, B]
            cost_t = (P * C).sum()
        else:
            cost_t = z_pseudo.new_tensor(0.0)
            start = 0
            while start < Ns:
                end = min(start + sc_chunk_size, Ns)
                z_s = z_s_full[start:end]  # [Nc, D]
                Nc = z_s.size(0)
                if Nc == 0:
                    start = end
                    continue

                a = torch.full((Nc,), 1.0 / Nc, device=z_s.device)  # [Nc]

                C = torch.cdist(z_s, z_t, p=2.0)  # [Nc, B]
                C = C / C.mean()
                C = C ** 2
                K = torch.exp(-C / eps)

                u = torch.ones_like(a)
                v = torch.ones_like(b)

                for _ in range(n_iter):
                    Kv = K @ v + 1e-8
                    u = a / Kv
                    KTu = K.t() @ u + 1e-8
                    v = b / KTu

                P = u.unsqueeze(1) * K * v.unsqueeze(0)  # [Nc, B]
                cost_chunk = (P * C).sum()

                weight_k = float(Nc) / float(Ns)
                cost_t = cost_t + weight_k * cost_chunk

                start = end

        total_cost = total_cost + mass_t * cost_t
        total_mass = total_mass + mass_t

    if total_mass.item() <= 0.0:
        return z_pseudo.new_tensor(0.0)
    return total_cost / total_mass


# ============================================================
# sc reference prep (unchanged)
# ============================================================

def prepare_sc_reference(
    adata_sc_ctrl,
    ae_model,
    cell_type_col: str,
    device: torch.device | None = None,
):
    """Compute sc-side latent reference stats and gene basis."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(adata_sc_ctrl.X, "toarray"):
        X_sc_norm = adata_sc_ctrl.X.toarray().astype(np.float32)
    else:
        X_sc_norm = np.asarray(adata_sc_ctrl.X, dtype=np.float32)

    sc_data_tensor = torch.tensor(X_sc_norm, dtype=torch.float32, device=device)

    ae_model = ae_model.to(device)
    ae_model.eval()
    with torch.no_grad():
        z_sc_ctrl = ae_model.encode(sc_data_tensor)

    for p in ae_model.parameters():
        p.requires_grad_(False)

    ct_series = adata_sc_ctrl.obs[cell_type_col].astype("category")
    cell_type_categories = ct_series.cat.categories
    cell_type_index_np = ct_series.cat.codes.to_numpy()
    num_cell_types = len(cell_type_categories)

    if "raw_counts" not in adata_sc_ctrl.layers:
        raise KeyError(
            "Canonical frequency L_mix requires adata_sc_ctrl.layers['raw_counts']"
        )
    X_sc_raw = adata_sc_ctrl.layers["raw_counts"]
    raw_library = np.asarray(X_sc_raw.sum(axis=1)).reshape(-1).astype(np.float64)
    if np.any(~np.isfinite(raw_library)) or np.any(raw_library <= 0):
        raise ValueError("scRNA raw library sizes must be finite and positive")

    R_gene_list = []
    R_gene_frequency_list = []
    for t in range(num_cell_types):
        mask = cell_type_index_np == t
        if not np.any(mask):
            raise ValueError(f"cell_type index {t} has no cells")
        prof_t_norm = X_sc_norm[mask].mean(axis=0)
        R_gene_list.append(prof_t_norm)
        raw_t = X_sc_raw[mask]
        if hasattr(raw_t, "multiply"):
            freq_t = raw_t.astype(np.float64).multiply(
                (1.0 / raw_library[mask])[:, None]
            )
            prof_t_frequency = np.asarray(freq_t.mean(axis=0)).reshape(-1)
        else:
            raw_t = np.asarray(raw_t, dtype=np.float64)
            prof_t_frequency = (
                raw_t / raw_library[mask, None]
            ).mean(axis=0)
        profile_sum = float(prof_t_frequency.sum())
        if not np.isfinite(profile_sum) or profile_sum <= 0:
            raise ValueError(f"Invalid frequency profile for cell type {t}")
        R_gene_frequency_list.append(prof_t_frequency / profile_sum)
    R_gene = np.stack(R_gene_list, axis=0)
    R_gene_frequency = np.stack(R_gene_frequency_list, axis=0)
    R_gene_torch = torch.as_tensor(R_gene, dtype=torch.float32, device=device)
    R_gene_frequency_torch = torch.as_tensor(
        R_gene_frequency, dtype=torch.float32, device=device
    )

    cell_type_index_sc = torch.as_tensor(cell_type_index_np, device=device, dtype=torch.long)

    with torch.no_grad():
        mu_sc_type_list = []
        for t in range(num_cell_types):
            mask_t = cell_type_index_sc == t
            if not mask_t.any():
                raise ValueError(f"cell_type index {t} has no cells in latent space")
            mu_t = z_sc_ctrl[mask_t].mean(dim=0, keepdim=True)
            mu_sc_type_list.append(mu_t)
        mu_sc_types = torch.cat(mu_sc_type_list, dim=0)
        mu_sc_types.requires_grad_(False)

    with torch.no_grad():
        z_sc_by_type = []
        for t in range(num_cell_types):
            mask_t = cell_type_index_sc == t
            z_t = z_sc_ctrl[mask_t].detach()
            if z_t.size(0) == 0:
                raise ValueError(f"Type {t} has no cells")
            z_sc_by_type.append(z_t)

    return {
        "z_sc_ctrl": z_sc_ctrl,
        "z_sc_by_type": z_sc_by_type,
        "mu_sc_types": mu_sc_types,
        "R_gene_torch": R_gene_torch,
        "R_gene_frequency_torch": R_gene_frequency_torch,
        "cell_type_index_sc": cell_type_index_sc,
        "cell_type_categories": cell_type_categories,
        "num_cell_types": num_cell_types,
        "latent_dim": z_sc_ctrl.shape[1],
    }


def configure_film_stats(encoder: STMixEncoder, adata_sc_ctrl, device: torch.device | None = None) -> None:
    """Set FiLM conditioning stats from sc library sizes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        if "lib_raw" in adata_sc_ctrl.obs.columns:
            lib_sc = adata_sc_ctrl.obs["lib_raw"].values.astype(np.float32)
        else:
            RAW_sc = adata_sc_ctrl.layers["raw_counts"]
            if hasattr(RAW_sc, "toarray"):
                RAW_sc = RAW_sc.toarray()
            lib_sc = np.asarray(RAW_sc.sum(axis=1)).astype(np.float32)

        sc_log_lib = np.log(np.clip(lib_sc, 1.0, None))
        c_mean = float(sc_log_lib.mean())
        c_std = float(sc_log_lib.std())

        if isinstance(encoder.film, SafeCondFiLM):
            encoder.film.set_condition_stats(c_mean, c_std)
            encoder.film.calibrate_g0(device=device)


def init_stmix_model(
    adata_sc_ctrl,
    ae_model,
    cell_type_col: str = "leiden_ctrl",
    device: torch.device | None = None,
    target_sum: float = 1e4,
    hidden_units: list[int] | tuple[int, ...] = (128, 128),
    res_scale: float = 0.25,
    use_sn: bool = True,
    film_eta: float = 0.1,
    lmix_mode: str = "frequency",
):
    """Initialize STMix model and compute sc reference quantities."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sc_ref = prepare_sc_reference(
        adata_sc_ctrl=adata_sc_ctrl,
        ae_model=ae_model,
        cell_type_col=cell_type_col,
        device=device,
    )

    n_genes = adata_sc_ctrl.n_vars
    n_types_plus_bg = sc_ref["num_cell_types"] + 1

    encoder = STMixEncoder(
        input_dim=n_genes,
        n_types_plus_bg=n_types_plus_bg,
        latent_dim=sc_ref["latent_dim"],
        hidden_units=list(hidden_units),
        res_scale=res_scale,
        use_sn=use_sn,
        film_eta=film_eta,
    ).to(device)

    configure_film_stats(encoder, adata_sc_ctrl, device=device)

    st_mix_model = STMixModel(
        encoder=encoder,
        R_gene=sc_ref["R_gene_torch"],
        mu_sc_types=sc_ref["mu_sc_types"],
        z_sc_all=sc_ref["z_sc_ctrl"],
        cell_type_index_sc=sc_ref["cell_type_index_sc"],
        decoder=ae_model.decode,
        target_sum=target_sum,
        R_gene_frequency=sc_ref["R_gene_frequency_torch"],
        lmix_mode=lmix_mode,
    ).to(device)

    return st_mix_model, sc_ref


# ============================================================
# DataLoader factory
# ============================================================

def make_st_dataloader(
    adata_st,
    target_sum: float = 1e4,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    spatial_key: str = "X_spatial",
):
    """Create STDataset and DataLoader."""
    st_dataset = STDataset(adata_st, target_sum=target_sum, spatial_key=spatial_key)
    st_loader = DataLoader(
        st_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return st_dataset, st_loader


@torch.no_grad()
def compute_global_residual_center(
    st_mix_model: STMixModel,
    st_dataset: STDataset,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> torch.Tensor:
    """Return the raw-residual mean over every spot in ``st_dataset``.

    The encoder is evaluated in deterministic inference mode and only the
    residual head is collected.  Accumulation is performed in float64 on CPU
    so that the same center can be reused by arbitrary downstream chunking.
    During training this dataset statistic is refreshed once per epoch and is
    intentionally treated as a fixed centering statistic for that epoch.
    """
    if len(st_dataset) == 0:
        raise ValueError("Cannot compute a residual center for an empty ST dataset")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    dev = torch.device(device) if isinstance(device, str) else device
    loader = DataLoader(
        st_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    was_training = st_mix_model.training
    st_mix_model.eval()
    residual_sum = torch.zeros(st_mix_model.D, dtype=torch.float64)
    n_spots = 0
    try:
        for x_norm, _x_raw, log_lib, _spot_idx, _coords in loader:
            _p, _logits, raw_residual, _gamma = st_mix_model.encoder(
                x_norm.to(dev), log_lib.to(dev)
            )
            residual_sum += raw_residual.detach().to("cpu", dtype=torch.float64).sum(dim=0)
            n_spots += int(raw_residual.shape[0])
    finally:
        st_mix_model.train(was_training)

    if n_spots != len(st_dataset):
        raise AssertionError(
            f"Global residual center saw {n_spots} spots, expected {len(st_dataset)}"
        )
    center = (residual_sum / float(n_spots)).to(dtype=torch.float32, device=dev)
    if not torch.isfinite(center).all():
        raise FloatingPointError("Global residual center contains NaN/Inf")
    return center.unsqueeze(0)


def default_stmix_config() -> dict[str, float | int | str]:
    """Default training config for STMix.

    Note: For spatial regularization to work well, prefer DataLoader shuffle=False.
    """
    return {
        "lambda_nb": 5.0,
        "lambda_mix_gene": 0.2,
        "lmix_mode": "frequency",
        "require_frequency_lmix": True,
        "lambda_ot": 20.0,
        "lambda_pbg": 1.0,
        "target_bg_mean": 0.05,
        "lambda_res": 2e-1,
        "lambda_gamma": 1.0,
        "lambda_proto_sc": 5.0,
        # Standard model: L_pseudo_proto is retained only as a
        # diagnostic quantity and has no contribution to the training loss.
        "lambda_pseudo_proto": 0.0,
        "proto_sc_batch_size": 512,
        "ot_eps": 0.5,
        "ot_max_iter": 5,
        "ot_interval": 5,
        "ot_sc_chunk_size": 500,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "n_epochs": 150,

        # One residual mean is computed over every spatial spot at the start
        # of each training epoch.  Inference recomputes the all-spot mean for
        # the final checkpoint.  Minibatch centering is not permitted.
        "residual_centering_scope": "global_spot",
        "residual_center_refresh": "each_epoch",
        "residual_center_batch_size": 256,

        # ---- spatial gamma regularization ----
        "lambda_sp_gamma": 1e-1,   # set 0.0 to disable
        # Keep the canonical source equal to the value used in those runs.
        "sp_k": 10,
        "sp_gate": "sqrt",         # "sqrt" or "min"
    }


# ============================================================
# Training
# ============================================================

def train_stmix_model(
    st_mix_model: STMixModel,
    st_loader: DataLoader,
    z_sc_by_type,
    config: dict[str, float | int | str],
    device: torch.device | None = None,
    clip_grad: float = 1.0,
    log_every: int = 20,
):
    """Train STMix model and return loss history."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configured_lmix_mode = str(config.get("lmix_mode", ""))
    model_lmix_mode = str(getattr(st_mix_model, "lmix_mode", ""))
    require_frequency_lmix = bool(config.get("require_frequency_lmix", True))
    if require_frequency_lmix and (
        configured_lmix_mode != "frequency" or model_lmix_mode != "frequency"
    ):
        raise RuntimeError(
            "Standard STMix requires lmix_mode='frequency'; "
            f"config={configured_lmix_mode!r}, model={model_lmix_mode!r}"
        )

    optimizer = torch.optim.Adam(
        st_mix_model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    # ---- spatial graph (build once) ----
    lam_sp = float(config.get("lambda_sp_gamma", 0.0))
    lam_pseudo_proto = float(config.get("lambda_pseudo_proto", 0.0))
    if lam_sp > 0.0:
        coords_np = st_loader.dataset.coords.detach().cpu().numpy()
        neigh_idx, neigh_w = build_knn_graph(coords_np, k=int(config.get("sp_k", 10)))
        sp_gate = str(config.get("sp_gate", "sqrt"))
    else:
        neigh_idx, neigh_w = None, None
        sp_gate = "sqrt"

    history = {
        "total": [],
        "L_nb": [],
        "L_mix_gene": [],
        "L_ot": [],
        "L_pbg": [],
        "L_res": [],
        "L_gamma": [],
        "L_proto_sc": [],
        "L_pseudo_proto": [],
        "L_sp_gamma": [],
        "theta_mean": [],
        "bg_mean": [],
        "delta_mu_norm": [],
        "edge_ratio": [],  # fraction of batch edges found (diagnostic)
    }

    print("\n=== Train ST Mix-Model (with optional spatial gamma) ===")
    for epoch in range(int(config["n_epochs"])):
        if str(config.get("residual_centering_scope")) != "global_spot":
            raise ValueError("Canonical STMix requires residual_centering_scope='global_spot'")
        if str(config.get("residual_center_refresh")) != "each_epoch":
            raise ValueError("Canonical STMix requires residual_center_refresh='each_epoch'")
        residual_center = compute_global_residual_center(
            st_mix_model,
            st_loader.dataset,
            device=device,
            batch_size=int(config.get("residual_center_batch_size", 256)),
        )
        st_mix_model.train()
        sum_total = sum_nb = sum_mix = sum_ot = 0.0
        sum_pbg = sum_res = 0.0
        sum_gamma = sum_proto = 0.0
        sum_pseudo_proto = 0.0
        sum_spg = 0.0
        sum_bg_mean = 0.0
        sum_edge_ratio = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(st_loader):
            # batch = (x_norm, x_raw, log_lib, spot_idx, coords)
            x_norm, x_raw, log_lib, spot_idx, _coords = batch

            x_norm = x_norm.to(device)
            x_raw = x_raw.to(device)
            log_lib = log_lib.to(device)
            spot_idx = torch.as_tensor(spot_idx, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            out = st_mix_model(
                x_norm,
                x_raw,
                log_lib,
                residual_center=residual_center,
            )

            L_nb = negative_binomial_nll(x_raw, out["mu_pred"], out["theta"])
            L_mix_gene = F.mse_loss(out["X_mix_pred"], x_norm)

            p_bg = out["p_bg"]
            bg_mean_batch = p_bg.mean()
            L_pbg = (bg_mean_batch - float(config["target_bg_mean"])) ** 2

            L_res = out["L_res"]
            L_gamma = out["L_gamma"]

            L_proto_sc = st_mix_model.compute_L_proto_sc(
                batch_size=int(config["proto_sc_batch_size"])
            )
            L_pseudo_proto = st_mix_model.compute_L_pseudo_proto(
                out["z_pseudo"], out["p_known"]
            )

            if float(config["lambda_ot"]) > 0.0 and (batch_idx % int(config["ot_interval"]) == 0):
                sc_chunk_size = config.get("ot_sc_chunk_size")
                if sc_chunk_size is not None:
                    sc_chunk_size = int(sc_chunk_size)
                L_ot = sinkhorn_ot_loss_typewise(
                    z_sc_by_type,
                    out["z_pseudo"],
                    out["p_known"].detach(),
                    eps=float(config["ot_eps"]),
                    n_iter=int(config["ot_max_iter"]),
                    sc_chunk_size=sc_chunk_size,
                )
            else:
                L_ot = out["z_st"].new_tensor(0.0)

            # ---- spatial gamma loss (in-batch edges) ----
            if lam_sp > 0.0 and (neigh_idx is not None) and (neigh_w is not None):
                edges = make_batch_edges(spot_idx, neigh_idx, neigh_w)
                if edges is None:
                    L_sp_gamma = out["z_st"].new_tensor(0.0)
                    edge_ratio = 0.0
                else:
                    src, dst, w = edges
                    L_sp_gamma = spatial_gamma_loss(
                        gamma=out["gamma"],
                        p_known=out["p_known"],
                        src=src,
                        dst=dst,
                        w=w,
                        gate=sp_gate,
                    )
                    # diagnostic: how many edges did we get relative to B*k?
                    B = spot_idx.numel()
                    k = int(config.get("sp_k", 10))
                    edge_ratio = float(src.numel()) / float(max(B * k, 1))
            else:
                L_sp_gamma = out["z_st"].new_tensor(0.0)
                edge_ratio = 0.0

            loss = (
                float(config["lambda_nb"]) * L_nb
                + float(config["lambda_mix_gene"]) * L_mix_gene
                + float(config["lambda_ot"]) * L_ot
                + float(config["lambda_pbg"]) * L_pbg
                + float(config["lambda_res"]) * L_res
                + float(config["lambda_gamma"]) * L_gamma
                + float(config["lambda_proto_sc"]) * L_proto_sc
                + lam_sp * L_sp_gamma
            )
            # Keep L_pseudo_proto available in history for diagnostics and
            # legacy reproduction, but exclude it from the canonical graph.
            if lam_pseudo_proto != 0.0:
                loss = loss + lam_pseudo_proto * L_pseudo_proto

            loss.backward()
            torch.nn.utils.clip_grad_norm_(st_mix_model.parameters(), max_norm=clip_grad)
            optimizer.step()

            sum_total += loss.item()
            sum_nb += L_nb.item()
            sum_mix += L_mix_gene.item()
            sum_ot += L_ot.item()
            sum_pbg += L_pbg.item()
            sum_res += L_res.item()
            sum_gamma += L_gamma.item()
            sum_proto += L_proto_sc.item()
            sum_pseudo_proto += L_pseudo_proto.item()
            sum_spg += L_sp_gamma.item()
            sum_bg_mean += bg_mean_batch.item()
            sum_edge_ratio += edge_ratio
            n_batches += 1

        history["total"].append(sum_total / n_batches)
        history["L_nb"].append(sum_nb / n_batches)
        history["L_mix_gene"].append(sum_mix / n_batches)
        history["L_ot"].append(sum_ot / n_batches)
        history["L_pbg"].append(sum_pbg / n_batches)
        history["L_res"].append(sum_res / n_batches)
        history["L_gamma"].append(sum_gamma / n_batches)
        history["L_proto_sc"].append(sum_proto / n_batches)
        history["L_pseudo_proto"].append(sum_pseudo_proto / n_batches)
        history["L_sp_gamma"].append(sum_spg / n_batches)
        history["edge_ratio"].append(sum_edge_ratio / n_batches)

        with torch.no_grad():
            theta_proc = F.softplus(st_mix_model.theta_raw) + 1e-8
            delta_mu_norm = st_mix_model.delta_mu.norm().item()
            history["theta_mean"].append(theta_proc.mean().item())
            history["bg_mean"].append(sum_bg_mean / n_batches)
            history["delta_mu_norm"].append(delta_mu_norm)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"total={history['total'][-1]:.4f} | "
                f"NB={float(config['lambda_nb']) * history['L_nb'][-1]:.3f} | "
                f"mix_gene={float(config['lambda_mix_gene']) * history['L_mix_gene'][-1]:.4f} | "
                f"OT={float(config['lambda_ot']) * history['L_ot'][-1]:.4f} | "
                f"pseudo_proto={float(config['lambda_pseudo_proto']) * history['L_pseudo_proto'][-1]:.4f} | "
                f"pbg={float(config['lambda_pbg']) * history['L_pbg'][-1]:.4f} | "
                f"pres={float(config['lambda_res']) * history['L_res'][-1]:.4f} | "
                f"gamma={float(config['lambda_gamma']) * history['L_gamma'][-1]:.4f} | "
                f"proto={float(config['lambda_proto_sc']) * history['L_proto_sc'][-1]:.4f} | "
                f"sp_gamma={lam_sp * history['L_sp_gamma'][-1]:.4f} | "
                f"edge_ratio={history['edge_ratio'][-1]:.2f} | "
                f"bg_mean={history['bg_mean'][-1]:.3f} | "
                f"||delta_mu||={history['delta_mu_norm'][-1]:.3f}"
            )

    print("=== Training finished ===")
    return history


# ============================================================
# Plotting
# ============================================================

def plot_stmix_history(history: dict[str, list[float]]) -> None:
    """Plot loss curves for STMix training."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(22, 12))

    axes[0, 0].plot(history["total"])
    axes[0, 0].set_title("Total loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["L_nb"])
    axes[0, 1].set_title("NB NLL (raw)")
    axes[0, 1].grid(True)

    axes[0, 2].plot(history["theta_mean"])
    axes[0, 2].set_title("theta mean")
    axes[0, 2].grid(True)

    axes[0, 3].plot(history["edge_ratio"])
    axes[0, 3].set_title("edge_ratio (batch)")
    axes[0, 3].grid(True)

    axes[1, 0].plot(history["L_mix_gene"])
    axes[1, 0].set_title("L_mix_gene (raw)")
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["L_ot"])
    axes[1, 1].set_title("L_OT (type-wise OT, pseudo)")
    axes[1, 1].grid(True)

    axes[1, 2].plot(history["bg_mean"])
    axes[1, 2].set_title("bg_mean")
    axes[1, 2].grid(True)

    axes[1, 3].plot(history["L_sp_gamma"])
    axes[1, 3].set_title("L_sp_gamma (raw)")
    axes[1, 3].grid(True)

    axes[2, 0].plot(history["L_pbg"])
    axes[2, 0].set_title("L_pbg (bg mass penalty, raw)")
    axes[2, 0].grid(True)

    axes[2, 1].plot(history["L_res"])
    axes[2, 1].set_title("L_res (residual L2, raw)")
    axes[2, 1].grid(True)

    axes[2, 2].plot(history["L_gamma"])
    axes[2, 2].set_title("L_gamma (state L2, raw)")
    axes[2, 2].grid(True)

    axes[2, 3].plot(history["L_proto_sc"])
    axes[2, 3].set_title("L_proto_sc (raw)")
    axes[2, 3].grid(True)

    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    axes2[0].plot(history["L_pseudo_proto"])
    axes2[0].set_title("L_pseudo_proto (raw)")
    axes2[0].grid(True)
    axes2[1].plot(history.get("delta_mu_norm", []))
    axes2[1].set_title("||delta_mu||")
    axes2[1].grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# Latent extraction (updated unpack)
# ============================================================

def get_st_pseudo_latents(
    st_adata,
    st_mix_model: STMixModel,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
):
    """Extract pseudo-cell latents, residuals, and known mix p for all spots."""
    st_mix_model.eval()
    dataset = STDataset(st_adata, target_sum=1e4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    z_pseudo_list = []
    p_known_list = []
    r_list = []

    dev = torch.device(device) if isinstance(device, str) else device
    st_mix_model.to(dev)
    residual_center = compute_global_residual_center(
        st_mix_model,
        dataset,
        device=dev,
        batch_size=batch_size,
    )

    with torch.no_grad():
        for x_norm, x_raw, log_lib, spot_idx, coords in loader:
            x_norm = x_norm.to(dev)
            x_raw = x_raw.to(dev)
            log_lib = log_lib.to(dev)

            out = st_mix_model(
                x_norm,
                x_raw,
                log_lib,
                residual_center=residual_center,
            )
            z_pseudo_list.append(out["z_pseudo"].cpu())
            p_known_list.append(out["p_known"].cpu())
            r_list.append(out["r"].cpu())

    z_pseudo_all = torch.cat(z_pseudo_list, dim=0).numpy()
    p_known_all = torch.cat(p_known_list, dim=0).numpy()
    r_all = torch.cat(r_list, dim=0).numpy()
    residual_mean = r_all.astype(np.float64).mean(axis=0)
    if float(np.max(np.abs(residual_mean))) > 1e-5:
        raise AssertionError(
            "Global-centered inference residual does not have zero all-spot mean: "
            f"max_abs_mean={float(np.max(np.abs(residual_mean))):.3e}"
        )
    return z_pseudo_all, r_all, p_known_all


def build_pseudo_adata_from_latents(
    st_adata,
    z_pseudo_all,
    r_all,
    p_known_all,
    cell_type_categories,
    use_residual: bool = False,
    mass_threshold: float = 0.05,
):
    """Create pseudo-cell AnnData from (spot, type) latents."""
    import scanpy as sc

    S, T_known, D = z_pseudo_all.shape
    if p_known_all.shape != (S, T_known):
        raise ValueError("p_known_all shape mismatch")
    if r_all.shape != (S, D):
        raise ValueError("r_all shape mismatch")

    mask = p_known_all > mass_threshold
    spot_idx, type_idx = np.where(mask)

    z_flat = z_pseudo_all[spot_idx, type_idx, :]
    if use_residual:
        z_flat = z_flat + r_all[spot_idx, :]

    if hasattr(st_adata.X, "toarray"):
        X_st = st_adata.X.toarray()
    else:
        X_st = np.asarray(st_adata.X)
    X_sel = X_st[spot_idx, :]

    pseudo_adata = sc.AnnData(X_sel, var=st_adata.var.copy())

    pseudo_adata.obsm["Z_latent"] = z_flat
    pseudo_adata.obs["spot_index"] = spot_idx
    pseudo_adata.obs["type_index"] = type_idx
    pseudo_adata.obs["cell_type"] = [str(cell_type_categories[t]) for t in type_idx]
    pseudo_adata.obsm["R_spot"] = r_all[spot_idx, :]
    pseudo_adata.uns["mass_threshold"] = float(mass_threshold)
    pseudo_adata.uns["use_residual"] = bool(use_residual)
    pseudo_adata.obsm["p_known"] = p_known_all[spot_idx, :]
    pseudo_adata.obs["cell_type"] = pseudo_adata.obs["cell_type"].astype("category")
    return pseudo_adata


# Optional: export list (not required)
__all__ = [
    "STDataset",
    "default_stmix_config",
    "compute_global_residual_center",
    "init_stmix_model",
    "negative_binomial_nll",
    "plot_stmix_history",
    "sinkhorn_ot_loss_typewise",
    "make_st_dataloader",
    "train_stmix_model",
    "get_st_pseudo_latents",
    "build_pseudo_adata_from_latents",
]
