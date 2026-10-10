"""Normalization used by the evaluation and the prediction helpers: total-count scaling of each
spot or cell to a fixed sum, followed by log1p."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp


def _matrix(adata: ad.AnnData) -> sp.csr_matrix:
    return adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)


def normalize_total_linear(X: Any, target_sum: float = 1e4) -> sp.csr_matrix:
    """Library-normalize a spot/cell by gene matrix without mutating input."""
    Y = X.tocsr().astype(np.float64, copy=True) if sp.issparse(X) else sp.csr_matrix(
        np.asarray(X, dtype=np.float64)
    )
    totals = np.asarray(Y.sum(axis=1)).ravel()
    scale = np.divide(
        float(target_sum),
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0,
    )
    return sp.diags(scale) @ Y


def normalize_log1p(X: Any, target_sum: float = 1e4) -> sp.csr_matrix:
    Y = normalize_total_linear(X, target_sum=target_sum)
    Y.data = np.log1p(Y.data)
    return Y
