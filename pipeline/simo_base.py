#!/usr/bin/env python3
"""Generate a SIMO mapping for one explicitly allowed v2 scenario.

Run with SIMO-env. This script only assigns control scRNA cells to control ST
spots. Perturbation transport is performed later with the canonical CellOT in
COT-env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from simo import alignment_1_batch, assign_coord_1, find_marker, process_anndata


EXPECTED_G = 8000
EXPECTED_SC = 20000
EXPECTED_SPOTS = 2500
MODEL_SEED = 1


def set_seed(seed: int = MODEL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _decode_strings(values) -> np.ndarray:
    values = np.asarray(values)
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def _read_h5ad_matrix(node):
    """Read only the standard dense/CSR/CSC encodings needed by v2 inputs."""
    if isinstance(node, h5py.Dataset):
        return node[()]
    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    shape = tuple(int(value) for value in node.attrs["shape"])
    data = node["data"][()]
    indices = node["indices"][()]
    indptr = node["indptr"][()]
    if encoding == "csr_matrix":
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    if encoding == "csc_matrix":
        return sp.csc_matrix((data, indices, indptr), shape=shape)
    raise RuntimeError(f"Unsupported H5AD X encoding for SIMO compatibility: {encoding}")


def _dataframe_index(group: h5py.Group) -> pd.Index:
    index_key = group.attrs.get("_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")
    if not index_key or index_key not in group:
        raise RuntimeError("H5AD dataframe index metadata is missing")
    return pd.Index(_decode_strings(group[index_key][()]), name=str(index_key))


def read_v2_input_without_new_anndata_encoding(
    path: Path, *, require_spatial: bool
) -> ad.AnnData:
    """Load v2 inputs in old SIMO-env without parsing new ``dict`` groups.

    New AnnData writes empty ``layers/uns/obsp`` containers with
    ``encoding-type=dict``.  The old AnnData bundled with SIMO cannot decode
    that container type.  SIMO needs only X, observation IDs, gene IDs and ST
    coordinates, so these fields are read directly with h5py.  The frozen
    source H5AD is opened read-only and is never rewritten.
    """
    with h5py.File(path, "r") as handle:
        obs_columns = _decode_strings(handle["obs"].attrs.get("column-order", []))
        forbidden = {"true_group", "p_true_Group1"}.intersection(obs_columns)
        if forbidden:
            raise RuntimeError(f"Truth labels found in SIMO input: {sorted(forbidden)}")
        obs_index = _dataframe_index(handle["obs"])
        var_index = _dataframe_index(handle["var"])
        matrix = _read_h5ad_matrix(handle["X"])
        spatial = None
        if require_spatial:
            if "obsm" not in handle or "spatial" not in handle["obsm"]:
                raise RuntimeError("ST input has no obsm/spatial coordinates")
            spatial = np.asarray(handle["obsm"]["spatial"][()], dtype=np.float32)

    value = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=obs_index),
        var=pd.DataFrame(index=var_index),
    )
    if spatial is not None:
        value.obsm["spatial"] = spatial
    return value
