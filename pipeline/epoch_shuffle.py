#!/usr/bin/env python3
"""Mini-batch order of the spot model: the spots are reshuffled in every epoch with a standard
PyTorch ``RandomSampler`` (the implementation used by ``DataLoader(..., shuffle=True)``).

The custom batch sampler only observes the already-generated random order so
that cumulative OT-spot and in-batch spatial-edge coverage can be recorded. It
does not generate another permutation and does not alter the loss calculation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from perturbed_spatial_omics.stmix import build_knn_graph


ALLOWED_SCENARIOS = ("unpaired_group", "unpaired_group_modality")
BATCH_SIZE = 64
DROP_LAST = False
OT_INTERVAL = 5
SP_K = 10
EXPECTED_SPOTS = 2_500


@dataclass
class CoverageTracker:
    """Track the batches that actually enter one completed training epoch."""

    neigh_idx: np.ndarray
    n_spots: int
    batch_size: int
    ot_interval: int
    unique_ot_spots: set[int] = field(default_factory=set)
    unique_spatial_edges: set[tuple[int, int]] = field(default_factory=set)
    unique_spatial_spots: set[int] = field(default_factory=set)
    records: list[dict[str, int | float]] = field(default_factory=list)
    _epoch: int = 0

    @property
    def n_total_directed_spatial_edges(self) -> int:
        return int(self.neigh_idx.size)

    def observe_batch(self, batch_index: int, indices: Sequence[int]) -> None:
        indices = [int(value) for value in indices]
        if batch_index % self.ot_interval == 0:
            self.unique_ot_spots.update(indices)

        members = set(indices)
        for source in indices:
            for destination in self.neigh_idx[source]:
                destination = int(destination)
                if destination in members:
                    self.unique_spatial_edges.add((source, destination))
                    self.unique_spatial_spots.add(source)
                    self.unique_spatial_spots.add(destination)

    def finish_epoch(self) -> None:
        self._epoch += 1
        self.records.append(
            {
                "epoch": self._epoch,
                "n_unique_ot_spots_seen": len(self.unique_ot_spots),
                "fraction_unique_ot_spots_seen": (
                    len(self.unique_ot_spots) / float(self.n_spots)
                ),
                "n_unique_spatial_edges_seen": len(self.unique_spatial_edges),
                "fraction_unique_spatial_edges_seen": (
                    len(self.unique_spatial_edges)
                    / float(self.n_total_directed_spatial_edges)
                ),
                "n_unique_spatial_spots_seen": len(self.unique_spatial_spots),
                "fraction_unique_spatial_spots_seen": (
                    len(self.unique_spatial_spots) / float(self.n_spots)
                ),
            }
        )

    def fixed_order_reference(self) -> dict[str, int | float]:
        """Coverage obtained when the same sequential order is reused forever."""
        ot_spots: set[int] = set()
        spatial_edges: set[tuple[int, int]] = set()
        spatial_spots: set[int] = set()
        order = list(range(self.n_spots))
        for batch_index, start in enumerate(range(0, self.n_spots, self.batch_size)):
            batch = order[start : start + self.batch_size]
            if batch_index % self.ot_interval == 0:
                ot_spots.update(batch)
            members = set(batch)
            for source in batch:
                for destination in self.neigh_idx[source]:
                    destination = int(destination)
                    if destination in members:
                        spatial_edges.add((source, destination))
                        spatial_spots.add(source)
                        spatial_spots.add(destination)
        return {
            "n_unique_ot_spots_seen": len(ot_spots),
            "fraction_unique_ot_spots_seen": len(ot_spots) / float(self.n_spots),
            "n_unique_spatial_edges_seen": len(spatial_edges),
            "fraction_unique_spatial_edges_seen": (
                len(spatial_edges) / float(self.n_total_directed_spatial_edges)
            ),
            "n_unique_spatial_spots_seen": len(spatial_spots),
            "fraction_unique_spatial_spots_seen": (
                len(spatial_spots) / float(self.n_spots)
            ),
        }


class TrackingBatchSampler(BatchSampler):
    """A normal random BatchSampler with passive, batch-level coverage logging."""

    def __init__(self, sampler: RandomSampler, tracker: CoverageTracker):
        if not isinstance(sampler, RandomSampler) or sampler.replacement:
            raise TypeError("A non-replacement PyTorch RandomSampler is required")
        super().__init__(sampler, batch_size=BATCH_SIZE, drop_last=DROP_LAST)
        self.tracker = tracker

    def __iter__(self) -> Iterator[list[int]]:
        for batch_index, batch in enumerate(super().__iter__()):
            self.tracker.observe_batch(batch_index, batch)
            yield batch
        self.tracker.finish_epoch()


ACTIVE_TRACKER: CoverageTracker | None = None


def install_shuffle_true_patch(core) -> None:
    """Replace only the canonical loader order and attach passive coverage."""
    original_train = core.train_stmix_model

    def initialize_stmix_shuffle_true(
        prepared: dict[str, Any], device: torch.device, vae_model
    ):
        global ACTIVE_TRACKER
        model, sc_ref = core.init_stmix_model(
            prepared["rna_ctrl"],
            vae_model,
            cell_type_col="cell_type",
            device=device,
            target_sum=core.TARGET_SUM,
        )
        dataset, canonical_shuffled_loader = core.make_st_dataloader(
            prepared["st_ctrl"],
            target_sum=core.TARGET_SUM,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=DROP_LAST,
            spatial_key="X_spatial",
        )
        if len(dataset) != EXPECTED_SPOTS:
            raise AssertionError(
                f"Expected {EXPECTED_SPOTS} spots, observed {len(dataset)}"
            )
        coords = dataset.coords.detach().cpu().numpy()
        neigh_idx, _neigh_w = build_knn_graph(coords, k=SP_K)
        ACTIVE_TRACKER = CoverageTracker(
            neigh_idx=neigh_idx,
            n_spots=len(dataset),
            batch_size=BATCH_SIZE,
            ot_interval=OT_INTERVAL,
        )
        if not isinstance(canonical_shuffled_loader.sampler, RandomSampler):
            raise AssertionError(
                "make_st_dataloader(..., shuffle=True) did not create RandomSampler"
            )
        batch_sampler = TrackingBatchSampler(
            canonical_shuffled_loader.sampler, ACTIVE_TRACKER
        )
        loader = DataLoader(dataset, batch_sampler=batch_sampler)
        # With an explicit batch_sampler, DataLoader._index_sampler is this
        # batch sampler; DataLoader.sampler itself is not the active source of
        # indices.  Assert the sampler that is actually consumed.
        if not isinstance(loader.batch_sampler.sampler, RandomSampler):
            raise AssertionError("Effective sampler is not PyTorch RandomSampler")
        return model, sc_ref, loader

    def train_stmix_with_coverage(
        st_mix_model,
        st_loader,
        z_sc_by_type,
        config,
        device=None,
        clip_grad=1.0,
        log_every=20,
    ):
        expected = {
            "n_epochs": 150,
            "ot_interval": OT_INTERVAL,
            "ot_eps": 0.5,
            "sp_k": SP_K,
            "residual_centering_scope": "global_spot",
            "lambda_pseudo_proto": 0.0,
        }
        for key, value in expected.items():
            if config[key] != value:
                raise AssertionError(
                    f"Frozen setting changed: {key}={config[key]!r}, expected {value!r}"
                )
        if ACTIVE_TRACKER is None:
            raise RuntimeError("Coverage tracker was not initialized")
        history = original_train(
            st_mix_model,
            st_loader,
            z_sc_by_type,
            config,
            device=device,
            clip_grad=clip_grad,
            log_every=log_every,
        )
        if len(ACTIVE_TRACKER.records) != int(config["n_epochs"]):
            raise AssertionError(
                "Coverage epoch count does not match completed training epochs"
            )
        history["shuffle_true_coverage"] = ACTIVE_TRACKER.records
        history["effective_shuffle"] = True
        history["shuffle_implementation"] = (
            "torch.utils.data.RandomSampler(replacement=False)"
        )
        return history

    core.initialize_stmix = initialize_stmix_shuffle_true
    core.train_stmix_model = train_stmix_with_coverage
