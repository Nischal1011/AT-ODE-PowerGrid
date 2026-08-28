from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lib.simbench_lgode_data import (
    NormalizationStats,
    PowerGridDataLoaders,
    _build_encoder_graph,
    _build_physical_graph,
    collate_powergrid_lgode,
)


TaskName = Literal["interpolation", "extrapolation"]
SplitName = Literal["train", "validation", "test"]
ScaleName = Literal["smoke", "development", "publication"]
MASK_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IEEE39ScenarioRecord:
    trajectory_id: int
    split: str
    scenario_index: int
    start: int
    stop: int
    context_stop: int
    target_start: int
    target_stop: int


@dataclass(frozen=True)
class IEEE39TransientArchive:
    path: Path
    source_path: Path
    scenario_state: Tensor
    timestamps_seconds: Tensor
    labels: Tensor
    scenario_ids: Tensor
    split_indices: Dict[str, Tensor]
    bus_indices: Tensor
    edge_index: Tensor
    edge_type: Tensor
    bus_feature_names: Tuple[str, ...]
    generator_names: Tuple[str, ...]
    feature_units: Tuple[str, ...]
    normalization_mean: Tensor
    normalization_std: Tensor
    metadata: Dict[str, Any]

    @property
    def num_scenarios(self) -> int:
        return int(self.scenario_state.shape[0])

    @property
    def num_timesteps(self) -> int:
        return int(self.scenario_state.shape[1])

    @property
    def num_nodes(self) -> int:
        return int(self.scenario_state.shape[2])

    @property
    def input_dim(self) -> int:
        return int(self.scenario_state.shape[3])

    @property
    def train_end(self) -> int:
        return int(self.split_indices["train"].numel())

    @property
    def validation_end(self) -> int:
        return self.train_end + int(
            self.split_indices["validation"].numel()
        )


def load_ieee39_archive(
    processed_path: Path,
    metadata_path: Optional[Path] = None,
) -> IEEE39TransientArchive:
    processed_path = Path(processed_path).resolve()
    metadata_path = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else processed_path.with_name("ieee39_transient_v1_metadata.json")
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(processed_path)
    if actual_hash != metadata.get("processed_sha256"):
        raise ValueError("IEEE39 processed cache SHA256 does not match metadata.")
    source_path = Path(metadata["source_path"]).resolve()
    if sha256_file(source_path) != metadata.get("source_sha256"):
        raise ValueError("IEEE39 raw source SHA256 does not match metadata.")
    with np.load(processed_path, allow_pickle=False) as cache:
        required = {
            "trajectories", "timestamps_seconds", "labels", "scenario_ids",
            "train_indices", "validation_indices", "test_indices",
            "normalization_mean", "normalization_std", "generator_names",
            "feature_names", "feature_units", "edge_index", "edge_type",
        }
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f"IEEE39 cache missing fields: {sorted(missing)}")
        trajectories = np.asarray(cache["trajectories"], dtype=np.float32)
        timestamps = np.asarray(cache["timestamps_seconds"], dtype=np.float64)
        labels = np.asarray(cache["labels"], dtype=np.uint8)
        scenario_ids = np.asarray(cache["scenario_ids"], dtype=np.int64)
        splits = {
            name: np.asarray(cache[f"{name}_indices"], dtype=np.int64)
            for name in ("train", "validation", "test")
        }
        mean = np.asarray(cache["normalization_mean"], dtype=np.float64)
        std = np.asarray(cache["normalization_std"], dtype=np.float64)
        generators = tuple(str(value) for value in cache["generator_names"])
        features = tuple(str(value) for value in cache["feature_names"])
        units = tuple(str(value) for value in cache["feature_units"])
        edge_index = np.asarray(cache["edge_index"], dtype=np.int64)
        edge_type = np.asarray(cache["edge_type"], dtype=np.int64)
    if trajectories.shape != (12852, 60, 10, 5):
        raise ValueError(f"Unexpected IEEE39 cache shape: {trajectories.shape}")
    if timestamps.shape != (12852, 60):
        raise ValueError(f"Unexpected IEEE39 timestamp shape: {timestamps.shape}")
    if not np.isfinite(trajectories).all():
        raise ValueError("IEEE39 cache contains NaN or infinity.")
    if edge_index.shape != (2, 90):
        raise ValueError("IEEE39 candidate graph must have shape [2, 90].")
    all_indices = np.concatenate(tuple(splits.values()))
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("IEEE39 scenario splits overlap.")
    if set(all_indices.tolist()) != set(range(len(trajectories))):
        raise ValueError("IEEE39 scenario splits do not cover every scenario.")
    return IEEE39TransientArchive(
        path=processed_path,
        source_path=source_path,
        scenario_state=torch.from_numpy(trajectories),
        timestamps_seconds=torch.from_numpy(timestamps),
        labels=torch.from_numpy(labels),
        scenario_ids=torch.from_numpy(scenario_ids),
        split_indices={name: torch.from_numpy(value) for name, value in splits.items()},
        bus_indices=torch.arange(10, dtype=torch.long),
        edge_index=torch.from_numpy(edge_index),
        edge_type=torch.from_numpy(edge_type),
        bus_feature_names=features,
        generator_names=generators,
        feature_units=units,
        normalization_mean=torch.from_numpy(mean),
        normalization_std=torch.from_numpy(std),
        metadata=metadata,
    )


def _subset_count(scale: ScaleName, split: SplitName) -> Optional[int]:
    if scale == "smoke":
        return {"train": 16, "validation": 8, "test": 8}[split]
    if scale == "development":
        return {"train": 512, "validation": 128, "test": 128}[split]
    return None


def _mask_for_scenario(
    scenario_id: int,
    num_nodes: int,
    length: int,
    fraction: float,
    mask_seed: int,
    task: TaskName,
) -> np.ndarray:
    mask = np.zeros((num_nodes, length), dtype=np.bool_)
    base_count = int(math.floor(fraction * length + 0.5))
    for node in range(num_nodes):
        seed = np.random.SeedSequence(
            [mask_seed, scenario_id, node, 0x493339, int(task == "extrapolation")]
        )
        generator = np.random.default_rng(seed)
        count = int(np.clip(base_count + generator.integers(-1, 2), 2, length - 1))
        selected = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                generator.choice(
                    np.arange(1, length), size=count - 1, replace=False
                ),
            )
        )
        midpoint = length // 2
        if not np.any(selected >= midpoint):
            selected[-1] = int(generator.integers(midpoint, length))
        mask[node, np.unique(selected)] = True
        while int(mask[node].sum()) < count:
            candidate = int(generator.integers(0, length))
            mask[node, candidate] = True
    return mask


def _mask_cache_path(
    processed_path: Path,
    task: TaskName,
    fraction: float,
    mask_seed: int,
    scale: ScaleName,
) -> Path:
    return processed_path.parent / "masks" / (
        f"{task}__obs{int(round(100*fraction))}__seed{mask_seed}__{scale}.npz"
    )


def _load_or_create_masks(
    archive: IEEE39TransientArchive,
    selected: Dict[str, Tensor],
    task: TaskName,
    fraction: float,
    mask_seed: int,
    scale: ScaleName,
) -> Dict[str, Tensor]:
    path = _mask_cache_path(archive.path, task, fraction, mask_seed, scale)
    domain_length = 60 if task == "interpolation" else 30
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            cached_processed_hash = (
                str(cache["processed_sha256"].item())
                if "processed_sha256" in cache.files
                else None
            )
            cached_schema = (
                int(cache["mask_schema_version"].item())
                if "mask_schema_version" in cache.files
                else None
            )
            if (
                cached_processed_hash == archive.metadata["processed_sha256"]
                and cached_schema == MASK_SCHEMA_VERSION
            ):
                masks = {
                    split: torch.from_numpy(np.asarray(cache[f"{split}_masks"], dtype=np.bool_))
                    for split in selected
                }
                for split, indices in selected.items():
                    cached_indices = np.asarray(cache[f"{split}_indices"], dtype=np.int64)
                    if not np.array_equal(cached_indices, indices.numpy()):
                        raise ValueError(f"Cached IEEE39 {split} mask indices differ.")
                return masks
        path.unlink()
    arrays: Dict[str, np.ndarray] = {}
    arrays["processed_sha256"] = np.asarray(
        archive.metadata["processed_sha256"], dtype="U64"
    )
    arrays["mask_schema_version"] = np.asarray(
        MASK_SCHEMA_VERSION, dtype=np.int64
    )
    result = {}
    for split, indices in selected.items():
        split_masks = np.stack([
            _mask_for_scenario(
                int(archive.scenario_ids[index]), archive.num_nodes,
                domain_length, fraction, mask_seed, task,
            )
            for index in indices.tolist()
        ])
        arrays[f"{split}_masks"] = split_masks
        arrays[f"{split}_indices"] = indices.numpy()
        result[split] = torch.from_numpy(split_masks)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return result


class IEEE39TransientDataset(Dataset):
    def __init__(
        self,
        archive: IEEE39TransientArchive,
        split: SplitName,
        indices: Tensor,
        masks: Tensor,
        task: TaskName,
        normalization: NormalizationStats,
        max_temporal_gap: Optional[float],
    ) -> None:
        self.archive = archive
        self.split = split
        self.indices = indices.long()
        self.masks = masks.bool()
        self.task = task
        self.normalization = normalization
        self.max_temporal_gap = max_temporal_gap
        self.windows = tuple(
            IEEE39ScenarioRecord(
                trajectory_id=int(archive.scenario_ids[index]),
                split=split,
                scenario_index=int(index),
                start=0,
                stop=60,
                context_stop=60 if task == "interpolation" else 30,
                target_start=0 if task == "interpolation" else 30,
                target_stop=60,
            )
            for index in self.indices.tolist()
        )
        self._graph_cache: OrderedDict[int, Any] = OrderedDict()
        self._graph_cache_limit = min(len(self.windows), 512)
        self._physical_graph = _build_physical_graph(archive, 0)

    def __len__(self) -> int:
        return len(self.windows)

    def observation_mask(self, index: int) -> Tensor:
        return self.masks[index]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.windows[index]
        values = self.archive.scenario_state[record.scenario_index]
        mean = self.normalization.mean.to(values.dtype)
        std = self.normalization.std.to(values.dtype)
        normalized = (values - mean) / std
        normalized = normalized.permute(1, 0, 2).contiguous()
        timestamps = self.archive.timestamps_seconds[
            record.scenario_index
        ].to(torch.float32)
        mask = self.masks[index]
        if self.task == "interpolation":
            context_values = normalized
            encoder_times = timestamps - timestamps[0]
            target_values = normalized
            target_times = encoder_times
            withheld = (~mask).unsqueeze(-1).expand_as(target_values).clone()
            future = torch.zeros_like(withheld)
            training_mask = withheld
        else:
            context_values = normalized[:, :30]
            encoder_times = timestamps[:30] - timestamps[29]
            target_values = normalized[:, 30:]
            target_times = timestamps[30:] - timestamps[29]
            withheld = torch.zeros_like(target_values, dtype=torch.bool)
            future = torch.ones_like(target_values, dtype=torch.bool)
            training_mask = future
        if index in self._graph_cache:
            encoder_graph = self._graph_cache.pop(index)
            self._graph_cache[index] = encoder_graph
        else:
            encoder_graph = _build_encoder_graph(
                context_values, encoder_times, mask, self.archive.edge_index,
                self.max_temporal_gap,
            )
            encoder_graph.scenario_label = self.archive.labels[
                record.scenario_index
            ].view(1)
            if self.task == "interpolation":
                encoder_graph.latest_observation_time = torch.zeros(
                    self.archive.num_nodes, dtype=torch.float32
                )
            self._graph_cache[index] = encoder_graph
            if len(self._graph_cache) > self._graph_cache_limit:
                self._graph_cache.popitem(last=False)
        target_mask = torch.ones_like(target_values, dtype=torch.bool)
        return {
            "encoder_graph": encoder_graph,
            "physical_graph": self._physical_graph,
            "target_values": target_values,
            "target_times": target_times,
            "target_mask": target_mask,
            "observed_event_mask": mask,
            "encoder_observation_mask": mask,
            "training_loss_mask": training_mask,
            "interpolation_withheld_mask": withheld,
            "extrapolation_future_mask": future,
            "scenario_label": self.archive.labels[
                record.scenario_index
            ].long(),
            "trajectory_id": torch.tensor(record.trajectory_id, dtype=torch.long),
        }


def build_ieee39_dataloaders(
    data_path: Path,
    *,
    task: TaskName,
    observed_fraction: float,
    batch_size: int,
    seed: int,
    mask_seed: int,
    scale: ScaleName = "publication",
    max_temporal_gap: Optional[float] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
    **_: Any,
) -> PowerGridDataLoaders:
    archive = load_ieee39_archive(Path(data_path))
    selected = {}
    for split in ("train", "validation", "test"):
        indices = archive.split_indices[split]
        limit = _subset_count(scale, split)
        selected[split] = indices if limit is None else indices[:limit]
    selected_train = archive.scenario_state[selected["train"]].double()
    mean = selected_train.mean(dim=(0, 1, 2))
    std = selected_train.std(dim=(0, 1, 2), unbiased=False)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    normalization = NormalizationStats(
        mean=mean,
        std=std,
        count=int(selected_train.shape[0] * 60 * 10),
        fitted_start=0,
        fitted_end=len(selected["train"]),
    )
    masks = _load_or_create_masks(
        archive, selected, task, observed_fraction, mask_seed, scale
    )
    datasets = {
        split: IEEE39TransientDataset(
            archive, split, selected[split], masks[split], task,
            normalization, max_temporal_gap,
        )
        for split in ("train", "validation", "test")
    }
    generator = torch.Generator().manual_seed(int(seed))
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_powergrid_lgode,
    )
    return PowerGridDataLoaders(
        train=DataLoader(
            datasets["train"], shuffle=True, drop_last=drop_last_train,
            generator=generator, **common,
        ),
        validation=DataLoader(datasets["validation"], shuffle=False, **common),
        test=DataLoader(datasets["test"], shuffle=False, **common),
        normalization=normalization,
        archive=archive,
    )


def observation_statistics(dataset: IEEE39TransientDataset) -> Dict[str, Any]:
    mask = dataset.masks
    counts = mask.sum(dim=-1).float()
    gaps = []
    simultaneous = []
    multi_generator = []
    for scenario_mask in mask:
        observations_per_time = scenario_mask.sum(dim=0)
        simultaneous.append(
            float((observations_per_time == scenario_mask.shape[0]).float().mean())
        )
        multi_generator.append(
            float((observations_per_time > 1).float().mean())
        )
        for node_mask in scenario_mask:
            indices = torch.nonzero(node_mask, as_tuple=False).flatten().float()
            if indices.numel() > 1:
                gaps.append(indices.diff())
    gap_values = torch.cat(gaps)
    return {
        "observations_per_generator_min": int(counts.min()),
        "observations_per_generator_mean": float(counts.mean()),
        "observations_per_generator_max": int(counts.max()),
        "realized_observed_fraction": float(mask.float().mean()),
        "gap_samples_mean": float(gap_values.mean()),
        "gap_samples_std": float(gap_values.std(unbiased=False)),
        "gap_samples_quantiles": {
            str(value): float(torch.quantile(gap_values, value))
            for value in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "simultaneous_observation_fraction": float(np.mean(simultaneous)),
        "multi_generator_observation_fraction": float(
            np.mean(multi_generator)
        ),
        "asynchronous": bool(
            any(not torch.equal(row[0], row[node]) for row in mask for node in range(1, row.shape[0]))
        ),
    }


__all__ = [
    "IEEE39TransientArchive",
    "IEEE39TransientDataset",
    "build_ieee39_dataloaders",
    "load_ieee39_archive",
    "observation_statistics",
]