from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

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

# Version 3 removes forced timestep-zero/midpoint observations and changes
# interpolation training supervision from withheld entries to observed entries.
MASK_SCHEMA_VERSION = 3

_ALLOWED_OBSERVED_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
_SPLITS: Tuple[SplitName, ...] = ("train", "validation", "test")


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


def _decode_strings(values: np.ndarray) -> Tuple[str, ...]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return tuple(result)


def load_ieee39_archive(
    processed_path: Path,
    metadata_path: Optional[Path] = None,
) -> IEEE39TransientArchive:
    processed_path = Path(processed_path).expanduser().resolve()
    metadata_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else processed_path.with_name(
            "ieee39_transient_v1_metadata.json"
        )
    )

    if not processed_path.is_file():
        raise FileNotFoundError(
            f"Processed IEEE39 cache not found: {processed_path}"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"IEEE39 metadata not found: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    actual_processed_hash = sha256_file(processed_path)
    expected_processed_hash = metadata.get("processed_sha256")
    if actual_processed_hash != expected_processed_hash:
        raise ValueError(
            "IEEE39 processed cache SHA256 does not match metadata: "
            f"{actual_processed_hash} != {expected_processed_hash}"
        )

    source_path = Path(metadata["source_path"]).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Raw IEEE39 source referenced by metadata is missing: "
            f"{source_path}"
        )

    # Hashing the 328 MB pickle for every model process is expensive.
    # The single shell entrypoint can set IEEE39_VERIFY_RAW_HASH=1 for a
    # publication preflight. Processed-cache validation remains mandatory.
    if os.environ.get("IEEE39_VERIFY_RAW_HASH", "0") == "1":
        actual_source_hash = sha256_file(source_path)
        expected_source_hash = metadata.get("source_sha256")
        if actual_source_hash != expected_source_hash:
            raise ValueError(
                "IEEE39 raw source SHA256 does not match metadata: "
                f"{actual_source_hash} != {expected_source_hash}"
            )

    with np.load(processed_path, allow_pickle=False) as cache:
        required = {
            "trajectories",
            "timestamps_seconds",
            "labels",
            "scenario_ids",
            "train_indices",
            "validation_indices",
            "test_indices",
            "normalization_mean",
            "normalization_std",
            "generator_names",
            "feature_names",
            "feature_units",
            "edge_index",
            "edge_type",
        }
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(
                f"IEEE39 cache missing fields: {sorted(missing)}"
            )

        trajectories = np.asarray(
            cache["trajectories"], dtype=np.float32
        )
        timestamps = np.asarray(
            cache["timestamps_seconds"], dtype=np.float64
        )
        labels = np.asarray(cache["labels"], dtype=np.uint8)
        scenario_ids = np.asarray(
            cache["scenario_ids"], dtype=np.int64
        )
        splits = {
            split: np.asarray(
                cache[f"{split}_indices"], dtype=np.int64
            )
            for split in _SPLITS
        }
        mean = np.asarray(
            cache["normalization_mean"], dtype=np.float64
        )
        std = np.asarray(
            cache["normalization_std"], dtype=np.float64
        )
        generator_names = _decode_strings(
            cache["generator_names"]
        )
        feature_names = _decode_strings(cache["feature_names"])
        feature_units = _decode_strings(cache["feature_units"])
        edge_index = np.asarray(
            cache["edge_index"], dtype=np.int64
        )
        edge_type = np.asarray(
            cache["edge_type"], dtype=np.int64
        ).reshape(-1)

    if trajectories.shape != (12852, 60, 10, 5):
        raise ValueError(
            f"Unexpected IEEE39 cache shape: {trajectories.shape}"
        )
    if timestamps.shape != (12852, 60):
        raise ValueError(
            f"Unexpected IEEE39 timestamp shape: {timestamps.shape}"
        )
    if labels.shape != (12852,):
        raise ValueError(
            f"Unexpected IEEE39 label shape: {labels.shape}"
        )
    if scenario_ids.shape != (12852,):
        raise ValueError(
            f"Unexpected IEEE39 scenario-ID shape: "
            f"{scenario_ids.shape}"
        )
    if mean.shape != (5,) or std.shape != (5,):
        raise ValueError(
            "IEEE39 normalization statistics must have shape [5]."
        )
    if not np.isfinite(trajectories).all():
        raise ValueError(
            "IEEE39 cache contains NaN or infinity."
        )
    if not np.isfinite(timestamps).all():
        raise ValueError(
            "IEEE39 timestamps contain NaN or infinity."
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError(
            "IEEE39 normalization statistics are non-finite."
        )
    if np.any(std <= 0.0):
        raise ValueError(
            "IEEE39 normalization standard deviations must be positive."
        )
    if not np.all(np.diff(timestamps, axis=1) > 0.0):
        raise ValueError(
            "Every IEEE39 timestamp grid must be strictly increasing."
        )

    if len(generator_names) != 10:
        raise ValueError(
            "IEEE39 cache must contain ten generator names."
        )
    if len(feature_names) != 5 or len(feature_units) != 5:
        raise ValueError(
            "IEEE39 cache must contain five feature names and units."
        )
    if edge_index.shape != (2, 90):
        raise ValueError(
            "IEEE39 candidate graph must have shape [2, 90]."
        )
    if edge_type.shape != (90,):
        raise ValueError(
            "IEEE39 edge_type must have shape [90]."
        )
    if np.any(edge_index < 0) or np.any(edge_index >= 10):
        raise ValueError(
            "IEEE39 candidate graph contains an invalid node index."
        )
    if np.any(edge_index[0] == edge_index[1]):
        raise ValueError(
            "IEEE39 candidate graph must not contain self-loops."
        )
    if len(set(map(tuple, edge_index.T.tolist()))) != 90:
        raise ValueError(
            "IEEE39 candidate graph contains duplicate edges."
        )

    all_indices = np.concatenate(tuple(splits.values()))
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("IEEE39 scenario splits overlap.")
    if set(all_indices.tolist()) != set(range(len(trajectories))):
        raise ValueError(
            "IEEE39 scenario splits do not cover every scenario."
        )

    return IEEE39TransientArchive(
        path=processed_path,
        source_path=source_path,
        scenario_state=torch.from_numpy(
            np.ascontiguousarray(trajectories)
        ).float(),
        timestamps_seconds=torch.from_numpy(
            np.ascontiguousarray(timestamps)
        ).double(),
        labels=torch.from_numpy(
            np.ascontiguousarray(labels)
        ).long(),
        scenario_ids=torch.from_numpy(
            np.ascontiguousarray(scenario_ids)
        ).long(),
        split_indices={
            split: torch.from_numpy(
                np.ascontiguousarray(indices)
            ).long()
            for split, indices in splits.items()
        },
        bus_indices=torch.arange(10, dtype=torch.long),
        edge_index=torch.from_numpy(
            np.ascontiguousarray(edge_index)
        ).long(),
        edge_type=torch.from_numpy(
            np.ascontiguousarray(edge_type)
        ).long(),
        bus_feature_names=feature_names,
        generator_names=generator_names,
        feature_units=feature_units,
        normalization_mean=torch.from_numpy(
            np.ascontiguousarray(mean)
        ).float(),
        normalization_std=torch.from_numpy(
            np.ascontiguousarray(std)
        ).float(),
        metadata=metadata,
    )


def _subset_count(
    scale: ScaleName,
    split: SplitName,
) -> Optional[int]:
    if scale == "smoke":
        return {
            "train": 16,
            "validation": 8,
            "test": 8,
        }[split]
    if scale == "development":
        return {
            "train": 512,
            "validation": 128,
            "test": 128,
        }[split]
    if scale == "publication":
        return None
    raise ValueError(f"Unknown IEEE39 scale: {scale!r}")


def _validate_fraction(fraction: float) -> float:
    value = float(fraction)
    for allowed in _ALLOWED_OBSERVED_FRACTIONS:
        if math.isclose(
            value,
            allowed,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return float(allowed)

    raise ValueError(
        "observed_fraction must be one of "
        f"{_ALLOWED_OBSERVED_FRACTIONS}; got {fraction}"
    )


def _observation_count(
    length: int,
    fraction: float,
) -> int:
    if length < 2:
        raise ValueError(
            "Observation domain must contain at least two timestamps."
        )

    count = int(math.floor(float(fraction) * length + 0.5))
    return int(np.clip(count, 1, length - 1))


def _mask_for_scenario(
    scenario_id: int,
    num_nodes: int,
    length: int,
    fraction: float,
    mask_seed: int,
    task: TaskName,
) -> np.ndarray:
    """
    Independently subsample each generator's legal observation domain.

    This follows the LG-ODE experimental idea: the underlying trajectory may
    be regularly sampled, while each object retains its own independently
    selected subset of observation times.

    No timestamp is forcibly shared across generators. In particular:
    - timestep zero is not forced;
    - the final context timestamp is not forced;
    - interpolation observations are not forced into either half;
    - extrapolation observations are sampled only from the legal context.

    Every generator receives the same deterministic observation count, but
    independently selected timestamps.
    """

    fraction = _validate_fraction(fraction)

    if task not in ("interpolation", "extrapolation"):
        raise ValueError(f"Unsupported task: {task!r}")
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive.")
    if length < 2:
        raise ValueError("length must be at least two.")

    count = _observation_count(length, fraction)
    mask = np.zeros((num_nodes, length), dtype=np.bool_)

    for node in range(num_nodes):
        seed_sequence = np.random.SeedSequence(
            [
                int(mask_seed) & 0xFFFFFFFF,
                int(scenario_id) & 0xFFFFFFFF,
                int(node) & 0xFFFFFFFF,
                0x493339,  # IEEE39 namespace
                int(task == "extrapolation"),
                MASK_SCHEMA_VERSION,
            ]
        )
        generator = np.random.default_rng(seed_sequence)
        selected = generator.choice(
            length,
            size=count,
            replace=False,
        )
        mask[node, selected] = True

    if not np.all(mask.sum(axis=1) == count):
        raise RuntimeError(
            "IEEE39 mask construction produced an invalid count."
        )

    return mask


def _mask_cache_path(
    processed_path: Path,
    task: TaskName,
    fraction: float,
    mask_seed: int,
    scale: ScaleName,
) -> Path:
    observation_tag = int(round(100 * fraction))
    return processed_path.parent / "masks" / (
        f"maskv{MASK_SCHEMA_VERSION}__{task}"
        f"__obs{observation_tag}"
        f"__seed{mask_seed}"
        f"__{scale}.npz"
    )


def _atomic_save_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_or_create_masks(
    archive: IEEE39TransientArchive,
    selected: Mapping[str, Tensor],
    task: TaskName,
    fraction: float,
    mask_seed: int,
    scale: ScaleName,
) -> Dict[str, Tensor]:
    path = _mask_cache_path(
        archive.path,
        task,
        fraction,
        mask_seed,
        scale,
    )
    domain_length = 60 if task == "interpolation" else 30

    if path.exists():
        cache_is_valid = False
        try:
            with np.load(path, allow_pickle=False) as cache:
                required = {
                    "processed_sha256",
                    "mask_schema_version",
                    "task",
                    "observed_fraction",
                    "mask_seed",
                    "domain_length",
                }
                for split in selected:
                    required.add(f"{split}_masks")
                    required.add(f"{split}_indices")

                if required.issubset(cache.files):
                    cached_hash = str(
                        cache["processed_sha256"].item()
                    )
                    cached_schema = int(
                        cache["mask_schema_version"].item()
                    )
                    cached_task = str(cache["task"].item())
                    cached_fraction = float(
                        cache["observed_fraction"].item()
                    )
                    cached_seed = int(cache["mask_seed"].item())
                    cached_length = int(
                        cache["domain_length"].item()
                    )

                    cache_is_valid = (
                        cached_hash
                        == archive.metadata["processed_sha256"]
                        and cached_schema == MASK_SCHEMA_VERSION
                        and cached_task == task
                        and math.isclose(
                            cached_fraction,
                            fraction,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        and cached_seed == int(mask_seed)
                        and cached_length == domain_length
                    )

                    if cache_is_valid:
                        result: Dict[str, Tensor] = {}
                        for split, indices in selected.items():
                            cached_indices = np.asarray(
                                cache[f"{split}_indices"],
                                dtype=np.int64,
                            )
                            expected_indices = indices.cpu().numpy()

                            if not np.array_equal(
                                cached_indices,
                                expected_indices,
                            ):
                                raise ValueError(
                                    f"Cached IEEE39 {split} mask "
                                    "indices differ from selected scenarios."
                                )

                            cached_masks = np.asarray(
                                cache[f"{split}_masks"],
                                dtype=np.bool_,
                            )
                            expected_shape = (
                                len(indices),
                                archive.num_nodes,
                                domain_length,
                            )
                            if cached_masks.shape != expected_shape:
                                raise ValueError(
                                    f"Cached IEEE39 {split} masks have "
                                    f"shape {cached_masks.shape}, expected "
                                    f"{expected_shape}."
                                )

                            result[split] = torch.from_numpy(
                                np.ascontiguousarray(cached_masks)
                            ).bool()

                        return result
        except (OSError, ValueError, KeyError):
            cache_is_valid = False

        if not cache_is_valid:
            path.unlink(missing_ok=True)

    arrays: Dict[str, np.ndarray] = {
        "processed_sha256": np.asarray(
            archive.metadata["processed_sha256"],
            dtype="U64",
        ),
        "mask_schema_version": np.asarray(
            MASK_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "task": np.asarray(task, dtype="U16"),
        "observed_fraction": np.asarray(
            fraction,
            dtype=np.float64,
        ),
        "mask_seed": np.asarray(mask_seed, dtype=np.int64),
        "domain_length": np.asarray(
            domain_length,
            dtype=np.int64,
        ),
    }

    result = {}
    for split, indices in selected.items():
        split_masks = np.stack(
            [
                _mask_for_scenario(
                    scenario_id=int(
                        archive.scenario_ids[index].item()
                    ),
                    num_nodes=archive.num_nodes,
                    length=domain_length,
                    fraction=fraction,
                    mask_seed=mask_seed,
                    task=task,
                )
                for index in indices.tolist()
            ],
            axis=0,
        )

        arrays[f"{split}_masks"] = split_masks
        arrays[f"{split}_indices"] = (
            indices.cpu().numpy().astype(np.int64, copy=False)
        )
        result[split] = torch.from_numpy(
            np.ascontiguousarray(split_masks)
        ).bool()

    _atomic_save_npz(path, arrays)
    return result


def _normalization_from_archive(
    archive: IEEE39TransientArchive,
) -> NormalizationStats:
    train_count = int(
        archive.split_indices["train"].numel()
        * archive.num_timesteps
        * archive.num_nodes
    )
    return NormalizationStats(
        mean=archive.normalization_mean.float().clone(),
        std=archive.normalization_std.float().clone(),
        count=train_count,
        fitted_start=0,
        fitted_end=int(
            archive.split_indices["train"].numel()
        ),
        eps=1e-8,
    )


def _relative_interpolation_times(
    timestamps: Tensor,
) -> Tensor:
    timestamps = timestamps.to(torch.float64)
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        raise ValueError(
            "Interpolation timestamps must be increasing."
        )
    return (
        (timestamps - timestamps[0]) / duration
    ).to(torch.float32)


def _relative_extrapolation_times(
    context_timestamps: Tensor,
    target_timestamps: Tensor,
) -> Tuple[Tensor, Tensor]:
    context_timestamps = context_timestamps.to(torch.float64)
    target_timestamps = target_timestamps.to(torch.float64)

    scale = context_timestamps[-1] - context_timestamps[0]
    if scale <= 0:
        raise ValueError(
            "Extrapolation context timestamps must be increasing."
        )
    if target_timestamps[0] <= context_timestamps[-1]:
        raise ValueError(
            "Extrapolation targets must occur after the context."
        )

    origin = context_timestamps[-1]
    encoder_times = (
        (context_timestamps - origin) / scale
    ).to(torch.float32)
    target_times = (
        (target_timestamps - origin) / scale
    ).to(torch.float32)
    return encoder_times, target_times


class IEEE39TransientDataset(Dataset):
    """Scenario-level IEEE-39 transient dataset for LG-ODE experiments."""

    def __init__(
        self,
        archive: IEEE39TransientArchive,
        split: SplitName,
        task: TaskName,
        observed_fraction: float,
        scenario_indices: Tensor,
        masks: Tensor,
        *,
        max_temporal_gap: Optional[float] = None,
    ) -> None:
        super().__init__()

        if split not in _SPLITS:
            raise ValueError(f"Unknown split: {split!r}")
        if task not in ("interpolation", "extrapolation"):
            raise ValueError(f"Unknown task: {task!r}")
        if max_temporal_gap is not None and max_temporal_gap < 0:
            raise ValueError(
                "max_temporal_gap must be non-negative or None."
            )

        self.archive = archive
        self.split = split
        self.task = task
        self.observed_fraction = _validate_fraction(
            observed_fraction
        )
        self.scenario_indices = (
            scenario_indices.clone().long()
        )
        self.masks = masks.clone().bool()
        self.max_temporal_gap = max_temporal_gap
        self.normalization = _normalization_from_archive(archive)

        domain_length = 60 if task == "interpolation" else 30
        expected_mask_shape = (
            len(self.scenario_indices),
            archive.num_nodes,
            domain_length,
        )
        if tuple(self.masks.shape) != expected_mask_shape:
            raise ValueError(
                f"IEEE39 {split} mask shape {tuple(self.masks.shape)} "
                f"does not match expected {expected_mask_shape}."
            )
        if torch.any(self.masks.sum(dim=-1) < 1):
            raise ValueError(
                "Every IEEE39 generator must have at least one "
                "legal observation."
            )

        records = []
        for local_index, scenario_index in enumerate(
            self.scenario_indices.tolist()
        ):
            trajectory_id = int(
                archive.scenario_ids[scenario_index].item()
            )
            if task == "interpolation":
                context_stop = 60
                target_start = 0
                target_stop = 60
            else:
                context_stop = 30
                target_start = 30
                target_stop = 60

            records.append(
                IEEE39ScenarioRecord(
                    trajectory_id=trajectory_id,
                    split=split,
                    scenario_index=int(scenario_index),
                    start=0,
                    stop=60,
                    context_stop=context_stop,
                    target_start=target_start,
                    target_stop=target_stop,
                )
            )

        self.windows = tuple(records)

    @property
    def num_nodes(self) -> int:
        return self.archive.num_nodes

    @property
    def input_dim(self) -> int:
        return self.archive.input_dim

    @property
    def bus_feature_names(self) -> Tuple[str, ...]:
        return self.archive.bus_feature_names

    def __len__(self) -> int:
        return len(self.windows)

    def get_window_record(
        self,
        index: int,
    ) -> IEEE39ScenarioRecord:
        return self.windows[index]

    def observation_mask(self, index: int) -> Tensor:
        return self.masks[index].clone()

    def _normalize(self, values: Tensor) -> Tensor:
        mean = self.archive.normalization_mean.to(
            dtype=values.dtype
        )
        std = self.archive.normalization_std.to(
            dtype=values.dtype
        )
        return (values - mean) / std

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.windows[index]
        scenario_index = record.scenario_index

        full_values = self.archive.scenario_state[
            scenario_index
        ].float()
        full_timestamps = self.archive.timestamps_seconds[
            scenario_index
        ].double()

        normalized_values = self._normalize(full_values)
        observed_event_mask = self.masks[index].clone()

        if self.task == "interpolation":
            context_values = normalized_values.permute(
                1, 0, 2
            ).contiguous()
            encoder_times = _relative_interpolation_times(
                full_timestamps
            )

            target_values = context_values.clone()
            target_times = encoder_times.clone()
        else:
            context_values = normalized_values[:30].permute(
                1, 0, 2
            ).contiguous()
            target_values = normalized_values[30:].permute(
                1, 0, 2
            ).contiguous()

            encoder_times, target_times = (
                _relative_extrapolation_times(
                    full_timestamps[:30],
                    full_timestamps[30:],
                )
            )

        encoder_graph = _build_encoder_graph(
            observed_values=context_values,
            observed_times=encoder_times,
            observed_mask=observed_event_mask,
            physical_edge_index=self.archive.edge_index,
            max_temporal_gap=self.max_temporal_gap,
        )

        physical_graph = _build_physical_graph(
            self.archive,
            trajectory_id=record.trajectory_id,
        )

        target_mask = torch.ones_like(
            target_values,
            dtype=torch.bool,
        )

        if self.task == "interpolation":
            observed_feature_mask = (
                observed_event_mask.unsqueeze(-1)
                .expand_as(target_values)
                .clone()
            )
            interpolation_withheld_mask = (
                ~observed_event_mask
            ).unsqueeze(-1).expand_as(target_values).clone()

            # LG-ODE-faithful partial-observation training:
            # likelihood is evaluated only where observations are available.
            # Withheld values are reserved for validation/test interpolation
            # evaluation and never supervise interpolation training.
            training_loss_mask = observed_feature_mask
            extrapolation_future_mask = torch.zeros_like(
                target_mask
            )
        else:
            interpolation_withheld_mask = torch.zeros_like(
                target_mask
            )
            extrapolation_future_mask = target_mask.clone()

            # Forecast targets are available as supervision for training
            # scenarios, but are never passed to the encoder.
            training_loss_mask = extrapolation_future_mask.clone()

        encoder_graph.trajectory_id = torch.tensor(
            [record.trajectory_id],
            dtype=torch.long,
        )
        encoder_graph.window_start = torch.tensor(
            [record.start],
            dtype=torch.long,
        )
        encoder_graph.window_stop = torch.tensor(
            [record.stop],
            dtype=torch.long,
        )
        encoder_graph.observation_domain_length = torch.tensor(
            [observed_event_mask.shape[-1]],
            dtype=torch.long,
        )
        encoder_graph.scenario_index = torch.tensor(
            [scenario_index],
            dtype=torch.long,
        )

        return {
            "encoder_graph": encoder_graph,
            "physical_graph": physical_graph,
            "target_values": target_values,
            "target_times": target_times,
            "target_mask": target_mask,
            "observed_event_mask": observed_event_mask,
            "encoder_observation_mask": observed_event_mask.clone(),
            "training_loss_mask": training_loss_mask,
            "interpolation_withheld_mask": (
                interpolation_withheld_mask
            ),
            "extrapolation_future_mask": (
                extrapolation_future_mask
            ),
            "scenario_label": self.archive.labels[
                scenario_index
            ].clone().long(),
            "trajectory_id": torch.tensor(
                record.trajectory_id,
                dtype=torch.long,
            ),
        }


def _select_scenarios(
    archive: IEEE39TransientArchive,
    scale: ScaleName,
) -> "OrderedDict[str, Tensor]":
    selected: "OrderedDict[str, Tensor]" = OrderedDict()

    for split in _SPLITS:
        indices = archive.split_indices[split].clone().long()
        count = _subset_count(scale, split)
        if count is not None:
            if len(indices) < count:
                raise ValueError(
                    f"IEEE39 {split} split contains {len(indices)} "
                    f"scenarios but {scale} mode requires {count}."
                )
            indices = indices[:count]
        selected[split] = indices

    return selected


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def build_ieee39_datasets(
    data_path: Path,
    *,
    task: TaskName,
    observed_fraction: float,
    seed: int = 0,
    mask_seed: Optional[int] = None,
    scale: ScaleName = "publication",
    max_temporal_gap: Optional[float] = None,
    trajectory_length: int = 60,
    context_length: int = 30,
    forecast_length: int = 30,
    stride: int = 1,
    normalization_eps: float = 1e-8,
) -> Dict[str, IEEE39TransientDataset]:
    del seed, normalization_eps

    if task == "interpolation" and trajectory_length != 60:
        raise ValueError(
            "IEEE39 interpolation requires trajectory_length=60."
        )
    if task == "extrapolation":
        if context_length != 30 or forecast_length != 30:
            raise ValueError(
                "IEEE39 extrapolation currently requires "
                "context_length=30 and forecast_length=30."
            )
    if stride != 1:
        raise ValueError(
            "IEEE39 scenarios are independent trajectories; stride must be 1."
        )
    if scale not in ("smoke", "development", "publication"):
        raise ValueError(f"Unknown scenario scale: {scale!r}")

    fraction = _validate_fraction(observed_fraction)
    effective_mask_seed = (
        0 if mask_seed is None else int(mask_seed)
    )

    archive = load_ieee39_archive(Path(data_path))
    selected = _select_scenarios(archive, scale)
    masks = _load_or_create_masks(
        archive=archive,
        selected=selected,
        task=task,
        fraction=fraction,
        mask_seed=effective_mask_seed,
        scale=scale,
    )

    return {
        split: IEEE39TransientDataset(
            archive=archive,
            split=split,
            task=task,
            observed_fraction=fraction,
            scenario_indices=selected[split],
            masks=masks[split],
            max_temporal_gap=max_temporal_gap,
        )
        for split in _SPLITS
    }


def build_ieee39_dataloaders(
    data_path: Path,
    *,
    task: TaskName,
    observed_fraction: float,
    batch_size: int,
    seed: int = 0,
    mask_seed: Optional[int] = None,
    scale: ScaleName = "publication",
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
    max_temporal_gap: Optional[float] = None,
    trajectory_length: int = 60,
    context_length: int = 30,
    forecast_length: int = 30,
    stride: int = 1,
    normalization_eps: float = 1e-8,
) -> PowerGridDataLoaders:
    if batch_size < 1:
        raise ValueError(
            f"batch_size must be positive; got {batch_size}"
        )
    if num_workers < 0:
        raise ValueError(
            f"num_workers must be non-negative; got {num_workers}"
        )

    effective_mask_seed = (
        int(seed) if mask_seed is None else int(mask_seed)
    )

    datasets = build_ieee39_datasets(
        data_path=Path(data_path),
        task=task,
        observed_fraction=observed_fraction,
        seed=seed,
        mask_seed=effective_mask_seed,
        scale=scale,
        max_temporal_gap=max_temporal_gap,
        trajectory_length=trajectory_length,
        context_length=context_length,
        forecast_length=forecast_length,
        stride=stride,
        normalization_eps=normalization_eps,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    common_arguments = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_powergrid_lgode,
        "worker_init_fn": _seed_worker,
    }

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        drop_last=bool(drop_last_train),
        generator=generator,
        **common_arguments,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )
    test_loader = DataLoader(
        datasets["test"],
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )

    normalization = datasets["train"].normalization

    return PowerGridDataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        normalization=normalization,
        archive=datasets["train"].archive,
    )


def observation_statistics(
    dataset: IEEE39TransientDataset,
) -> Dict[str, Any]:
    masks = dataset.masks.bool()

    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks [S,N,T], got {tuple(masks.shape)}."
        )

    counts = masks.sum(dim=-1)
    observations_per_time = masks.sum(dim=1)
    common_observation_mask = masks.all(dim=1)

    gaps = []
    first_indices = []
    last_indices = []
    asynchronous_scenarios = []

    for scenario_mask in masks:
        asynchronous_scenarios.append(
            any(
                not torch.equal(
                    scenario_mask[0],
                    scenario_mask[node],
                )
                for node in range(1, scenario_mask.shape[0])
            )
        )

        for node_mask in scenario_mask:
            indices = torch.nonzero(
                node_mask,
                as_tuple=False,
            ).flatten()

            if indices.numel() == 0:
                raise RuntimeError(
                    "A generator has no legal observations."
                )

            first_indices.append(indices[0].float())
            last_indices.append(indices[-1].float())

            if indices.numel() > 1:
                gaps.append(indices.diff().float())

    gap_values = (
        torch.cat(gaps)
        if gaps
        else torch.empty(0, dtype=torch.float32)
    )
    first_values = torch.stack(first_indices)
    last_values = torch.stack(last_indices)

    if gap_values.numel():
        gap_quantiles = {
            str(value): float(
                torch.quantile(gap_values, value)
            )
            for value in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        gap_mean = float(gap_values.mean())
        gap_std = float(
            gap_values.std(unbiased=False)
        )
    else:
        gap_quantiles = {}
        gap_mean = 0.0
        gap_std = 0.0

    return {
        "mask_schema_version": MASK_SCHEMA_VERSION,
        "task": dataset.task,
        "scenario_count": int(masks.shape[0]),
        "generator_count": int(masks.shape[1]),
        "observation_domain_length": int(masks.shape[2]),
        "observations_per_generator_min": int(counts.min()),
        "observations_per_generator_mean": float(
            counts.float().mean()
        ),
        "observations_per_generator_max": int(counts.max()),
        "realized_observed_fraction": float(
            masks.float().mean()
        ),
        "gap_samples_mean": gap_mean,
        "gap_samples_std": gap_std,
        "gap_samples_quantiles": gap_quantiles,
        "first_observation_index_mean": float(
            first_values.mean()
        ),
        "first_observation_index_min": int(
            first_values.min()
        ),
        "first_observation_index_max": int(
            first_values.max()
        ),
        "last_observation_index_mean": float(
            last_values.mean()
        ),
        "last_observation_index_min": int(
            last_values.min()
        ),
        "last_observation_index_max": int(
            last_values.max()
        ),
        "timestep_zero_observation_fraction": float(
            masks[:, :, 0].float().mean()
        ),
        "final_timestep_observation_fraction": float(
            masks[:, :, -1].float().mean()
        ),
        "fully_synchronized_timestep_fraction": float(
            common_observation_mask.float().mean()
        ),
        "multi_generator_observation_fraction": float(
            (observations_per_time > 1).float().mean()
        ),
        "asynchronous_scenario_fraction": float(
            np.mean(asynchronous_scenarios)
        ),
        "all_scenarios_asynchronous": bool(
            all(asynchronous_scenarios)
        ),
        "forced_shared_endpoint": False,
    }


__all__ = [
    "IEEE39ScenarioRecord",
    "IEEE39TransientArchive",
    "IEEE39TransientDataset",
    "MASK_SCHEMA_VERSION",
    "_mask_for_scenario",
    "build_ieee39_dataloaders",
    "build_ieee39_datasets",
    "load_ieee39_archive",
    "observation_statistics",
    "sha256_file",
]
