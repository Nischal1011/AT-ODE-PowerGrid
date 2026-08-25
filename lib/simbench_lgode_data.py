# lib/simbench_lgode_data.py
"""
SimBench data pipeline for the power-grid LG-ODE experiments.

This module implements the common data realization shared by:

    persistence
    latentode
    lgode
    atode

Responsibilities
----------------
* Load a complete physical SimBench trajectory from an NPZ archive.
* Fit feature normalization using training timesteps only.
* Construct chronological train/validation/test windows.
* Sample asynchronous observations independently for every bus.
* Build the temporal event graph expected by the original GTrans encoder.
* Return complete interpolation targets or future extrapolation targets.
* Keep masks deterministic and independent of DataLoader iteration order.
* Never forward-fill or otherwise impute missing observations.

Expected NPZ fields
-------------------
bus_state:
    Complete physical trajectory with shape [T, N, F]. The loader also accepts
    [N, T, F] when the time dimension can be inferred unambiguously.

timestamps_hours:
    One-dimensional timestamps with shape [T].

bus_indices:
    Original SimBench bus identifiers with shape [N].

edge_index:
    Physical directed edges with shape [2, E] or [E, 2].

edge_type:
    Edge types with shape [E]. A scalar is broadcast to all edges.

train_end:
    Exclusive integer end index of the training split.

validation_end:
    Exclusive integer end index of the validation split. The test split is
    [validation_end, T).

bus_feature_names:
    Feature names with shape [F].

Batch contract
--------------
Each batch is a PowerGridBatch containing:

    encoder_graph
    physical_graph
    target_values
    target_times
    target_mask
    observed_event_mask
    trajectory_id

The encoder graph is a torch_geometric Batch whose component Data objects
contain the original LG-ODE fields:

    x
    pos
    edge_index
    edge_attr
    edge_same
    batch          # added by torch_geometric batching
    y

Additional non-required metadata is attached where useful, including
latest_observation_time and event_bus.

Shape conventions
-----------------
target_values:
    [batch, num_buses, target_length, num_features]

target_times:
    [batch, target_length]

target_mask:
    [batch, num_buses, target_length, num_features]

observed_event_mask:
    [batch, num_buses, observation_domain_length]

trajectory_id:
    [batch]

For interpolation, the observation domain and target domain are the complete
trajectory window.

For extrapolation, the observation domain is the context interval and the
target domain is the future forecast interval.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric.data import Batch, Data
except ImportError as exc:  # pragma: no cover - dependency failure is explicit
    raise ImportError(
        "lib.simbench_lgode_data requires torch_geometric. "
        "Install a torch-geometric version compatible with your PyTorch build."
    ) from exc


TaskName = Literal["interpolation", "extrapolation"]
SplitName = Literal["train", "validation", "test"]


_REQUIRED_NPZ_FIELDS = {
    "bus_state",
    "timestamps_hours",
    "bus_indices",
    "edge_index",
    "edge_type",
    "train_end",
    "validation_end",
    "bus_feature_names",
}

_BATCH_KEYS = (
    "encoder_graph",
    "physical_graph",
    "target_values",
    "target_times",
    "target_mask",
    "observed_event_mask",
    "encoder_observation_mask",
    "training_loss_mask",
    "interpolation_withheld_mask",
    "extrapolation_future_mask",
    "trajectory_id",
)


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature normalization fitted on training timesteps only."""

    mean: Tensor
    std: Tensor
    count: int
    fitted_start: int
    fitted_end: int
    eps: float = 1e-8

    def normalize(self, values: Tensor) -> Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def denormalize(self, values: Tensor) -> Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return values * std + mean

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.clone(),
            "std": self.std.clone(),
            "count": self.count,
            "fitted_start": self.fitted_start,
            "fitted_end": self.fitted_end,
            "eps": self.eps,
        }


@dataclass(frozen=True)
class SimBenchArchive:
    """Validated in-memory representation of a SimBench NPZ archive."""

    path: Path
    bus_state: Tensor
    timestamps_hours: Tensor
    bus_indices: Tensor
    edge_index: Tensor
    edge_type: Tensor
    train_end: int
    validation_end: int
    bus_feature_names: Tuple[str, ...]
    metadata: Dict[str, Any]

    @property
    def num_timesteps(self) -> int:
        return int(self.bus_state.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.bus_state.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.bus_state.shape[2])


@dataclass(frozen=True)
class WindowRecord:
    """One chronological trajectory window."""

    trajectory_id: int
    split: SplitName
    start: int
    stop: int
    context_stop: int
    target_start: int
    target_stop: int


@dataclass
class PowerGridBatch(Mapping[str, Any]):
    """Dictionary-compatible batch used by every power-grid model."""

    encoder_graph: Batch
    physical_graph: Batch
    target_values: Tensor
    target_times: Tensor
    target_mask: Tensor
    observed_event_mask: Tensor
    encoder_observation_mask: Tensor
    training_loss_mask: Tensor
    interpolation_withheld_mask: Tensor
    extrapolation_future_mask: Tensor
    trajectory_id: Tensor

    def __getitem__(self, key: str) -> Any:
        if key not in _BATCH_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_BATCH_KEYS)

    def __len__(self) -> int:
        return len(_BATCH_KEYS)

    def to(
        self,
        device: Union[str, torch.device],
        non_blocking: bool = False,
    ) -> "PowerGridBatch":
        return PowerGridBatch(
            encoder_graph=self.encoder_graph.to(device),
            physical_graph=self.physical_graph.to(device),
            target_values=self.target_values.to(
                device, non_blocking=non_blocking
            ),
            target_times=self.target_times.to(
                device, non_blocking=non_blocking
            ),
            target_mask=self.target_mask.to(
                device, non_blocking=non_blocking
            ),
            observed_event_mask=self.observed_event_mask.to(
                device, non_blocking=non_blocking
            ),
            encoder_observation_mask=self.encoder_observation_mask.to(
                device, non_blocking=non_blocking
            ),
            training_loss_mask=self.training_loss_mask.to(
                device, non_blocking=non_blocking
            ),
            interpolation_withheld_mask=(
                self.interpolation_withheld_mask.to(
                    device, non_blocking=non_blocking
                )
            ),
            extrapolation_future_mask=self.extrapolation_future_mask.to(
                device, non_blocking=non_blocking
            ),
            trajectory_id=self.trajectory_id.to(
                device, non_blocking=non_blocking
            ),
        )

    def pin_memory(self) -> "PowerGridBatch":
        # torch_geometric's pin_memory implementation varies by release, so
        # explicitly pin tensor attributes through Data.apply when available.
        if hasattr(self.encoder_graph, "pin_memory"):
            encoder_graph = self.encoder_graph.pin_memory()
        else:  # pragma: no cover
            encoder_graph = self.encoder_graph

        if hasattr(self.physical_graph, "pin_memory"):
            physical_graph = self.physical_graph.pin_memory()
        else:  # pragma: no cover
            physical_graph = self.physical_graph

        return PowerGridBatch(
            encoder_graph=encoder_graph,
            physical_graph=physical_graph,
            target_values=self.target_values.pin_memory(),
            target_times=self.target_times.pin_memory(),
            target_mask=self.target_mask.pin_memory(),
            observed_event_mask=self.observed_event_mask.pin_memory(),
            encoder_observation_mask=(
                self.encoder_observation_mask.pin_memory()
            ),
            training_loss_mask=self.training_loss_mask.pin_memory(),
            interpolation_withheld_mask=(
                self.interpolation_withheld_mask.pin_memory()
            ),
            extrapolation_future_mask=(
                self.extrapolation_future_mask.pin_memory()
            ),
            trajectory_id=self.trajectory_id.pin_memory(),
        )


class PowerGridDataLoaders(NamedTuple):
    """Train/validation/test loaders plus shared dataset metadata."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    normalization: NormalizationStats
    archive: SimBenchArchive


def _scalar_integer(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)

    if array.size != 1:
        raise ValueError(f"{name} must be a scalar; got shape {array.shape}")

    raw = array.reshape(-1)[0]

    if isinstance(raw, (np.integer, int)):
        return int(raw)

    if isinstance(raw, (np.floating, float)) and float(raw).is_integer():
        return int(raw)

    raise ValueError(f"{name} must contain an integer index; got {raw!r}")


def _decode_feature_names(values: np.ndarray) -> Tuple[str, ...]:
    values = np.asarray(values).reshape(-1)
    names: List[str] = []

    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))

    return tuple(names)


def _load_optional_metadata(npz_path: Path) -> Dict[str, Any]:
    json_path = npz_path.with_suffix(".json")
    if not json_path.exists():
        return {}

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read SimBench metadata file {json_path}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"SimBench metadata must be a JSON object: {json_path}"
        )

    return value


def resolve_simbench_npz(
    data_path: Union[str, Path],
    simbench_code: Optional[str] = None,
) -> Path:
    """
    Resolve either a direct NPZ path or ``<data_path>/<simbench_code>.npz``.
    """

    path = Path(data_path).expanduser()

    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Expected an .npz file, got: {path}")
        return path.resolve()

    if simbench_code is None:
        raise ValueError(
            "simbench_code is required when data_path is a directory"
        )

    candidate = path / f"{simbench_code}.npz"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"SimBench archive not found: {candidate}"
        )

    return candidate.resolve()


def load_simbench_npz(
    data_path: Union[str, Path],
    simbench_code: Optional[str] = None,
) -> SimBenchArchive:
    """
    Load and validate a complete SimBench trajectory.

    Object arrays are intentionally rejected. The dataset generator should save
    ``bus_feature_names`` as a fixed-width Unicode array, not an object array.
    """

    npz_path = resolve_simbench_npz(data_path, simbench_code)

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = _REQUIRED_NPZ_FIELDS.difference(archive.files)
            if missing:
                raise ValueError(
                    f"{npz_path} is missing required fields: "
                    f"{sorted(missing)}"
                )

            bus_state_np = np.asarray(
                archive["bus_state"], dtype=np.float32
            )
            timestamps_np = np.asarray(
                archive["timestamps_hours"], dtype=np.float64
            ).reshape(-1)
            bus_indices_np = np.asarray(
                archive["bus_indices"]
            ).reshape(-1)
            edge_index_np = np.asarray(
                archive["edge_index"], dtype=np.int64
            )
            edge_type_np = np.asarray(
                archive["edge_type"], dtype=np.int64
            ).reshape(-1)
            train_end = _scalar_integer(archive["train_end"], "train_end")
            validation_end = _scalar_integer(
                archive["validation_end"], "validation_end"
            )
            feature_names = _decode_feature_names(
                archive["bus_feature_names"]
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not load SimBench archive {npz_path}"
        ) from exc

    if bus_state_np.ndim != 3:
        raise ValueError(
            "bus_state must have three dimensions [T, N, F] or [N, T, F]; "
            f"got {bus_state_np.shape}"
        )

    num_times = timestamps_np.shape[0]
    num_buses = bus_indices_np.shape[0]

    if (
        bus_state_np.shape[0] == num_times
        and bus_state_np.shape[1] == num_buses
    ):
        pass
    elif (
        bus_state_np.shape[0] == num_buses
        and bus_state_np.shape[1] == num_times
    ):
        bus_state_np = np.transpose(bus_state_np, (1, 0, 2))
    else:
        raise ValueError(
            "Could not align bus_state with timestamps_hours and bus_indices. "
            f"bus_state={bus_state_np.shape}, timestamps={num_times}, "
            f"buses={num_buses}"
        )

    if num_times < 3:
        raise ValueError("The trajectory must contain at least three timesteps")

    if num_buses < 1:
        raise ValueError("The trajectory must contain at least one bus")

    num_features = bus_state_np.shape[2]
    if len(feature_names) != num_features:
        raise ValueError(
            "bus_feature_names length does not match bus_state feature "
            f"dimension: {len(feature_names)} != {num_features}"
        )

    if not np.isfinite(bus_state_np).all():
        bad = np.argwhere(~np.isfinite(bus_state_np))[0].tolist()
        raise ValueError(
            "bus_state must be a complete finite physical trajectory; "
            f"first invalid location is {bad}"
        )

    if not np.isfinite(timestamps_np).all():
        raise ValueError("timestamps_hours contains a non-finite value")

    time_deltas = np.diff(timestamps_np)
    if not np.all(time_deltas > 0.0):
        raise ValueError(
            "timestamps_hours must be strictly increasing"
        )

    if not (0 < train_end < validation_end < num_times):
        raise ValueError(
            "Split boundaries must satisfy "
            f"0 < train_end < validation_end < T; got "
            f"train_end={train_end}, validation_end={validation_end}, "
            f"T={num_times}"
        )

    if edge_index_np.ndim != 2:
        raise ValueError(
            f"edge_index must be rank 2; got {edge_index_np.shape}"
        )

    if edge_index_np.shape[0] == 2:
        pass
    elif edge_index_np.shape[1] == 2:
        edge_index_np = edge_index_np.T
    else:
        raise ValueError(
            "edge_index must have shape [2, E] or [E, 2]; "
            f"got {edge_index_np.shape}"
        )

    num_edges = edge_index_np.shape[1]

    if edge_type_np.size == 1 and num_edges != 1:
        edge_type_np = np.full(
            num_edges, int(edge_type_np[0]), dtype=np.int64
        )

    if edge_type_np.shape[0] != num_edges:
        raise ValueError(
            f"edge_type must have one entry per physical edge: "
            f"{edge_type_np.shape[0]} != {num_edges}"
        )

    if num_edges:
        minimum_node = int(edge_index_np.min())
        maximum_node = int(edge_index_np.max())
        if minimum_node < 0 or maximum_node >= num_buses:
            raise ValueError(
                "edge_index contains an invalid bus index: "
                f"min={minimum_node}, max={maximum_node}, "
                f"num_buses={num_buses}"
            )

        edge_pairs = [
            (int(edge_index_np[0, index]), int(edge_index_np[1, index]))
            for index in range(num_edges)
        ]
        if any(sender == receiver for sender, receiver in edge_pairs):
            raise ValueError("edge_index must not contain self-edges")
        if len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError("edge_index contains duplicate directed edges")
        pair_set = set(edge_pairs)
        missing_reverse = [
            (sender, receiver)
            for sender, receiver in edge_pairs
            if (receiver, sender) not in pair_set
        ]
        if missing_reverse:
            raise ValueError(
                "The power-grid graph must contain both directions for every "
                f"physical connection; missing reverse for {missing_reverse[0]}."
            )

    if np.unique(bus_indices_np).shape[0] != num_buses:
        raise ValueError("bus_indices must be unique")

    bus_indices_numeric: np.ndarray
    try:
        bus_indices_numeric = bus_indices_np.astype(np.int64)
    except (TypeError, ValueError):
        # Preserve deterministic local integer identifiers if the generator
        # saved non-numeric external bus labels.
        bus_indices_numeric = np.arange(num_buses, dtype=np.int64)

    return SimBenchArchive(
        path=npz_path,
        bus_state=torch.from_numpy(
            np.ascontiguousarray(bus_state_np)
        ).float(),
        timestamps_hours=torch.from_numpy(
            np.ascontiguousarray(timestamps_np)
        ).double(),
        bus_indices=torch.from_numpy(
            np.ascontiguousarray(bus_indices_numeric)
        ).long(),
        edge_index=torch.from_numpy(
            np.ascontiguousarray(edge_index_np)
        ).long(),
        edge_type=torch.from_numpy(
            np.ascontiguousarray(edge_type_np)
        ).long(),
        train_end=train_end,
        validation_end=validation_end,
        bus_feature_names=feature_names,
        metadata=_load_optional_metadata(npz_path),
    )


def fit_training_normalization(
    archive: SimBenchArchive,
    eps: float = 1e-8,
) -> NormalizationStats:
    """Fit per-feature mean and standard deviation on [0, train_end)."""

    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}")

    training = archive.bus_state[: archive.train_end].to(torch.float64)
    flat = training.reshape(-1, training.shape[-1])

    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False)

    # Constant features remain finite and map to zero.
    std = torch.where(std > eps, std, torch.ones_like(std))

    return NormalizationStats(
        mean=mean.float(),
        std=std.float(),
        count=int(flat.shape[0]),
        fitted_start=0,
        fitted_end=archive.train_end,
        eps=float(eps),
    )


def _split_bounds(
    archive: SimBenchArchive,
    split: SplitName,
) -> Tuple[int, int]:
    if split == "train":
        return 0, archive.train_end
    if split == "validation":
        return archive.train_end, archive.validation_end
    if split == "test":
        return archive.validation_end, archive.num_timesteps
    raise ValueError(f"Unknown split: {split!r}")


def _normalized_elapsed_time(
    timestamps: Tensor,
    origin: Tensor,
    scale: Tensor,
) -> Tensor:
    return ((timestamps - origin) / scale).to(torch.float32)


def _window_time_scale(timestamps: Tensor) -> Tensor:
    """
    Return a non-zero characteristic duration for a window.

    Normally this is the elapsed time from the first to last timestamp.
    """

    if timestamps.numel() < 2:
        return torch.ones((), dtype=timestamps.dtype)

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        raise ValueError("A window has non-increasing timestamps")
    return duration


def _exact_observation_count(
    num_candidates: int,
    observed_fraction: float,
) -> int:
    """
    Convert a retained fraction into a deterministic observation count.

    The official configurations use lengths for which 0.4, 0.6 and 0.8 are
    integral. For other lengths, nearest-integer rounding is used because an
    exact fractional count is mathematically impossible.
    """

    if num_candidates < 1:
        raise ValueError("Observation domain must contain at least one point")

    raw_count = observed_fraction * num_candidates
    count = int(math.floor(raw_count + 0.5))
    return min(num_candidates, max(1, count))


def _sample_observation_mask(
    num_nodes: int,
    num_candidates: int,
    observed_fraction: float,
    mask_seed: int,
    trajectory_id: int,
) -> Tensor:
    """
    Independently sample a fixed-size observation set for each bus.

    A separate SeedSequence is created per bus, making masks independent of
    DataLoader order, worker count, model choice, and global RNG state.
    """

    count = _exact_observation_count(
        num_candidates=num_candidates,
        observed_fraction=observed_fraction,
    )

    mask = np.zeros((num_nodes, num_candidates), dtype=np.bool_)

    for bus in range(num_nodes):
        seed_sequence = np.random.SeedSequence(
            [
                int(mask_seed) & 0xFFFFFFFF,
                int(trajectory_id) & 0xFFFFFFFF,
                int(bus) & 0xFFFFFFFF,
                0x4C474F44,  # ASCII-inspired LGOD namespace constant
            ]
        )
        rng = np.random.default_rng(seed_sequence)
        selected = rng.choice(
            num_candidates,
            size=count,
            replace=False,
        )
        mask[bus, selected] = True

    return torch.from_numpy(mask)


def _temporal_pairs(
    source_times: Tensor,
    destination_times: Tensor,
    max_gap: Optional[float],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build causal source-to-destination event pairs.

    An edge is retained when source_time <= destination_time. This matches the
    original LG-ODE interpolation temporal-edge direction: information flows
    from an earlier or equal event to a later event.
    """

    if source_times.numel() == 0 or destination_times.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=torch.float32)
        return empty, empty, empty_float

    delta = source_times[:, None] - destination_times[None, :]
    keep = delta <= 1e-12

    if max_gap is not None:
        keep &= delta.abs() <= float(max_gap) + 1e-12

    source_local, destination_local = torch.nonzero(
        keep, as_tuple=True
    )

    edge_delta = delta[source_local, destination_local].to(torch.float32)
    return source_local.long(), destination_local.long(), edge_delta


def _build_encoder_graph(
    observed_values: Tensor,
    observed_times: Tensor,
    observed_mask: Tensor,
    physical_edge_index: Tensor,
    max_temporal_gap: Optional[float],
) -> Data:
    """
    Construct one temporal observation graph.

    Parameters
    ----------
    observed_values:
        Normalized values [N, C, F]. Values at unobserved entries are never
        placed in the graph.
    observed_times:
        Relative observation-domain times [C].
    observed_mask:
        Boolean event mask [N, C].
    physical_edge_index:
        Physical directed graph [2, E].
    max_temporal_gap:
        Optional maximum absolute normalized time gap for temporal edges.
    """

    num_nodes, _, num_features = observed_values.shape

    event_values: List[Tensor] = []
    event_times: List[Tensor] = []
    event_buses: List[Tensor] = []
    event_indices_by_bus: List[Tensor] = []
    event_times_by_bus: List[Tensor] = []
    counts = torch.zeros(num_nodes, dtype=torch.long)

    offset = 0
    for bus in range(num_nodes):
        time_indices = torch.nonzero(
            observed_mask[bus], as_tuple=False
        ).flatten()

        # The mask generator guarantees this, but retain an explicit invariant.
        if time_indices.numel() == 0:
            raise RuntimeError(
                f"Bus {bus} has no observations in a trajectory"
            )

        values = observed_values[bus, time_indices]
        times = observed_times[time_indices]

        count = int(time_indices.numel())
        counts[bus] = count

        event_values.append(values)
        event_times.append(times)
        event_buses.append(
            torch.full((count,), bus, dtype=torch.long)
        )
        event_indices_by_bus.append(
            torch.arange(offset, offset + count, dtype=torch.long)
        )
        event_times_by_bus.append(times)
        offset += count

    x = torch.cat(event_values, dim=0).reshape(-1, num_features)
    pos = torch.cat(event_times, dim=0).to(torch.float32)
    event_bus = torch.cat(event_buses, dim=0)

    source_parts: List[Tensor] = []
    destination_parts: List[Tensor] = []
    edge_attr_parts: List[Tensor] = []
    edge_same_parts: List[Tensor] = []

    # Same-object temporal relationships are always present, independently of
    # the physical grid's self-loop representation.
    for bus in range(num_nodes):
        source_local, destination_local, delta = _temporal_pairs(
            event_times_by_bus[bus],
            event_times_by_bus[bus],
            max_temporal_gap,
        )

        if source_local.numel() == 0:
            continue

        source_parts.append(event_indices_by_bus[bus][source_local])
        destination_parts.append(
            event_indices_by_bus[bus][destination_local]
        )
        edge_attr_parts.append(delta)
        edge_same_parts.append(
            torch.ones(delta.numel(), dtype=torch.float32)
        )

    # Cross-object temporal relationships follow only existing physical edges.
    if physical_edge_index.numel():
        for edge_number in range(physical_edge_index.shape[1]):
            source_bus = int(physical_edge_index[0, edge_number])
            destination_bus = int(physical_edge_index[1, edge_number])

            # Same-bus edges were already added exactly once above.
            if source_bus == destination_bus:
                continue

            source_local, destination_local, delta = _temporal_pairs(
                event_times_by_bus[source_bus],
                event_times_by_bus[destination_bus],
                max_temporal_gap,
            )

            if source_local.numel() == 0:
                continue

            source_parts.append(
                event_indices_by_bus[source_bus][source_local]
            )
            destination_parts.append(
                event_indices_by_bus[destination_bus][destination_local]
            )
            edge_attr_parts.append(delta)
            edge_same_parts.append(
                torch.zeros(delta.numel(), dtype=torch.float32)
            )

    if source_parts:
        edge_index = torch.stack(
            [
                torch.cat(source_parts),
                torch.cat(destination_parts),
            ],
            dim=0,
        )
        edge_attr = torch.cat(edge_attr_parts).to(torch.float32)
        edge_same = torch.cat(edge_same_parts).to(torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float32)
        edge_same = torch.empty((0,), dtype=torch.float32)

    latest_observation_time = torch.stack(
        [times[-1] for times in event_times_by_bus]
    ).to(torch.float32)

    graph = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_same=edge_same,
        y=counts,
    )

    # Extra metadata used by IndependentLatentODE and transport diagnostics.
    graph.event_bus = event_bus
    graph.latest_observation_time = latest_observation_time
    graph.num_objects = torch.tensor([num_nodes], dtype=torch.long)

    return graph


def _build_physical_graph(
    archive: SimBenchArchive,
    trajectory_id: int,
) -> Data:
    """Construct one immutable physical-graph realization."""

    num_nodes = archive.num_nodes
    num_edges = archive.edge_index.shape[1]

    graph = Data(
        x=torch.ones((num_nodes, 1), dtype=torch.float32),
        edge_index=archive.edge_index.clone(),
        edge_type=archive.edge_type.clone(),
        edge_weight=torch.ones(num_edges, dtype=torch.float32),
        num_nodes=num_nodes,
    )
    graph.bus_indices = archive.bus_indices.clone()
    graph.trajectory_id = torch.tensor(
        [trajectory_id], dtype=torch.long
    )
    return graph


class SimBenchLGODEDataset(Dataset):
    """
    Chronological SimBench window dataset for interpolation or extrapolation.

    Parameters
    ----------
    archive:
        Loaded SimBench archive.
    split:
        ``train``, ``validation`` or ``test``.
    task:
        ``interpolation`` or ``extrapolation``.
    observed_fraction:
        One of 0.4, 0.6 or 0.8.
    normalization:
        Statistics fitted on training timesteps. If omitted, statistics are
        fitted from ``archive[:train_end]``.
    trajectory_length:
        Interpolation window length.
    context_length:
        Extrapolation context length.
    forecast_length:
        Extrapolation future target length.
    stride:
        Number of physical timesteps between window starts.
    mask_seed:
        Seed controlling asynchronous observations.
    max_temporal_gap:
        Optional maximum normalized time difference for temporal encoder
        edges. ``None`` retains all causal temporal edges.
    """

    def __init__(
        self,
        archive: SimBenchArchive,
        split: SplitName,
        task: TaskName,
        observed_fraction: float,
        *,
        normalization: Optional[NormalizationStats] = None,
        trajectory_length: Optional[int] = None,
        context_length: Optional[int] = None,
        forecast_length: Optional[int] = None,
        stride: int = 1,
        mask_seed: int = 0,
        max_temporal_gap: Optional[float] = None,
    ) -> None:
        super().__init__()

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split!r}")

        if task not in {"interpolation", "extrapolation"}:
            raise ValueError(f"Unknown task: {task!r}")

        if not any(
            math.isclose(
                observed_fraction,
                allowed,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for allowed in (0.4, 0.6, 0.8)
        ):
            raise ValueError(
                "observed_fraction must be one of 0.4, 0.6 or 0.8; "
                f"got {observed_fraction}"
            )

        if stride < 1:
            raise ValueError(f"stride must be positive; got {stride}")

        if max_temporal_gap is not None and max_temporal_gap < 0:
            raise ValueError(
                "max_temporal_gap must be non-negative or None"
            )

        if task == "interpolation":
            if trajectory_length is None or trajectory_length < 2:
                raise ValueError(
                    "Interpolation requires trajectory_length >= 2"
                )

            total_length = int(trajectory_length)
            effective_context_length = total_length
            effective_forecast_length = 0
        else:
            if context_length is None or context_length < 2:
                raise ValueError(
                    "Extrapolation requires context_length >= 2"
                )
            if forecast_length is None or forecast_length < 1:
                raise ValueError(
                    "Extrapolation requires forecast_length >= 1"
                )

            effective_context_length = int(context_length)
            effective_forecast_length = int(forecast_length)
            total_length = (
                effective_context_length + effective_forecast_length
            )

        self.archive = archive
        self.split = split
        self.task = task
        self.observed_fraction = float(observed_fraction)
        self.trajectory_length = (
            int(trajectory_length)
            if trajectory_length is not None
            else total_length
        )
        self.context_length = effective_context_length
        self.forecast_length = effective_forecast_length
        self.total_length = total_length
        self.stride = int(stride)
        self.mask_seed = int(mask_seed)
        self.max_temporal_gap = max_temporal_gap

        self.normalization = (
            normalization
            if normalization is not None
            else fit_training_normalization(archive)
        )

        if (
            self.normalization.fitted_start != 0
            or self.normalization.fitted_end != archive.train_end
        ):
            raise ValueError(
                "Normalization must be fitted only on training timesteps "
                f"[0, {archive.train_end}); got "
                f"[{self.normalization.fitted_start}, "
                f"{self.normalization.fitted_end})"
            )

        self.normalized_bus_state = self.normalization.normalize(
            archive.bus_state
        )

        split_start, split_stop = _split_bounds(archive, split)
        available = split_stop - split_start

        if total_length > available:
            raise ValueError(
                f"{split} split has {available} timesteps but the requested "
                f"window needs {total_length}"
            )

        records: List[WindowRecord] = []
        last_start = split_stop - total_length

        for start in range(split_start, last_start + 1, self.stride):
            stop = start + total_length

            if task == "interpolation":
                context_stop = stop
                target_start = start
                target_stop = stop
            else:
                context_stop = start + effective_context_length
                target_start = context_stop
                target_stop = stop

            # The global start index is a stable unique trajectory ID because
            # chronological splits do not overlap.
            trajectory_id = int(start)

            records.append(
                WindowRecord(
                    trajectory_id=trajectory_id,
                    split=split,
                    start=start,
                    stop=stop,
                    context_stop=context_stop,
                    target_start=target_start,
                    target_stop=target_stop,
                )
            )

        if not records:
            raise ValueError(
                f"No {split} windows could be constructed"
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

    @property
    def split_bounds(self) -> Tuple[int, int]:
        return _split_bounds(self.archive, self.split)

    def __len__(self) -> int:
        return len(self.windows)

    def get_window_record(self, index: int) -> WindowRecord:
        return self.windows[index]

    def observation_mask(self, index: int) -> Tensor:
        record = self.windows[index]
        return _sample_observation_mask(
            num_nodes=self.num_nodes,
            num_candidates=self.context_length,
            observed_fraction=self.observed_fraction,
            mask_seed=self.mask_seed,
            trajectory_id=record.trajectory_id,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.windows[index]

        context_values = self.normalized_bus_state[
            record.start : record.context_stop
        ].permute(1, 0, 2).contiguous()

        context_timestamps = self.archive.timestamps_hours[
            record.start : record.context_stop
        ]

        observed_event_mask = self.observation_mask(index)

        if self.task == "interpolation":
            origin = context_timestamps[0]
            scale = _window_time_scale(context_timestamps)

            encoder_times = _normalized_elapsed_time(
                context_timestamps,
                origin,
                scale,
            )

            target_values = context_values.clone()
            target_times = encoder_times.clone()
        else:
            # The latent state represents the context endpoint. Observed event
            # times are therefore non-positive, and future target times are
            # positive. This also exposes per-bus staleness directly.
            origin = context_timestamps[-1]
            scale = _window_time_scale(context_timestamps)

            encoder_times = _normalized_elapsed_time(
                context_timestamps,
                origin,
                scale,
            )

            target_values = self.normalized_bus_state[
                record.target_start : record.target_stop
            ].permute(1, 0, 2).contiguous()

            target_timestamps = self.archive.timestamps_hours[
                record.target_start : record.target_stop
            ]
            target_times = _normalized_elapsed_time(
                target_timestamps,
                origin,
                scale,
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
            target_values, dtype=torch.bool
        )
        if self.task == "interpolation":
            interpolation_withheld_mask = (
                ~observed_event_mask
            ).unsqueeze(-1).expand_as(target_values).clone()
            extrapolation_future_mask = torch.zeros_like(target_mask)
            training_loss_mask = interpolation_withheld_mask
        else:
            interpolation_withheld_mask = torch.zeros_like(target_mask)
            extrapolation_future_mask = target_mask.clone()
            training_loss_mask = extrapolation_future_mask

        # Metadata useful for protocol checks and diagnostics. Required fields
        # remain exactly the standard LG-ODE fields listed in the module docs.
        encoder_graph.trajectory_id = torch.tensor(
            [record.trajectory_id], dtype=torch.long
        )
        encoder_graph.window_start = torch.tensor(
            [record.start], dtype=torch.long
        )
        encoder_graph.window_stop = torch.tensor(
            [record.stop], dtype=torch.long
        )
        encoder_graph.observation_domain_length = torch.tensor(
            [self.context_length], dtype=torch.long
        )

        return {
            "encoder_graph": encoder_graph,
            "physical_graph": physical_graph,
            "target_values": target_values,
            "target_times": target_times,
            "target_mask": target_mask,
            "observed_event_mask": observed_event_mask,
            "encoder_observation_mask": observed_event_mask,
            "training_loss_mask": training_loss_mask,
            "interpolation_withheld_mask": interpolation_withheld_mask,
            "extrapolation_future_mask": extrapolation_future_mask,
            "trajectory_id": torch.tensor(
                record.trajectory_id, dtype=torch.long
            ),
        }


def collate_powergrid_lgode(
    samples: Sequence[Mapping[str, Any]],
) -> PowerGridBatch:
    """Collate dataset samples without imputing any missing observations."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    encoder_graph = Batch.from_data_list(
        [sample["encoder_graph"] for sample in samples]
    )
    physical_graph = Batch.from_data_list(
        [sample["physical_graph"] for sample in samples]
    )

    target_values = torch.stack(
        [sample["target_values"] for sample in samples],
        dim=0,
    )
    target_times = torch.stack(
        [sample["target_times"] for sample in samples],
        dim=0,
    )
    target_mask = torch.stack(
        [sample["target_mask"] for sample in samples],
        dim=0,
    )
    observed_event_mask = torch.stack(
        [sample["observed_event_mask"] for sample in samples],
        dim=0,
    )
    encoder_observation_mask = torch.stack(
        [sample["encoder_observation_mask"] for sample in samples],
        dim=0,
    )
    training_loss_mask = torch.stack(
        [sample["training_loss_mask"] for sample in samples],
        dim=0,
    )
    interpolation_withheld_mask = torch.stack(
        [sample["interpolation_withheld_mask"] for sample in samples],
        dim=0,
    )
    extrapolation_future_mask = torch.stack(
        [sample["extrapolation_future_mask"] for sample in samples],
        dim=0,
    )
    trajectory_id = torch.stack(
        [sample["trajectory_id"] for sample in samples],
        dim=0,
    ).long()

    return PowerGridBatch(
        encoder_graph=encoder_graph,
        physical_graph=physical_graph,
        target_values=target_values,
        target_times=target_times,
        target_mask=target_mask,
        observed_event_mask=observed_event_mask,
        encoder_observation_mask=encoder_observation_mask,
        training_loss_mask=training_loss_mask,
        interpolation_withheld_mask=interpolation_withheld_mask,
        extrapolation_future_mask=extrapolation_future_mask,
        trajectory_id=trajectory_id,
    )


def _seed_worker(worker_id: int) -> None:
    """
    Seed worker-local NumPy state for unrelated augmentations.

    Observation masks do not depend on this state; they use deterministic
    per-trajectory SeedSequence instances.
    """

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def build_simbench_datasets(
    data_path: Union[str, Path],
    simbench_code: Optional[str],
    *,
    task: TaskName,
    observed_fraction: float,
    trajectory_length: Optional[int] = None,
    context_length: Optional[int] = None,
    forecast_length: Optional[int] = None,
    stride: int = 1,
    mask_seed: int = 0,
    max_temporal_gap: Optional[float] = None,
    normalization_eps: float = 1e-8,
) -> Dict[SplitName, SimBenchLGODEDataset]:
    """
    Build all chronological datasets with one shared training-only normalizer.
    """

    archive = load_simbench_npz(data_path, simbench_code)
    normalization = fit_training_normalization(
        archive,
        eps=normalization_eps,
    )

    common = {
        "archive": archive,
        "task": task,
        "observed_fraction": observed_fraction,
        "normalization": normalization,
        "trajectory_length": trajectory_length,
        "context_length": context_length,
        "forecast_length": forecast_length,
        "stride": stride,
        "mask_seed": mask_seed,
        "max_temporal_gap": max_temporal_gap,
    }

    return {
        "train": SimBenchLGODEDataset(split="train", **common),
        "validation": SimBenchLGODEDataset(
            split="validation", **common
        ),
        "test": SimBenchLGODEDataset(split="test", **common),
    }


def build_simbench_dataloaders(
    data_path: Union[str, Path],
    simbench_code: Optional[str],
    *,
    task: TaskName,
    observed_fraction: float,
    batch_size: int,
    trajectory_length: Optional[int] = None,
    context_length: Optional[int] = None,
    forecast_length: Optional[int] = None,
    stride: int = 1,
    seed: int = 0,
    mask_seed: Optional[int] = None,
    max_temporal_gap: Optional[float] = None,
    normalization_eps: float = 1e-8,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
) -> PowerGridDataLoaders:
    """
    Build deterministic train/validation/test DataLoaders.

    Training-window order uses ``seed``. Sparse observation masks use
    ``mask_seed`` and are therefore unchanged by model choice or shuffle order.
    """

    if batch_size < 1:
        raise ValueError(
            f"batch_size must be positive; got {batch_size}"
        )
    if num_workers < 0:
        raise ValueError(
            f"num_workers must be non-negative; got {num_workers}"
        )

    effective_mask_seed = seed if mask_seed is None else mask_seed

    datasets = build_simbench_datasets(
        data_path=data_path,
        simbench_code=simbench_code,
        task=task,
        observed_fraction=observed_fraction,
        trajectory_length=trajectory_length,
        context_length=context_length,
        forecast_length=forecast_length,
        stride=stride,
        mask_seed=effective_mask_seed,
        max_temporal_gap=max_temporal_gap,
        normalization_eps=normalization_eps,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    common_loader_arguments = {
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
        **common_loader_arguments,
    )

    validation_loader = DataLoader(
        datasets["validation"],
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    test_loader = DataLoader(
        datasets["test"],
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    return PowerGridDataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        normalization=datasets["train"].normalization,
        archive=datasets["train"].archive,
    )


# More explicit alias for callers that use the project-specific naming.
build_powergrid_lgode_dataloaders = build_simbench_dataloaders


class SimBenchLGODEDataModule:
    """
    Small framework-independent data-module wrapper.

    It is intentionally not tied to PyTorch Lightning. The runner can create
    it once and pass its loaders unchanged to all four models.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        simbench_code: Optional[str],
        *,
        task: TaskName,
        observed_fraction: float,
        batch_size: int,
        trajectory_length: Optional[int] = None,
        context_length: Optional[int] = None,
        forecast_length: Optional[int] = None,
        stride: int = 1,
        seed: int = 0,
        mask_seed: Optional[int] = None,
        max_temporal_gap: Optional[float] = None,
        normalization_eps: float = 1e-8,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last_train: bool = False,
    ) -> None:
        self.data_path = data_path
        self.simbench_code = simbench_code
        self.task = task
        self.observed_fraction = observed_fraction
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.stride = stride
        self.seed = seed
        self.mask_seed = seed if mask_seed is None else mask_seed
        self.max_temporal_gap = max_temporal_gap
        self.normalization_eps = normalization_eps
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train

        self.archive = load_simbench_npz(
            data_path,
            simbench_code,
        )
        self.normalization = fit_training_normalization(
            self.archive,
            eps=normalization_eps,
        )

        common = {
            "archive": self.archive,
            "task": task,
            "observed_fraction": observed_fraction,
            "normalization": self.normalization,
            "trajectory_length": trajectory_length,
            "context_length": context_length,
            "forecast_length": forecast_length,
            "stride": stride,
            "mask_seed": self.mask_seed,
            "max_temporal_gap": max_temporal_gap,
        }

        self.train_dataset = SimBenchLGODEDataset(
            split="train", **common
        )
        self.validation_dataset = SimBenchLGODEDataset(
            split="validation", **common
        )
        self.test_dataset = SimBenchLGODEDataset(
            split="test", **common
        )

    @property
    def input_dim(self) -> int:
        return self.archive.input_dim

    @property
    def num_nodes(self) -> int:
        return self.archive.num_nodes

    @property
    def edge_index(self) -> Tensor:
        return self.archive.edge_index

    @property
    def edge_type(self) -> Tensor:
        return self.archive.edge_type

    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(int(self.seed))

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_powergrid_lgode,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    def validation_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_powergrid_lgode,
            worker_init_fn=_seed_worker,
        )

    def val_dataloader(self) -> DataLoader:
        return self.validation_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_powergrid_lgode,
            worker_init_fn=_seed_worker,
        )

    def dataloaders(self) -> PowerGridDataLoaders:
        return PowerGridDataLoaders(
            train=self.train_dataloader(),
            validation=self.validation_dataloader(),
            test=self.test_dataloader(),
            normalization=self.normalization,
            archive=self.archive,
        )


__all__ = [
    "NormalizationStats",
    "PowerGridBatch",
    "PowerGridDataLoaders",
    "SimBenchArchive",
    "SimBenchLGODEDataModule",
    "SimBenchLGODEDataset",
    "WindowRecord",
    "build_powergrid_lgode_dataloaders",
    "build_simbench_dataloaders",
    "build_simbench_datasets",
    "collate_powergrid_lgode",
    "fit_training_normalization",
    "load_simbench_npz",
    "resolve_simbench_npz",
]
