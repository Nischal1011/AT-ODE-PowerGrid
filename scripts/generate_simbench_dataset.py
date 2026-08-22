#!/usr/bin/env python3
"""
Generate reproducible SimBench AC power-flow trajectories for LG-ODE tasks.

This script generates complete physical trajectories only. It does not create
observation masks, missing values, interpolation splits, or extrapolation
contexts. Those belong in the downstream LG-ODE task loader.

One compressed NPZ file and one JSON metadata file are generated for each
requested SimBench grid.

NPZ fields
----------
bus_state
    Float32 array with shape [time, buses, bus_features].

timestamps_hours
    Float64 array with shape [time]. Timestamps correspond to the original
    SimBench profile positions and therefore preserve gaps if failed
    power-flow steps are skipped.

profile_steps
    Int64 array containing the original SimBench profile indices.

bus_indices
    Int64 array containing the original pandapower bus indices. The position
    of an entry in this array is the node index used by edge_index.

bus_vn_kv
    Float32 nominal voltage level for every retained bus.

edge_index
    Int64 directed physical topology with shape [2, directed_edges].
    Node indices are zero-based positions into bus_indices, not raw
    pandapower bus labels.

edge_type
    Int64 edge type for every directed edge:
        0: line
        1: transformer
        2: closed bus-bus switch

train_end
    Scalar index marking the exclusive end of the chronological training
    interval.

validation_end
    Scalar index marking the exclusive end of the validation interval.

bus_feature_names
    Names of features in bus_state.

Optional NPZ fields
-------------------
line_state
    Float32 array with shape [time, active_lines, line_features].

line_indices
    Original pandapower line indices corresponding to line_state.

line_feature_names
    Names of line-state features.

Design principles
-----------------
- SimBench annual profiles are converted to absolute power values.
- An AC power flow is run for every selected profile timestep.
- Normalization is not performed here.
- Missingness is not simulated here.
- No values are forward-filled.
- Train/validation/test boundaries are chronological.
- Dataset provenance and a SHA-256 checksum are stored in JSON metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandapower as pp
import simbench as sb


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

FORMAT_VERSION = 2

BUS_FEATURE_NAMES: Tuple[str, ...] = (
    "vm_pu",
    "va_degree",
    "p_mw",
    "q_mvar",
)

LINE_FEATURE_NAMES: Tuple[str, ...] = (
    "loading_percent",
    "i_from_ka",
    "i_to_ka",
    "p_from_mw",
    "q_from_mvar",
    "pl_mw",
)

EDGE_TYPE_NAMES: Tuple[str, ...] = (
    "line",
    "transformer",
    "closed_bus_switch",
)

EDGE_TYPE_LINE = 0
EDGE_TYPE_TRANSFORMER = 1
EDGE_TYPE_BUS_SWITCH = 2


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate complete SimBench AC power-flow trajectories for "
            "LG-ODE interpolation and extrapolation experiments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--codes",
        nargs="+",
        default=["1-MV-rural--0-sw"],
        help="One or more SimBench grid codes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simbench"),
        help="Directory for NPZ datasets and JSON metadata.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First annual-profile timestep to process.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help=(
            "Number of selected profile timesteps. Zero means use all "
            "available timesteps after --start."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride over SimBench profile timesteps.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=15.0,
        help="Time represented by one unstrided SimBench profile step.",
    )
    parser.add_argument(
        "--split-fractions",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.70, 0.15, 0.15),
        help="Chronological train/validation/test fractions.",
    )
    parser.add_argument(
        "--on-failure",
        choices=("error", "skip"),
        default="error",
        help=(
            "Whether to stop at the first failed power flow or omit failed "
            "timesteps. Skipped steps remain visible as timestamp gaps."
        ),
    )
    parser.add_argument(
        "--no-line-results",
        action="store_true",
        help="Do not save dynamic line-result tensors.",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable pandapower's optional numba acceleration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing NPZ and JSON files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run an eight-timestep smoke test using only the first requested "
            "grid."
        ),
    )

    args = parser.parse_args()

    if not args.codes:
        parser.error("--codes must contain at least one SimBench code.")

    if args.start < 0:
        parser.error("--start must be nonnegative.")

    if args.steps < 0:
        parser.error("--steps must be nonnegative.")

    if args.stride < 1:
        parser.error("--stride must be at least one.")

    if args.interval_minutes <= 0.0:
        parser.error("--interval-minutes must be positive.")

    if len(args.split_fractions) != 3:
        parser.error("--split-fractions requires exactly three values.")

    if any(value <= 0.0 for value in args.split_fractions):
        parser.error("All split fractions must be positive.")

    if not math.isclose(
        sum(args.split_fractions),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        parser.error("--split-fractions must sum to one.")

    if args.smoke:
        args.codes = args.codes[:1]
        args.steps = 8

    return args


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def sanitize_code(code: str) -> str:
    """Convert a SimBench code into a safe filename."""

    return "".join(
        character
        if character.isalnum() or character in "-_."
        else "_"
        for character in code
    )


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def package_version(module: object) -> str:
    """Return a package version without assuming it is defined."""

    return str(getattr(module, "__version__", "unknown"))


def parse_profile_key(key: object) -> Optional[Tuple[str, str]]:
    """
    Parse a SimBench absolute-profile key.

    SimBench versions may use either:

        ("load", "p_mw")

    or:

        "load.p_mw"
    """

    if isinstance(key, tuple) and len(key) == 2:
        return str(key[0]), str(key[1])

    if isinstance(key, str) and "." in key:
        element, variable = key.split(".", maxsplit=1)
        return element, variable

    return None


def available_profile_steps(
    absolute_values: Mapping[object, object],
) -> int:
    """
    Return the largest common profile horizon.

    Empty optional profile tables are ignored.
    """

    lengths: List[int] = []

    for key, frame in absolute_values.items():
        parsed = parse_profile_key(key)

        if parsed is None or not hasattr(frame, "__len__"):
            continue

        length = len(frame)

        if length > 0:
            lengths.append(int(length))

    if not lengths:
        raise RuntimeError(
            "SimBench returned no nonempty absolute profile tables."
        )

    return min(lengths)


def select_profile_steps(
    available: int,
    start: int,
    count: int,
    stride: int,
) -> np.ndarray:
    """Construct the requested profile-step indices."""

    if available < 1:
        raise ValueError("The available profile length must be positive.")

    if start >= available:
        raise ValueError(
            f"--start={start} is outside the available profile range "
            f"[0, {available})."
        )

    if count == 0:
        stop = available
    else:
        stop = min(
            available,
            start + count * stride,
        )

    selected = np.arange(
        start,
        stop,
        stride,
        dtype=np.int64,
    )

    if selected.size == 0:
        raise RuntimeError("No profile timesteps were selected.")

    return selected


def chronological_boundaries(
    count: int,
    fractions: Sequence[float],
) -> Tuple[int, int]:
    """Calculate nonempty chronological split boundaries."""

    if count < 3:
        raise ValueError(
            "At least three successful timesteps are required for "
            "nonempty train, validation, and test splits."
        )

    train_end = int(math.floor(count * float(fractions[0])))
    validation_size = int(math.floor(count * float(fractions[1])))
    validation_end = train_end + validation_size

    train_end = max(train_end, 1)
    validation_end = max(validation_end, train_end + 1)
    validation_end = min(validation_end, count - 1)

    if train_end >= validation_end:
        raise RuntimeError("Training or validation split is empty.")

    if validation_end >= count:
        raise RuntimeError("Test split is empty.")

    return train_end, validation_end


# ---------------------------------------------------------------------------
# SimBench profile application
# ---------------------------------------------------------------------------

def apply_profile_row(
    net,
    absolute_values: Mapping[object, object],
    step: int,
) -> None:
    """
    Apply one row from every available absolute SimBench profile.

    Values are assigned by pandapower element index, not by column order.
    """

    for raw_key, frame in absolute_values.items():
        parsed = parse_profile_key(raw_key)

        if parsed is None:
            continue

        element, variable = parsed

        if element not in net:
            continue

        table = net[element]

        if variable not in table.columns:
            continue

        if not hasattr(frame, "iloc"):
            continue

        frame_length = len(frame)

        # Some optional element classes have empty profile tables.
        if frame_length == 0:
            continue

        if step >= frame_length:
            raise IndexError(
                f"Profile step {step} exceeds {raw_key!r}, which contains "
                f"{frame_length} rows."
            )

        row = frame.iloc[step]

        if not hasattr(row, "index"):
            raise TypeError(
                f"Profile {raw_key!r} did not produce an indexed row."
            )

        direct_indices = table.index.intersection(row.index)

        if len(direct_indices) > 0:
            values = row.loc[direct_indices].to_numpy(
                dtype=np.float64,
            )

            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"Profile {raw_key!r} contains nonfinite values at "
                    f"step {step}."
                )

            table.loc[direct_indices, variable] = values

        # Handle index-type mismatch, for example integer table indices
        # against string profile columns.
        assigned = set(direct_indices.tolist())

        profile_column_by_string = {
            str(column): column
            for column in row.index
        }

        table_indices: List[object] = []
        profile_columns: List[object] = []

        for table_index in table.index:
            if table_index in assigned:
                continue

            profile_column = profile_column_by_string.get(
                str(table_index)
            )

            if profile_column is not None:
                table_indices.append(table_index)
                profile_columns.append(profile_column)

        if table_indices:
            values = row.loc[profile_columns].to_numpy(
                dtype=np.float64,
            )

            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"Profile {raw_key!r} contains nonfinite values at "
                    f"step {step}."
                )

            table.loc[table_indices, variable] = values


# ---------------------------------------------------------------------------
# Physical topology
# ---------------------------------------------------------------------------

def active_bus_indices(net) -> List[int]:
    """Return in-service buses in deterministic table order."""

    buses: List[int] = []

    for index, row in net.bus.iterrows():
        if bool(row.get("in_service", True)):
            buses.append(int(index))

    if not buses:
        raise RuntimeError("The SimBench network contains no active buses.")

    return buses


def open_element_indices(net, element_type: str) -> Set[int]:
    """
    Return line or transformer indices opened by switches.

    Pandapower switch element types:
        l: line
        t: transformer
        b: bus-bus switch
    """

    if "switch" not in net or len(net.switch) == 0:
        return set()

    switches = net.switch

    if "et" not in switches.columns:
        return set()

    if "closed" not in switches.columns:
        return set()

    selected = switches[
        (switches["et"] == element_type)
        & (~switches["closed"].astype(bool))
    ]

    return {
        int(value)
        for value in selected["element"].tolist()
    }


def active_line_indices(net) -> List[int]:
    """
    Return in-service lines not disconnected by an open line switch.
    """

    if "line" not in net:
        return []

    open_lines = open_element_indices(net, "l")
    indices: List[int] = []

    for index, row in net.line.iterrows():
        index = int(index)

        if index in open_lines:
            continue

        if not bool(row.get("in_service", True)):
            continue

        indices.append(index)

    return indices


def build_directed_topology(
    net,
    bus_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the directed active physical topology.

    The returned edge_index uses zero-based node positions corresponding to
    bus_indices. Raw pandapower bus labels are never used directly as neural
    network node indices.
    """

    bus_to_position = {
        int(bus_index): position
        for position, bus_index in enumerate(bus_indices)
    }

    active_bus_set = set(bus_to_position)
    open_lines = open_element_indices(net, "l")
    open_transformers = open_element_indices(net, "t")

    # Each tuple contains raw pandapower bus labels and an edge type.
    undirected_edges: List[Tuple[int, int, int]] = []

    if "line" in net:
        for index, row in net.line.iterrows():
            index = int(index)

            if index in open_lines:
                continue

            if not bool(row.get("in_service", True)):
                continue

            source = int(row["from_bus"])
            target = int(row["to_bus"])

            if source in active_bus_set and target in active_bus_set:
                undirected_edges.append(
                    (source, target, EDGE_TYPE_LINE)
                )

    if "trafo" in net:
        for index, row in net.trafo.iterrows():
            index = int(index)

            if index in open_transformers:
                continue

            if not bool(row.get("in_service", True)):
                continue

            source = int(row["hv_bus"])
            target = int(row["lv_bus"])

            if source in active_bus_set and target in active_bus_set:
                undirected_edges.append(
                    (source, target, EDGE_TYPE_TRANSFORMER)
                )

    if "switch" in net and len(net.switch) > 0:
        for _, row in net.switch.iterrows():
            if str(row.get("et")) != "b":
                continue

            if not bool(row.get("closed", True)):
                continue

            source = int(row["bus"])
            target = int(row["element"])

            if source in active_bus_set and target in active_bus_set:
                undirected_edges.append(
                    (source, target, EDGE_TYPE_BUS_SWITCH)
                )

    # Remove duplicate entries of the same edge type while retaining
    # distinct physical connection types.
    unique_edges = sorted(
        {
            (
                min(source, target),
                max(source, target),
                edge_type,
            )
            for source, target, edge_type in undirected_edges
            if source != target
        }
    )

    directed_edges: List[Tuple[int, int]] = []
    directed_types: List[int] = []

    for raw_source, raw_target, edge_type in unique_edges:
        source = bus_to_position[raw_source]
        target = bus_to_position[raw_target]

        directed_edges.append((source, target))
        directed_types.append(edge_type)

        directed_edges.append((target, source))
        directed_types.append(edge_type)

    if not directed_edges:
        raise RuntimeError(
            "No active physical edges were found in the SimBench grid."
        )

    edge_index = np.asarray(
        directed_edges,
        dtype=np.int64,
    ).T

    edge_type = np.asarray(
        directed_types,
        dtype=np.int64,
    )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError(
            f"Invalid edge_index shape: {edge_index.shape}."
        )

    if edge_type.shape != (edge_index.shape[1],):
        raise RuntimeError(
            "edge_type length does not match the directed edge count."
        )

    if edge_index.min() < 0:
        raise RuntimeError("edge_index contains a negative node index.")

    if edge_index.max() >= len(bus_indices):
        raise RuntimeError(
            "edge_index contains a node index outside bus_indices."
        )

    return edge_index, edge_type


# ---------------------------------------------------------------------------
# Power-flow result extraction
# ---------------------------------------------------------------------------

def extract_bus_state(
    net,
    bus_indices: Sequence[int],
) -> np.ndarray:
    """Extract bus features in the fixed bus_indices order."""

    frame = net.res_bus.reindex(bus_indices)

    missing_columns = [
        name
        for name in BUS_FEATURE_NAMES
        if name not in frame.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "pandapower res_bus is missing required columns: "
            f"{missing_columns}."
        )

    state = frame.loc[
        :,
        list(BUS_FEATURE_NAMES),
    ].to_numpy(dtype=np.float32)

    expected_shape = (
        len(bus_indices),
        len(BUS_FEATURE_NAMES),
    )

    if state.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected bus-state shape {state.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(state).all():
        raise FloatingPointError(
            "Nonfinite values appeared in the bus power-flow results."
        )

    return state


def extract_line_state(
    net,
    line_indices: Sequence[int],
) -> np.ndarray:
    """Extract active-line result features in a fixed order."""

    if not line_indices:
        return np.empty(
            (0, len(LINE_FEATURE_NAMES)),
            dtype=np.float32,
        )

    frame = net.res_line.reindex(line_indices)

    missing_columns = [
        name
        for name in LINE_FEATURE_NAMES
        if name not in frame.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "pandapower res_line is missing required columns: "
            f"{missing_columns}."
        )

    state = frame.loc[
        :,
        list(LINE_FEATURE_NAMES),
    ].to_numpy(dtype=np.float32)

    expected_shape = (
        len(line_indices),
        len(LINE_FEATURE_NAMES),
    )

    if state.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected line-state shape {state.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(state).all():
        raise FloatingPointError(
            "Nonfinite values appeared in the line power-flow results."
        )

    return state


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_one(
    code: str,
    output_dir: Path,
    start: int,
    count: int,
    stride: int,
    interval_minutes: float,
    split_fractions: Sequence[float],
    on_failure: str,
    save_line_results: bool,
    use_numba: bool,
    overwrite: bool,
) -> Tuple[Path, Path]:
    """Generate one complete SimBench trajectory."""

    safe_code = sanitize_code(code)

    npz_path = output_dir / f"{safe_code}.npz"
    metadata_path = output_dir / f"{safe_code}.json"

    existing = [
        path
        for path in (npz_path, metadata_path)
        if path.exists()
    ]

    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)

        raise FileExistsError(
            f"Output already exists: {formatted}. "
            "Pass --overwrite to replace it."
        )

    print()
    print(f"Loading SimBench network: {code}", flush=True)

    net = sb.get_simbench_net(code)

    absolute_values = sb.get_absolute_values(
        net,
        profiles_instead_of_study_cases=True,
    )

    available_steps = available_profile_steps(
        absolute_values
    )

    requested_steps = select_profile_steps(
        available=available_steps,
        start=start,
        count=count,
        stride=stride,
    )

    bus_indices = active_bus_indices(net)
    line_indices = active_line_indices(net)

    edge_index, edge_type = build_directed_topology(
        net,
        bus_indices,
    )

    bus_states: List[np.ndarray] = []
    line_states: List[np.ndarray] = []
    successful_steps: List[int] = []
    failed_steps: List[Dict[str, object]] = []

    previous_step_converged = False
    started = time.monotonic()

    for position, profile_step_value in enumerate(requested_steps):
        profile_step = int(profile_step_value)

        apply_profile_row(
            net=net,
            absolute_values=absolute_values,
            step=profile_step,
        )

        try:
            pp.runpp(
                net,
                init=(
                    "results"
                    if previous_step_converged
                    else "auto"
                ),
                calculate_voltage_angles=True,
                check_connectivity=True,
                numba=use_numba,
            )

            if not bool(net.converged):
                raise RuntimeError(
                    "pandapower reported that the power flow did not "
                    "converge."
                )

            bus_state = extract_bus_state(
                net,
                bus_indices,
            )

            if save_line_results:
                line_state = extract_line_state(
                    net,
                    line_indices,
                )
            else:
                line_state = None

            bus_states.append(bus_state)

            if line_state is not None:
                line_states.append(line_state)

            successful_steps.append(profile_step)
            previous_step_converged = True

        except Exception as error:
            previous_step_converged = False

            failure = {
                "profile_step": profile_step,
                "error_type": type(error).__name__,
                "message": str(error),
            }

            failed_steps.append(failure)

            message = (
                f"Power flow failed for grid {code!r} at profile "
                f"step {profile_step}: {type(error).__name__}: {error}"
            )

            if on_failure == "error":
                raise RuntimeError(message) from error

            print(
                f"WARNING: {message}",
                file=sys.stderr,
                flush=True,
            )

        completed = position + 1

        if (
            completed == 1
            or completed % 100 == 0
            or completed == len(requested_steps)
        ):
            elapsed = time.monotonic() - started
            rate = completed / max(elapsed, 1.0e-12)

            print(
                f"{code}: requested={completed}/"
                f"{len(requested_steps)}, "
                f"converged={len(successful_steps)}, "
                f"failed={len(failed_steps)}, "
                f"rate={rate:.2f} steps/s",
                flush=True,
            )

    if len(bus_states) < 3:
        raise RuntimeError(
            f"Only {len(bus_states)} successful power-flow states were "
            "generated. At least three are required."
        )

    bus_state_array = np.stack(
        bus_states,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if save_line_results:
        if len(line_states) != len(bus_states):
            raise RuntimeError(
                "Line-state and bus-state trajectory lengths differ."
            )

        line_state_array = np.stack(
            line_states,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )
    else:
        line_state_array = np.empty(
            (len(bus_states), 0, 0),
            dtype=np.float32,
        )

    profile_steps_array = np.asarray(
        successful_steps,
        dtype=np.int64,
    )

    timestamps_hours = (
        profile_steps_array.astype(np.float64)
        * float(interval_minutes)
        / 60.0
    )

    if np.any(np.diff(timestamps_hours) <= 0.0):
        raise RuntimeError(
            "Generated timestamps are not strictly increasing."
        )

    train_end, validation_end = chronological_boundaries(
        count=len(successful_steps),
        fractions=split_fractions,
    )

    bus_vn_kv = (
        net.bus
        .reindex(bus_indices)["vn_kv"]
        .to_numpy(dtype=np.float32)
    )

    if not np.isfinite(bus_vn_kv).all():
        raise FloatingPointError(
            "Nonfinite nominal bus voltages were found."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        npz_path,
        bus_state=bus_state_array,
        line_state=line_state_array,
        timestamps_hours=timestamps_hours,
        profile_steps=profile_steps_array,
        bus_indices=np.asarray(
            bus_indices,
            dtype=np.int64,
        ),
        bus_vn_kv=bus_vn_kv,
        line_indices=np.asarray(
            line_indices,
            dtype=np.int64,
        ),
        edge_index=edge_index,
        edge_type=edge_type,
        train_end=np.asarray(
            train_end,
            dtype=np.int64,
        ),
        validation_end=np.asarray(
            validation_end,
            dtype=np.int64,
        ),
        bus_feature_names=np.asarray(
            BUS_FEATURE_NAMES,
            dtype=np.str_,
        ),
        line_feature_names=np.asarray(
            LINE_FEATURE_NAMES,
            dtype=np.str_,
        ),
        edge_type_names=np.asarray(
            EDGE_TYPE_NAMES,
            dtype=np.str_,
        ),
    )

    metadata: Dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "simbench_code": code,
        "simbench_version": package_version(sb),
        "pandapower_version": package_version(pp),
        "available_profile_steps": int(available_steps),
        "requested_profile_steps": int(len(requested_steps)),
        "successful_profile_steps": int(len(successful_steps)),
        "failed_profile_step_count": int(len(failed_steps)),
        "failed_profile_steps": failed_steps,
        "profile_start": int(start),
        "profile_stride": int(stride),
        "interval_minutes": float(interval_minutes),
        "timestamps_preserve_skipped_gaps": True,
        "bus_count": int(len(bus_indices)),
        "line_count": int(len(line_indices)),
        "directed_edge_count": int(edge_index.shape[1]),
        "bus_features": list(BUS_FEATURE_NAMES),
        "line_features": (
            list(LINE_FEATURE_NAMES)
            if save_line_results
            else []
        ),
        "edge_types": {
            str(index): name
            for index, name in enumerate(EDGE_TYPE_NAMES)
        },
        "split_policy": "chronological",
        "train_range": [0, int(train_end)],
        "validation_range": [
            int(train_end),
            int(validation_end),
        ],
        "test_range": [
            int(validation_end),
            int(len(successful_steps)),
        ],
        "split_fractions_requested": [
            float(value)
            for value in split_fractions
        ],
        "normalization": (
            "Not applied during generation. Downstream loaders must fit "
            "normalization using the training interval only."
        ),
        "missingness": (
            "Not generated. Downstream task loaders must generate "
            "deterministic observation subsets."
        ),
        "npz_file": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "generated_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved NPZ:      {npz_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Bus state:      {bus_state_array.shape}")
    print(f"Line state:     {line_state_array.shape}")
    print(f"Edge index:     {edge_index.shape}")
    print(
        "Chronological splits: "
        f"train=[0,{train_end}), "
        f"validation=[{train_end},{validation_end}), "
        f"test=[{validation_end},{len(successful_steps)})"
    )

    return npz_path, metadata_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("SimBench dataset generation")
    print("===========================")
    print(f"Codes:       {', '.join(args.codes)}")
    print(f"Output:      {args.output_dir.resolve()}")
    print(f"Start:       {args.start}")
    print(f"Steps:       {args.steps or 'all'}")
    print(f"Stride:      {args.stride}")
    print(f"Interval:    {args.interval_minutes} minutes")
    print(f"Smoke test:  {args.smoke}")
    print(f"Line states: {not args.no_line_results}")
    print(f"Numba:       {not args.no_numba}")

    generated: List[Tuple[Path, Path]] = []

    for code in args.codes:
        generated.append(
            generate_one(
                code=code,
                output_dir=args.output_dir,
                start=args.start,
                count=args.steps,
                stride=args.stride,
                interval_minutes=args.interval_minutes,
                split_fractions=args.split_fractions,
                on_failure=args.on_failure,
                save_line_results=not args.no_line_results,
                use_numba=not args.no_numba,
                overwrite=args.overwrite,
            )
        )

    manifest = {
        "format_version": FORMAT_VERSION,
        "generated_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "codes": list(args.codes),
        "files": [
            {
                "npz": str(npz_path),
                "metadata": str(metadata_path),
                "npz_sha256": sha256_file(npz_path),
            }
            for npz_path, metadata_path in generated
        ],
    }

    manifest_path = args.output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Manifest: {manifest_path}")
    print("Generation complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
