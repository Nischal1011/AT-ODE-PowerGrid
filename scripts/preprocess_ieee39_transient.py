#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


SOURCE_SHA256 = "f190efdd64ea7652a1f8c84c3823b3f3477c80a83f32861dd45ffdf1e968a711"
GENERATORS = tuple(f"G{index:02d}" for index in range(1, 11))
FEATURES = (
    "active_power_mw",
    "terminal_voltage_pu",
    "excitation_current_pu",
    "rotor_speed_pu",
    "relative_rotor_angle_deg",
)
RAW_FEATURES = (
    "P in MW",
    "ut in p.u.",
    "ie in p.u.",
    "xspeed in p.u.",
    "firel in deg",
)
UNITS = ("MW", "p.u.", "p.u.", "p.u.", "deg")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_columns() -> Tuple[str, ...]:
    return tuple(
        f"{generator} {feature}"
        for feature in RAW_FEATURES
        for generator in GENERATORS
    )


def stratified_split(
    labels: np.ndarray,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    partitions = {"train": [], "validation": [], "test": []}
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        train_count = int(np.floor(0.70 * len(indices)))
        validation_count = int(np.floor(0.15 * len(indices)))
        partitions["train"].append(indices[:train_count])
        partitions["validation"].append(
            indices[train_count : train_count + validation_count]
        )
        partitions["test"].append(indices[train_count + validation_count :])
    result = []
    for split in ("train", "validation", "test"):
        values = np.concatenate(partitions[split]).astype(np.int64)
        generator.shuffle(values)
        result.append(values)
    return tuple(result)


def complete_directed_graph(num_nodes: int = 10) -> np.ndarray:
    senders = []
    receivers = []
    for receiver in range(num_nodes):
        for sender in range(num_nodes):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    return np.asarray([senders, receivers], dtype=np.int64)


def load_and_convert(raw_path: Path):
    with raw_path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("tsa_data.pkl must contain (trajectories, labels).")
    frames, labels_series = value
    if len(frames) != 12852 or len(labels_series) != len(frames):
        raise ValueError("Unexpected IEEE39 scenario count.")
    columns = expected_columns()
    timestamps = np.empty((len(frames), 60), dtype=np.float64)
    trajectories = np.empty((len(frames), 60, 10, 5), dtype=np.float32)
    for scenario_index, frame in enumerate(frames):
        if tuple(str(column) for column in frame.columns) != columns:
            raise ValueError(
                f"Scenario {scenario_index} has unexpected feature columns."
            )
        if frame.shape != (60, 50):
            raise ValueError(
                f"Scenario {scenario_index} has shape {frame.shape}, expected (60, 50)."
            )
        frame_times = np.asarray(frame.index, dtype=np.float64)
        if not np.allclose(
            np.diff(frame_times), 0.01, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"Scenario {scenario_index} does not use 0.01-second spacing."
            )
        timestamps[scenario_index] = frame_times
        raw = frame.to_numpy(dtype=np.float32, copy=False)
        trajectories[scenario_index] = raw.reshape(60, 5, 10).transpose(0, 2, 1)
    labels = np.asarray(labels_series, dtype=np.uint8)
    if not np.isfinite(trajectories).all():
        raise ValueError("Converted trajectories contain NaN or infinity.")
    if not np.array_equal(np.unique(labels), np.asarray([0, 1], dtype=np.uint8)):
        raise ValueError("Stability labels must be binary 0/1.")
    return trajectories, timestamps, labels


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
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


def atomic_save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def preprocess(raw_path: Path, output_path: Path, metadata_path: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    source_hash = sha256_file(raw_path)
    if source_hash != SOURCE_SHA256:
        raise ValueError(
            f"Raw SHA256 mismatch: {source_hash}; expected {SOURCE_SHA256}."
        )
    trajectories, timestamps, labels = load_and_convert(raw_path)
    train, validation, test = stratified_split(labels)
    training = trajectories[train].astype(np.float64)
    normalization_mean = training.mean(axis=(0, 1, 2))
    normalization_std = training.std(axis=(0, 1, 2))
    normalization_std = np.where(normalization_std > 1e-8, normalization_std, 1.0)
    edge_index = complete_directed_graph(10)
    edge_hash = hashlib.sha256(edge_index.tobytes()).hexdigest()
    atomic_save_npz(
        output_path,
        trajectories=trajectories,
        timestamps_seconds=timestamps,
        labels=labels,
        scenario_ids=np.arange(len(labels), dtype=np.int64),
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        generator_names=np.asarray(GENERATORS, dtype="U3"),
        feature_names=np.asarray(FEATURES, dtype="U32"),
        feature_units=np.asarray(UNITS, dtype="U8"),
        edge_index=edge_index,
        edge_type=np.ones(edge_index.shape[1], dtype=np.int64),
    )
    metadata = {
        "schema_version": 1,
        "dataset_doi": "10.17632/p992nhb8ss.1",
        "license": "CC BY 4.0",
        "source_path": str(raw_path),
        "source_sha256": source_hash,
        "processed_path": str(output_path),
        "processed_sha256": sha256_file(output_path),
        "scenario_count": int(len(labels)),
        "trajectory_shape": [60, 10, 5],
        "timestamps_seconds": [float(timestamps.min()), float(timestamps.max())],
        "timestamp_grid_count": int(
            np.unique(np.round(timestamps, 12), axis=0).shape[0]
        ),
        "timestamp_spacing_seconds": 0.01,
        "generator_order": list(GENERATORS),
        "feature_order": list(FEATURES),
        "feature_units": list(UNITS),
        "label_counts": {
            str(int(label)): int((labels == label).sum())
            for label in np.unique(labels)
        },
        "split_seed": 2026,
        "split_stratification": "stability_label_only",
        "split_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "graph_mode": "complete_directed_generator_candidate",
        "graph_node_order": list(GENERATORS),
        "graph_edge_count": int(edge_index.shape[1]),
        "graph_sha256": edge_hash,
        "generator_bus_mapping": None,
        "mapping_limitation": (
            "Dataset generator-to-bus mapping is unavailable; graph is a "
            "mapping-free complete directed candidate graph, not electrical topology."
        ),
        "scenario_metadata_available": ["stability_label"],
        "preprocessing_seconds": time.perf_counter() - started,
    }
    atomic_save_json(metadata_path, metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-path", type=Path,
        default=Path("data/ieee39_transient/raw/tsa_data.pkl"),
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=Path("data/ieee39_transient/processed/ieee39_transient_v1.npz"),
    )
    parser.add_argument(
        "--metadata-path", type=Path,
        default=Path("data/ieee39_transient/processed/ieee39_transient_v1_metadata.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (args.output_path.exists() or args.metadata_path.exists()) and not args.force:
        raise FileExistsError("Processed cache exists; pass --force to rebuild it.")
    metadata = preprocess(args.raw_path, args.output_path, args.metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())