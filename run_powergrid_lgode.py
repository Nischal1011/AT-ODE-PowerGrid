#!/usr/bin/env python3
"""
Train and evaluate power-grid forecasting models on SimBench trajectories.

Supported models
----------------
persistence
    Non-learned carry-forward baseline.

latentode
    Graph-free independent Latent ODE baseline.

lgode
    LG-ODE with fixed unit weights on physical edges.

atode
    LG-ODE with solver-safe attention-transport edge weights.

Supported tasks
---------------
interpolation
    Reconstruct the complete trajectory window from sparse asynchronous
    observations distributed throughout that window.

extrapolation
    Encode sparse asynchronous observations from a context interval and
    forecast a complete future interval.

Scientific protocol
-------------------
* All models use the same dataset realization and observation masks.
* Normalization is fitted on training timesteps only.
* Validation normalized MSE selects the best checkpoint.
* The test split is evaluated exactly once, after restoring that checkpoint.
* LG-ODE and AT-ODE use the same encoder, decoder, ODE network, solver,
  tolerances, dimensions, dropout, prior, and initialization seed.
* Their only intended generative-model difference is edge weighting.
* Evaluation targets are complete; missing observations are not imputed and
  fed to the encoder.
* Result JSON files are compatible with
  scripts/summarize_powergrid_lgode.py.

Examples
--------
Smoke tests:

python run_powergrid_lgode.py \
    --data-path data/simbench/1-MV-rural--0-sw.npz \
    --model persistence \
    --task extrapolation \
    --observed-fraction 0.4 \
    --context-length 12 \
    --forecast-length 12 \
    --niters 2 \
    --batch-size 4 \
    --seed 42 \
    --mask-seed 42

python run_powergrid_lgode.py \
    --data-path data/simbench/1-MV-rural--0-sw.npz \
    --model latentode \
    --task extrapolation \
    --observed-fraction 0.4 \
    --context-length 12 \
    --forecast-length 12 \
    --niters 2 \
    --batch-size 4 \
    --seed 42 \
    --mask-seed 42

python run_powergrid_lgode.py \
    --data-path data/simbench/1-MV-rural--0-sw.npz \
    --model lgode \
    --task extrapolation \
    --observed-fraction 0.4 \
    --context-length 12 \
    --forecast-length 12 \
    --niters 2 \
    --batch-size 4 \
    --seed 42 \
    --mask-seed 42

python run_powergrid_lgode.py \
    --data-path data/simbench/1-MV-rural--0-sw.npz \
    --model atode \
    --task extrapolation \
    --observed-fraction 0.4 \
    --context-length 12 \
    --forecast-length 12 \
    --niters 2 \
    --batch-size 4 \
    --seed 42 \
    --mask-seed 42

Interpolation example:

python run_powergrid_lgode.py \
    --data-path data/simbench/1-MV-rural--0-sw.npz \
    --model atode \
    --task interpolation \
    --observed-fraction 0.4 \
    --trajectory-length 24 \
    --niters 50 \
    --batch-size 16 \
    --seed 1 \
    --mask-seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from lib.powergrid_model_factory import (
    SUPPORTED_POWERGRID_MODELS,
    assert_lgode_atode_protocol_match,
    build_powergrid_lgode_model,
    count_total_parameters,
    count_trainable_parameters,
    shared_graph_state_dict,
)
from lib.simbench_lgode_data import (
    NormalizationStats,
    PowerGridBatch,
    PowerGridDataLoaders,
    build_simbench_dataloaders,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_TASKS = (
    "interpolation",
    "extrapolation",
)

SUPPORTED_OBSERVED_FRACTIONS = (
    0.4,
    0.6,
    0.8,
)

DEFAULT_RESULTS_DIRECTORY = Path("results/powergrid_lgode")
DEFAULT_CHECKPOINT_DIRECTORY = Path("checkpoints/powergrid_lgode")

_EPS = 1.0e-12


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(
    seed: int,
    *,
    deterministic: bool,
) -> None:
    """Seed Python, NumPy, CPU PyTorch and CUDA PyTorch."""

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            torch.use_deterministic_algorithms(True)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()

    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    return device


def git_commit() -> str:
    """Return the current Git commit when available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def sha256_tensor(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sha256_state(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(sha256_tensor(value).encode("ascii"))
    return digest.hexdigest()


def dataset_protocol_manifest(
    loaders: PowerGridDataLoaders,
) -> Dict[str, Any]:
    datasets = {
        "train": loaders.train.dataset,
        "validation": loaders.validation.dataset,
        "test": loaders.test.dataset,
    }
    window_ids: Dict[str, Any] = {}
    window_hashes: Dict[str, str] = {}
    observation_mask_hashes: Dict[str, str] = {}

    for split, dataset in datasets.items():
        records = [asdict(record) for record in dataset.windows]
        ids = [int(record["trajectory_id"]) for record in records]
        window_ids[split] = ids
        window_hashes[split] = hashlib.sha256(
            json.dumps(
                records,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        mask_digest = hashlib.sha256()
        for index, trajectory_id in enumerate(ids):
            mask_digest.update(str(trajectory_id).encode("ascii"))
            mask_digest.update(
                sha256_tensor(dataset.observation_mask(index)).encode(
                    "ascii"
                )
            )
        observation_mask_hashes[split] = mask_digest.hexdigest()

    return {
        "window_counts": {
            split: len(ids) for split, ids in window_ids.items()
        },
        "window_ids": window_ids,
        "window_hashes": window_hashes,
        "observation_mask_hashes": observation_mask_hashes,
    }


def enforce_protocol_fingerprint(
    path: Optional[Path],
    fingerprint: Mapping[str, Any],
) -> None:
    if path is None:
        return

    resolved = path.expanduser().resolve()
    expected = to_json_compatible(fingerprint)
    if resolved.exists():
        with resolved.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != expected:
            raise RuntimeError(
                "Protocol fingerprint differs from the first model for this "
                f"controlled run: {resolved}"
            )
        return

    atomic_write_json(fingerprint, resolved)


# ---------------------------------------------------------------------------
# JSON and checkpoint utilities
# ---------------------------------------------------------------------------

def to_json_compatible(value: Any) -> Any:
    """Recursively convert common scientific-Python objects to JSON."""

    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, Tensor):
        detached = value.detach().cpu()

        if detached.numel() == 1:
            scalar = detached.item()
            if isinstance(scalar, float) and not math.isfinite(scalar):
                return None
            return scalar

        return detached.tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if is_dataclass(value):
        return to_json_compatible(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            to_json_compatible(item)
            for item in value
        ]

    return str(value)


def atomic_write_json(
    value: Mapping[str, Any],
    path: Path,
) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(file_descriptor)

    temporary_path = Path(temporary_name)

    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                to_json_compatible(value),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")

        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_torch_save(
    value: Mapping[str, Any],
    path: Path,
) -> None:
    """Write a PyTorch checkpoint atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(file_descriptor)

    temporary_path = Path(temporary_name)

    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Mapping[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch versions without weights_only.
        return torch.load(
            path,
            map_location=device,
        )


# ---------------------------------------------------------------------------
# Physical candidate-edge graph
# ---------------------------------------------------------------------------

def build_candidate_graph(
    num_nodes: int,
    physical_edge_index: Tensor,
    graph_mode: str = "physical_sparse",
) -> Tuple[Tensor, Tensor]:
    """
    Use the physical directed edges as the graph-model candidate pairs.

    Returns
    -------
    candidate_edge_index:
        [2, E], identical to the physical directed edge index.

    candidate_edge_labels:
        [E], with every physical edge marked active.
    """

    physical_edge_index = torch.as_tensor(
        physical_edge_index,
        dtype=torch.long,
    )

    if physical_edge_index.ndim != 2:
        raise ValueError("physical_edge_index must be rank two.")

    if physical_edge_index.shape[0] == 2:
        pass
    elif physical_edge_index.shape[1] == 2:
        physical_edge_index = (
            physical_edge_index.transpose(0, 1).contiguous()
        )
    else:
        raise ValueError(
            "physical_edge_index must have shape [2,E] or [E,2]."
        )

    if physical_edge_index.numel() == 0:
        raise ValueError(
            "No physical directed edges were found in the SimBench archive."
        )

    minimum = int(physical_edge_index.min().item())
    maximum = int(physical_edge_index.max().item())
    if minimum < 0 or maximum >= num_nodes:
        raise ValueError(
            "physical_edge_index contains an invalid node index: "
            f"minimum={minimum}, maximum={maximum}, num_nodes={num_nodes}."
        )

    if torch.any(physical_edge_index[0] == physical_edge_index[1]):
        raise ValueError(
            "physical_edge_index must not contain self-edges."
        )

    if graph_mode == "physical_sparse":
        candidate_edge_index = physical_edge_index.contiguous().clone()
        candidate_edge_labels = torch.ones(
            candidate_edge_index.shape[1],
            dtype=torch.long,
        )
    elif graph_mode == "all_pairs_nri":
        physical_pairs = {
            (int(sender), int(receiver))
            for sender, receiver in physical_edge_index.transpose(0, 1)
            .tolist()
        }
        senders = []
        receivers = []
        labels = []
        for receiver in range(num_nodes):
            for sender in range(num_nodes):
                if sender == receiver:
                    continue
                senders.append(sender)
                receivers.append(receiver)
                labels.append(
                    int((sender, receiver) in physical_pairs)
                )
        candidate_edge_index = torch.tensor(
            [senders, receivers], dtype=torch.long
        )
        candidate_edge_labels = torch.tensor(labels, dtype=torch.long)
    else:
        raise ValueError(
            "graph_mode must be 'physical_sparse' or 'all_pairs_nri'."
        )

    if graph_mode == "physical_sparse" and (
        candidate_edge_index.shape != physical_edge_index.shape
        or not torch.equal(candidate_edge_index, physical_edge_index)
    ):
        raise AssertionError(
            "AT-ODE candidate edge_index must equal physical_edge_index."
        )

    if (
        graph_mode == "physical_sparse"
        and candidate_edge_index.shape[1] != physical_edge_index.shape[1]
    ):
        raise AssertionError(
            "Candidate-pair count must equal physical-edge count."
        )

    if candidate_edge_labels.numel() != candidate_edge_index.shape[1]:
        raise AssertionError(
            "candidate_labels must contain one entry per candidate edge."
        )

    if graph_mode == "physical_sparse" and not torch.all(
        candidate_edge_labels == 1
    ):
        raise AssertionError(
            "Every candidate label must represent an active physical line."
        )

    return candidate_edge_index, candidate_edge_labels


def batch_candidate_labels(
    labels: Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    return labels.to(device).unsqueeze(0).expand(
        batch_size,
        -1,
    )


# ---------------------------------------------------------------------------
# Model inputs and predictions
# ---------------------------------------------------------------------------

def graph_decoder_dictionary(
    batch: PowerGridBatch,
    *,
    training: bool = False,
) -> Dict[str, Tensor]:
    """
    Convert [B,N,T,F] targets to the flattened original LG-ODE layout.
    """

    batch_size, num_nodes, num_times, input_dim = (
        batch.target_values.shape
    )

    return {
        "data": batch.target_values.reshape(
            batch_size * num_nodes,
            num_times,
            input_dim,
        ),
        "mask": (
            batch.training_loss_mask if training else batch.target_mask
        ).reshape(
            batch_size * num_nodes,
            num_times,
            input_dim,
        ).to(dtype=batch.target_values.dtype),
        "target_times": batch.target_times,
    }


def graph_prediction_to_powergrid_shape(
    predictions: Tensor,
    batch: PowerGridBatch,
) -> Tensor:
    """
    Convert [S,B*N,T,F] predictions to [S,B,N,T,F].
    """

    if predictions.ndim != 4:
        raise ValueError(
            "Graph model predictions must have shape [S,B*N,T,F]; "
            f"got {tuple(predictions.shape)}."
        )

    batch_size, num_nodes, num_times, input_dim = (
        batch.target_values.shape
    )

    expected = (
        predictions.shape[0],
        batch_size * num_nodes,
        num_times,
        input_dim,
    )

    if tuple(predictions.shape) != expected:
        raise ValueError(
            "Graph prediction shape does not match targets: "
            f"expected {expected}, got {tuple(predictions.shape)}."
        )

    return predictions.reshape(
        predictions.shape[0],
        batch_size,
        num_nodes,
        num_times,
        input_dim,
    )


def compute_training_loss(
    model: nn.Module,
    model_name: str,
    batch: PowerGridBatch,
    candidate_labels: Tensor,
    *,
    n_traj_samples: int,
    kl_coefficient: float,
) -> Dict[str, Any]:
    """Compute the model-specific training objective."""

    if model_name == "persistence":
        return model.compute_all_losses(batch)

    if model_name == "latentode":
        return model.compute_all_losses(
            batch,
            n_traj_samples=n_traj_samples,
            kl_coef=kl_coefficient,
            sample_z0=True,
        )

    batch_size = int(batch.target_values.shape[0])
    graph_labels = batch_candidate_labels(
        candidate_labels,
        batch_size,
        batch.target_values.device,
    )

    return model.compute_all_losses(
        batch.encoder_graph,
        graph_decoder_dictionary(batch, training=True),
        graph_labels,
        n_traj_samples=n_traj_samples,
        kl_coef=kl_coefficient,
        sample_z0=True,
    )


def predict_batch(
    model: nn.Module,
    model_name: str,
    batch: PowerGridBatch,
    candidate_labels: Tensor,
    *,
    n_traj_samples: int,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Return predictions [S,B,N,T,F] and optional diagnostics.
    """

    if model_name == "persistence":
        predictions = model(
            batch,
            n_traj_samples=1,
        )
        return predictions, {}

    if model_name == "latentode":
        predictions = model(
            batch,
            n_traj_samples=n_traj_samples,
            sample=False,
        )
        return predictions, {}

    batch_size = int(batch.target_values.shape[0])
    graph_labels = batch_candidate_labels(
        candidate_labels,
        batch_size,
        batch.target_values.device,
    )

    predictions, extra_info, _ = model.get_reconstruction(
        batch.encoder_graph,
        graph_decoder_dictionary(batch),
        graph_labels,
        n_traj_samples=n_traj_samples,
        sample_z0=False,
    )

    predictions = graph_prediction_to_powergrid_shape(
        predictions,
        batch,
    )

    diagnostics: Dict[str, Any] = {}

    if isinstance(extra_info, Mapping):
        transport = extra_info.get("transport_diagnostics")
        if transport is not None:
            diagnostics["transport"] = transport

    solver = getattr(model, "diffeq_solver", None)
    if solver is not None:
        diagnostics["solver"] = getattr(
            solver,
            "last_solver_diagnostics",
            {},
        )

    return predictions, diagnostics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """Accumulate normalized and physical-space errors exactly."""

    def __init__(
        self,
        normalization: NormalizationStats,
        feature_names: Sequence[str],
    ) -> None:
        self.mean = normalization.mean.detach().cpu().double()
        self.std = normalization.std.detach().cpu().double()
        self.feature_names = tuple(str(name) for name in feature_names)

        feature_count = len(self.feature_names)

        self.normalized_squared_error = 0.0
        self.normalized_absolute_error = 0.0
        self.normalized_count = 0.0

        self.physical_squared_error = torch.zeros(
            feature_count,
            dtype=torch.float64,
        )
        self.physical_absolute_error = torch.zeros(
            feature_count,
            dtype=torch.float64,
        )
        self.physical_count = torch.zeros(
            feature_count,
            dtype=torch.float64,
        )

        self.horizon_squared_error: Optional[Tensor] = None
        self.horizon_absolute_error: Optional[Tensor] = None
        self.horizon_count: Optional[Tensor] = None
        self.node_squared_error: Optional[Tensor] = None
        self.node_absolute_error: Optional[Tensor] = None
        self.node_count: Optional[Tensor] = None
        self.feature_squared_error = torch.zeros(
            feature_count, dtype=torch.float64
        )
        self.feature_absolute_error = torch.zeros(
            feature_count, dtype=torch.float64
        )
        self.feature_count = torch.zeros(
            feature_count, dtype=torch.float64
        )

        self.number_of_windows = 0
        self.number_of_batches = 0

    def update(
        self,
        prediction: Tensor,
        truth: Tensor,
        mask: Tensor,
    ) -> None:
        """
        Parameters have shape [B,N,T,F].
        """

        prediction = prediction.detach().cpu().double()
        truth = truth.detach().cpu().double()
        mask = mask.detach().cpu().bool()

        if prediction.shape != truth.shape:
            raise ValueError(
                "Prediction and truth shapes differ: "
                f"{tuple(prediction.shape)} versus {tuple(truth.shape)}."
            )

        if mask.shape != truth.shape:
            raise ValueError(
                "Mask and truth shapes differ."
            )

        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                "Evaluation predictions contain NaN or infinity."
            )

        normalized_error = prediction - truth
        mask_float = mask.to(dtype=torch.float64)

        self.normalized_squared_error += float(
            (
                normalized_error.square() * mask_float
            ).sum().item()
        )
        self.normalized_absolute_error += float(
            (
                normalized_error.abs() * mask_float
            ).sum().item()
        )
        self.normalized_count += float(mask_float.sum().item())

        squared = normalized_error.square() * mask_float
        absolute = normalized_error.abs() * mask_float
        horizon_squared = squared.sum(dim=(0, 1, 3))
        horizon_absolute = absolute.sum(dim=(0, 1, 3))
        horizon_count = mask_float.sum(dim=(0, 1, 3))
        node_squared = squared.sum(dim=(0, 2, 3))
        node_absolute = absolute.sum(dim=(0, 2, 3))
        node_count = mask_float.sum(dim=(0, 2, 3))

        if self.horizon_squared_error is None:
            self.horizon_squared_error = torch.zeros_like(horizon_squared)
            self.horizon_absolute_error = torch.zeros_like(horizon_absolute)
            self.horizon_count = torch.zeros_like(horizon_count)
            self.node_squared_error = torch.zeros_like(node_squared)
            self.node_absolute_error = torch.zeros_like(node_absolute)
            self.node_count = torch.zeros_like(node_count)

        self.horizon_squared_error += horizon_squared
        self.horizon_absolute_error += horizon_absolute
        self.horizon_count += horizon_count
        self.node_squared_error += node_squared
        self.node_absolute_error += node_absolute
        self.node_count += node_count
        self.feature_squared_error += squared.sum(dim=(0, 1, 2))
        self.feature_absolute_error += absolute.sum(dim=(0, 1, 2))
        self.feature_count += mask_float.sum(dim=(0, 1, 2))

        standard_deviation = self.std.reshape(1, 1, 1, -1)
        physical_error = normalized_error * standard_deviation

        reduce_dimensions = (0, 1, 2)

        self.physical_squared_error += (
            physical_error.square() * mask_float
        ).sum(dim=reduce_dimensions)

        self.physical_absolute_error += (
            physical_error.abs() * mask_float
        ).sum(dim=reduce_dimensions)

        self.physical_count += mask_float.sum(
            dim=reduce_dimensions
        )

        self.number_of_windows += int(truth.shape[0])
        self.number_of_batches += 1

    def compute(self) -> Dict[str, Any]:
        if self.normalized_count <= 0.0:
            raise RuntimeError("No target elements were evaluated.")

        normalized_mse = (
            self.normalized_squared_error
            / self.normalized_count
        )
        normalized_mae = (
            self.normalized_absolute_error
            / self.normalized_count
        )

        if self.horizon_count is None or self.node_count is None:
            raise RuntimeError("No structured target metrics were accumulated.")

        horizon_denominator = self.horizon_count.clamp_min(1.0)
        node_denominator = self.node_count.clamp_min(1.0)
        normalized_feature_denominator = self.feature_count.clamp_min(1.0)
        per_horizon = {
            "mse": (
                self.horizon_squared_error / horizon_denominator
            ).tolist(),
            "mae": (
                self.horizon_absolute_error / horizon_denominator
            ).tolist(),
            "count": self.horizon_count.long().tolist(),
        }
        per_node = {
            "mse": (self.node_squared_error / node_denominator).tolist(),
            "mae": (self.node_absolute_error / node_denominator).tolist(),
            "count": self.node_count.long().tolist(),
        }
        normalized_per_feature = {
            self.feature_names[index]: {
                "mse": float(
                    self.feature_squared_error[index]
                    / normalized_feature_denominator[index]
                ),
                "mae": float(
                    self.feature_absolute_error[index]
                    / normalized_feature_denominator[index]
                ),
                "count": int(self.feature_count[index].item()),
            }
            for index in range(len(self.feature_names))
        }

        feature_denominator = self.physical_count.clamp_min(1.0)
        feature_mse = (
            self.physical_squared_error / feature_denominator
        )
        feature_mae = (
            self.physical_absolute_error / feature_denominator
        )
        feature_rmse = torch.sqrt(feature_mse)

        per_feature = {}

        for feature_index, feature_name in enumerate(
            self.feature_names
        ):
            per_feature[feature_name] = {
                "mse": float(feature_mse[feature_index].item()),
                "rmse": float(feature_rmse[feature_index].item()),
                "mae": float(feature_mae[feature_index].item()),
                "count": int(
                    self.physical_count[feature_index].item()
                ),
            }

        return {
            "normalized_mse": float(normalized_mse),
            "normalized_mae": float(normalized_mae),
            "physical_per_feature": per_feature,
            "normalized_per_horizon": per_horizon,
            "normalized_per_node": per_node,
            "normalized_per_feature": normalized_per_feature,
            "number_of_windows": self.number_of_windows,
            "number_of_batches": self.number_of_batches,
            "number_of_target_elements": int(
                self.normalized_count
            ),
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    loader: Iterable[PowerGridBatch],
    candidate_labels: Tensor,
    normalization: NormalizationStats,
    feature_names: Sequence[str],
    device: torch.device,
    *,
    task: str,
    n_traj_samples: int,
    evaluation_seed: int,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a split without changing model parameters.

    A fixed RNG scope makes graph-model posterior sampling reproducible across
    validation epochs. Latent ODE evaluation uses posterior means.
    """

    model.eval()

    full_accumulator = MetricAccumulator(
        normalization=normalization,
        feature_names=feature_names,
    )
    unobserved_accumulator = MetricAccumulator(
        normalization=normalization,
        feature_names=feature_names,
    )

    last_diagnostics: Dict[str, Any] = {}

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(
        devices=cuda_devices,
        enabled=True,
    ):
        torch.manual_seed(int(evaluation_seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(evaluation_seed))

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = batch.to(
                device,
                non_blocking=device.type == "cuda",
            )

            samples, diagnostics = predict_batch(
                model,
                model_name,
                batch,
                candidate_labels,
                n_traj_samples=n_traj_samples,
            )

            if samples.ndim != 5:
                raise ValueError(
                    "Predictions must have shape [S,B,N,T,F]; "
                    f"got {tuple(samples.shape)}."
                )

            # The predictive mean is the authoritative evaluation prediction.
            mean_prediction = samples.mean(dim=0)

            full_accumulator.update(
                mean_prediction,
                batch.target_values,
                batch.target_mask,
            )

            if task == "interpolation":
                expected_mask_shape = batch.target_values.shape
                if tuple(batch.interpolation_withheld_mask.shape) != tuple(
                    expected_mask_shape
                ):
                    raise ValueError(
                        "interpolation_withheld_mask must match targets; "
                        f"expected {tuple(expected_mask_shape)}, got "
                        f"{tuple(batch.interpolation_withheld_mask.shape)}."
                    )
                observed_target_mask = batch.observed_event_mask.to(
                    dtype=torch.bool
                ).unsqueeze(-1).expand_as(batch.target_values)
                unobserved_mask = batch.interpolation_withheld_mask.bool()
                if torch.any(unobserved_mask & observed_target_mask):
                    raise AssertionError(
                        "Interpolation unobserved targets overlap observed "
                        "encoder events."
                    )
                if not torch.any(unobserved_mask):
                    raise ValueError(
                        "Interpolation contains no unobserved target entries."
                    )
            else:
                unobserved_mask = batch.extrapolation_future_mask.bool()

            unobserved_accumulator.update(
                mean_prediction,
                batch.target_values,
                unobserved_mask,
            )

            if diagnostics:
                last_diagnostics = diagnostics

    full_metrics = full_accumulator.compute()
    unobserved_metrics = unobserved_accumulator.compute()
    metrics = dict(full_metrics)
    metrics.update(
        {
            "normalized_mse_full": full_metrics["normalized_mse"],
            "normalized_mae_full": full_metrics["normalized_mae"],
            "normalized_mse_unobserved": unobserved_metrics[
                "normalized_mse"
            ],
            "normalized_mae_unobserved": unobserved_metrics[
                "normalized_mae"
            ],
            "normalized_per_horizon_full": full_metrics[
                "normalized_per_horizon"
            ],
            "normalized_per_node_full": full_metrics[
                "normalized_per_node"
            ],
            "normalized_per_feature_full": full_metrics[
                "normalized_per_feature"
            ],
            "normalized_per_horizon_unobserved": unobserved_metrics[
                "normalized_per_horizon"
            ],
            "normalized_per_node_unobserved": unobserved_metrics[
                "normalized_per_node"
            ],
            "normalized_per_feature_unobserved": unobserved_metrics[
                "normalized_per_feature"
            ],
            "physical_per_feature_full": full_metrics[
                "physical_per_feature"
            ],
            "physical_per_feature_unobserved": unobserved_metrics[
                "physical_per_feature"
            ],
        }
    )
    return metrics, last_diagnostics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def kl_coefficient(
    epoch: int,
    *,
    maximum: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs <= 0:
        return float(maximum)

    fraction = min(
        1.0,
        float(epoch) / float(warmup_epochs),
    )
    return float(maximum) * fraction


def train_one_epoch(
    model: nn.Module,
    model_name: str,
    loader: Iterable[PowerGridBatch],
    candidate_labels: Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    n_traj_samples: int,
    kl_coef: float,
    gradient_clip: float,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_batches = 0
    total_mse = 0.0
    total_kl = 0.0
    total_likelihood = 0.0
    total_gradient_norm = 0.0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(
            device,
            non_blocking=device.type == "cuda",
        )

        optimizer.zero_grad(set_to_none=True)

        losses = compute_training_loss(
            model,
            model_name,
            batch,
            candidate_labels,
            n_traj_samples=n_traj_samples,
            kl_coefficient=kl_coef,
        )

        loss = losses.get("loss")

        if not isinstance(loss, Tensor):
            raise TypeError(
                "compute_all_losses must return tensor key 'loss'."
            )

        if loss.numel() != 1:
            loss = loss.mean()

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss: {loss.detach().cpu().item()}."
            )

        loss.backward()

        for parameter_name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    "Non-finite gradient in parameter "
                    f"{parameter_name!r}."
                )

        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=(
                gradient_clip if gradient_clip > 0.0 else float("inf")
            ),
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                "Non-finite aggregate gradient norm."
            )

        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_mse += float(losses.get("mse", float("nan")))
        total_kl += float(
            losses.get("kl_first_p", float("nan"))
        )
        total_likelihood += float(
            losses.get("likelihood", float("nan"))
        )
        total_gradient_norm += float(gradient_norm.detach().cpu().item())
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("The training loader produced no batches.")

    return {
        "loss": total_loss / total_batches,
        "model_mse_diagnostic": total_mse / total_batches,
        "kl_first_p": total_kl / total_batches,
        "reconstruction_likelihood": total_likelihood / total_batches,
        "gradient_norm_pre_clip": total_gradient_norm / total_batches,
        "kl_coefficient": float(kl_coef),
        "number_of_batches": total_batches,
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    validation_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": dict(validation_metrics),
        "arguments": vars(args),
        "git_commit": git_commit(),
    }


# ---------------------------------------------------------------------------
# Naming and reporting
# ---------------------------------------------------------------------------

def fraction_tag(value: float) -> str:
    return str(int(round(100.0 * float(value))))


def default_run_name(
    args: argparse.Namespace,
    simbench_code: str,
) -> str:
    components = [
        simbench_code,
        args.task,
        f"obs{fraction_tag(args.observed_fraction)}",
        args.model,
        f"seed{args.seed}",
        f"mask{args.mask_seed}",
    ]

    if args.alias:
        components.append(args.alias)

    return "__".join(
        component.replace("/", "_").replace(" ", "_")
        for component in components
    )


def print_epoch(
    epoch: int,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    learning_rate: float,
    best_validation_mse: float,
    elapsed: float,
) -> None:
    print(
        f"epoch={epoch:04d} "
        f"loss={train_metrics['loss']:.6f} "
        f"train_diag_mse="
        f"{train_metrics['model_mse_diagnostic']:.6f} "
        f"val_mse={validation_metrics['normalized_mse']:.6f} "
        f"val_mae={validation_metrics['normalized_mae']:.6f} "
        f"best_val_mse={best_validation_mse:.6f} "
        f"lr={learning_rate:.3e} "
        f"time={elapsed:.2f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def observed_fraction_type(value: str) -> float:
    fraction = float(value)

    if fraction in (40.0, 60.0, 80.0):
        fraction /= 100.0

    for allowed in SUPPORTED_OBSERVED_FRACTIONS:
        if math.isclose(
            fraction,
            allowed,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return allowed

    raise argparse.ArgumentTypeError(
        "observed fraction must be one of 0.4, 0.6, 0.8, 40, 60 or 80"
    )


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Persistence, Latent ODE, LG-ODE or AT-ODE on a "
            "chronologically split SimBench trajectory."
        )
    )

    # Dataset and task.
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help=(
            "Direct SimBench NPZ path or a directory containing "
            "<simbench-code>.npz."
        ),
    )
    parser.add_argument(
        "--simbench-code",
        type=str,
        default=None,
        help=(
            "SimBench archive stem when --data-path is a directory."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_POWERGRID_MODELS,
        required=True,
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=SUPPORTED_TASKS,
        required=True,
    )
    parser.add_argument(
        "--observed-fraction",
        type=observed_fraction_type,
        required=True,
    )

    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=24,
        help="Interpolation window length.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=12,
        help="Extrapolation context length.",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=12,
        help="Extrapolation forecast length.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-temporal-gap",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--normalization-eps",
        type=float,
        default=1.0e-8,
    )

    # Training.
    parser.add_argument(
        "--niters",
        "--epochs",
        dest="niters",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5.0e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adam",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience; zero disables early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
    )

    # Shared architecture.
    parser.add_argument(
        "--latent-dim",
        "--latents",
        dest="latent_dim",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--recognition-dim",
        "--rec-dims",
        dest="recognition_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--ode-hidden-dim",
        "--ode-dims",
        dest="ode_hidden_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--augmentation-dim",
        "--augment-dim",
        dest="augmentation_dim",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--encoder-layers",
        "--rec-layers",
        dest="encoder_layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--ode-layers",
        "--gen-layers",
        dest="ode_layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--attention-heads",
        "--n-heads",
        dest="attention_heads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--edge-types",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--graph-mode",
        choices=("physical_sparse", "all_pairs_nri"),
        default="physical_sparse",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--ode-dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--observation-std",
        "--obsrv-std",
        dest="observation_std",
        type=float,
        default=0.01,
    )

    # ODE solver.
    parser.add_argument(
        "--solver",
        type=str,
        default="dopri5",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-4,
    )

    # AT-ODE.
    parser.add_argument(
        "--transport-bins",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--transport-max-age",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--transport-hidden-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--transport-attention-dim",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--transport-heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--transport-speed",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--transport-decay",
        type=float,
        default=1.0,
    )

    # Reproducibility and loading.
    parser.add_argument(
        "--seed",
        "--random-seed",
        dest="seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
    )
    parser.add_argument(
        "--drop-last-train",
        action="store_true",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    # Output.
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
    )
    parser.add_argument(
        "--protocol-fingerprint-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    args = parser.parse_args(argv)

    if args.mask_seed is None:
        args.mask_seed = args.seed

    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "batch_size",
        "trajectory_length",
        "context_length",
        "forecast_length",
        "latent_dim",
        "recognition_dim",
        "ode_hidden_dim",
        "encoder_layers",
        "ode_layers",
        "attention_heads",
        "edge_types",
        "train_samples",
        "eval_samples",
        "transport_bins",
        "transport_hidden_dim",
        "transport_attention_dim",
        "transport_heads",
    )

    for field in positive_integer_fields:
        if int(getattr(args, field)) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")

    if args.niters < 1 and args.model != "persistence":
        raise ValueError("--niters must be positive.")

    if args.stride < 1:
        raise ValueError("--stride must be positive.")

    for field in ("max_train_batches", "max_eval_batches"):
        value = getattr(args, field)
        if value is not None and value < 1:
            raise ValueError(
                f"--{field.replace('_', '-')} must be positive when set."
            )

    if args.lr <= 0.0:
        raise ValueError("--lr must be positive.")

    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay cannot be negative.")

    if args.gradient_clip < 0.0:
        raise ValueError("--gradient-clip cannot be negative.")

    if args.kl_coef < 0.0:
        raise ValueError("--kl-coef cannot be negative.")

    if args.rtol <= 0.0 or args.atol <= 0.0:
        raise ValueError("--rtol and --atol must be positive.")

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0,1).")

    if args.ode_dropout != 0.0:
        raise ValueError("--ode-dropout must be exactly zero.")

    if args.observation_std <= 0.0:
        raise ValueError("--observation-std must be positive.")

    if args.transport_bins < 2:
        raise ValueError("--transport-bins must be at least two.")

    if args.transport_max_age <= 0.0:
        raise ValueError("--transport-max-age must be positive.")

    if args.transport_speed <= 0.0:
        raise ValueError("--transport-speed must be positive.")

    if args.transport_decay <= 0.0:
        raise ValueError("--transport-decay must be positive.")

    if args.recognition_dim % args.attention_heads != 0:
        raise ValueError(
            "--recognition-dim must be divisible by --attention-heads."
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    experiment_start = time.perf_counter()
    seed_everything(
        args.seed,
        deterministic=args.deterministic,
    )

    device = resolve_device(args.device)

    data_arguments: Dict[str, Any] = {
        "data_path": args.data_path,
        "simbench_code": args.simbench_code,
        "task": args.task,
        "observed_fraction": args.observed_fraction,
        "batch_size": args.batch_size,
        "stride": args.stride,
        "seed": args.seed,
        "mask_seed": args.mask_seed,
        "max_temporal_gap": args.max_temporal_gap,
        "normalization_eps": args.normalization_eps,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last_train": args.drop_last_train,
    }

    if args.task == "interpolation":
        data_arguments.update(
            {
                "trajectory_length": args.trajectory_length,
                "context_length": None,
                "forecast_length": None,
            }
        )
    else:
        data_arguments.update(
            {
                "trajectory_length": None,
                "context_length": args.context_length,
                "forecast_length": args.forecast_length,
            }
        )

    loaders: PowerGridDataLoaders = (
        build_simbench_dataloaders(**data_arguments)
    )

    archive = loaders.archive
    resolved_data_path = archive.path.resolve()

    simbench_code = (
        args.simbench_code
        if args.simbench_code is not None
        else resolved_data_path.stem
    )

    args.num_nodes = archive.num_nodes
    args.input_dim = archive.input_dim

    # DiffeqSolver reads model/model_name from args to determine its mode.
    args.model_name = args.model
    args.model_type = args.model

    candidate_edge_index, candidate_edge_labels = (
        build_candidate_graph(
            archive.num_nodes,
            archive.edge_index,
            args.graph_mode,
        )
    )

    data_sha256 = sha256_file(resolved_data_path)
    protocol_manifest = dataset_protocol_manifest(loaders)
    edge_index_hash = sha256_tensor(candidate_edge_index)
    protocol_fingerprint = {
        "data_sha256": data_sha256,
        "task": args.task,
        "observed_fraction": args.observed_fraction,
        "trajectory_length": args.trajectory_length,
        "context_length": args.context_length,
        "forecast_length": args.forecast_length,
        "stride": args.stride,
        "seed": args.seed,
        "mask_seed": args.mask_seed,
        "graph_mode": args.graph_mode,
        "train_end": archive.train_end,
        "validation_end": archive.validation_end,
        "edge_index_hash": edge_index_hash,
        **protocol_manifest,
    }
    enforce_protocol_fingerprint(
        args.protocol_fingerprint_path,
        protocol_fingerprint,
    )

    model_edge_index = (
        candidate_edge_index
        if args.model in {"lgode", "atode"}
        else archive.edge_index
    )

    model = build_powergrid_lgode_model(
        model_name=args.model,
        input_dim=archive.input_dim,
        num_nodes=archive.num_nodes,
        edge_index=model_edge_index,
        args=args,
        device=device,
    )

    shared_initialization_hash = None

    if args.model in {"lgode", "atode"}:
        model_edges = getattr(model, "powergrid_edge_index", None)
        if not isinstance(model_edges, Tensor) or not torch.equal(
            model_edges.cpu(), candidate_edge_index.cpu()
        ):
            raise AssertionError(
                "Graph model edge order differs from physical_edge_index."
            )
        model_edge_count = int(
            model.diffeq_solver.candidate_edge_count
        )
        if model_edge_count != candidate_edge_index.shape[1]:
            raise AssertionError(
                "Graph model edge count differs from physical-edge count."
            )

    if args.model in {"lgode", "atode"}:
        counterpart_name = "atode" if args.model == "lgode" else "lgode"
        counterpart = build_powergrid_lgode_model(
            model_name=counterpart_name,
            input_dim=archive.input_dim,
            num_nodes=archive.num_nodes,
            edge_index=model_edge_index,
            args=args,
            device=device,
        )
        if args.model == "lgode":
            assert_lgode_atode_protocol_match(model, counterpart)
            model_shared_state = shared_graph_state_dict(model)
            counterpart_shared_state = shared_graph_state_dict(counterpart)
        else:
            assert_lgode_atode_protocol_match(counterpart, model)
            model_shared_state = shared_graph_state_dict(model)
            counterpart_shared_state = shared_graph_state_dict(counterpart)
        model_initialization_hash = sha256_state(model_shared_state)
        counterpart_initialization_hash = sha256_state(
            counterpart_shared_state
        )
        if model_initialization_hash != counterpart_initialization_hash:
            raise AssertionError(
                "LG-ODE and AT-ODE shared initialization hashes differ."
            )
        shared_initialization_hash = model_initialization_hash
        del counterpart

    total_parameters = count_total_parameters(model)
    trainable_parameters = count_trainable_parameters(model)

    run_name = (
        args.run_name
        if args.run_name
        else default_run_name(args, simbench_code)
    )

    result_path = (
        args.results_dir.expanduser().resolve()
        / f"{run_name}.json"
    )
    checkpoint_path = (
        args.checkpoint_dir.expanduser().resolve()
        / f"{run_name}.pt"
    )

    if result_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Result already exists: {result_path}. "
            "Use --overwrite to replace it."
        )

    if not args.quiet:
        print("=" * 78)
        print("Power-grid LG-ODE benchmark")
        print(f"model:                 {args.model}")
        print(f"task:                  {args.task}")
        print(f"observed fraction:     {args.observed_fraction}")
        print(f"data:                  {resolved_data_path}")
        print(f"SimBench code:         {simbench_code}")
        print(f"nodes:                 {archive.num_nodes}")
        print(f"features:              {archive.input_dim}")
        print(f"physical edges:        {archive.edge_index.shape[1]}")
        print(
            f"candidate pairs:        "
            f"{candidate_edge_index.shape[1]}"
        )
        print(f"device:                {device}")
        print(f"parameters:            {total_parameters}")
        print(f"trainable parameters:  {trainable_parameters}")
        print(
            "windows train/val/test: "
            f"{len(loaders.train.dataset)}/"
            f"{len(loaders.validation.dataset)}/"
            f"{len(loaders.test.dataset)}"
        )
        print(f"result:                {result_path}")
        print(f"checkpoint:            {checkpoint_path}")
        print("=" * 78)

    training_history = []
    best_validation_mse = math.inf
    best_validation_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    if args.model == "persistence":
        validation_metrics, validation_diagnostics = evaluate(
            model,
            args.model,
            loaders.validation,
            candidate_edge_labels,
            loaders.normalization,
            archive.bus_feature_names,
            device,
            task=args.task,
            n_traj_samples=1,
            evaluation_seed=args.eval_seed,
            max_batches=args.max_eval_batches,
        )

        checkpoint_metric = (
            "normalized_mse_unobserved"
            if args.task == "interpolation"
            else "normalized_mse_full"
        )
        best_validation_mse = float(validation_metrics[checkpoint_metric])
        best_validation_epoch = 0
        training_time_seconds = 0.0

    else:
        optimizer_class = (
            torch.optim.AdamW
            if args.optimizer == "adamw"
            else torch.optim.Adam
        )
        optimizer = optimizer_class(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

        validation_metrics = {}
        validation_diagnostics = {}

        for epoch in range(1, args.niters + 1):
            epoch_start = time.perf_counter()

            current_kl_coefficient = kl_coefficient(
                epoch,
                maximum=args.kl_coef,
                warmup_epochs=args.kl_warmup_epochs,
            )

            train_metrics = train_one_epoch(
                model,
                args.model,
                loaders.train,
                candidate_edge_labels,
                optimizer,
                device,
                n_traj_samples=args.train_samples,
                kl_coef=current_kl_coefficient,
                gradient_clip=args.gradient_clip,
                max_batches=args.max_train_batches,
            )

            validation_metrics, validation_diagnostics = evaluate(
                model,
                args.model,
                loaders.validation,
                candidate_edge_labels,
                loaders.normalization,
                archive.bus_feature_names,
                device,
                task=args.task,
                n_traj_samples=args.eval_samples,
                evaluation_seed=args.eval_seed,
                max_batches=args.max_eval_batches,
            )

            checkpoint_metric = (
                "normalized_mse_unobserved"
                if args.task == "interpolation"
                else "normalized_mse_full"
            )
            validation_mse = float(validation_metrics[checkpoint_metric])

            scheduler.step(validation_mse)

            improved = (
                validation_mse
                < best_validation_mse - args.min_delta
            )

            if improved:
                best_validation_mse = validation_mse
                best_validation_epoch = epoch
                epochs_without_improvement = 0

                atomic_torch_save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        args,
                        epoch,
                        validation_metrics,
                    ),
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            epoch_elapsed = time.perf_counter() - epoch_start
            current_learning_rate = float(
                optimizer.param_groups[0]["lr"]
            )

            training_history.append(
                {
                    "epoch": epoch,
                    "training": train_metrics,
                    "validation": validation_metrics,
                    "learning_rate": current_learning_rate,
                    "improved": improved,
                    "epoch_time_seconds": epoch_elapsed,
                }
            )

            if not args.quiet:
                print_epoch(
                    epoch,
                    train_metrics,
                    validation_metrics,
                    current_learning_rate,
                    best_validation_mse,
                    epoch_elapsed,
                )

            if (
                args.patience > 0
                and epochs_without_improvement >= args.patience
            ):
                if not args.quiet:
                    print(
                        "Early stopping: validation MSE did not improve "
                        f"for {args.patience} epochs.",
                        flush=True,
                    )
                break

        training_time_seconds = (
            time.perf_counter() - training_start
        )

        if not checkpoint_path.is_file():
            raise RuntimeError(
                "Training completed without producing a validation-selected "
                "checkpoint."
            )

        # Restore the validation-selected checkpoint before any test access.
        checkpoint = load_checkpoint(
            checkpoint_path,
            device,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        best_validation_epoch = int(checkpoint["epoch"])
        best_validation_mse = float(
            checkpoint["validation_metrics"][checkpoint_metric]
        )

        # Recompute validation only to report the restored model consistently.
        validation_metrics, validation_diagnostics = evaluate(
            model,
            args.model,
            loaders.validation,
            candidate_edge_labels,
            loaders.normalization,
            archive.bus_feature_names,
            device,
            task=args.task,
            n_traj_samples=args.eval_samples,
            evaluation_seed=args.eval_seed,
            max_batches=args.max_eval_batches,
        )

    # This is the only test-set evaluation in the entire script.
    test_start = time.perf_counter()

    test_metrics, test_diagnostics = evaluate(
        model,
        args.model,
        loaders.test,
        candidate_edge_labels,
        loaders.normalization,
        archive.bus_feature_names,
        device,
        task=args.task,
        n_traj_samples=(
            1 if args.model == "persistence"
            else args.eval_samples
        ),
        evaluation_seed=args.eval_seed,
        max_batches=args.max_eval_batches,
    )

    test_time_seconds = time.perf_counter() - test_start
    total_runtime_seconds = time.perf_counter() - experiment_start

    normalizer = {
        "mean": loaders.normalization.mean.tolist(),
        "std": loaders.normalization.std.tolist(),
        "count": loaders.normalization.count,
        "fitted_start": loaders.normalization.fitted_start,
        "fitted_end": loaders.normalization.fitted_end,
        "eps": loaders.normalization.eps,
    }

    result = {
        "schema_version": 1,
        "run_name": run_name,
        "alias": args.alias,
        "model": args.model,
        "model_name": args.model,
        "architecture_name": (
            "IndependentGRULatentODE"
            if args.model == "latentode"
            else args.model
        ),
        "task": args.task,
        "observed_fraction": args.observed_fraction,
        "seed": args.seed,
        "mask_seed": args.mask_seed,
        "eval_seed": args.eval_seed,
        "evaluation_mode": (
            "posterior_mean"
            if args.model != "persistence"
            else "deterministic"
        ),
        "eval_samples": args.eval_samples,
        "simbench_code": simbench_code,
        "git_commit": git_commit(),
        "dataset": {
            "path": str(resolved_data_path),
            "sha256": data_sha256,
            "simbench_code": simbench_code,
            "num_timesteps": archive.num_timesteps,
            "num_nodes": archive.num_nodes,
            "input_dim": archive.input_dim,
            "feature_names": list(archive.bus_feature_names),
            "physical_directed_edge_count": int(
                archive.edge_index.shape[1]
            ),
            "candidate_directed_edge_count": int(
                candidate_edge_index.shape[1]
            ),
            "active_candidate_edge_count": int(
                candidate_edge_labels.sum().item()
            ),
            "train_end": archive.train_end,
            "validation_end": archive.validation_end,
            "test_end": archive.num_timesteps,
            "edge_index_sha256": edge_index_hash,
            **protocol_manifest,
            "metadata": archive.metadata,
            "normalization": normalizer,
        },
        "config": {
            **vars(args),
            "data_path": str(args.data_path),
            "device_resolved": str(device),
        },
        "trajectory_length": (
            args.trajectory_length
            if args.task == "interpolation"
            else None
        ),
        "context_length": (
            args.context_length
            if args.task == "extrapolation"
            else None
        ),
        "forecast_length": (
            args.forecast_length
            if args.task == "extrapolation"
            else None
        ),
        "stride": args.stride,
        "batch_size": args.batch_size,
        "parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "best_validation_epoch": best_validation_epoch,
        "best_validation_mse": best_validation_mse,
        "checkpoint_selection_metric": checkpoint_metric,
        "shared_initialization_sha256": shared_initialization_hash,
        "training_time_seconds": training_time_seconds,
        "test_time_seconds": test_time_seconds,
        "runtime_seconds": total_runtime_seconds,
        "validation": validation_metrics,
        "test": test_metrics,
        "diagnostics": {
            "validation": validation_diagnostics,
            "test": test_diagnostics,
        },
        "training": {
            "best_epoch": best_validation_epoch,
            "best_validation_mse": best_validation_mse,
            "time_seconds": training_time_seconds,
            "epochs_completed": (
                0
                if args.model == "persistence"
                else len(training_history)
            ),
            "history": training_history,
        },
        "checkpoint": {
            "path": (
                None
                if args.model == "persistence"
                else str(checkpoint_path)
            ),
            "selected_using": f"validation.{checkpoint_metric}",
            "test_used_for_selection": False,
        },
        "optimizer": {
            "name": (
                args.optimizer.upper()
                if args.model != "persistence"
                else None
            ),
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "lr_patience": args.lr_patience,
            "lr_factor": args.lr_factor,
        },
        "protocol": {
            "normalization_training_only": True,
            "complete_target_evaluation": True,
            "validation_checkpoint_selection": True,
            "primary_metric": checkpoint_metric,
            "test_evaluations": 1,
            "observation_masks_model_independent": True,
            "missing_observations_imputed_into_encoder": False,
            "lgode_atode_difference": (
                "time-dependent edge transport only"
            ),
            "graph_mode": args.graph_mode,
            "interpolation_semantics": (
                "noncausal_smoothing_over_observations_across_window"
                if args.task == "interpolation"
                else None
            ),
        },
    }

    atomic_write_json(
        result,
        result_path,
    )

    if not args.quiet:
        print("=" * 78)
        print("Final result")
        print(f"best epoch:       {best_validation_epoch}")
        print(f"validation MSE:   {best_validation_mse:.8f}")
        print(
            f"test MSE:         "
            f"{test_metrics['normalized_mse']:.8f}"
        )
        print(
            f"test MAE:         "
            f"{test_metrics['normalized_mae']:.8f}"
        )
        print(f"training seconds: {training_time_seconds:.2f}")
        print(f"test seconds:     {test_time_seconds:.2f}")
        print(f"wrote:            {result_path}")
        print("=" * 78)

    return result


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(
            f"Error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
