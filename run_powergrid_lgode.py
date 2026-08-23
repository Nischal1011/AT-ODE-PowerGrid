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

import lib.powergrid_model_factory as powergrid_factory
from lib.powergrid_model_factory import (
    SUPPORTED_POWERGRID_MODELS,
    build_powergrid_lgode_model,
    count_total_parameters,
    count_trainable_parameters,
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
# Compatibility adapter for the current repository
# ---------------------------------------------------------------------------

class RepositoryAttentionTransportAdapter(nn.Module):
    """
    Adapt SolverSafeAttentionTransport to the factory/solver interface.

    The current repository has three interface differences:

    1. The factory imports ``AttentionTransport``, but the implementation
       exports ``SolverSafeAttentionTransport``.
    2. DiffeqSolver passes rel_send and rel_rec to the transport forward call,
       while SolverSafeAttentionTransport does not consume them.
    3. DiffeqSolver expects a callable edge-weight provider, while
       SolverSafeAttentionTransport returns an AttentionTransportCache whose
       ``edge_weights_at`` method is the callable provider.

    This adapter changes interfaces only. It does not modify transport
    mathematics.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        edge_index: Tensor,
        num_nodes: int,
        num_bins: int,
        max_age: float,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        initial_speed: float,
        initial_decay: float,
        dropout: float,
    ) -> None:
        super().__init__()

        from lib.attention_transport import (
            SolverSafeAttentionTransport,
        )

        self.transport = SolverSafeAttentionTransport(
            latent_dim=int(latent_dim),
            edge_index=edge_index,
            num_nodes=int(num_nodes),
            num_bins=int(num_bins),
            max_age=float(max_age),
            hidden_dim=int(hidden_dim),
            attention_dim=int(attention_dim),
            num_heads=int(num_heads),
            initial_speed=float(initial_speed),
            initial_decay=float(initial_decay),
            learnable_speed=True,
            learnable_decay=True,
            dropout=float(dropout),
        )

    def forward(
        self,
        *,
        z0: Tensor,
        latest_observation_time: Tensor,
        time_grid: Tensor,
        physical_edge_mask: Optional[Tensor] = None,
        rel_send: Optional[Tensor] = None,
        rel_rec: Optional[Tensor] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        del rel_send, rel_rec

        cache = self.transport(
            z0=z0,
            latest_observation_time=(
                latest_observation_time.squeeze(-1)
                if latest_observation_time.ndim == 3
                and latest_observation_time.shape[-1] == 1
                else latest_observation_time
            ),
            time_grid=time_grid,
            physical_edge_mask=(
                physical_edge_mask.squeeze(-1)
                if physical_edge_mask is not None
                and physical_edge_mask.ndim == 3
                and physical_edge_mask.shape[-1] == 1
                else physical_edge_mask
            ),
        )


        return {
            "provider": cache.edge_weights_at,
            "diagnostics": cache.diagnostics,
            "cache": cache,
        }


def install_transport_factory_adapter(
    args: argparse.Namespace,
) -> None:
    """
    Install a constructor compatible with powergrid_model_factory.py.

    This can be removed after powergrid_model_factory directly imports and
    correctly configures SolverSafeAttentionTransport.
    """

    def constructor(**keywords: Any) -> nn.Module:
        latent_dim = int(
            keywords.get(
                "latent_dim",
                keywords.get("state_dim", args.latent_dim),
            )
        )
        edge_index = keywords.get("edge_index")

        if edge_index is None:
            raise ValueError(
                "AT-ODE transport construction requires edge_index."
            )

        return RepositoryAttentionTransportAdapter(
            latent_dim=latent_dim,
            edge_index=torch.as_tensor(
                edge_index,
                dtype=torch.long,
            ),
            num_nodes=int(
                keywords.get("num_nodes", args.num_nodes)
            ),
            num_bins=int(
                keywords.get(
                    "num_bins",
                    keywords.get(
                        "transport_bins",
                        args.transport_bins,
                    ),
                )
            ),
            max_age=float(
                keywords.get(
                    "max_age",
                    keywords.get(
                        "transport_max_age",
                        args.transport_max_age,
                    ),
                )
            ),
            hidden_dim=int(
                keywords.get(
                    "hidden_dim",
                    keywords.get(
                        "transport_hidden_dim",
                        args.transport_hidden_dim,
                    ),
                )
            ),
            attention_dim=int(args.transport_attention_dim),
            num_heads=int(args.transport_heads),
            initial_speed=float(args.transport_speed),
            initial_decay=float(args.transport_decay),
            dropout=float(
                keywords.get("dropout", args.dropout)
            ),
        )

    powergrid_factory.AttentionTransport = constructor
    powergrid_factory._ATTENTION_TRANSPORT_IMPORT_ERROR = None


def canonicalize_graph_model_runtime(
    model: nn.Module,
    model_name: str,
) -> None:
    """
    Convert the factory's legacy LG-ODE ``fixed`` label to ``ones``.

    DiffeqSolver accepts ``fixed`` during construction as an alias, but the
    factory subsequently overwrites the normalized value with ``fixed``.
    The solver forward path requires the canonical value ``ones``.
    """

    if model_name != "lgode":
        return

    model.edge_weight_mode = "ones"

    solver = getattr(model, "diffeq_solver", None)
    if solver is None:
        raise RuntimeError("LG-ODE model does not expose diffeq_solver.")

    solver.edge_weight_mode = "ones"

    ode_function = getattr(model, "generative_ode_function", None)
    if ode_function is None:
        ode_function = getattr(solver, "ode_func", None)

    if ode_function is not None:
        ode_function.edge_weight_mode = "ones"

        ode_network = getattr(
            ode_function,
            "ode_func_net",
            None,
        )

        if ode_network is not None and hasattr(ode_network, "gcs"):
            for layer in ode_network.gcs:
                convolution = getattr(layer, "base_conv", None)
                if (
                    convolution is not None
                    and hasattr(convolution, "set_edge_weight_mode")
                ):
                    convolution.set_edge_weight_mode("ones")


# ---------------------------------------------------------------------------
# Complete candidate-pair graph
# ---------------------------------------------------------------------------

def build_candidate_graph(
    num_nodes: int,
    physical_edge_index: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Build complete directed non-self candidate pairs.

    Ordering exactly matches DiffeqSolver.compute_rec_send():

        receiver 0: sender 1, sender 2, ...
        receiver 1: sender 0, sender 2, ...
        ...

    Returns
    -------
    candidate_edge_index:
        [2, N*(N-1)], with row zero containing senders and row one receivers.

    candidate_edge_labels:
        [N*(N-1)], where one means a physical edge and zero means no edge.
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

    physical_pairs = {
        (
            int(physical_edge_index[0, edge].item()),
            int(physical_edge_index[1, edge].item()),
        )
        for edge in range(physical_edge_index.shape[1])
        if int(physical_edge_index[0, edge].item())
        != int(physical_edge_index[1, edge].item())
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
                1 if (sender, receiver) in physical_pairs else 0
            )

    candidate_edge_index = torch.tensor(
        [senders, receivers],
        dtype=torch.long,
    )
    candidate_edge_labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    expected = num_nodes * (num_nodes - 1)

    if candidate_edge_index.shape != (2, expected):
        raise RuntimeError(
            "Candidate edge construction produced an invalid shape."
        )

    if candidate_edge_labels.shape != (expected,):
        raise RuntimeError(
            "Candidate edge labels have an invalid shape."
        )

    if candidate_edge_labels.sum() == 0:
        raise ValueError(
            "No physical directed edges were found in the SimBench archive."
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
        "mask": batch.target_mask.reshape(
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
        )

    batch_size = int(batch.target_values.shape[0])
    graph_labels = batch_candidate_labels(
        candidate_labels,
        batch_size,
        batch.target_values.device,
    )

    return model.compute_all_losses(
        batch.encoder_graph,
        graph_decoder_dictionary(batch),
        graph_labels,
        n_traj_samples=n_traj_samples,
        kl_coef=kl_coefficient,
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
    n_traj_samples: int,
    evaluation_seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a split without changing model parameters.

    A fixed RNG scope makes graph-model posterior sampling reproducible across
    validation epochs. Latent ODE evaluation uses posterior means.
    """

    model.eval()

    accumulator = MetricAccumulator(
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

        for batch in loader:
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

            accumulator.update(
                mean_prediction,
                batch.target_values,
                batch.target_mask,
            )

            if diagnostics:
                last_diagnostics = diagnostics

    return accumulator.compute(), last_diagnostics


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
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_batches = 0
    total_mse = 0.0
    total_kl = 0.0

    for batch in loader:
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

        if gradient_clip > 0.0:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )

        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_mse += float(losses.get("mse", float("nan")))
        total_kl += float(
            losses.get("kl_first_p", float("nan"))
        )
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("The training loader produced no batches.")

    return {
        "loss": total_loss / total_batches,
        "model_mse_diagnostic": total_mse / total_batches,
        "kl_first_p": total_kl / total_batches,
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
        "--dropout",
        type=float,
        default=0.2,
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
        )
    )

    if args.model == "atode":
        install_transport_factory_adapter(args)

    # The transport implementation operates over complete non-self candidate
    # pairs. Physical edges remain selected by candidate_edge_labels.
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

    canonicalize_graph_model_runtime(
        model,
        args.model,
    )

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
            n_traj_samples=1,
            evaluation_seed=args.eval_seed,
        )

        best_validation_mse = float(
            validation_metrics["normalized_mse"]
        )
        best_validation_epoch = 0
        training_time_seconds = 0.0

    else:
        optimizer = torch.optim.Adam(
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
            )

            validation_metrics, validation_diagnostics = evaluate(
                model,
                args.model,
                loaders.validation,
                candidate_edge_labels,
                loaders.normalization,
                archive.bus_feature_names,
                device,
                n_traj_samples=args.eval_samples,
                evaluation_seed=args.eval_seed,
            )

            validation_mse = float(
                validation_metrics["normalized_mse"]
            )

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
            checkpoint["validation_metrics"]["normalized_mse"]
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
            n_traj_samples=args.eval_samples,
            evaluation_seed=args.eval_seed,
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
        n_traj_samples=(
            1 if args.model == "persistence"
            else args.eval_samples
        ),
        evaluation_seed=args.eval_seed,
    )

    test_time_seconds = time.perf_counter() - test_start

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
        "task": args.task,
        "observed_fraction": args.observed_fraction,
        "seed": args.seed,
        "mask_seed": args.mask_seed,
        "eval_seed": args.eval_seed,
        "simbench_code": simbench_code,
        "git_commit": git_commit(),
        "dataset": {
            "path": str(resolved_data_path),
            "sha256": sha256_file(resolved_data_path),
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
        "training_time_seconds": training_time_seconds,
        "test_time_seconds": test_time_seconds,
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
            "selected_using": "validation.normalized_mse",
            "test_used_for_selection": False,
        },
        "protocol": {
            "normalization_training_only": True,
            "complete_target_evaluation": True,
            "validation_checkpoint_selection": True,
            "test_evaluations": 1,
            "observation_masks_model_independent": True,
            "missing_observations_imputed_into_encoder": False,
            "lgode_atode_difference": (
                "time-dependent edge transport only"
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
