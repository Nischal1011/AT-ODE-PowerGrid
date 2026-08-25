# lib/powergrid_baselines.py
"""
Power-grid baselines for the SimBench LG-ODE experiments.

This module contains exactly two model families:

1. PersistenceBaseline
   * Interpolation:
       Carry the latest real observation forward. Before the first observation,
       use the first available real observation.
   * Extrapolation:
       Repeat the latest observed context state throughout the forecast.
   * Missing observations are never forward-filled and fed back as inputs.

2. IndependentLatentODE
   * Encodes every bus's sparse event sequence independently.
   * Infers a variational posterior over a latent initial state per bus.
   * Evolves every bus with the same node-wise ODE.
   * Does not inspect or use the physical graph.
   * Uses the original LG-ODE linear Decoder and a compatible variational
     Gaussian reconstruction objective.

Common prediction shape
-----------------------
Both models return predictions with shape:

    [num_trajectory_samples, batch, num_buses, target_length, input_dim]

Persistence has no stochastic latent state, so num_trajectory_samples is
always one.

The module supports the new PowerGridBatch interface. IndependentLatentODE
also exposes ``get_reconstruction`` and ``compute_all_losses`` methods shaped
similarly to the original LatentGraphODE/VAE_Baseline API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence
from torch.nn.utils.rnn import pack_padded_sequence

try:
    from torchdiffeq import odeint
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lib.powergrid_baselines requires torchdiffeq. "
        "Install it with `pip install torchdiffeq`."
    ) from exc

from lib.base_models import VAE_Baseline
from lib.encoder_decoder import Decoder

try:
    from lib.simbench_lgode_data import PowerGridBatch
except ImportError:  # pragma: no cover
    PowerGridBatch = Any  # type: ignore[misc,assignment]


_EPS = 1e-8


@dataclass
class IndependentLatentODEOutput:
    """Structured output from an IndependentLatentODE forward pass."""

    predictions: Tensor
    posterior_mean: Tensor
    posterior_std: Tensor
    latent_initial_state: Tensor
    latent_trajectory: Tensor
    target_times: Tensor

    def as_extra_info(self) -> Dict[str, Any]:
        """
        Return the metadata layout expected by the original VAE code.

        The first-point tensors are flattened from [B, N, D] to [B*N, D].
        """

        num_samples, batch_size, num_nodes, latent_dim = (
            self.latent_initial_state.shape
        )

        mean_flat = self.posterior_mean.reshape(
            batch_size * num_nodes, latent_dim
        )
        std_flat = self.posterior_std.reshape(
            batch_size * num_nodes, latent_dim
        )
        sample_flat = self.latent_initial_state.reshape(
            num_samples, batch_size * num_nodes, latent_dim
        )

        return {
            "first_point": (
                mean_flat.unsqueeze(0),
                std_flat.unsqueeze(0),
                sample_flat,
            ),
            "latent_traj": self.latent_trajectory,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "target_times": self.target_times,
        }


def _is_powergrid_batch(value: Any) -> bool:
    required = (
        "encoder_graph",
        "target_values",
        "target_times",
        "target_mask",
        "observed_event_mask",
        "trajectory_id",
    )
    return all(hasattr(value, name) for name in required)


def _batch_size_from_encoder_graph(encoder_graph: Any) -> int:
    if hasattr(encoder_graph, "num_graphs"):
        return int(encoder_graph.num_graphs)

    if hasattr(encoder_graph, "ptr"):
        return int(encoder_graph.ptr.numel() - 1)

    if hasattr(encoder_graph, "batch") and encoder_graph.batch.numel() > 0:
        return int(encoder_graph.batch.max().item()) + 1

    return 1


def _event_graph_ptr(encoder_graph: Any) -> Tensor:
    """Return event-node boundaries for every trajectory graph."""

    if hasattr(encoder_graph, "ptr"):
        return encoder_graph.ptr.long()

    num_events = int(encoder_graph.x.shape[0])
    return torch.tensor(
        [0, num_events],
        dtype=torch.long,
        device=encoder_graph.x.device,
    )


def _counts_matrix(
    encoder_graph: Any,
    batch_size: int,
    num_nodes: int,
) -> Tensor:
    """
    Return the number of observed events for each [trajectory, bus].

    The SimBench temporal graph stores this information in ``y``.
    """

    if not hasattr(encoder_graph, "y"):
        raise ValueError(
            "The encoder graph must contain y with one event count per bus"
        )

    counts = encoder_graph.y.long().reshape(-1)
    expected = batch_size * num_nodes

    if counts.numel() != expected:
        raise ValueError(
            "encoder_graph.y must contain one count per trajectory and bus; "
            f"expected {expected}, got {counts.numel()}"
        )

    counts = counts.reshape(batch_size, num_nodes)

    if torch.any(counts < 1):
        bad = torch.nonzero(counts < 1, as_tuple=False)[0].tolist()
        raise ValueError(
            "Every bus must have at least one real observation; "
            f"first empty [trajectory, bus] is {bad}"
        )

    return counts


def _extract_sparse_bus_sequences(
    encoder_graph: Any,
    batch_size: int,
    num_nodes: int,
) -> Tuple[List[List[Tensor]], List[List[Tensor]], Tensor]:
    """
    Extract observed values and times without constructing dense imputations.

    Returns
    -------
    values:
        Nested ``values[trajectory][bus]`` tensors with shape [events, F].
    times:
        Nested ``times[trajectory][bus]`` tensors with shape [events].
    counts:
        Integer tensor [B, N].
    """

    x = encoder_graph.x
    pos = encoder_graph.pos.reshape(-1).to(dtype=x.dtype)
    ptr = _event_graph_ptr(encoder_graph)
    counts = _counts_matrix(encoder_graph, batch_size, num_nodes)

    if ptr.numel() != batch_size + 1:
        raise ValueError(
            f"Encoder graph contains {ptr.numel() - 1} graph segments, "
            f"but batch size is {batch_size}"
        )

    all_values: List[List[Tensor]] = []
    all_times: List[List[Tensor]] = []

    for trajectory in range(batch_size):
        graph_start = int(ptr[trajectory].item())
        graph_stop = int(ptr[trajectory + 1].item())
        graph_event_count = graph_stop - graph_start

        expected_events = int(counts[trajectory].sum().item())
        if graph_event_count != expected_events:
            raise ValueError(
                "Temporal graph event count does not match encoder_graph.y: "
                f"trajectory={trajectory}, graph={graph_event_count}, "
                f"counts={expected_events}"
            )

        trajectory_values: List[Tensor] = []
        trajectory_times: List[Tensor] = []
        offset = graph_start

        for bus in range(num_nodes):
            count = int(counts[trajectory, bus].item())
            stop = offset + count

            bus_values = x[offset:stop]
            bus_times = pos[offset:stop]

            if bus_values.shape[0] != count:
                raise ValueError(
                    f"Could not extract all observations for trajectory "
                    f"{trajectory}, bus {bus}"
                )

            if bus_times.numel() > 1 and torch.any(
                bus_times[1:] < bus_times[:-1]
            ):
                order = torch.argsort(bus_times, stable=True)
                bus_times = bus_times[order]
                bus_values = bus_values[order]

            trajectory_values.append(bus_values)
            trajectory_times.append(bus_times)
            offset = stop

        if offset != graph_stop:
            raise RuntimeError(
                "Internal sparse-event extraction did not consume the graph"
            )

        all_values.append(trajectory_values)
        all_times.append(trajectory_times)

    return all_values, all_times, counts


def _validate_target_shapes(batch: Any) -> Tuple[int, int, int, int]:
    target_values = batch.target_values
    target_times = batch.target_times
    target_mask = batch.target_mask

    if target_values.ndim != 4:
        raise ValueError(
            "target_values must have shape [B, N, T, F]; "
            f"got {tuple(target_values.shape)}"
        )

    batch_size, num_nodes, num_times, input_dim = target_values.shape

    if target_times.shape != (batch_size, num_times):
        raise ValueError(
            "target_times must have shape [B, T]; "
            f"expected {(batch_size, num_times)}, "
            f"got {tuple(target_times.shape)}"
        )

    if target_mask.shape != target_values.shape:
        raise ValueError(
            "target_mask must have the same shape as target_values; "
            f"got {tuple(target_mask.shape)} and "
            f"{tuple(target_values.shape)}"
        )

    return batch_size, num_nodes, num_times, input_dim


def _expand_target_mask(target_mask: Tensor, num_samples: int) -> Tensor:
    return target_mask.to(dtype=torch.bool).unsqueeze(0).expand(
        num_samples, *target_mask.shape
    )


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_float = mask.to(dtype=values.dtype)
    denominator = mask_float.sum().clamp_min(1.0)
    return (values * mask_float).sum() / denominator


class PersistenceBaseline(nn.Module):
    """
    Non-learned persistence baseline operating directly on sparse events.

    No imputed value is ever inserted into the encoder input. Predictions are
    calculated from the original sparse events when ``forward`` is called.

    Parameters
    ----------
    task:
        ``interpolation`` or ``extrapolation``.
    """

    def __init__(self, task: str) -> None:
        super().__init__()

        if task not in {"interpolation", "extrapolation"}:
            raise ValueError(
                "task must be 'interpolation' or 'extrapolation'; "
                f"got {task!r}"
            )

        self.task = task
        self.input_dim: Optional[int] = None
        self.latent_dim = 0

    @property
    def is_learned_model(self) -> bool:
        return False

    def predict(self, batch: Any) -> Tensor:
        """
        Generate predictions with shape [B, N, T, F].
        """

        if not _is_powergrid_batch(batch):
            raise TypeError(
                "PersistenceBaseline expects a PowerGridBatch-compatible "
                "object"
            )

        batch_size, num_nodes, num_targets, input_dim = (
            _validate_target_shapes(batch)
        )
        self.input_dim = input_dim

        values, times, _ = _extract_sparse_bus_sequences(
            batch.encoder_graph,
            batch_size=batch_size,
            num_nodes=num_nodes,
        )

        predictions = torch.empty_like(batch.target_values)

        for trajectory in range(batch_size):
            target_times = batch.target_times[trajectory].to(
                device=batch.target_values.device,
                dtype=batch.target_values.dtype,
            )

            for bus in range(num_nodes):
                observed_values = values[trajectory][bus].to(
                    device=batch.target_values.device,
                    dtype=batch.target_values.dtype,
                )
                observed_times = times[trajectory][bus].to(
                    device=batch.target_values.device,
                    dtype=batch.target_values.dtype,
                )

                if self.task == "extrapolation":
                    # The sparse context events are sorted. Repeating the last
                    # event does not create or consume a forward-filled input.
                    latest_value = observed_values[-1]
                    predictions[trajectory, bus] = latest_value.unsqueeze(
                        0
                    ).expand(num_targets, input_dim)
                    continue

                # searchsorted returns the insertion point to the right of
                # observations at the exact target timestamp. Subtracting one
                # therefore selects the latest observation at or before target.
                latest_indices = torch.searchsorted(
                    observed_times.contiguous(),
                    target_times.contiguous(),
                    right=True,
                ) - 1

                # For targets before the first observed event, use the first
                # available real observation.
                latest_indices = latest_indices.clamp(
                    min=0,
                    max=observed_values.shape[0] - 1,
                )

                predictions[trajectory, bus] = observed_values[
                    latest_indices
                ]

        return predictions

    def forward(
        self,
        batch: Any,
        n_traj_samples: int = 1,
        **_: Any,
    ) -> Tensor:
        """
        Return [1, B, N, T, F].

        ``n_traj_samples`` is accepted for interface compatibility but
        persistence is deterministic and always returns one sample.
        """

        del n_traj_samples
        return self.predict(batch).unsqueeze(0)

    def compute_all_losses(
        self,
        batch: Any,
        *_: Any,
        **__: Any,
    ) -> Dict[str, Any]:
        """
        Return metric keys compatible with the learned-model runner.

        Persistence has no ELBO or KL term, so its optimization loss is MSE.
        """

        prediction = self.predict(batch)
        mask = batch.training_loss_mask.to(dtype=torch.bool)

        squared_error = (prediction - batch.target_values).square()
        absolute_error = (prediction - batch.target_values).abs()

        mse = _masked_mean(squared_error, mask)
        mae = _masked_mean(absolute_error, mask)

        return {
            "loss": mse,
            "likelihood": float("nan"),
            "mse": mse.detach().item(),
            "mae": mae.detach().item(),
            "kl_first_p": 0.0,
            "std_first_p": 0.0,
            "predictions": prediction.unsqueeze(0),
            "all_extra_info": {
                "baseline": "persistence",
                "task": self.task,
            },
        }


class IndependentSparseEncoder(nn.Module):
    """
    Graph-free recognition encoder for independent sparse bus trajectories.

    Every bus is encoded as a separate variable-length sequence. The physical
    graph and neighboring buses are never read.

    Inputs at each observed event are:

        [normalized bus features, relative timestamp, elapsed time since
         previous observation]

    A shared GRU is applied independently to all bus sequences.
    """

    def __init__(
        self,
        input_dim: int,
        recognition_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        minimum_std: float = 1e-4,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if recognition_dim < 1:
            raise ValueError("recognition_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if minimum_std <= 0.0:
            raise ValueError("minimum_std must be positive")

        self.input_dim = int(input_dim)
        self.recognition_dim = int(recognition_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.minimum_std = float(minimum_std)

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim + 2, recognition_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=recognition_dim,
            hidden_size=recognition_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

        self.posterior_mean = nn.Linear(recognition_dim, latent_dim)
        self.posterior_raw_scale = nn.Linear(
            recognition_dim, latent_dim
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

        # Start with a moderate posterior standard deviation.
        nn.init.constant_(self.posterior_raw_scale.bias, -2.0)

    def forward(
        self,
        encoder_graph: Any,
        batch_size: int,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor]:
        values, times, counts = _extract_sparse_bus_sequences(
            encoder_graph,
            batch_size=batch_size,
            num_nodes=num_nodes,
        )

        sequence_count = batch_size * num_nodes
        maximum_length = int(counts.max().item())
        device = encoder_graph.x.device
        dtype = encoder_graph.x.dtype

        padded = torch.zeros(
            sequence_count,
            maximum_length,
            self.input_dim + 2,
            device=device,
            dtype=dtype,
        )
        lengths = torch.empty(
            sequence_count,
            device=device,
            dtype=torch.long,
        )

        sequence_index = 0
        for trajectory in range(batch_size):
            for bus in range(num_nodes):
                bus_values = values[trajectory][bus]
                bus_times = times[trajectory][bus].to(
                    device=device, dtype=dtype
                )
                length = int(bus_values.shape[0])

                if bus_values.shape[-1] != self.input_dim:
                    raise ValueError(
                        "Encoder event feature dimension does not match "
                        f"input_dim: {bus_values.shape[-1]} != "
                        f"{self.input_dim}"
                    )

                elapsed = torch.zeros_like(bus_times)
                if length > 1:
                    elapsed[1:] = bus_times[1:] - bus_times[:-1]

                event_input = torch.cat(
                    [
                        bus_values,
                        bus_times.unsqueeze(-1),
                        elapsed.unsqueeze(-1),
                    ],
                    dim=-1,
                )

                padded[sequence_index, :length] = event_input
                lengths[sequence_index] = length
                sequence_index += 1

        projected = self.input_projection(padded)

        # enforce_sorted=False keeps the stable trajectory/bus ordering needed
        # to reshape the posterior back to [B, N, D].
        packed = pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        # Deterministic multi-layer cuDNN GRU teardown exits 127 on the
        # supported Windows CUDA stack. The native PyTorch CUDA path is
        # deterministic and preserves gradients without that process failure.
        if projected.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                _, final_hidden = self.gru(packed)
        else:
            _, final_hidden = self.gru(packed)
        representation = final_hidden[-1]

        posterior_mean = self.posterior_mean(representation)
        posterior_std = F.softplus(
            self.posterior_raw_scale(representation)
        ) + self.minimum_std

        posterior_mean = posterior_mean.reshape(
            batch_size, num_nodes, self.latent_dim
        )
        posterior_std = posterior_std.reshape(
            batch_size, num_nodes, self.latent_dim
        )

        return posterior_mean, posterior_std


class NodeWiseODEFunc(nn.Module):
    """
    Shared graph-free ODE function applied independently to every bus.

    No physical-edge argument is accepted or stored. Consequently, changing,
    permuting, or removing physical edges cannot change this function's output.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.nfe = 0

        layers: List[nn.Module] = []

        if num_layers == 1:
            layers.append(nn.Linear(state_dim, state_dim))
        else:
            layers.extend(
                [
                    nn.Linear(state_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )

            for _ in range(num_layers - 2):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                    ]
                )

            layers.append(nn.Linear(hidden_dim, state_dim))

        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def reset_nfe(self) -> None:
        self.nfe = 0

    def forward(self, time: Tensor, state: Tensor) -> Tensor:
        del time
        self.nfe += 1

        original_shape = state.shape
        flattened = state.reshape(-1, original_shape[-1])
        derivative = self.network(flattened)
        return derivative.reshape(original_shape)


class IndependentLatentODE(VAE_Baseline):
    """
    Graph-free Latent ODE baseline for sparse per-bus trajectories.

    Parameters
    ----------
    input_dim:
        Number of normalized bus-state features.
    latent_dim:
        Per-bus latent state dimension decoded into bus features.
    recognition_dim:
        Hidden dimension of the independent sparse-trajectory encoder.
    ode_hidden_dim:
        Hidden dimension of the shared node-wise ODE.
    encoder_layers:
        Number of GRU layers.
    ode_layers:
        Number of layers in the node-wise ODE network.
    augment_dim:
        Optional augmented ODE dimensions. Only the first ``latent_dim``
        dimensions are passed to the decoder.
    solver:
        torchdiffeq solver name.
    rtol, atol:
        ODE solver tolerances.
    dropout:
        Encoder dropout. Dropout is deliberately not placed inside the
        adaptive ODE right-hand side because repeated evaluations at identical
        ``(time, state)`` must be deterministic.
    z0_prior:
        Optional prior distribution. If omitted, a standard normal is used.
    obsrv_std:
        Gaussian observation standard deviation.
    device:
        Initial model device.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        recognition_dim: int,
        ode_hidden_dim: int,
        *,
        encoder_layers: int = 1,
        ode_layers: int = 2,
        augment_dim: int = 0,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        dropout: float = 0.0,
        z0_prior: Optional[Any] = None,
        obsrv_std: float = 0.01,
        device: Union[str, torch.device] = "cpu",
        minimum_posterior_std: float = 1e-4,
    ) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if augment_dim < 0:
            raise ValueError("augment_dim must be non-negative")
        if rtol <= 0.0 or atol <= 0.0:
            raise ValueError("rtol and atol must be positive")
        if obsrv_std <= 0.0:
            raise ValueError("obsrv_std must be positive")

        model_device = torch.device(device)

        if z0_prior is None:
            prior_mean = torch.zeros(
                1, latent_dim, device=model_device
            )
            prior_std = torch.ones(
                1, latent_dim, device=model_device
            )
            z0_prior = Normal(prior_mean, prior_std)

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            z0_prior=z0_prior,
            device=model_device,
            obsrv_std=obsrv_std,
        )

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.recognition_dim = int(recognition_dim)
        self.ode_hidden_dim = int(ode_hidden_dim)
        self.encoder_layers = int(encoder_layers)
        self.ode_layers = int(ode_layers)
        self.augment_dim = int(augment_dim)
        self.solver = str(solver)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.dropout = float(dropout)
        self.minimum_posterior_std = float(minimum_posterior_std)

        self.encoder_z0 = IndependentSparseEncoder(
            input_dim=input_dim,
            recognition_dim=recognition_dim,
            latent_dim=latent_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            minimum_std=minimum_posterior_std,
        )

        ode_state_dim = latent_dim + augment_dim
        self.ode_func = NodeWiseODEFunc(
            state_dim=ode_state_dim,
            hidden_dim=ode_hidden_dim,
            num_layers=ode_layers,
        )

        # Reuse the same linear decoder implementation as LG-ODE.
        self.decoder = Decoder(
            latent_dim=latent_dim,
            input_dim=input_dim,
        )

        # Register a device-aware observation scale. The original base class
        # stores obsrv_std as a plain tensor; this buffer is used here.
        if hasattr(self, "obsrv_std"):
            del self.obsrv_std
        self.register_buffer(
            "obsrv_std",
            torch.tensor(float(obsrv_std), dtype=torch.float32),
        )

        self.to(model_device)

    @property
    def is_learned_model(self) -> bool:
        return True

    def _sample_posterior(
        self,
        posterior_mean: Tensor,
        posterior_std: Tensor,
        n_traj_samples: int,
        sample: bool,
    ) -> Tensor:
        if n_traj_samples < 1:
            raise ValueError("n_traj_samples must be at least one")

        expanded_mean = posterior_mean.unsqueeze(0).expand(
            n_traj_samples, *posterior_mean.shape
        )
        expanded_std = posterior_std.unsqueeze(0).expand_as(
            expanded_mean
        )

        if sample:
            noise = torch.randn_like(expanded_mean)
            return expanded_mean + noise * expanded_std

        return expanded_mean

    def _augment_initial_state(self, latent_state: Tensor) -> Tensor:
        if self.augment_dim == 0:
            return latent_state

        augmentation = torch.zeros(
            *latent_state.shape[:-1],
            self.augment_dim,
            device=latent_state.device,
            dtype=latent_state.dtype,
        )
        return torch.cat([latent_state, augmentation], dim=-1)

    def _solve_one_time_grid(
        self,
        initial_state: Tensor,
        target_times: Tensor,
    ) -> Tensor:
        """
        Solve one trajectory time grid.

        Parameters
        ----------
        initial_state:
            [S, N, latent_dim + augment_dim]
        target_times:
            [T]

        Returns
        -------
        Tensor:
            [S, N, T, latent_dim + augment_dim]
        """

        if target_times.ndim != 1:
            raise ValueError("target_times must be one-dimensional")

        if target_times.numel() == 0:
            raise ValueError("target_times cannot be empty")

        if not torch.isfinite(target_times).all():
            raise ValueError("target_times contains non-finite values")

        if target_times.numel() > 1 and torch.any(
            target_times[1:] <= target_times[:-1]
        ):
            raise ValueError(
                "target_times must be strictly increasing"
            )

        # The inferred state is defined at t=0. Extrapolation targets usually
        # begin after zero, so prepend zero for the solve and remove it later.
        prepend_zero = not torch.isclose(
            target_times[0],
            torch.zeros(
                (),
                device=target_times.device,
                dtype=target_times.dtype,
            ),
            rtol=0.0,
            atol=1e-8,
        )

        if target_times[0] < -1e-8:
            raise ValueError(
                "Target times must be non-negative relative to the latent "
                "initial-state time"
            )

        if prepend_zero:
            integration_times = torch.cat(
                [target_times.new_zeros(1), target_times],
                dim=0,
            )
        else:
            integration_times = target_times

        self.ode_func.reset_nfe()

        solution = odeint(
            self.ode_func,
            initial_state,
            integration_times,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        # odeint: [T_integrate, S, N, D]

        if prepend_zero:
            solution = solution[1:]

        return solution.permute(1, 2, 0, 3).contiguous()

    def _solve_batch(
        self,
        initial_state: Tensor,
        target_times: Tensor,
    ) -> Tensor:
        """
        Solve all trajectories without assuming identical timestamp grids.

        Parameters
        ----------
        initial_state:
            [S, B, N, D]
        target_times:
            [B, T]

        Returns
        -------
        Tensor:
            [S, B, N, T, D]
        """

        if initial_state.ndim != 4:
            raise ValueError(
                "initial_state must have shape [S, B, N, D]"
            )
        if target_times.ndim != 2:
            raise ValueError(
                "target_times must have shape [B, T]"
            )

        _, batch_size, _, _ = initial_state.shape

        if target_times.shape[0] != batch_size:
            raise ValueError(
                "initial_state and target_times batch sizes differ"
            )

        # Use one ODE solve when all trajectories share the same grid.
        shared_grid = torch.allclose(
            target_times,
            target_times[0:1].expand_as(target_times),
            rtol=0.0,
            atol=1e-7,
        )

        if shared_grid:
            samples, trajectories, nodes, state_dim = (
                initial_state.shape
            )
            merged = initial_state.reshape(
                samples * trajectories, nodes, state_dim
            )
            solved = self._solve_one_time_grid(
                merged,
                target_times[0],
            )
            return solved.reshape(
                samples,
                trajectories,
                nodes,
                target_times.shape[1],
                state_dim,
            )

        solved_trajectories: List[Tensor] = []
        for trajectory in range(batch_size):
            solved = self._solve_one_time_grid(
                initial_state[:, trajectory],
                target_times[trajectory],
            )
            solved_trajectories.append(solved)

        return torch.stack(solved_trajectories, dim=1)

    def reconstruct(
        self,
        batch: Any,
        *,
        n_traj_samples: int = 1,
        sample: Optional[bool] = None,
    ) -> IndependentLatentODEOutput:
        """
        Infer independent latent states and reconstruct/predict all targets.
        """

        if not _is_powergrid_batch(batch):
            raise TypeError(
                "IndependentLatentODE.reconstruct expects a "
                "PowerGridBatch-compatible object"
            )

        batch_size, num_nodes, _, input_dim = _validate_target_shapes(
            batch
        )

        if input_dim != self.input_dim:
            raise ValueError(
                f"Batch input dimension {input_dim} does not match model "
                f"input dimension {self.input_dim}"
            )

        # Crucially, no physical_graph value is read anywhere in this method.
        posterior_mean, posterior_std = self.encoder_z0(
            batch.encoder_graph,
            batch_size=batch_size,
            num_nodes=num_nodes,
        )

        if sample is None:
            sample = self.training

        latent_initial_state = self._sample_posterior(
            posterior_mean,
            posterior_std,
            n_traj_samples=n_traj_samples,
            sample=bool(sample),
        )

        augmented_initial_state = self._augment_initial_state(
            latent_initial_state
        )

        target_times = batch.target_times.to(
            device=augmented_initial_state.device,
            dtype=augmented_initial_state.dtype,
        )

        augmented_trajectory = self._solve_batch(
            augmented_initial_state,
            target_times,
        )

        latent_trajectory = augmented_trajectory[..., : self.latent_dim]
        predictions = self.decoder(latent_trajectory)

        return IndependentLatentODEOutput(
            predictions=predictions,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            latent_initial_state=latent_initial_state,
            latent_trajectory=latent_trajectory,
            target_times=target_times,
        )

    def forward(
        self,
        batch: Any,
        n_traj_samples: int = 1,
        sample: Optional[bool] = None,
        **_: Any,
    ) -> Tensor:
        """
        Return predictions with shape [S, B, N, T, F].
        """

        return self.reconstruct(
            batch,
            n_traj_samples=n_traj_samples,
            sample=sample,
        ).predictions

    def get_reconstruction(
        self,
        batch_en: Any,
        batch_de: Optional[Any] = None,
        batch_g: Optional[Any] = None,
        n_traj_samples: int = 1,
        run_backwards: bool = True,
    ) -> Tuple[Tensor, Dict[str, Any], None]:
        """
        Compatibility method mirroring ``LatentGraphODE.get_reconstruction``.

        Preferred usage is:

            predictions, info, weights = model.get_reconstruction(batch)

        For the new batch interface, predictions have shape [S, B, N, T, F].
        The physical graph argument is ignored by design.
        """

        del batch_g, run_backwards

        if _is_powergrid_batch(batch_en):
            batch = batch_en
        elif batch_de is not None and _is_powergrid_batch(batch_de):
            batch = batch_de
        else:
            raise TypeError(
                "IndependentLatentODE.get_reconstruction requires the new "
                "PowerGridBatch interface. Pass the complete batch as the "
                "first argument."
            )

        output = self.reconstruct(
            batch,
            n_traj_samples=n_traj_samples,
            sample=None,
        )

        return (
            output.predictions,
            output.as_extra_info(),
            None,
        )

    def _prior_kl(
        self,
        posterior_mean: Tensor,
        posterior_std: Tensor,
    ) -> Tensor:
        """
        Return per-bus KL values with shape [B, N, latent_dim].
        """

        posterior = Normal(posterior_mean, posterior_std)

        prior = self.z0_prior
        try:
            prior_mean = prior.loc.to(
                device=posterior_mean.device,
                dtype=posterior_mean.dtype,
            )
            prior_std = prior.scale.to(
                device=posterior_mean.device,
                dtype=posterior_mean.dtype,
            )
            prior_on_device = Normal(prior_mean, prior_std)
            return kl_divergence(posterior, prior_on_device)
        except (AttributeError, NotImplementedError):
            # The experiment uses a Gaussian prior, but retain a clear error
            # for unsupported custom distributions.
            try:
                return kl_divergence(posterior, prior)
            except NotImplementedError as exc:
                raise TypeError(
                    "IndependentLatentODE requires a prior with an analytic "
                    "KL divergence from a Normal posterior"
                ) from exc

    def compute_all_losses(
        self,
        batch: Any,
        batch_dict_decoder: Optional[Any] = None,
        batch_dict_graph: Optional[Any] = None,
        n_traj_samples: int = 1,
        kl_coef: float = 1.0,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Compute the LG-ODE-compatible variational objective and metrics.

        Reconstruction likelihood follows the original repository's effective
        Gaussian objective: squared error divided by ``2 * obsrv_std^2``,
        averaged over valid target times and features for each bus.

        The returned primary MSE is evaluated over complete target values.
        """

        del batch_dict_graph

        if not _is_powergrid_batch(batch):
            if (
                batch_dict_decoder is not None
                and _is_powergrid_batch(batch_dict_decoder)
            ):
                batch = batch_dict_decoder
            else:
                raise TypeError(
                    "compute_all_losses expects a PowerGridBatch-compatible "
                    "object"
                )

        output = self.reconstruct(
            batch,
            n_traj_samples=n_traj_samples,
            sample=True,
        )

        predictions = output.predictions
        truth = batch.target_values.to(
            device=predictions.device,
            dtype=predictions.dtype,
        )
        target_mask = batch.training_loss_mask.to(
            device=predictions.device,
            dtype=torch.bool,
        )

        truth_expanded = truth.unsqueeze(0).expand_as(predictions)
        mask_expanded = _expand_target_mask(
            target_mask,
            predictions.shape[0],
        )

        squared_error = (predictions - truth_expanded).square()
        absolute_error = (predictions - truth_expanded).abs()

        # [S, B, N, T, F] -> likelihood averaged over target T and F,
        # preserving sample, trajectory and bus axes.
        mask_float = mask_expanded.to(dtype=predictions.dtype)
        valid_per_bus = mask_float.sum(dim=(-2, -1)).clamp_min(1.0)

        gaussian_negative_log_likelihood = (
            squared_error / (2.0 * self.obsrv_std.square())
        )
        reconstruction_nll_per_bus = (
            gaussian_negative_log_likelihood * mask_float
        ).sum(dim=(-2, -1)) / valid_per_bus

        reconstruction_log_likelihood_per_sample = (
            -reconstruction_nll_per_bus.mean(dim=(-2, -1))
        )

        kl_values = self._prior_kl(
            output.posterior_mean,
            output.posterior_std,
        )
        kl_per_batch = kl_values.mean(dim=(-2, -1))
        mean_kl = kl_per_batch.mean()

        # The posterior is shared across Monte Carlo samples, matching the
        # standard latent-ODE ELBO.
        sample_elbo = (
            reconstruction_log_likelihood_per_sample
            - float(kl_coef) * mean_kl
        )

        # Stable negative Monte Carlo ELBO. Subtract log(S) so changing the
        # number of posterior samples does not introduce a constant shift.
        loss = -(
            torch.logsumexp(sample_elbo, dim=0)
            - math.log(predictions.shape[0])
        )

        mse = _masked_mean(squared_error, mask_expanded)
        mae = _masked_mean(absolute_error, mask_expanded)

        mean_prediction = predictions.mean(dim=0)
        deterministic_squared_error = (
            mean_prediction - truth
        ).square()
        deterministic_absolute_error = (
            mean_prediction - truth
        ).abs()

        deterministic_mse = _masked_mean(
            deterministic_squared_error,
            target_mask,
        )
        deterministic_mae = _masked_mean(
            deterministic_absolute_error,
            target_mask,
        )

        return {
            "loss": loss,
            "likelihood": (
                reconstruction_log_likelihood_per_sample.mean()
                .detach()
                .item()
            ),
            "mse": deterministic_mse.detach().item(),
            "mae": deterministic_mae.detach().item(),
            "sample_mse": mse.detach().item(),
            "sample_mae": mae.detach().item(),
            "kl_first_p": mean_kl.detach().item(),
            "std_first_p": (
                output.posterior_std.mean().detach().item()
            ),
            "predictions": predictions,
            "mean_prediction": mean_prediction,
            "all_extra_info": output.as_extra_info(),
        }


__all__ = [
    "IndependentLatentODE",
    "IndependentLatentODEOutput",
    "IndependentSparseEncoder",
    "NodeWiseODEFunc",
    "PersistenceBaseline",
]
