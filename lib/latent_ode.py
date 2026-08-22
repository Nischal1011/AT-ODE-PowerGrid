# lib/latent_ode.py

"""
High-level latent ODE model wrapper.

This module coordinates:

* sparse-trajectory encoding;
* posterior sampling;
* conditional graph/context routing to the ODE solver;
* decoding;
* transport diagnostic reporting.

It deliberately does not implement transport mathematics. AT-ODE transport
weights are constructed by DiffeqSolver and lib.attention_transport.

Interpolation note
------------------
LG-ODE-style interpolation may encode observations from across the complete
reconstruction interval. It is therefore not necessarily a strictly causal
prediction task. This wrapper does not alter the observation set supplied by
the data adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from lib.base_models import VAE_Baseline
import lib.utils as utils


SUPPORTED_MODEL_TYPES = {
    "latentode",
    "lgode",
    "atode",
}


def _normalize_model_type(model_type: str) -> str:
    normalized = (
        str(model_type)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
    )

    aliases = {
        "latentode": "latentode",
        "independentlatentode": "latentode",
        "lgode": "lgode",
        "atode": "atode",
        "attentiontransportode": "atode",
    }

    if normalized not in aliases:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. "
            f"Supported model types are "
            f"{sorted(SUPPORTED_MODEL_TYPES)}"
        )

    return aliases[normalized]


def _get_field(
    value: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """Read a field from either a mapping or an attribute-based object."""

    for name in names:
        if isinstance(value, Mapping):
            if name in value:
                result = value[name]
                if result is not None:
                    return result
        elif value is not None and hasattr(value, name):
            result = getattr(value, name)
            if result is not None:
                return result

    return default


def _detach_nested(value: Any) -> Any:
    """Detach tensors in nested diagnostic structures."""

    if isinstance(value, Tensor):
        return value.detach()

    if isinstance(value, Mapping):
        return {
            key: _detach_nested(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _detach_nested(item)
            for item in value
        )

    if isinstance(value, list):
        return [
            _detach_nested(item)
            for item in value
        ]

    return value


class LatentGraphODE(VAE_Baseline):
    """
    High-level wrapper for graph-free Latent ODE, LG-ODE, and AT-ODE.

    Parameters
    ----------
    input_dim:
        Number of observed features per bus.
    latent_dim:
        Latent state dimension per bus.
    encoder_z0:
        Configured recognition encoder. LG-ODE and AT-ODE must receive
        architecturally identical GTrans encoders. A graph-free Latent ODE
        must receive an independent per-bus encoder or an encoder graph
        containing only within-bus temporal edges.
    decoder:
        Shared latent-to-observation decoder.
    diffeq_solver:
        Configured DiffeqSolver.
    z0_prior:
        Prior over the latent initial state.
    device:
        Model device.
    obsrv_std:
        Observation likelihood standard deviation.
    model_type:
        ``latentode``, ``lgode`` or ``atode``.
    require_explicit_observation_time:
        If True, AT-ODE requires ``batch_en.latest_observation_time``.
        If False, a migration-only fallback reconstructs latest times from
        ``batch_en.pos`` and ``batch_en.y``.
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_z0,
        decoder,
        diffeq_solver,
        z0_prior,
        device,
        obsrv_std=None,
        model_type=None,
        require_explicit_observation_time=True,
    ):
        if obsrv_std is None:
            obsrv_std = 0.01

        super(LatentGraphODE, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            z0_prior=z0_prior,
            device=device,
            obsrv_std=obsrv_std,
        )

        if model_type is None:
            model_type = getattr(
                diffeq_solver,
                "model_type",
                "lgode",
            )

        self.model_type = _normalize_model_type(
            model_type
        )
        self.require_explicit_observation_time = bool(
            require_explicit_observation_time
        )

        solver_model_type = getattr(
            diffeq_solver,
            "model_type",
            self.model_type,
        )
        solver_model_type = _normalize_model_type(
            solver_model_type
        )

        if solver_model_type != self.model_type:
            raise ValueError(
                "LatentGraphODE model_type does not match the solver: "
                f"model={self.model_type!r}, "
                f"solver={solver_model_type!r}"
            )

        expected_graph_usage = self.model_type in {
            "lgode",
            "atode",
        }
        solver_graph_usage = bool(
            getattr(
                diffeq_solver,
                "uses_graph",
                expected_graph_usage,
            )
        )

        if solver_graph_usage != expected_graph_usage:
            raise ValueError(
                "Solver graph configuration does not match model_type: "
                f"model_type={self.model_type!r}, "
                f"solver.uses_graph={solver_graph_usage}"
            )

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.latent_dim = int(latent_dim)

        if self.latent_dim < 1:
            raise ValueError(
                f"latent_dim must be positive; got {latent_dim}"
            )

    def _run_encoder(
        self,
        batch_en: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run the configured recognition encoder.

        LG-ODE and AT-ODE use this exact same call path. No transport-specific
        values are supplied to the encoder.
        """

        required_fields = (
            "x",
            "edge_attr",
            "edge_index",
            "pos",
            "edge_same",
            "batch",
            "y",
        )

        missing = [
            name
            for name in required_fields
            if not hasattr(batch_en, name)
        ]
        if missing:
            raise ValueError(
                "Encoder batch is missing required LG-ODE fields: "
                f"{missing}"
            )

        first_point_mu, first_point_std = self.encoder_z0(
            batch_en.x,
            batch_en.edge_attr,
            batch_en.edge_index,
            batch_en.pos,
            batch_en.edge_same,
            batch_en.batch,
            batch_en.y,
        )

        return first_point_mu, first_point_std

    def _infer_expected_encoded_nodes(
        self,
        batch_en: Any,
        batch_de: Any,
    ) -> int:
        """
        Determine the expected number of encoded trajectory-bus objects.
        """

        num_nodes = int(
            getattr(
                self.diffeq_solver,
                "num_atoms",
                0,
            )
        )
        if num_nodes < 1:
            raise ValueError(
                "The ODE solver must expose a positive num_atoms value"
            )

        if hasattr(batch_en, "num_graphs"):
            batch_size = int(batch_en.num_graphs)
            if batch_size > 0:
                return batch_size * num_nodes

        if hasattr(batch_en, "ptr"):
            ptr = batch_en.ptr
            if isinstance(ptr, Tensor) and ptr.numel() >= 2:
                return int(ptr.numel() - 1) * num_nodes

        target_values = _get_field(
            batch_de,
            "target_values",
            "data",
            default=None,
        )

        if isinstance(target_values, Tensor):
            # New power-grid shape: [B, N, T, F].
            if (
                target_values.ndim == 4
                and target_values.shape[1] == num_nodes
            ):
                return (
                    int(target_values.shape[0])
                    * num_nodes
                )

            # Original flattened shape: [B*N, T, F].
            if target_values.ndim == 3:
                flattened = int(
                    target_values.shape[0]
                )
                if flattened % num_nodes == 0:
                    return flattened

        if hasattr(batch_en, "y"):
            counts = torch.as_tensor(
                batch_en.y
            ).reshape(-1)
            if counts.numel() % num_nodes == 0:
                return int(counts.numel())

        raise ValueError(
            "Could not infer B*N from the encoder and decoder batches"
        )

    def _validate_encoder_outputs(
        self,
        first_point_mu: Tensor,
        first_point_std: Tensor,
        expected_encoded_nodes: int,
    ) -> None:
        if not isinstance(first_point_mu, Tensor):
            raise TypeError(
                "The encoder posterior mean must be a tensor"
            )
        if not isinstance(first_point_std, Tensor):
            raise TypeError(
                "The encoder posterior standard deviation must be a tensor"
            )

        if first_point_mu.shape != first_point_std.shape:
            raise ValueError(
                "Posterior mean and standard deviation must have identical "
                f"shapes; got {tuple(first_point_mu.shape)} and "
                f"{tuple(first_point_std.shape)}"
            )

        if first_point_mu.ndim != 2:
            raise ValueError(
                "Encoder outputs must have shape [B*N, latent_dim]; "
                f"got {tuple(first_point_mu.shape)}"
            )

        if first_point_mu.shape[0] != expected_encoded_nodes:
            raise ValueError(
                "The number of encoded objects does not equal B*N: "
                f"expected {expected_encoded_nodes}, "
                f"got {first_point_mu.shape[0]}"
            )

        if first_point_mu.shape[1] != self.latent_dim:
            raise ValueError(
                "Encoder posterior dimension does not match latent_dim: "
                f"{first_point_mu.shape[1]} != {self.latent_dim}"
            )

        if not first_point_mu.is_floating_point():
            raise TypeError(
                "Posterior mean must have a floating-point dtype"
            )
        if not first_point_std.is_floating_point():
            raise TypeError(
                "Posterior standard deviation must have a floating-point "
                "dtype"
            )

        if not torch.isfinite(first_point_mu).all():
            raise ValueError(
                "Posterior mean contains NaN or infinity"
            )
        if not torch.isfinite(first_point_std).all():
            raise ValueError(
                "Posterior standard deviation contains NaN or infinity"
            )
        if torch.any(first_point_std <= 0):
            minimum = float(
                first_point_std.detach().min().item()
            )
            raise ValueError(
                "Posterior standard deviation must be strictly positive; "
                f"minimum value is {minimum}"
            )

    def _extract_time_grid(
        self,
        batch_de: Any,
    ) -> Tensor:
        time_steps = _get_field(
            batch_de,
            "time_steps",
            "target_times",
            default=None,
        )

        if time_steps is None:
            raise ValueError(
                "Decoder batch must provide time_steps or target_times"
            )

        if not isinstance(time_steps, Tensor):
            time_steps = torch.as_tensor(
                time_steps,
                dtype=torch.float32,
            )

        # New PowerGridBatch target times have shape [B, T]. The current ODE
        # solve has one shared grid, so verify that all trajectories agree.
        if time_steps.ndim == 2:
            if time_steps.shape[0] < 1:
                raise ValueError(
                    "target_times contains no trajectories"
                )

            reference = time_steps[0]
            if not torch.allclose(
                time_steps,
                reference.unsqueeze(0).expand_as(
                    time_steps
                ),
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError(
                    "LatentGraphODE currently requires a shared target-time "
                    "grid within each batch"
                )
            time_steps = reference

        if time_steps.ndim != 1:
            raise ValueError(
                "Prediction time grid must be one-dimensional or [B, T]; "
                f"got {tuple(time_steps.shape)}"
            )

        if time_steps.numel() == 0:
            raise ValueError(
                "Prediction time grid cannot be empty"
            )

        if not time_steps.is_floating_point():
            time_steps = time_steps.float()

        if not torch.isfinite(time_steps).all():
            raise ValueError(
                "Prediction time grid contains NaN or infinity"
            )

        if time_steps.numel() > 1:
            time_differences = (
                time_steps[1:] - time_steps[:-1]
            )
            if torch.any(time_differences < 0):
                raise ValueError(
                    "Prediction time grid must be monotonically "
                    "nondecreasing"
                )

        return time_steps

    def _sample_initial_state(
        self,
        first_point_mu: Tensor,
        first_point_std: Tensor,
        n_traj_samples: int,
    ) -> Tensor:
        if not isinstance(n_traj_samples, int):
            raise TypeError(
                "n_traj_samples must be an integer"
            )
        if n_traj_samples < 1:
            raise ValueError(
                "n_traj_samples must be at least one"
            )

        means_z0 = first_point_mu.unsqueeze(0).expand(
            n_traj_samples,
            *first_point_mu.shape,
        )
        sigmas_z0 = first_point_std.unsqueeze(0).expand(
            n_traj_samples,
            *first_point_std.shape,
        )

        first_point_enc = (
            utils.sample_standard_gaussian(
                means_z0,
                sigmas_z0,
            )
        )

        if first_point_enc.shape != means_z0.shape:
            raise RuntimeError(
                "Posterior sampler returned an invalid shape: "
                f"expected {tuple(means_z0.shape)}, "
                f"got {tuple(first_point_enc.shape)}"
            )

        if not torch.isfinite(first_point_enc).all():
            raise ValueError(
                "Sampled latent initial state contains NaN or infinity"
            )

        return first_point_enc

    def _fallback_latest_observation_time(
        self,
        batch_en: Any,
        expected_encoded_nodes: int,
    ) -> Tensor:
        """
        Migration-only fallback using grouped event positions.

        The production SimBench path should provide
        ``batch_en.latest_observation_time`` explicitly.
        """

        if not hasattr(batch_en, "pos"):
            raise ValueError(
                "Cannot reconstruct latest observation times without pos"
            )
        if not hasattr(batch_en, "y"):
            raise ValueError(
                "Cannot reconstruct latest observation times without y"
            )

        positions = torch.as_tensor(
            batch_en.pos
        ).reshape(-1)
        counts = torch.as_tensor(
            batch_en.y,
            device=positions.device,
            dtype=torch.long,
        ).reshape(-1)

        if counts.numel() != expected_encoded_nodes:
            raise ValueError(
                "batch_en.y does not contain one event count per encoded "
                f"object: expected {expected_encoded_nodes}, "
                f"got {counts.numel()}"
            )
        if torch.any(counts < 1):
            raise ValueError(
                "Every encoded bus must have at least one observation"
            )
        if int(counts.sum().item()) != positions.numel():
            raise ValueError(
                "sum(batch_en.y) does not match the number of event "
                "timestamps"
            )

        latest_times = []
        offset = 0

        for count_tensor in counts:
            count = int(count_tensor.item())
            stop = offset + count
            group_times = positions[offset:stop]

            if group_times.numel() == 0:
                raise ValueError(
                    "An encoded bus has no observation timestamps"
                )

            latest_times.append(
                group_times.max()
            )
            offset = stop

        return torch.stack(
            latest_times,
            dim=0,
        ).reshape(
            expected_encoded_nodes,
            1,
        )

    def _extract_latest_observation_time(
        self,
        batch_en: Any,
        expected_encoded_nodes: int,
    ) -> Tensor:
        latest_time = _get_field(
            batch_en,
            "latest_observation_time",
            "latest_valid_observation_time",
            default=None,
        )

        if latest_time is None:
            if self.require_explicit_observation_time:
                raise ValueError(
                    "AT-ODE requires explicit "
                    "batch_en.latest_observation_time metadata with shape "
                    "[B*N, 1]. The production data adapter must provide it."
                )

            latest_time = (
                self._fallback_latest_observation_time(
                    batch_en,
                    expected_encoded_nodes,
                )
            )

        if not isinstance(latest_time, Tensor):
            latest_time = torch.as_tensor(
                latest_time,
                device=batch_en.x.device,
                dtype=batch_en.x.dtype,
            )
        else:
            latest_time = latest_time.to(
                device=batch_en.x.device,
                dtype=batch_en.x.dtype,
            )

        # PyG concatenates an [N] graph attribute into [B*N]. Accept this
        # unambiguous representation and canonicalize it to [B*N, 1].
        if latest_time.ndim == 1:
            latest_time = latest_time.unsqueeze(-1)

        if latest_time.ndim == 3:
            # Some adapters retain [B, N, 1].
            latest_time = latest_time.reshape(
                -1,
                latest_time.shape[-1],
            )

        expected_shape = (
            expected_encoded_nodes,
            1,
        )
        if tuple(latest_time.shape) != expected_shape:
            raise ValueError(
                "latest_observation_time must have shape [B*N, 1]; "
                f"expected {expected_shape}, "
                f"got {tuple(latest_time.shape)}"
            )

        if not torch.isfinite(latest_time).all():
            raise ValueError(
                "latest_observation_time contains NaN or infinity"
            )

        return latest_time

    def _extract_graph(
        self,
        batch_g: Any,
    ) -> Any:
        """
        Extract candidate-pair edge labels from the graph batch.

        The solver expects binary labels [B, N*(N-1)] in deterministic
        candidate-pair order. Dataset/model adapters may supply this tensor
        directly or expose it as a graph attribute.
        """

        if batch_g is None:
            return None

        if isinstance(batch_g, Tensor):
            return batch_g

        graph = _get_field(
            batch_g,
            "candidate_edge_type",
            "candidate_edge_labels",
            "graph",
            default=None,
        )

        if graph is not None:
            return graph

        # ``edge_type`` is accepted only as a final compatibility path. The
        # power-grid adapter should preferably expose candidate_edge_type so it
        # is clear that these are complete directed candidate pairs rather
        # than a sparse physical edge list.
        graph = _get_field(
            batch_g,
            "edge_type",
            default=None,
        )
        if graph is not None:
            return graph

        raise ValueError(
            "Graph batch must provide candidate edge labels through a tensor, "
            "candidate_edge_type, candidate_edge_labels, graph, or edge_type"
        )

    def get_reconstruction(
        self,
        batch_en,
        batch_de,
        batch_g=None,
        n_traj_samples=1,
    ):
        """
        Reconstruct or forecast trajectories.

        Returns
        -------
        pred_x:
            Decoded predictions.
        all_extra_info:
            Posterior, latent trajectory, and optional transport diagnostics.
        temporal_weights:
            None, retained for compatibility with VAE_Baseline.
        """

        first_point_mu, first_point_std = (
            self._run_encoder(batch_en)
        )

        expected_encoded_nodes = (
            self._infer_expected_encoded_nodes(
                batch_en,
                batch_de,
            )
        )

        self._validate_encoder_outputs(
            first_point_mu,
            first_point_std,
            expected_encoded_nodes,
        )

        first_point_enc = (
            self._sample_initial_state(
                first_point_mu,
                first_point_std,
                n_traj_samples,
            )
        )

        time_steps_to_predict = (
            self._extract_time_grid(batch_de)
        ).to(
            device=first_point_enc.device,
            dtype=first_point_enc.dtype,
        )

        if not torch.isfinite(
            time_steps_to_predict
        ).all():
            raise ValueError(
                "Prediction time grid contains NaN or infinity after "
                "device conversion"
            )

        if self.model_type == "latentode":
            if batch_g is not None:
                raise ValueError(
                    "Graph-free Latent ODE must not receive batch_g"
                )

            sol_y = self.diffeq_solver(
                first_point_enc,
                time_steps_to_predict,
                graph=None,
            )

        elif self.model_type == "lgode":
            graph = self._extract_graph(batch_g)
            if graph is None:
                raise ValueError(
                    "LG-ODE requires a physical graph realization"
                )

            sol_y = self.diffeq_solver(
                first_point_enc,
                time_steps_to_predict,
                graph=graph,
            )

        elif self.model_type == "atode":
            graph = self._extract_graph(batch_g)
            if graph is None:
                raise ValueError(
                    "AT-ODE requires a physical graph realization"
                )

            latest_observation_time = (
                self._extract_latest_observation_time(
                    batch_en,
                    expected_encoded_nodes,
                )
            )

            observation_context = {
                "latest_observation_time": (
                    latest_observation_time
                ),
                "trajectory_id": _get_field(
                    batch_en,
                    "trajectory_id",
                    default=None,
                ),
            }

            sol_y = self.diffeq_solver(
                first_point_enc,
                time_steps_to_predict,
                graph=graph,
                latest_observation_time=(
                    latest_observation_time
                ),
                observation_context=observation_context,
            )

        else:  # Defensive guard against post-construction mutation.
            raise ValueError(
                f"Unsupported model_type: {self.model_type!r}"
            )

        if not isinstance(sol_y, Tensor):
            raise TypeError(
                "The differential-equation solver must return a tensor"
            )
        if sol_y.ndim != 4:
            raise ValueError(
                "Solver output must have shape [S, B*N, T, D]; "
                f"got {tuple(sol_y.shape)}"
            )
        if sol_y.shape[0] != n_traj_samples:
            raise ValueError(
                "Solver output sample dimension is incorrect: "
                f"{sol_y.shape[0]} != {n_traj_samples}"
            )
        if sol_y.shape[1] != expected_encoded_nodes:
            raise ValueError(
                "Solver output object dimension is incorrect: "
                f"{sol_y.shape[1]} != {expected_encoded_nodes}"
            )
        if sol_y.shape[2] != time_steps_to_predict.numel():
            raise ValueError(
                "Solver output time dimension is incorrect: "
                f"{sol_y.shape[2]} != "
                f"{time_steps_to_predict.numel()}"
            )
        if sol_y.shape[-1] != self.latent_dim:
            raise ValueError(
                "Solver output latent dimension is incorrect: "
                f"{sol_y.shape[-1]} != {self.latent_dim}"
            )
        if not torch.isfinite(sol_y).all():
            raise ValueError(
                "Latent ODE trajectory contains NaN or infinity"
            )

        pred_x = self.decoder(sol_y)

        if not isinstance(pred_x, Tensor):
            raise TypeError(
                "Decoder must return a tensor"
            )
        if not torch.isfinite(pred_x).all():
            raise ValueError(
                "Decoded prediction contains NaN or infinity"
            )

        if self.model_type == "atode":
            transport_diagnostics = _detach_nested(
                getattr(
                    self.diffeq_solver,
                    "last_transport_diagnostics",
                    {},
                )
            )
        else:
            transport_diagnostics = None

        all_extra_info: Dict[str, Any] = {
            "first_point": (
                first_point_mu.unsqueeze(0),
                first_point_std.unsqueeze(0),
                first_point_enc,
            ),
            "latent_traj": sol_y.detach(),
            "transport_diagnostics": (
                transport_diagnostics
            ),
            "model_type": self.model_type,
        }

        return pred_x, all_extra_info, None


__all__ = [
    "LatentGraphODE",
    "SUPPORTED_MODEL_TYPES",
]
