# lib/latent_ode.py

"""
High-level latent ODE model wrapper.

This module coordinates:

* sparse-trajectory encoding;
* posterior sampling during training;
* deterministic posterior-mean prediction during evaluation;
* conditional graph/context routing to the ODE solver;
* decoding;
* transport diagnostic reporting.

It deliberately does not implement transport mathematics. AT-ODE transport
weights are constructed by DiffeqSolver and lib.attention_transport.

Evaluation policy
-----------------
By default:

* model.train():
    Sample z0 from q(z0 | observations).

* model.eval():
    Use the posterior mean as z0.

This makes validation, checkpoint selection, and test evaluation deterministic.
The behavior can be overridden explicitly with ``sample_posterior``.

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
    """Normalize and validate a supported model-type name."""

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
            f"{sorted(SUPPORTED_MODEL_TYPES)}."
        )

    canonical = aliases[normalized]

    if canonical not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Normalized model_type {canonical!r} is not supported."
        )

    return canonical


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
        architecturally identical GTrans encoders.

    decoder:
        Shared latent-to-observation decoder.

    diffeq_solver:
        Configured DiffeqSolver.

    z0_prior:
        Prior over the latent initial state.

    device:
        Model device.

    obsrv_std:
        Observation-likelihood standard deviation.

    model_type:
        One of ``latentode``, ``lgode``, or ``atode``.

    Notes
    -----
    AT-ODE requires ``batch_en.latest_observation_time`` explicitly. This
    wrapper does not infer latest observation times from sparse event data.
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
                f"solver={solver_model_type!r}."
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
                f"solver.uses_graph={solver_graph_usage}."
            )

        if encoder_z0 is None:
            raise ValueError(
                "encoder_z0 cannot be None."
            )

        if decoder is None:
            raise ValueError(
                "decoder cannot be None."
            )

        if diffeq_solver is None:
            raise ValueError(
                "diffeq_solver cannot be None."
            )

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.latent_dim = int(latent_dim)

        if self.latent_dim < 1:
            raise ValueError(
                f"latent_dim must be positive; got {latent_dim}."
            )

    def _run_encoder(
        self,
        batch_en: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run the configured recognition encoder.

        LG-ODE and AT-ODE use this exact same call path. No AT-ODE transport
        values, latest-observation times, or sparse transport inputs are
        supplied to the encoder.
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
                f"{missing}."
            )

        # Shape from the original LG-ODE implementation:
        # first_point_mu:  [num_ball, 10]
        # first_point_std: [num_ball, 10]
        #
        # In batched power-grid experiments:
        # first_point_mu:  [B*N, latent_dim]
        # first_point_std: [B*N, latent_dim]
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
        """Determine the expected number of encoded trajectory-bus objects."""

        num_nodes = int(
            getattr(
                self.diffeq_solver,
                "num_atoms",
                0,
            )
        )

        if num_nodes < 1:
            raise ValueError(
                "The ODE solver must expose a positive num_atoms value."
            )

        if hasattr(batch_en, "num_graphs"):
            batch_size = int(batch_en.num_graphs)

            if batch_size > 0:
                return batch_size * num_nodes

        if hasattr(batch_en, "ptr"):
            ptr = batch_en.ptr

            if isinstance(ptr, Tensor) and ptr.numel() >= 2:
                batch_size = int(ptr.numel() - 1)

                if batch_size > 0:
                    return batch_size * num_nodes

        target_values = _get_field(
            batch_de,
            "target_values",
            "data",
            default=None,
        )

        if isinstance(target_values, Tensor):
            # Power-grid target shape: [B, N, T, F].
            if (
                target_values.ndim == 4
                and int(target_values.shape[1]) == num_nodes
            ):
                return (
                    int(target_values.shape[0])
                    * num_nodes
                )

            # Flattened LG-ODE target shape: [B*N, T, F].
            if target_values.ndim == 3:
                flattened_nodes = int(
                    target_values.shape[0]
                )

                if flattened_nodes % num_nodes == 0:
                    return flattened_nodes

        if hasattr(batch_en, "y"):
            counts = torch.as_tensor(
                batch_en.y
            ).reshape(-1)

            if (
                counts.numel() > 0
                and counts.numel() % num_nodes == 0
            ):
                return int(counts.numel())

        raise ValueError(
            "Could not infer the expected B*N encoder-output dimension "
            "from the encoder and decoder batches."
        )

    def _validate_encoder_outputs(
        self,
        first_point_mu: Tensor,
        first_point_std: Tensor,
        expected_encoded_nodes: int,
    ) -> None:
        """Validate the posterior tensors returned by the encoder."""

        if not isinstance(first_point_mu, Tensor):
            raise TypeError(
                "The encoder posterior mean must be a tensor."
            )

        if not isinstance(first_point_std, Tensor):
            raise TypeError(
                "The encoder posterior standard deviation must be a tensor."
            )

        if first_point_mu.shape != first_point_std.shape:
            raise ValueError(
                "Posterior mean and standard deviation must have identical "
                f"shapes; got {tuple(first_point_mu.shape)} and "
                f"{tuple(first_point_std.shape)}."
            )

        if first_point_mu.ndim != 2:
            raise ValueError(
                "Encoder outputs must have shape [B*N, latent_dim]; "
                f"got {tuple(first_point_mu.shape)}."
            )

        if int(first_point_mu.shape[0]) != expected_encoded_nodes:
            raise ValueError(
                "The number of encoded objects does not equal B*N: "
                f"expected {expected_encoded_nodes}, "
                f"got {first_point_mu.shape[0]}."
            )

        if int(first_point_mu.shape[1]) != self.latent_dim:
            raise ValueError(
                "Encoder posterior dimension does not match latent_dim: "
                f"{first_point_mu.shape[1]} != {self.latent_dim}."
            )

        if not first_point_mu.is_floating_point():
            raise TypeError(
                "Posterior mean must have a floating-point dtype."
            )

        if not first_point_std.is_floating_point():
            raise TypeError(
                "Posterior standard deviation must have a floating-point "
                "dtype."
            )

        if first_point_mu.device != first_point_std.device:
            raise ValueError(
                "Posterior mean and standard deviation must be on the same "
                f"device; got {first_point_mu.device} and "
                f"{first_point_std.device}."
            )

        if first_point_mu.dtype != first_point_std.dtype:
            raise ValueError(
                "Posterior mean and standard deviation must have the same "
                f"dtype; got {first_point_mu.dtype} and "
                f"{first_point_std.dtype}."
            )

        if not torch.isfinite(first_point_mu).all():
            raise FloatingPointError(
                "Posterior mean contains NaN or infinity."
            )

        if not torch.isfinite(first_point_std).all():
            raise FloatingPointError(
                "Posterior standard deviation contains NaN or infinity."
            )

        if torch.any(first_point_std <= 0):
            minimum = float(
                first_point_std.detach().min().item()
            )

            raise ValueError(
                "Posterior standard deviation must be strictly positive; "
                f"minimum value is {minimum}."
            )

    def _extract_time_grid(
        self,
        batch_de: Any,
    ) -> Tensor:
        """Extract and validate the shared prediction-time grid."""

        time_steps = _get_field(
            batch_de,
            "time_steps",
            "target_times",
            default=None,
        )

        if time_steps is None:
            raise ValueError(
                "Decoder batch must provide time_steps or target_times."
            )

        if not isinstance(time_steps, Tensor):
            time_steps = torch.as_tensor(
                time_steps,
                dtype=torch.float32,
            )

        # PowerGridBatch target times may have shape [B, T]. The current ODE
        # solver uses one shared grid, so every trajectory in the batch must
        # have the same normalized target times.
        if time_steps.ndim == 2:
            if int(time_steps.shape[0]) < 1:
                raise ValueError(
                    "target_times contains no trajectories."
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
                    "LatentGraphODE requires a shared target-time grid "
                    "within each batch."
                )

            time_steps = reference

        if time_steps.ndim != 1:
            raise ValueError(
                "Prediction time grid must have shape [T] or [B, T]; "
                f"got {tuple(time_steps.shape)}."
            )

        if time_steps.numel() == 0:
            raise ValueError(
                "Prediction time grid cannot be empty."
            )

        if not time_steps.is_floating_point():
            time_steps = time_steps.float()

        if not torch.isfinite(time_steps).all():
            raise FloatingPointError(
                "Prediction time grid contains NaN or infinity."
            )

        if time_steps.numel() > 1:
            time_differences = (
                time_steps[1:] - time_steps[:-1]
            )

            if torch.any(time_differences <= 0):
                raise ValueError(
                    "Prediction time grid must be strictly increasing."
                )

        return time_steps

    def _construct_initial_state(
        self,
        first_point_mu: Tensor,
        first_point_std: Tensor,
        n_traj_samples: int,
        sample_posterior: Optional[bool],
    ) -> Tuple[Tensor, bool]:
        """
        Construct latent initial states.

        If ``sample_posterior`` is None, posterior sampling follows the module
        mode:

        * training mode: sample;
        * evaluation mode: use the posterior mean.
        """

        if isinstance(n_traj_samples, bool) or not isinstance(
            n_traj_samples,
            int,
        ):
            raise TypeError(
                "n_traj_samples must be an integer."
            )

        if n_traj_samples < 1:
            raise ValueError(
                "n_traj_samples must be at least one."
            )

        if sample_posterior is None:
            should_sample = bool(self.training)
        elif isinstance(sample_posterior, bool):
            should_sample = sample_posterior
        else:
            raise TypeError(
                "sample_posterior must be True, False, or None."
            )

        # Shape:
        # means_z0:  [S, B*N, latent_dim]
        # sigmas_z0: [S, B*N, latent_dim]
        means_z0 = first_point_mu.unsqueeze(0).expand(
            n_traj_samples,
            *first_point_mu.shape,
        )

        sigmas_z0 = first_point_std.unsqueeze(0).expand(
            n_traj_samples,
            *first_point_std.shape,
        )

        if should_sample:
            # Training path:
            # first_point_enc = mu + std * epsilon.
            first_point_enc = (
                utils.sample_standard_gaussian(
                    means_z0,
                    sigmas_z0,
                )
            )
        else:
            # Deterministic validation/test path. Clone avoids exposing an
            # expanded zero-stride view to solver implementations that might
            # expect contiguous writable storage.
            first_point_enc = means_z0.clone()

        if not isinstance(first_point_enc, Tensor):
            raise TypeError(
                "The posterior initial-state constructor must return a tensor."
            )

        if first_point_enc.shape != means_z0.shape:
            raise RuntimeError(
                "Posterior initial-state constructor returned an invalid "
                f"shape: expected {tuple(means_z0.shape)}, "
                f"got {tuple(first_point_enc.shape)}."
            )

        if not torch.isfinite(first_point_enc).all():
            raise FloatingPointError(
                "Latent initial state contains NaN or infinity."
            )

        return first_point_enc, should_sample

    def _extract_latest_observation_time(
        self,
        batch_en: Any,
        expected_encoded_nodes: int,
    ) -> Tensor:
        """
        Extract explicit latest-observation times for AT-ODE.

        The data adapter must provide one value for every trajectory-bus
        object. Latest times are not inferred in this model wrapper.
        """

        latest_time = _get_field(
            batch_en,
            "latest_observation_time",
            default=None,
        )

        if latest_time is None:
            raise ValueError(
                "AT-ODE requires explicit "
                "batch_en.latest_observation_time metadata with shape "
                "[B*N, 1]. The data adapter must calculate and provide it."
            )

        encoder_x = getattr(
            batch_en,
            "x",
            None,
        )

        if not isinstance(encoder_x, Tensor):
            raise TypeError(
                "batch_en.x must be a tensor."
            )

        if not isinstance(latest_time, Tensor):
            latest_time = torch.as_tensor(
                latest_time,
                device=encoder_x.device,
                dtype=encoder_x.dtype,
            )
        else:
            latest_time = latest_time.to(
                device=encoder_x.device,
                dtype=encoder_x.dtype,
            )

        # PyG may concatenate an [N] graph attribute into [B*N].
        if latest_time.ndim == 1:
            latest_time = latest_time.unsqueeze(-1)

        # Some data adapters retain [B, N, 1].
        if latest_time.ndim == 3:
            if int(latest_time.shape[-1]) != 1:
                raise ValueError(
                    "Three-dimensional latest_observation_time must have "
                    f"shape [B, N, 1]; got {tuple(latest_time.shape)}."
                )

            latest_time = latest_time.reshape(
                -1,
                1,
            )

        expected_shape = (
            expected_encoded_nodes,
            1,
        )

        if tuple(latest_time.shape) != expected_shape:
            raise ValueError(
                "latest_observation_time must have shape [B*N, 1]; "
                f"expected {expected_shape}, "
                f"got {tuple(latest_time.shape)}."
            )

        if not latest_time.is_floating_point():
            raise TypeError(
                "latest_observation_time must have a floating-point dtype."
            )

        if not torch.isfinite(latest_time).all():
            raise FloatingPointError(
                "latest_observation_time contains NaN or infinity."
            )

        return latest_time

    def _extract_graph(
        self,
        batch_g: Any,
    ) -> Tensor:
        """
        Extract complete directed candidate-pair edge labels.

        The solver expects binary candidate-edge labels with shape
        [B, N*(N-1)] or another explicitly supported equivalent layout.
        """

        if batch_g is None:
            raise ValueError(
                f"{self.model_type.upper()} requires a graph realization."
            )

        if isinstance(batch_g, Tensor):
            graph = batch_g
        else:
            graph = _get_field(
                batch_g,
                "candidate_edge_type",
                "candidate_edge_labels",
                "graph",
                default=None,
            )

            if graph is None:
                # Compatibility path. The preferred adapter field is
                # candidate_edge_type because edge_type may otherwise refer
                # only to a sparse physical edge list.
                graph = _get_field(
                    batch_g,
                    "edge_type",
                    default=None,
                )

        if graph is None:
            raise ValueError(
                "Graph batch must provide candidate edge labels through a "
                "tensor, candidate_edge_type, candidate_edge_labels, graph, "
                "or edge_type."
            )

        if not isinstance(graph, Tensor):
            graph = torch.as_tensor(
                graph,
            )

        if graph.numel() == 0:
            raise ValueError(
                "Graph candidate-edge labels cannot be empty."
            )

        if not torch.isfinite(
            graph.to(dtype=torch.float32)
        ).all():
            raise FloatingPointError(
                "Graph candidate-edge labels contain NaN or infinity."
            )

        return graph

    def get_reconstruction(
        self,
        batch_en,
        batch_de,
        batch_g=None,
        n_traj_samples=1,
        sample_posterior: Optional[bool] = None,
    ):
        """
        Reconstruct or forecast trajectories.

        Parameters
        ----------
        batch_en:
            Sparse encoder batch.

        batch_de:
            Decoder target/time batch.

        batch_g:
            Candidate-edge labels for LG-ODE and AT-ODE. Must be None for
            graph-free Latent ODE.

        n_traj_samples:
            Number of latent trajectories.

        sample_posterior:
            Posterior initial-state policy:

            * True:
                Sample z0 from q(z0 | observations).

            * False:
                Use posterior mean deterministically.

            * None:
                Sample when ``self.training`` is True and use the posterior
                mean when ``self.training`` is False.

        Returns
        -------
        pred_x:
            Decoded predictions with shape [S, B*N, T, input_dim].

        all_extra_info:
            Posterior information, latent trajectory, sampling policy, and
            optional AT-ODE transport diagnostics.

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

        first_point_enc, posterior_sampled = (
            self._construct_initial_state(
                first_point_mu,
                first_point_std,
                n_traj_samples,
                sample_posterior,
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
            raise FloatingPointError(
                "Prediction time grid contains NaN or infinity after "
                "device conversion."
            )

        if self.model_type == "latentode":
            if batch_g is not None:
                raise ValueError(
                    "Graph-free Latent ODE must not receive batch_g."
                )

            # Graph-free route: do not pass graph metadata.
            sol_y = self.diffeq_solver(
                first_point_enc,
                time_steps_to_predict,
            )

        elif self.model_type == "lgode":
            graph = self._extract_graph(
                batch_g
            ).to(
                device=first_point_enc.device,
            )

            # Fixed-graph LG-ODE route.
            sol_y = self.diffeq_solver(
                first_point_enc,
                time_steps_to_predict,
                graph=graph,
            )

        elif self.model_type == "atode":
            graph = self._extract_graph(
                batch_g
            ).to(
                device=first_point_enc.device,
            )

            latest_observation_time = (
                self._extract_latest_observation_time(
                    batch_en,
                    expected_encoded_nodes,
                )
            ).to(
                device=first_point_enc.device,
                dtype=first_point_enc.dtype,
            )

            # Attention-transport route. The latest-observation tensor is
            # supplied directly by the data adapter and is not inferred here.
            sol_y = self.diffeq_solver(
                first_point_enc,
                time_steps_to_predict,
                graph=graph,
                latest_observation_time=latest_observation_time,
            )

        else:
            # Defensive guard against post-construction mutation.
            raise ValueError(
                f"Unsupported model_type: {self.model_type!r}."
            )

        if not isinstance(sol_y, Tensor):
            raise TypeError(
                "The differential-equation solver must return a tensor."
            )

        # Expected shape:
        # [n_traj_samples, B*N, n_timepoints, latent_dim]
        if sol_y.ndim != 4:
            raise ValueError(
                "Solver output must have shape [S, B*N, T, D]; "
                f"got {tuple(sol_y.shape)}."
            )

        if int(sol_y.shape[0]) != n_traj_samples:
            raise ValueError(
                "Solver output sample dimension is incorrect: "
                f"{sol_y.shape[0]} != {n_traj_samples}."
            )

        if int(sol_y.shape[1]) != expected_encoded_nodes:
            raise ValueError(
                "Solver output object dimension is incorrect: "
                f"{sol_y.shape[1]} != {expected_encoded_nodes}."
            )

        if int(sol_y.shape[2]) != time_steps_to_predict.numel():
            raise ValueError(
                "Solver output time dimension is incorrect: "
                f"{sol_y.shape[2]} != "
                f"{time_steps_to_predict.numel()}."
            )

        if int(sol_y.shape[-1]) != self.latent_dim:
            raise ValueError(
                "Solver output latent dimension is incorrect: "
                f"{sol_y.shape[-1]} != {self.latent_dim}."
            )

        if not sol_y.is_floating_point():
            raise TypeError(
                "Solver output must have a floating-point dtype."
            )

        if not torch.isfinite(sol_y).all():
            raise FloatingPointError(
                "Latent ODE trajectory contains NaN or infinity."
            )

        pred_x = self.decoder(
            sol_y
        )

        if not isinstance(pred_x, Tensor):
            raise TypeError(
                "Decoder must return a tensor."
            )

        if pred_x.ndim != 4:
            raise ValueError(
                "Decoder output must have shape [S, B*N, T, input_dim]; "
                f"got {tuple(pred_x.shape)}."
            )

        expected_prediction_shape = (
            n_traj_samples,
            expected_encoded_nodes,
            int(time_steps_to_predict.numel()),
            int(self.input_dim),
        )

        if tuple(pred_x.shape) != expected_prediction_shape:
            raise ValueError(
                "Decoder output shape is incorrect: "
                f"expected {expected_prediction_shape}, "
                f"got {tuple(pred_x.shape)}."
            )

        if not pred_x.is_floating_point():
            raise TypeError(
                "Decoded predictions must have a floating-point dtype."
            )

        if not torch.isfinite(pred_x).all():
            raise FloatingPointError(
                "Decoded prediction contains NaN or infinity."
            )

        if self.model_type == "atode":
            raw_transport_diagnostics = getattr(
                self.diffeq_solver,
                "last_transport_diagnostics",
                None,
            )

            transport_diagnostics = _detach_nested(
                raw_transport_diagnostics
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
            "transport_diagnostics": transport_diagnostics,
            "model_type": self.model_type,
            "posterior_sampled": posterior_sampled,
            "evaluation_is_deterministic": (
                not posterior_sampled
            ),
        }

        return pred_x, all_extra_info, None


__all__ = [
    "LatentGraphODE",
    "SUPPORTED_MODEL_TYPES",
]
