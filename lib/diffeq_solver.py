# lib/diffeq_solver.py

"""
ODE solvers for graph-free Latent ODE, fixed-graph LG-ODE, and AT-ODE.

The central AT-ODE rule implemented here is:

    Transport trajectories are constructed once per solver forward pass.

The adaptive ODE solver may evaluate the vector field repeatedly, at times that
are out of order, or more than once at the same time. GraphODEFunc therefore
queries a deterministic, side-effect-free edge-weight provider rather than
advancing mutable transport state during each vector-field evaluation.

Model modes
-----------
latentode:
    Graph-free dynamics. No relation matrices, edge types, physical masks, or
    transport provider are installed.

lgode:
    Graph dynamics over the original complete directed candidate-pair
    representation. Existing physical lines are selected by an explicit mask.
    Their effective edge-weight mode is ``ones``.

atode:
    Identical graph ODE architecture to LG-ODE, but with a precomputed
    solver-safe transport-weight trajectory. Direct-autograd odeint is used so
    gradients through

        z0 -> transport weights -> ODE trajectory

    are preserved.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchdiffeq import odeint as odeint_direct
from torchdiffeq import odeint_adjoint

import lib.utils as utils

try:
    from lib.attention_transport import SolverSafeAttentionTransport
except ImportError:
    try:
        # Compatibility with an implementation that exports the transport
        # module under the shorter project-level name.
        from lib.attention_transport import (
            AttentionTransport as SolverSafeAttentionTransport,
        )
    except ImportError as exc:  # pragma: no cover
        SolverSafeAttentionTransport = None  # type: ignore[assignment,misc]
        _TRANSPORT_IMPORT_ERROR = exc
    else:
        _TRANSPORT_IMPORT_ERROR = None
else:
    _TRANSPORT_IMPORT_ERROR = None


SUPPORTED_MODEL_TYPES = {
    "latentode",
    "lgode",
    "atode",
}

SUPPORTED_EDGE_WEIGHT_MODES = {
    "ones",
    "transport",
}


@dataclass
class TransportRuntimeOutput:
    """Normalized output returned by the transport precomputation."""

    provider: Any
    diagnostics: Dict[str, Any]


def _read_argument(
    args: Any,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if isinstance(args, Mapping):
            if name in args and args[name] is not None:
                return args[name]
        elif args is not None and hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value

    return default


def _normalize_model_type(model_type: Optional[str]) -> str:
    if model_type is None:
        return "lgode"

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
            f"Unsupported model type: {model_type!r}. "
            f"Supported model types are {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    return aliases[normalized]


def _normalize_edge_weight_mode(
    mode: Optional[str],
    model_type: str,
) -> str:
    if mode is None:
        return "transport" if model_type == "atode" else "ones"

    normalized = (
        str(mode)
        .strip()
        .lower()
        .replace("-", "_")
    )

    # ``fixed`` appeared in an earlier factory draft. Treat it as the canonical
    # ``ones`` mode without exposing a third scientific model.
    aliases = {
        "fixed": "ones",
        "unit": "ones",
        "unit_weights": "ones",
        "ones": "ones",
        "transport": "transport",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_EDGE_WEIGHT_MODES:
        raise ValueError(
            f"Unsupported edge-weight mode: {mode!r}. "
            f"Supported modes are {sorted(SUPPORTED_EDGE_WEIGHT_MODES)}"
        )

    if model_type == "latentode" and normalized != "ones":
        raise ValueError(
            "Graph-free Latent ODE cannot use transport edge weights"
        )
    if model_type == "lgode" and normalized != "ones":
        raise ValueError(
            "LG-ODE must use edge_weight_mode='ones'"
        )
    if model_type == "atode" and normalized != "transport":
        raise ValueError(
            "AT-ODE must use edge_weight_mode='transport'"
        )

    return normalized


def _construct_with_supported_keywords(
    constructor: Any,
    **keywords: Any,
) -> Any:
    """Call a constructor using only keywords supported by its signature."""

    signature = inspect.signature(constructor)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_kwargs:
        selected = keywords
    else:
        selected = {
            name: value
            for name, value in keywords.items()
            if name in signature.parameters
        }

    return constructor(**selected)


def _detach_diagnostics(value: Any) -> Any:
    """Recursively detach diagnostics without affecting training tensors."""

    if isinstance(value, Tensor):
        return value.detach()

    if isinstance(value, Mapping):
        return {
            key: _detach_diagnostics(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _detach_diagnostics(item)
            for item in value
        )

    if isinstance(value, list):
        return [
            _detach_diagnostics(item)
            for item in value
        ]

    return value


def _extract_from_context(
    observation_context: Any,
    *names: str,
) -> Any:
    if observation_context is None:
        return None

    for name in names:
        if isinstance(observation_context, Mapping):
            if name in observation_context:
                return observation_context[name]
        elif hasattr(observation_context, name):
            return getattr(observation_context, name)

    return None


class DiffeqSolver(nn.Module):
    """
    ODE integration wrapper for all power-grid model modes.

    Parameters
    ----------
    ode_func:
        GraphODEFunc for LG-ODE/AT-ODE or ODEFunc for graph-free Latent ODE.
    method:
        torchdiffeq integration method.
    args:
        Experiment argument namespace or mapping.
    odeint_rtol, odeint_atol:
        Solver tolerances.
    device:
        Initial device.
    model_type:
        ``latentode``, ``lgode`` or ``atode``. If omitted, read from args and
        fall back to LG-ODE for compatibility with the original repository.
    edge_weight_mode:
        ``ones`` or ``transport``. It must agree with model_type.
    attention_transport:
        Optional preconstructed transport module for AT-ODE.
    edge_weight_provider:
        Compatibility alias for a preconstructed transport module. A per-solve
        provider is still created by calling this module before odeint.
    """

    def __init__(
        self,
        ode_func,
        method,
        args,
        odeint_rtol=1e-3,
        odeint_atol=1e-4,
        device=torch.device("cpu"),
        model_type=None,
        edge_weight_mode=None,
        attention_transport=None,
        edge_weight_provider=None,
        physical_edge_index=None,
        edge_index=None,
        **kwargs,
    ):
        super(DiffeqSolver, self).__init__()

        del kwargs

        self.ode_method = str(method)
        self.device = torch.device(device)
        self.ode_func = ode_func
        self.args = args

        inferred_model_type = model_type
        if inferred_model_type is None:
            inferred_model_type = _read_argument(
                args,
                "model_type",
                "model",
                "model_name",
                default=None,
            )

        # Original LG-ODE arguments often do not contain a model field.
        if inferred_model_type is None:
            inferred_model_type = (
                "latentode"
                if isinstance(ode_func, ODEFunc)
                and not isinstance(ode_func, GraphODEFunc)
                else "lgode"
            )

        self.model_type = _normalize_model_type(
            inferred_model_type
        )
        self.edge_weight_mode = _normalize_edge_weight_mode(
            edge_weight_mode,
            self.model_type,
        )
        self.uses_graph = self.model_type in {
            "lgode",
            "atode",
        }

        self.num_atoms = int(
            _read_argument(
                args,
                "num_nodes",
                "n_balls",
                "num_atoms",
                default=0,
            )
        )
        if self.num_atoms < 1:
            raise ValueError(
                "The solver requires a positive number of nodes through "
                "args.num_nodes, args.n_balls, or args.num_atoms"
            )

        self.odeint_rtol = float(odeint_rtol)
        self.odeint_atol = float(odeint_atol)

        if self.odeint_rtol <= 0.0:
            raise ValueError("odeint_rtol must be positive")
        if self.odeint_atol <= 0.0:
            raise ValueError("odeint_atol must be positive")

        self.edge_types = int(
            _read_argument(args, "edge_types", default=2)
        )
        if self.uses_graph and self.edge_types != 2:
            raise ValueError(
                "The controlled power-grid protocol requires exactly two "
                "candidate-edge types: 0=no line and 1=physical line"
            )

        self.augment_dim = int(
            _read_argument(
                args,
                "augmentation_dim",
                "augment_dim",
                default=0,
            )
        )
        if self.augment_dim < 0:
            raise ValueError(
                "augmentation dimension cannot be negative"
            )

        latent_dim = int(
            _read_argument(
                args,
                "latent_dim",
                "latents",
                default=0,
            )
        )
        self.latent_dim = latent_dim

        self.transport_hidden_dim = int(
            _read_argument(
                args,
                "transport_hidden_dim",
                default=64,
            )
        )
        self.transport_bins = int(
            _read_argument(
                args,
                "transport_bins",
                default=32,
            )
        )
        self.transport_max_age = float(
            _read_argument(
                args,
                "transport_max_age",
                default=1.0,
            )
        )
        self.transport_speed = float(
            _read_argument(
                args,
                "transport_speed",
                default=1.0,
            )
        )
        self.transport_decay = float(
            _read_argument(
                args,
                "transport_decay",
                default=0.0,
            )
        )

        if self.transport_hidden_dim < 1:
            raise ValueError(
                "transport_hidden_dim must be positive"
            )
        if self.transport_bins < 2:
            raise ValueError(
                "transport_bins must be at least two"
            )
        if self.transport_max_age <= 0.0:
            raise ValueError(
                "transport_max_age must be positive"
            )
        if self.transport_speed <= 0.0:
            raise ValueError(
                "transport_speed must be positive"
            )
        if self.transport_decay < 0.0:
            raise ValueError(
                "transport_decay cannot be negative"
            )

        supplied_physical_edges = (
            physical_edge_index
            if physical_edge_index is not None
            else edge_index
        )
        if supplied_physical_edges is not None:
            supplied_physical_edges = torch.as_tensor(
                supplied_physical_edges,
                dtype=torch.long,
                device=self.device,
            )
        self.physical_edge_index = supplied_physical_edges

        if self.uses_graph:
            rel_rec, rel_send = self.compute_rec_send()
            self.register_buffer(
                "rel_rec",
                rel_rec,
                persistent=True,
            )
            self.register_buffer(
                "rel_send",
                rel_send,
                persistent=True,
            )
        else:
            self.register_buffer(
                "rel_rec",
                torch.empty(
                    0,
                    self.num_atoms,
                    dtype=torch.float32,
                    device=self.device,
                ),
                persistent=False,
            )
            self.register_buffer(
                "rel_send",
                torch.empty(
                    0,
                    self.num_atoms,
                    dtype=torch.float32,
                    device=self.device,
                ),
                persistent=False,
            )

        if (
            attention_transport is not None
            and edge_weight_provider is not None
            and attention_transport is not edge_weight_provider
        ):
            raise ValueError(
                "Specify only one of attention_transport and "
                "edge_weight_provider"
            )

        externally_supplied_transport = (
            attention_transport
            if attention_transport is not None
            else edge_weight_provider
        )

        if self.edge_weight_mode == "transport":
            if externally_supplied_transport is not None:
                if not isinstance(
                    externally_supplied_transport,
                    nn.Module,
                ):
                    raise TypeError(
                        "The AT-ODE transport constructor must be an "
                        "nn.Module"
                    )
                self.attention_transport = (
                    externally_supplied_transport
                )
            else:
                self.attention_transport = (
                    self._build_attention_transport()
                )
        else:
            # Do not instantiate unused transport parameters for LG-ODE or
            # graph-free Latent ODE.
            if externally_supplied_transport is not None:
                raise ValueError(
                    "A transport module may be supplied only for AT-ODE"
                )
            self.attention_transport = None

        self.last_transport_diagnostics: Dict[str, Any] = {}
        self.last_solver_diagnostics: Dict[str, Any] = {}

    @property
    def candidate_edge_count(self) -> int:
        if not self.uses_graph:
            return 0
        return self.num_atoms * (self.num_atoms - 1)

    def compute_rec_send(self) -> Tuple[Tensor, Tensor]:
        """
        Deterministically construct all directed non-self candidate pairs.

        Ordering is receiver-major for compatibility with the original code:

            receiver 0: senders 1, 2, ..., N-1
            receiver 1: senders 0, 2, ..., N-1
            ...

        Returns
        -------
        rel_rec:
            [E, N] one-hot receiver matrix.
        rel_send:
            [E, N] one-hot sender matrix.
        """

        receiver_indices = []
        sender_indices = []

        for receiver in range(self.num_atoms):
            for sender in range(self.num_atoms):
                if sender == receiver:
                    continue
                receiver_indices.append(receiver)
                sender_indices.append(sender)

        receiver = torch.tensor(
            receiver_indices,
            dtype=torch.long,
            device=self.device,
        )
        sender = torch.tensor(
            sender_indices,
            dtype=torch.long,
            device=self.device,
        )

        rel_rec = F.one_hot(
            receiver,
            num_classes=self.num_atoms,
        ).to(torch.float32)
        rel_send = F.one_hot(
            sender,
            num_classes=self.num_atoms,
        ).to(torch.float32)

        expected_edges = self.num_atoms * (
            self.num_atoms - 1
        )
        if rel_rec.shape != (
            expected_edges,
            self.num_atoms,
        ):
            raise RuntimeError(
                "Internal relation-matrix construction failed"
            )

        return rel_rec, rel_send

    def encode_onehot(self, labels):
        """
        Deterministic compatibility replacement for the original helper.
        """

        labels = torch.as_tensor(
            labels,
            dtype=torch.long,
        ).reshape(-1)

        if labels.numel() == 0:
            return torch.empty(
                0,
                0,
                dtype=torch.int64,
            ).cpu().numpy()

        classes = torch.unique(
            labels,
            sorted=True,
        )
        class_positions = torch.searchsorted(
            classes,
            labels,
        )
        encoded = F.one_hot(
            class_positions,
            num_classes=classes.numel(),
        )
        return encoded.cpu().numpy()

    def _build_attention_transport(self) -> nn.Module:
        if SolverSafeAttentionTransport is None:
            raise ImportError(
                "AT-ODE requires "
                "lib.attention_transport.SolverSafeAttentionTransport"
            ) from _TRANSPORT_IMPORT_ERROR

        state_dim = self.latent_dim + self.augment_dim
        if state_dim < 1:
            raise ValueError(
                "AT-ODE requires a positive latent_dim"
            )

        module = _construct_with_supported_keywords(
            SolverSafeAttentionTransport,
            latent_dim=state_dim,
            state_dim=state_dim,
            hidden_dim=self.transport_hidden_dim,
            transport_hidden_dim=self.transport_hidden_dim,
            num_bins=self.transport_bins,
            transport_bins=self.transport_bins,
            max_age=self.transport_max_age,
            transport_max_age=self.transport_max_age,
            speed=self.transport_speed,
            transport_speed=self.transport_speed,
            decay=self.transport_decay,
            transport_decay=self.transport_decay,
            num_nodes=self.num_atoms,
            rel_send=self.rel_send,
            rel_rec=self.rel_rec,
        )

        if not isinstance(module, nn.Module):
            raise TypeError(
                "SolverSafeAttentionTransport must inherit nn.Module"
            )

        return module.to(self.device)

    def set_physical_graph(self, edge_index: Tensor) -> None:
        """
        Store physical edge metadata for auditing.

        Candidate-pair graph labels passed to forward remain authoritative for
        each trajectory batch.
        """

        edge_index = torch.as_tensor(
            edge_index,
            dtype=torch.long,
            device=self.rel_rec.device,
        )

        if edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape [2, E] or [E, 2]"
            )
        if edge_index.shape[0] == 2:
            pass
        elif edge_index.shape[1] == 2:
            edge_index = edge_index.transpose(
                0, 1
            ).contiguous()
        else:
            raise ValueError(
                "edge_index must have shape [2, E] or [E, 2]"
            )

        if edge_index.numel():
            if int(edge_index.min()) < 0:
                raise ValueError(
                    "edge_index contains a negative node index"
                )
            if int(edge_index.max()) >= self.num_atoms:
                raise ValueError(
                    "edge_index exceeds the configured node count"
                )

        self.physical_edge_index = edge_index

    def set_edge_weight_provider(
        self,
        provider: Optional[nn.Module],
        mode: Optional[str] = None,
    ) -> None:
        """
        Compatibility setter for model factories.

        The supplied object is the transport-construction module, not the
        per-solve provider returned by that module.
        """

        normalized_mode = _normalize_edge_weight_mode(
            mode,
            self.model_type,
        )

        if normalized_mode == "transport":
            if provider is None:
                raise ValueError(
                    "AT-ODE requires a transport module"
                )
            if not isinstance(provider, nn.Module):
                raise TypeError(
                    "Transport module must inherit nn.Module"
                )
            self.attention_transport = provider
        elif provider is not None:
            raise ValueError(
                "LG-ODE/Latent ODE cannot install a transport module"
            )

        self.edge_weight_mode = normalized_mode

    def _validate_first_point(
        self,
        first_point: Tensor,
    ) -> Tuple[int, int, int, int]:
        if not isinstance(first_point, Tensor):
            raise TypeError(
                "first_point must be a torch.Tensor"
            )
        if first_point.ndim != 3:
            raise ValueError(
                "first_point must have shape [S, B*N, D]; "
                f"got {tuple(first_point.shape)}"
            )
        if not first_point.is_floating_point():
            raise TypeError(
                "first_point must have a floating-point dtype"
            )
        if not torch.isfinite(first_point).all():
            raise ValueError(
                "first_point contains NaN or infinity"
            )

        num_samples, flattened_nodes, feature_dim = (
            first_point.shape
        )

        if num_samples < 1:
            raise ValueError(
                "first_point must contain at least one latent sample"
            )
        if feature_dim < 1:
            raise ValueError(
                "first_point latent dimension must be positive"
            )
        if flattened_nodes % self.num_atoms != 0:
            raise ValueError(
                "The flattened trajectory-node dimension must be divisible "
                f"by N={self.num_atoms}; got {flattened_nodes}"
            )

        batch_size = flattened_nodes // self.num_atoms
        if batch_size < 1:
            raise ValueError(
                "first_point contains no trajectories"
            )

        return (
            int(num_samples),
            int(batch_size),
            int(flattened_nodes),
            int(feature_dim),
        )

    def _validate_time_grid(
        self,
        time_steps_to_predict: Tensor,
    ) -> Tensor:
        if not isinstance(
            time_steps_to_predict,
            Tensor,
        ):
            time_steps_to_predict = torch.as_tensor(
                time_steps_to_predict,
                dtype=torch.float32,
                device=self.rel_rec.device,
            )

        if time_steps_to_predict.ndim != 1:
            raise ValueError(
                "time_steps_to_predict must be one-dimensional; "
                f"got {tuple(time_steps_to_predict.shape)}"
            )
        if time_steps_to_predict.numel() == 0:
            raise ValueError(
                "time_steps_to_predict cannot be empty"
            )
        if not time_steps_to_predict.is_floating_point():
            time_steps_to_predict = (
                time_steps_to_predict.float()
            )
        if not torch.isfinite(
            time_steps_to_predict
        ).all():
            raise ValueError(
                "time_steps_to_predict contains NaN or infinity"
            )

        if time_steps_to_predict.numel() > 1:
            differences = (
                time_steps_to_predict[1:]
                - time_steps_to_predict[:-1]
            )
            if torch.any(differences < 0):
                raise ValueError(
                    "time_steps_to_predict must be monotonically "
                    "nondecreasing"
                )

        return time_steps_to_predict

    def _prepare_integration_grid(
        self,
        requested_times: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Construct a strict odeint grid and an index restoring requested times.

        Repeated requested times are allowed by the public interface. They are
        collapsed before odeint and restored afterward.
        """

        zero = requested_times.new_zeros(1)

        if requested_times[0] < -1e-8:
            raise ValueError(
                "Prediction times must be nonnegative relative to z0"
            )

        if torch.isclose(
            requested_times[0],
            zero[0],
            rtol=0.0,
            atol=1e-8,
        ):
            padded_times = requested_times.clone()
            padded_times[0] = 0.0
        else:
            padded_times = torch.cat(
                [zero, requested_times],
                dim=0,
            )

        unique_times, inverse = torch.unique_consecutive(
            padded_times,
            return_inverse=True,
        )

        if unique_times.numel() > 1:
            differences = (
                unique_times[1:]
                - unique_times[:-1]
            )
            if torch.any(differences <= 0):
                raise ValueError(
                    "Internal integration grid is not strictly increasing"
                )

        if torch.isclose(
            requested_times[0],
            zero[0],
            rtol=0.0,
            atol=1e-8,
        ):
            requested_inverse = inverse
        else:
            requested_inverse = inverse[1:]

        return unique_times, requested_inverse

    def _prepare_initial_state(
        self,
        first_point: Tensor,
        num_samples: int,
        batch_size: int,
        feature_dim: int,
    ) -> Tuple[Tensor, int]:
        initial = first_point.reshape(
            num_samples,
            batch_size,
            self.num_atoms,
            feature_dim,
        )
        initial = initial.reshape(
            num_samples * batch_size,
            self.num_atoms,
            feature_dim,
        )

        if self.augment_dim > 0:
            augmentation = torch.zeros(
                initial.shape[0],
                initial.shape[1],
                self.augment_dim,
                device=initial.device,
                dtype=initial.dtype,
            )
            initial = torch.cat(
                [initial, augmentation],
                dim=-1,
            )

        effective_feature_dim = (
            feature_dim + self.augment_dim
        )
        return initial, effective_feature_dim

    def _prepare_graph(
        self,
        graph: Any,
        physical_edge_mask: Optional[Tensor],
        num_samples: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        if graph is None:
            raise ValueError(
                f"{self.model_type} requires graph labels [B, E]"
            )

        if hasattr(graph, "edge_type"):
            graph = graph.edge_type

        graph = torch.as_tensor(
            graph,
            device=device,
        )

        candidate_edges = self.candidate_edge_count

        if graph.ndim == 1:
            if graph.numel() != batch_size * candidate_edges:
                raise ValueError(
                    "Flattened graph has an invalid number of elements: "
                    f"expected {batch_size * candidate_edges}, "
                    f"got {graph.numel()}"
                )
            graph = graph.reshape(
                batch_size,
                candidate_edges,
            )
        elif graph.ndim == 2:
            if tuple(graph.shape) != (
                batch_size,
                candidate_edges,
            ):
                raise ValueError(
                    "graph must have shape [B, E]; expected "
                    f"{(batch_size, candidate_edges)}, got "
                    f"{tuple(graph.shape)}"
                )
        else:
            raise ValueError(
                "graph must have shape [B, E] or flattened [B*E]"
            )

        if not torch.isfinite(graph).all():
            raise ValueError(
                "graph contains NaN or infinity"
            )

        graph_long = graph.long()
        if not torch.equal(
            graph,
            graph_long.to(dtype=graph.dtype),
        ):
            raise ValueError(
                "graph edge labels must be integer-valued"
            )

        if torch.any(
            (graph_long != 0) & (graph_long != 1)
        ):
            raise ValueError(
                "Power-grid graph labels must be binary: "
                "0=no physical line, 1=physical line"
            )

        derived_mask = graph_long.ne(0).unsqueeze(-1)

        if physical_edge_mask is not None:
            mask = torch.as_tensor(
                physical_edge_mask,
                device=device,
            )

            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)

            expected_shape = (
                batch_size,
                candidate_edges,
                1,
            )
            if tuple(mask.shape) != expected_shape:
                raise ValueError(
                    "physical_edge_mask must have shape [B, E, 1]; "
                    f"expected {expected_shape}, got "
                    f"{tuple(mask.shape)}"
                )

            if not torch.isfinite(mask).all():
                raise ValueError(
                    "physical_edge_mask contains NaN or infinity"
                )

            mask_bool = mask.bool()
            if not torch.equal(
                mask_bool,
                derived_mask,
            ):
                raise ValueError(
                    "physical_edge_mask disagrees with graph.ne(0)"
                )
        else:
            mask_bool = derived_mask

        effective_graph = graph_long.unsqueeze(0).expand(
            num_samples,
            batch_size,
            candidate_edges,
        ).reshape(
            num_samples * batch_size,
            candidate_edges,
        )

        effective_mask = mask_bool.unsqueeze(0).expand(
            num_samples,
            batch_size,
            candidate_edges,
            1,
        ).reshape(
            num_samples * batch_size,
            candidate_edges,
            1,
        )

        rel_type = F.one_hot(
            effective_graph,
            num_classes=self.edge_types,
        ).to(
            device=device,
            dtype=dtype,
        )

        expected_rel_type = (
            num_samples * batch_size,
            candidate_edges,
            self.edge_types,
        )
        if tuple(rel_type.shape) != expected_rel_type:
            raise RuntimeError(
                "Internal relation-type construction produced an invalid "
                f"shape: {tuple(rel_type.shape)}"
            )

        return rel_type, effective_mask.to(dtype=dtype)

    def _prepare_latest_observation_time(
        self,
        latest_observation_time: Any,
        observation_context: Any,
        num_samples: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if latest_observation_time is None:
            latest_observation_time = _extract_from_context(
                observation_context,
                "latest_observation_time",
                "latest_valid_observation_time",
                "latest_time",
            )

        if latest_observation_time is None:
            raise ValueError(
                "AT-ODE requires latest_observation_time with shape "
                "[B*N, 1], [B, N], or [B, N, 1]"
            )

        latest = torch.as_tensor(
            latest_observation_time,
            device=device,
            dtype=dtype,
        )

        if latest.ndim == 1:
            if latest.numel() != batch_size * self.num_atoms:
                raise ValueError(
                    "latest_observation_time has an invalid length; "
                    f"expected {batch_size * self.num_atoms}, "
                    f"got {latest.numel()}"
                )
            latest = latest.reshape(
                batch_size,
                self.num_atoms,
                1,
            )

        elif latest.ndim == 2:
            if tuple(latest.shape) == (
                batch_size * self.num_atoms,
                1,
            ):
                latest = latest.reshape(
                    batch_size,
                    self.num_atoms,
                    1,
                )
            elif tuple(latest.shape) == (
                batch_size,
                self.num_atoms,
            ):
                latest = latest.unsqueeze(-1)
            else:
                raise ValueError(
                    "latest_observation_time must have shape [B*N, 1] "
                    f"or [B, N]; got {tuple(latest.shape)}"
                )

        elif latest.ndim == 3:
            expected = (
                batch_size,
                self.num_atoms,
                1,
            )
            if tuple(latest.shape) != expected:
                raise ValueError(
                    "latest_observation_time must have shape [B, N, 1]; "
                    f"expected {expected}, got {tuple(latest.shape)}"
                )

        else:
            raise ValueError(
                "latest_observation_time has an unsupported rank"
            )

        if not torch.isfinite(latest).all():
            raise ValueError(
                "latest_observation_time contains NaN or infinity"
            )

        effective = latest.unsqueeze(0).expand(
            num_samples,
            batch_size,
            self.num_atoms,
            1,
        ).reshape(
            num_samples * batch_size,
            self.num_atoms,
            1,
        )

        return effective

    def _normalize_transport_output(
        self,
        output: Any,
    ) -> TransportRuntimeOutput:
        provider = None
        diagnostics: Dict[str, Any] = {}

        if hasattr(output, "provider"):
            provider = output.provider
            raw_diagnostics = getattr(
                output,
                "diagnostics",
                {},
            )
            if raw_diagnostics is not None:
                diagnostics = dict(raw_diagnostics)

        elif isinstance(output, Mapping):
            provider = output.get(
                "provider",
                output.get("edge_weight_provider"),
            )
            raw_diagnostics = output.get(
                "diagnostics",
                {},
            )
            if raw_diagnostics is not None:
                diagnostics = dict(raw_diagnostics)

        elif isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(
                    "Transport output tuple must be "
                    "(provider, diagnostics)"
                )
            provider, raw_diagnostics = output
            if raw_diagnostics is not None:
                diagnostics = dict(raw_diagnostics)

        elif callable(output):
            provider = output

        if provider is None or not callable(provider):
            raise TypeError(
                "Attention transport must return a callable provider, an "
                "object with .provider, a mapping containing 'provider', "
                "or (provider, diagnostics)"
            )

        return TransportRuntimeOutput(
            provider=provider,
            diagnostics=diagnostics,
        )

    def _validate_provider_weight(
        self,
        weight: Tensor,
        effective_batch: int,
        physical_edge_mask: Tensor,
        reference: Tensor,
    ) -> Tensor:
        if not isinstance(weight, Tensor):
            raise TypeError(
                "Transport provider must return a torch.Tensor"
            )

        expected_shape = (
            effective_batch,
            self.candidate_edge_count,
            1,
        )
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                "Transport provider returned an invalid shape; "
                f"expected {expected_shape}, got {tuple(weight.shape)}"
            )

        if weight.device != reference.device:
            raise ValueError(
                "Transport weights must be on the same device as z0"
            )
        if not weight.is_floating_point():
            raise TypeError(
                "Transport weights must have a floating-point dtype"
            )
        if not torch.isfinite(weight).all():
            raise ValueError(
                "Transport weights contain NaN or infinity"
            )
        if torch.any(weight < 0):
            raise ValueError(
                "Transport weights must be nonnegative"
            )

        weight = weight.to(dtype=reference.dtype)

        # The provider is required to respect the physical graph. Mask again
        # defensively before passing weights to GraphODEFunc.
        return weight * physical_edge_mask

    def _precompute_transport(
        self,
        initial_state: Tensor,
        latest_observation_time: Tensor,
        integration_times: Tensor,
        physical_edge_mask: Tensor,
    ) -> TransportRuntimeOutput:
        if self.attention_transport is None:
            raise RuntimeError(
                "AT-ODE has no attention transport module"
            )

        output = self.attention_transport(
            z0=initial_state,
            latest_observation_time=latest_observation_time,
            rel_send=self.rel_send.to(
                device=initial_state.device,
                dtype=initial_state.dtype,
            ),
            rel_rec=self.rel_rec.to(
                device=initial_state.device,
                dtype=initial_state.dtype,
            ),
            time_grid=integration_times,
            physical_edge_mask=physical_edge_mask,
        )

        runtime = self._normalize_transport_output(
            output
        )

        # Validate the complete deterministic provider on the precomputed time
        # grid before handing it to the adaptive solver.
        diagnostic_weights = []
        for time in integration_times:
            weight = runtime.provider(time)
            weight = self._validate_provider_weight(
                weight,
                effective_batch=initial_state.shape[0],
                physical_edge_mask=physical_edge_mask,
                reference=initial_state,
            )
            diagnostic_weights.append(weight)

        weight_grid = torch.stack(
            diagnostic_weights,
            dim=0,
        )

        diagnostics = dict(runtime.diagnostics)
        diagnostics.setdefault(
            "transport_speed",
            torch.as_tensor(
                self.transport_speed,
                device=initial_state.device,
                dtype=initial_state.dtype,
            ),
        )
        diagnostics.setdefault(
            "decay_rate",
            torch.as_tensor(
                self.transport_decay,
                device=initial_state.device,
                dtype=initial_state.dtype,
            ),
        )

        physical = physical_edge_mask.unsqueeze(0).expand_as(
            weight_grid
        )
        physical_count = physical.sum().clamp_min(1.0)
        physical_weights = weight_grid * physical

        diagnostics.setdefault(
            "edge_weight_mean",
            physical_weights.sum() / physical_count,
        )

        if bool(physical.any()):
            selected = weight_grid[
                physical.to(dtype=torch.bool)
            ]
            diagnostics.setdefault(
                "edge_weight_min",
                selected.min(),
            )
            diagnostics.setdefault(
                "edge_weight_max",
                selected.max(),
            )
        else:
            zero = weight_grid.new_zeros(())
            diagnostics.setdefault(
                "edge_weight_min",
                zero,
            )
            diagnostics.setdefault(
                "edge_weight_max",
                zero,
            )

        initial_age = (
            integration_times[0]
            - latest_observation_time
        ).clamp_min(0.0)
        diagnostics.setdefault(
            "initial_age_mean",
            initial_age.mean(),
        )

        initial_mass = (
            physical_weights[0].sum(dim=1)
        )
        final_mass = (
            physical_weights[-1].sum(dim=1)
        )
        mass_denominator = initial_mass.abs().clamp_min(
            1e-8
        )
        diagnostics.setdefault(
            "transport_mass_drift",
            (
                (final_mass - initial_mass).abs()
                / mass_denominator
            ).mean(),
        )

        self.last_transport_diagnostics = (
            _detach_diagnostics(diagnostics)
        )

        return TransportRuntimeOutput(
            provider=runtime.provider,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        first_point,
        time_steps_to_predict,
        graph=None,
        latest_observation_time=None,
        physical_edge_mask=None,
        observation_context=None,
        backwards=False,
    ):
        """
        Integrate latent states over the requested prediction times.

        Parameters
        ----------
        first_point:
            Latent initial states [S, B*N, D].
        time_steps_to_predict:
            One-dimensional nondecreasing target time grid [T].
        graph:
            Candidate-pair binary labels [B, E] for LG-ODE and AT-ODE. It must
            be None or is ignored only for graph-free Latent ODE.
        latest_observation_time:
            Latest real observation time per bus. Required for AT-ODE. Accepted
            shapes are [B*N, 1], [B, N], and [B, N, 1].
        physical_edge_mask:
            Optional explicit [B, E, 1] mask. If supplied, it must agree with
            graph.ne(0).
        observation_context:
            Optional mapping/object from which latest observation metadata can
            be extracted.
        backwards:
            Negate the ODE vector field.
        """

        (
            num_samples,
            batch_size,
            flattened_nodes,
            feature_dim,
        ) = self._validate_first_point(first_point)

        requested_times = self._validate_time_grid(
            time_steps_to_predict
        ).to(
            device=first_point.device,
            dtype=first_point.dtype,
        )

        integration_times, requested_inverse = (
            self._prepare_integration_grid(
                requested_times
            )
        )

        initial_state, effective_feature_dim = (
            self._prepare_initial_state(
                first_point,
                num_samples,
                batch_size,
                feature_dim,
            )
        )

        effective_batch = num_samples * batch_size
        transport_runtime: Optional[
            TransportRuntimeOutput
        ] = None

        if not self.uses_graph:
            if graph is not None:
                # Ignore no graph data silently only when it is genuinely
                # absent. Passing a graph to Latent ODE usually signals a
                # protocol error.
                raise ValueError(
                    "Graph-free Latent ODE must not receive graph data"
                )
            if physical_edge_mask is not None:
                raise ValueError(
                    "Graph-free Latent ODE must not receive a physical mask"
                )
            if latest_observation_time is not None:
                raise ValueError(
                    "Graph-free Latent ODE does not use physical-edge "
                    "observation timing"
                )

            if hasattr(self.ode_func, "set_backwards"):
                self.ode_func.set_backwards(bool(backwards))

        else:
            rel_type, effective_physical_mask = (
                self._prepare_graph(
                    graph,
                    physical_edge_mask,
                    num_samples,
                    batch_size,
                    first_point.device,
                    first_point.dtype,
                )
            )

            effective_latest_time = None
            edge_weight_provider = None

            if self.edge_weight_mode == "transport":
                effective_latest_time = (
                    self._prepare_latest_observation_time(
                        latest_observation_time,
                        observation_context,
                        num_samples,
                        batch_size,
                        first_point.device,
                        first_point.dtype,
                    )
                )

                transport_runtime = (
                    self._precompute_transport(
                        initial_state=initial_state,
                        latest_observation_time=(
                            effective_latest_time
                        ),
                        integration_times=integration_times,
                        physical_edge_mask=(
                            effective_physical_mask
                        ),
                    )
                )
                edge_weight_provider = (
                    transport_runtime.provider
                )

            elif self.edge_weight_mode != "ones":
                raise ValueError(
                    "Unsupported edge-weight mode: "
                    f"{self.edge_weight_mode!r}"
                )

            self.ode_func.set_graph(
                rec_type=rel_type,
                rel_rec=self.rel_rec.to(
                    device=first_point.device,
                    dtype=first_point.dtype,
                ),
                rel_send=self.rel_send.to(
                    device=first_point.device,
                    dtype=first_point.dtype,
                ),
                edge_types=self.edge_types,
                physical_edge_mask=effective_physical_mask,
                edge_weight_mode=self.edge_weight_mode,
                edge_weight=None,
                edge_weight_provider=edge_weight_provider,
                latest_observation_time=effective_latest_time,
            )

            if hasattr(self.ode_func, "set_backwards"):
                self.ode_func.set_backwards(bool(backwards))

        # Direct autograd is mandatory for AT-ODE because the transport grid is
        # outside the ODE state but depends differentiably on z0.
        if self.edge_weight_mode == "transport":
            odeint_function = odeint_direct
        else:
            odeint_function = odeint_adjoint

        try:
            if integration_times.numel() == 1:
                # torchdiffeq requires at least two integration points. At
                # t=0, the exact solution is the initial state.
                unique_solution = initial_state.unsqueeze(0)
            else:
                unique_solution = odeint_function(
                    self.ode_func,
                    initial_state,
                    integration_times,
                    rtol=self.odeint_rtol,
                    atol=self.odeint_atol,
                    method=self.ode_method,
                )

            if not torch.isfinite(unique_solution).all():
                raise FloatingPointError(
                    "ODE integration produced NaN or infinity"
                )

            # Restore repeated requested times and remove any prepended t=0.
            solution = unique_solution[
                requested_inverse
            ]

        finally:
            # AT-ODE uses direct autograd, so clearing the module's references
            # does not sever the already-created autograd graph. LG-ODE has no
            # provider; its static relation tensors may remain available for
            # adjoint backward evaluation.
            if self.uses_graph:
                self.ode_func.clear_context(
                    preserve_graph=(
                        self.edge_weight_mode == "ones"
                    )
                )

        # [T, S*B, N, D] -> [S, B*N, T, D]
        solution = solution.reshape(
            requested_times.numel(),
            num_samples,
            batch_size,
            self.num_atoms,
            effective_feature_dim,
        )
        solution = solution.permute(
            1,
            2,
            3,
            0,
            4,
        ).contiguous()
        solution = solution.reshape(
            num_samples,
            flattened_nodes,
            requested_times.numel(),
            effective_feature_dim,
        )

        if self.augment_dim > 0:
            solution = solution[
                ...,
                :feature_dim,
            ]

        self.last_solver_diagnostics = {
            "model_type": self.model_type,
            "edge_weight_mode": self.edge_weight_mode,
            "uses_graph": self.uses_graph,
            "num_function_evaluations": int(
                getattr(self.ode_func, "nfe", 0)
            ),
            "requested_time_count": int(
                requested_times.numel()
            ),
            "integration_time_count": int(
                integration_times.numel()
            ),
            "transport": self.last_transport_diagnostics,
        }

        return solution


class GraphODEFunc(nn.Module):
    """
    Graph ODE vector field with solver-time edge-weight queries.
    """

    def __init__(
        self,
        ode_func_net,
        device=torch.device("cpu"),
        physical_edge_index=None,
        edge_index=None,
        edge_weight_mode="ones",
        edge_weight_provider=None,
        **kwargs,
    ):
        super(GraphODEFunc, self).__init__()

        del kwargs

        self.device = torch.device(device)
        self.ode_func_net = ode_func_net
        self.nfe = 0
        self.backwards = False

        self.physical_edge_index = (
            physical_edge_index
            if physical_edge_index is not None
            else edge_index
        )

        normalized_mode = str(
            edge_weight_mode
        ).lower()
        if normalized_mode == "fixed":
            normalized_mode = "ones"
        if normalized_mode not in SUPPORTED_EDGE_WEIGHT_MODES:
            raise ValueError(
                f"Unsupported edge-weight mode: {edge_weight_mode!r}"
            )

        self.edge_weight_mode = normalized_mode
        self.edge_weight_provider = edge_weight_provider
        self.edge_weight: Optional[Tensor] = None
        self.physical_edge_mask: Optional[Tensor] = None
        self.latest_observation_time: Optional[Tensor] = None

    def set_backwards(self, backwards: bool) -> None:
        self.backwards = bool(backwards)

    def _nri_layers(self):
        if not hasattr(self.ode_func_net, "gcs"):
            raise TypeError(
                "GraphODEFunc requires an ODE network with NRI layers"
            )

        for layer in self.ode_func_net.gcs:
            if not hasattr(layer, "base_conv"):
                raise TypeError(
                    "Graph ODE layer does not expose base_conv"
                )
            yield layer.base_conv

    def set_graph(
        self,
        rec_type,
        rel_rec,
        rel_send,
        edge_types,
        physical_edge_mask,
        edge_weight_mode="ones",
        edge_weight=None,
        edge_weight_provider=None,
        latest_observation_time=None,
    ):
        mode = str(edge_weight_mode).lower()
        if mode == "fixed":
            mode = "ones"
        if mode not in SUPPORTED_EDGE_WEIGHT_MODES:
            raise ValueError(
                f"Unsupported edge-weight mode: {edge_weight_mode!r}"
            )

        if rec_type.ndim != 3:
            raise ValueError(
                "rec_type must have shape [B, E, K]"
            )
        if rel_rec.ndim != 2 or rel_send.ndim != 2:
            raise ValueError(
                "rel_rec and rel_send must have shape [E, N]"
            )
        if rel_rec.shape != rel_send.shape:
            raise ValueError(
                "rel_rec and rel_send must have identical shapes"
            )
        if rec_type.shape[1] != rel_rec.shape[0]:
            raise ValueError(
                "rec_type edge count does not match relation matrices"
            )
        if int(edge_types) != rec_type.shape[-1]:
            raise ValueError(
                "edge_types does not match rec_type"
            )

        expected_mask_shape = (
            rec_type.shape[0],
            rec_type.shape[1],
            1,
        )
        if tuple(physical_edge_mask.shape) != expected_mask_shape:
            raise ValueError(
                "physical_edge_mask must have shape [B, E, 1]; "
                f"expected {expected_mask_shape}, got "
                f"{tuple(physical_edge_mask.shape)}"
            )

        if not torch.isfinite(rec_type).all():
            raise ValueError(
                "rec_type contains NaN or infinity"
            )
        if not torch.isfinite(
            physical_edge_mask
        ).all():
            raise ValueError(
                "physical_edge_mask contains NaN or infinity"
            )

        derived_mask = rec_type[..., 1:2].bool()
        if not torch.equal(
            derived_mask,
            physical_edge_mask.bool(),
        ):
            raise ValueError(
                "physical_edge_mask must equal rec_type[..., 1:2]"
            )

        if mode == "transport":
            if edge_weight_provider is None:
                raise ValueError(
                    "Transport mode requires edge_weight_provider"
                )
            if not callable(edge_weight_provider):
                raise TypeError(
                    "edge_weight_provider must be callable"
                )
        elif edge_weight_provider is not None:
            raise ValueError(
                "ones mode must not receive an edge_weight_provider"
            )

        self.edge_weight_mode = mode
        self.edge_weight = edge_weight
        self.edge_weight_provider = edge_weight_provider
        self.physical_edge_mask = physical_edge_mask
        self.latest_observation_time = latest_observation_time

        if hasattr(self.ode_func_net, "set_graph"):
            self.ode_func_net.set_graph(
                rel_type=rec_type,
                rel_rec=rel_rec,
                rel_send=rel_send,
                edge_types=edge_types,
            )
        else:
            for base_conv in self._nri_layers():
                base_conv.rel_type = rec_type
                base_conv.rel_rec = rel_rec
                base_conv.rel_send = rel_send
                base_conv.edge_types = edge_types

        for base_conv in self._nri_layers():
            base_conv.edge_weight_mode = mode
            base_conv.edge_weight = edge_weight
            base_conv.physical_edge_mask = (
                physical_edge_mask
            )
            base_conv.latest_observation_time = (
                latest_observation_time
            )
            base_conv.current_time = None

        self.nfe = 0

    def _validate_dynamic_weight(
        self,
        edge_weight: Tensor,
        z: Tensor,
    ) -> Tensor:
        if self.physical_edge_mask is None:
            raise RuntimeError(
                "GraphODEFunc physical edge mask is not installed"
            )
        if not isinstance(edge_weight, Tensor):
            raise TypeError(
                "edge_weight_provider must return a tensor"
            )

        expected_shape = tuple(
            self.physical_edge_mask.shape
        )
        if tuple(edge_weight.shape) != expected_shape:
            raise ValueError(
                "Dynamic edge weights must have shape [B, E, 1]; "
                f"expected {expected_shape}, got "
                f"{tuple(edge_weight.shape)}"
            )
        if edge_weight.device != z.device:
            raise ValueError(
                "Dynamic edge weights and ODE state must share a device"
            )
        if not edge_weight.is_floating_point():
            raise TypeError(
                "Dynamic edge weights must be floating point"
            )
        if not torch.isfinite(edge_weight).all():
            raise ValueError(
                "Dynamic edge weights contain NaN or infinity"
            )
        if torch.any(edge_weight < 0):
            raise ValueError(
                "Dynamic edge weights must be nonnegative"
            )

        return (
            edge_weight.to(dtype=z.dtype)
            * self.physical_edge_mask.to(
                device=z.device,
                dtype=z.dtype,
            )
        )

    def forward(
        self,
        t_local,
        z,
        backwards=False,
    ):
        self.nfe += 1

        dynamic_edge_weight = self.edge_weight

        if self.edge_weight_mode == "transport":
            if self.edge_weight_provider is None:
                raise RuntimeError(
                    "Transport provider is not installed"
                )

            dynamic_edge_weight = (
                self.edge_weight_provider(t_local)
            )
            dynamic_edge_weight = (
                self._validate_dynamic_weight(
                    dynamic_edge_weight,
                    z,
                )
            )

        elif self.edge_weight_mode == "ones":
            dynamic_edge_weight = None

        else:
            raise ValueError(
                "Unsupported edge-weight mode: "
                f"{self.edge_weight_mode!r}"
            )

        if hasattr(
            self.ode_func_net,
            "set_edge_runtime",
        ):
            self.ode_func_net.set_edge_runtime(
                edge_weight_mode=self.edge_weight_mode,
                edge_weight=dynamic_edge_weight,
                latest_observation_time=(
                    self.latest_observation_time
                ),
                current_time=t_local,
            )
        else:
            for base_conv in self._nri_layers():
                base_conv.edge_weight_mode = (
                    self.edge_weight_mode
                )
                base_conv.edge_weight = (
                    dynamic_edge_weight
                )
                base_conv.latest_observation_time = (
                    self.latest_observation_time
                )
                base_conv.current_time = t_local

        grad = self.ode_func_net(z)

        effective_backwards = (
            bool(backwards) or self.backwards
        )
        if effective_backwards:
            grad = -grad

        return grad

    def clear_context(
        self,
        preserve_graph=False,
    ) -> None:
        """
        Release per-batch transport tensors and providers.

        For adjoint LG-ODE integration, static graph tensors are preserved
        because the adjoint backward pass may reevaluate the vector field.
        AT-ODE uses direct autograd and can clear all per-batch context.
        """

        self.edge_weight_provider = None
        self.edge_weight = None
        self.latest_observation_time = None

        for base_conv in self._nri_layers():
            base_conv.edge_weight = None
            base_conv.latest_observation_time = None
            base_conv.current_time = None
            base_conv.last_effective_edge_weight = None

            if not preserve_graph:
                base_conv.rel_type = None
                base_conv.rel_rec = None
                base_conv.rel_send = None
                base_conv.physical_edge_mask = None

        if not preserve_graph:
            self.physical_edge_mask = None

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "nfe": self.nfe,
            "edge_weight_mode": self.edge_weight_mode,
        }


class ODEFunc(nn.Module):
    """Graph-free node-wise ODE function."""

    def __init__(
        self,
        input_dim,
        latent_dim,
        ode_func_net,
        device=torch.device("cpu"),
    ):
        super(ODEFunc, self).__init__()

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.device = torch.device(device)
        self.gradient_net = ode_func_net
        self.nfe = 0
        self.backwards = False

        utils.init_network_weights(
            self.gradient_net
        )

    def set_backwards(self, backwards: bool) -> None:
        self.backwards = bool(backwards)

    def reset_nfe(self) -> None:
        self.nfe = 0

    def forward(
        self,
        t_local,
        y,
        backwards=False,
    ):
        self.nfe += 1

        grad = self.get_ode_gradient_nn(
            t_local,
            y,
        )

        effective_backwards = (
            bool(backwards) or self.backwards
        )
        if effective_backwards:
            grad = -grad

        return grad

    def get_ode_gradient_nn(
        self,
        t_local,
        y,
    ):
        del t_local
        return self.gradient_net(y)


__all__ = [
    "DiffeqSolver",
    "GraphODEFunc",
    "ODEFunc",
    "SUPPORTED_EDGE_WEIGHT_MODES",
    "SUPPORTED_MODEL_TYPES",
    "TransportRuntimeOutput",
]
