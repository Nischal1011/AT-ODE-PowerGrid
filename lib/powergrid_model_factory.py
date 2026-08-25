# lib/powergrid_model_factory.py

"""
Consistent model factory for the SimBench LG-ODE experiments.

Exactly four model names are supported:

    persistence
    latentode
    lgode
    atode

LG-ODE and AT-ODE use the same recognition encoder, decoder, latent
dimensions, augmentation, graph-ODE architecture, solver settings,
tolerances, dropout, likelihood, and prior.

Their only intended generative-model difference is edge weighting:

    lgode: fixed unit weights on physical edges
    atode: time-dependent transport weights on physical edges

Important initialization guarantee
----------------------------------
For a fixed seed, every shared parameterized component is initialized before
AT-ODE's additional transport provider. Consequently, LG-ODE and AT-ODE
receive identical initial values for their shared encoder, decoder, and
graph-ODE parameters.

Paper runs should use ``build_lgode_atode_protocol_pair`` when constructing
the two graph models. That function automatically runs
``assert_lgode_atode_protocol_match`` before returning.
"""

from __future__ import annotations

import copy
import inspect
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from lib.diffeq_solver import DiffeqSolver, GraphODEFunc
from lib.encoder_decoder import Decoder
from lib.gnn_models import GNN
from lib.latent_ode import LatentGraphODE
from lib.powergrid_baselines import (
    IndependentLatentODE,
    PersistenceBaseline,
)

from lib.attention_transport import SolverSafeAttentionTransport


SUPPORTED_POWERGRID_MODELS = (
    "persistence",
    "latentode",
    "lgode",
    "atode",
)


@dataclass(frozen=True)
class PowerGridModelConfig:
    """Resolved architecture and numerical settings used by the factory."""

    model_name: str
    task: str

    input_dim: int
    num_nodes: int

    latent_dim: int
    recognition_dim: int
    ode_hidden_dim: int
    augmentation_dim: int

    encoder_layers: int
    ode_layers: int
    attention_heads: int
    edge_types: int

    encoder_type: str
    ode_type: str
    recognition_aggregation: str

    solver: str
    rtol: float
    atol: float
    dropout: float
    ode_dropout: float

    observation_std: float
    seed: int

    transport_bins: int
    transport_max_age: float
    transport_hidden_dim: int
    transport_attention_dim: int
    transport_heads: int
    transport_speed: float
    transport_decay: float

    @property
    def ode_state_dim(self) -> int:
        return self.latent_dim + self.augmentation_dim

    def common_graph_model_signature(self) -> Dict[str, Any]:
        """Return settings that must match between LG-ODE and AT-ODE."""

        return {
            "input_dim": self.input_dim,
            "num_nodes": self.num_nodes,
            "latent_dim": self.latent_dim,
            "recognition_dim": self.recognition_dim,
            "ode_hidden_dim": self.ode_hidden_dim,
            "augmentation_dim": self.augmentation_dim,
            "encoder_layers": self.encoder_layers,
            "ode_layers": self.ode_layers,
            "attention_heads": self.attention_heads,
            "edge_types": self.edge_types,
            "encoder_type": self.encoder_type,
            "ode_type": self.ode_type,
            "recognition_aggregation": self.recognition_aggregation,
            "solver": self.solver,
            "rtol": self.rtol,
            "atol": self.atol,
            "dropout": self.dropout,
            "ode_dropout": self.ode_dropout,
            "observation_std": self.observation_std,
        }


def _read_argument(
    args: Any,
    *names: str,
    default: Any = None,
    required: bool = False,
) -> Any:
    """Read the first available alias from a mapping or namespace."""

    for name in names:
        if isinstance(args, Mapping):
            if name in args and args[name] is not None:
                return args[name]
        elif hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value

    if required:
        joined = ", ".join(names)
        raise ValueError(
            f"Missing required model argument; expected one of: {joined}"
        )

    return default


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()

    aliases = {
        "copy-persistence": "persistence",
        "copypersistence": "persistence",
        "copy_persistence": "persistence",
        "latent-ode": "latentode",
        "latent_ode": "latentode",
        "lg-ode": "lgode",
        "lg_ode": "lgode",
        "at-ode": "atode",
        "at_ode": "atode",
    }

    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_POWERGRID_MODELS:
        raise ValueError(
            f"Unknown power-grid model {model_name!r}. Supported models are: "
            f"{', '.join(SUPPORTED_POWERGRID_MODELS)}"
        )

    return normalized


def _normalize_task(task: str) -> str:
    normalized = str(task).strip().lower()

    aliases = {
        "interp": "interpolation",
        "interpolate": "interpolation",
        "extrap": "extrapolation",
        "forecast": "extrapolation",
        "forecasting": "extrapolation",
    }

    normalized = aliases.get(normalized, normalized)

    if normalized not in {"interpolation", "extrapolation"}:
        raise ValueError(
            "task must be 'interpolation' or 'extrapolation'; "
            f"got {task!r}"
        )

    return normalized


def _resolve_config(
    model_name: str,
    input_dim: int,
    num_nodes: int,
    args: Any,
) -> PowerGridModelConfig:
    model_name = _normalize_model_name(model_name)

    if input_dim < 1:
        raise ValueError(f"input_dim must be positive; got {input_dim}")

    if num_nodes < 1:
        raise ValueError(f"num_nodes must be positive; got {num_nodes}")

    task = _normalize_task(
        _read_argument(args, "task", default="interpolation")
    )

    latent_dim = int(
        _read_argument(
            args,
            "latent_dim",
            "latents",
            default=16,
        )
    )

    recognition_dim = int(
        _read_argument(
            args,
            "recognition_dim",
            "rec_dims",
            "rec_dim",
            default=64,
        )
    )

    ode_hidden_dim = int(
        _read_argument(
            args,
            "ode_hidden_dim",
            "ode_dims",
            "ode_dim",
            default=128,
        )
    )

    augmentation_dim = int(
        _read_argument(
            args,
            "augmentation_dim",
            "augment_dim",
            default=0,
        )
    )

    encoder_layers = int(
        _read_argument(
            args,
            "encoder_layers",
            "rec_layers",
            default=2,
        )
    )

    ode_layers = int(
        _read_argument(
            args,
            "ode_layers",
            "gen_layers",
            default=1,
        )
    )

    attention_heads = int(
        _read_argument(
            args,
            "attention_heads",
            "n_heads",
            default=1,
        )
    )

    edge_types = int(
        _read_argument(
            args,
            "edge_types",
            default=2,
        )
    )

    encoder_type = str(
        _read_argument(
            args,
            "encoder_type",
            "z0_encoder",
            default="GTrans",
        )
    )

    ode_type = str(
        _read_argument(
            args,
            "ode_type",
            "odenet",
            default="NRI",
        )
    )

    recognition_aggregation = str(
        _read_argument(
            args,
            "recognition_aggregation",
            "rec_attention",
            default="attention",
        )
    )

    solver = str(
        _read_argument(
            args,
            "solver",
            default="dopri5",
        )
    )

    rtol = float(
        _read_argument(
            args,
            "rtol",
            default=1e-3,
        )
    )

    atol = float(
        _read_argument(
            args,
            "atol",
            default=1e-4,
        )
    )

    dropout = float(
        _read_argument(
            args,
            "dropout",
            default=0.2,
        )
    )

    ode_dropout = float(
        _read_argument(
            args,
            "ode_dropout",
            default=0.0,
        )
    )

    observation_std = float(
        _read_argument(
            args,
            "observation_std",
            "obsrv_std",
            default=0.01,
        )
    )

    seed = int(
        _read_argument(
            args,
            "seed",
            "random_seed",
            default=0,
        )
    )

    transport_bins = int(
        _read_argument(
            args,
            "transport_bins",
            default=32,
        )
    )

    transport_max_age = float(
        _read_argument(
            args,
            "transport_max_age",
            default=1.0,
        )
    )

    transport_hidden_dim = int(
        _read_argument(
            args,
            "transport_hidden_dim",
            default=64,
        )
    )

    transport_attention_dim = int(
        _read_argument(
            args,
            "transport_attention_dim",
            default=16,
        )
    )

    transport_heads = int(
        _read_argument(
            args,
            "transport_heads",
            default=4,
        )
    )

    transport_speed = float(
        _read_argument(
            args,
            "transport_speed",
            default=1.0,
        )
    )

    transport_decay = float(
        _read_argument(
            args,
            "transport_decay",
            default=0.0,
        )
    )

    if latent_dim < 1:
        raise ValueError("latent_dim must be positive")

    if recognition_dim < 1:
        raise ValueError("recognition_dim must be positive")

    if ode_hidden_dim < 1:
        raise ValueError("ode_hidden_dim must be positive")

    if augmentation_dim < 0:
        raise ValueError("augmentation_dim must be non-negative")

    if encoder_layers < 1:
        raise ValueError("encoder_layers must be positive")

    if ode_layers < 1:
        raise ValueError("ode_layers must be positive")

    if attention_heads < 1:
        raise ValueError("attention_heads must be positive")

    if edge_types < 1:
        raise ValueError("edge_types must be positive")

    if recognition_dim % attention_heads != 0:
        raise ValueError(
            "recognition_dim must be divisible by attention_heads for "
            f"GTrans; got {recognition_dim} and {attention_heads}"
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    if ode_dropout != 0.0:
        raise ValueError("ode_dropout must be exactly zero")

    if rtol <= 0.0:
        raise ValueError("rtol must be positive")

    if atol <= 0.0:
        raise ValueError("atol must be positive")

    if observation_std <= 0.0:
        raise ValueError("observation_std must be positive")

    if transport_bins < 2:
        raise ValueError("transport_bins must be at least two")

    if transport_max_age <= 0.0:
        raise ValueError("transport_max_age must be positive")

    if transport_hidden_dim < 1:
        raise ValueError("transport_hidden_dim must be positive")

    if transport_attention_dim < 1:
        raise ValueError("transport_attention_dim must be positive")

    if transport_heads < 1:
        raise ValueError("transport_heads must be positive")

    if transport_speed <= 0.0:
        raise ValueError("transport_speed must be positive")

    if transport_decay < 0.0:
        raise ValueError("transport_decay must be non-negative")

    if encoder_type.lower() != "gtrans":
        raise ValueError(
            "The power-grid protocol requires the shared GTrans encoder; "
            f"got {encoder_type!r}"
        )

    if ode_type.lower() != "nri":
        raise ValueError(
            "The power-grid protocol requires the NRI graph ODE; "
            f"got {ode_type!r}"
        )

    return PowerGridModelConfig(
        model_name=model_name,
        task=task,
        input_dim=int(input_dim),
        num_nodes=int(num_nodes),
        latent_dim=latent_dim,
        recognition_dim=recognition_dim,
        ode_hidden_dim=ode_hidden_dim,
        augmentation_dim=augmentation_dim,
        encoder_layers=encoder_layers,
        ode_layers=ode_layers,
        attention_heads=attention_heads,
        edge_types=edge_types,
        encoder_type="GTrans",
        ode_type="NRI",
        recognition_aggregation=recognition_aggregation,
        solver=solver,
        rtol=rtol,
        atol=atol,
        dropout=dropout,
        ode_dropout=ode_dropout,
        observation_std=observation_std,
        seed=seed,
        transport_bins=transport_bins,
        transport_max_age=transport_max_age,
        transport_hidden_dim=transport_hidden_dim,
        transport_attention_dim=transport_attention_dim,
        transport_heads=transport_heads,
        transport_speed=transport_speed,
        transport_decay=transport_decay,
    )


def _validate_edge_index(
    edge_index: Tensor,
    num_nodes: int,
) -> Tensor:
    if not isinstance(edge_index, Tensor):
        edge_index = torch.as_tensor(
            edge_index,
            dtype=torch.long,
        )

    edge_index = edge_index.detach().clone().long()

    if edge_index.ndim != 2:
        raise ValueError(
            "edge_index must have shape [2, E] or [E, 2]; "
            f"got {tuple(edge_index.shape)}"
        )

    if edge_index.shape[0] == 2:
        pass
    elif edge_index.shape[1] == 2:
        edge_index = edge_index.transpose(0, 1).contiguous()
    else:
        raise ValueError(
            "edge_index must have shape [2, E] or [E, 2]; "
            f"got {tuple(edge_index.shape)}"
        )

    if edge_index.numel():
        minimum = int(edge_index.min().item())
        maximum = int(edge_index.max().item())

        if minimum < 0 or maximum >= num_nodes:
            raise ValueError(
                "edge_index contains an invalid node index: "
                f"minimum={minimum}, maximum={maximum}, "
                f"num_nodes={num_nodes}"
            )

        if torch.any(edge_index[0] == edge_index[1]):
            raise ValueError("edge_index must not contain self-edges")

    return edge_index.contiguous()


def _solver_args_namespace(
    args: Any,
    config: PowerGridModelConfig,
) -> SimpleNamespace:
    """
    Make a private argument namespace without mutating the runner's args.
    """

    if isinstance(args, Mapping):
        values = dict(args)
    elif hasattr(args, "__dict__"):
        values = copy.copy(vars(args))
    else:
        values = {}

    canonical = {
        # New names
        "model_name": config.model_name,
        "model_type": config.model_name,
        "latent_dim": config.latent_dim,
        "recognition_dim": config.recognition_dim,
        "ode_hidden_dim": config.ode_hidden_dim,
        "augmentation_dim": config.augmentation_dim,
        "encoder_layers": config.encoder_layers,
        "ode_layers": config.ode_layers,
        "attention_heads": config.attention_heads,
        "rtol": config.rtol,
        "atol": config.atol,
        # Original repository aliases
        "latents": config.latent_dim,
        "rec_dims": config.recognition_dim,
        "ode_dims": config.ode_hidden_dim,
        "augment_dim": config.augmentation_dim,
        "rec_layers": config.encoder_layers,
        "gen_layers": config.ode_layers,
        "n_heads": config.attention_heads,
        "n_balls": config.num_nodes,
        "edge_types": config.edge_types,
        "z0_encoder": config.encoder_type,
        "odenet": config.ode_type,
        "rec_attention": config.recognition_aggregation,
        "dropout": config.dropout,
        "ode_dropout": config.ode_dropout,
        "solver": config.solver,
        "task": config.task,
    }

    values.update(canonical)
    return SimpleNamespace(**values)


@contextmanager
def _deterministic_initialization(
    seed: int,
    device: torch.device,
) -> Iterator[None]:
    """
    Initialize models deterministically without changing the caller's RNG.

    Each model built with the same seed begins from the same RNG state.
    """

    cuda_devices = []

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False"
            )

        cuda_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        cuda_devices = [cuda_index]

    with torch.random.fork_rng(
        devices=cuda_devices,
        enabled=True,
    ):
        torch.manual_seed(int(seed))

        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))

        yield


def _standard_normal_prior(
    latent_dim: int,
    device: torch.device,
) -> Normal:
    mean = torch.zeros(
        1,
        latent_dim,
        device=device,
    )
    std = torch.ones(
        1,
        latent_dim,
        device=device,
    )
    return Normal(mean, std)


def _construct_with_supported_keywords(
    constructor: Any,
    positional: Tuple[Any, ...],
    keywords: Dict[str, Any],
) -> Any:
    """
    Invoke a constructor while respecting its declared keyword signature.
    """

    signature = inspect.signature(constructor)
    parameters = signature.parameters

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_kwargs:
        selected = keywords
    else:
        selected = {
            name: value
            for name, value in keywords.items()
            if name in parameters
        }

    return constructor(*positional, **selected)


def _make_gtrans_encoder(
    config: PowerGridModelConfig,
) -> GNN:
    return GNN(
        in_dim=config.input_dim,
        n_hid=config.recognition_dim,
        out_dim=config.latent_dim,
        n_heads=config.attention_heads,
        n_layers=config.encoder_layers,
        dropout=config.dropout,
        conv_name=config.encoder_type,
        aggregate=config.recognition_aggregation,
    )


def _make_graph_ode_network(
    config: PowerGridModelConfig,
) -> GNN:
    return GNN(
        in_dim=config.ode_state_dim,
        n_hid=config.ode_hidden_dim,
        out_dim=config.ode_state_dim,
        n_heads=config.attention_heads,
        n_layers=config.ode_layers,
        dropout=config.ode_dropout,
        conv_name=config.ode_type,
        aggregate="add",
    )


def _make_decoder(
    config: PowerGridModelConfig,
) -> Decoder:
    return Decoder(
        latent_dim=config.latent_dim,
        input_dim=config.input_dim,
    )


def _make_graph_ode_function(
    config: PowerGridModelConfig,
    ode_network: GNN,
    edge_index: Tensor,
    device: torch.device,
    *,
    edge_weight_mode: str,
) -> GraphODEFunc:
    """
    Construct the graph ODE wrapper before any transport module.

    GraphODEFunc revisions may initialize or reinitialize the supplied ODE
    network. Therefore this step must occur before AttentionTransport is
    constructed, otherwise AT-ODE would consume random numbers before shared
    graph-ODE initialization.
    """

    ode_function = _construct_with_supported_keywords(
        GraphODEFunc,
        positional=(),
        keywords={
            "ode_func_net": ode_network,
            "device": device,
            "model_type": config.model_name,
            "physical_edge_index": edge_index,
            "edge_index": edge_index,
            "edge_weight_mode": edge_weight_mode,
            # Provider is deliberately absent during shared initialization.
            "edge_weight_provider": None,
        },
    )

    if not isinstance(ode_function, nn.Module):
        raise TypeError("GraphODEFunc must inherit torch.nn.Module")

    return ode_function.to(device)


def _make_transport_provider(
    config: PowerGridModelConfig,
    edge_index: Tensor,
    device: torch.device,
) -> nn.Module:
    """
    Construct AT-ODE's additional transport module.

    This function must be called only after every shared parameterized module
    has been initialized.
    """

    provider = SolverSafeAttentionTransport(
        latent_dim=config.ode_state_dim,
        edge_index=edge_index,
        num_nodes=config.num_nodes,
        num_bins=config.transport_bins,
        max_age=config.transport_max_age,
        hidden_dim=config.transport_hidden_dim,
        attention_dim=config.transport_attention_dim,
        num_heads=config.transport_heads,
        initial_speed=config.transport_speed,
        initial_decay=config.transport_decay,
        learnable_speed=True,
        learnable_decay=True,
        dropout=config.dropout,
    )

    if not isinstance(provider, nn.Module):
        raise TypeError(
            "AttentionTransport must inherit torch.nn.Module"
        )

    return provider.to(device)


def _make_graph_solver(
    config: PowerGridModelConfig,
    solver_args: SimpleNamespace,
    ode_function: GraphODEFunc,
    edge_index: Tensor,
    device: torch.device,
    *,
    edge_weight_mode: str,
    edge_weight_provider: Optional[nn.Module],
) -> DiffeqSolver:
    """
    Construct the ODE solver around an already initialized GraphODEFunc.

    DiffeqSolver is expected not to create shared neural-network parameters.
    The shared encoder, decoder, ODE network, and GraphODEFunc have already
    been initialized before an AT-ODE provider reaches this function.
    """

    solver_keywords = {
        "ode_func": ode_function,
        "method": config.solver,
        "args": solver_args,
        "odeint_rtol": config.rtol,
        "odeint_atol": config.atol,
        "device": device,
        "model_type": config.model_name,
        "uses_graph": True,
        "physical_edge_index": edge_index,
        "edge_index": edge_index,
        "edge_weight_mode": edge_weight_mode,
        "edge_weight_provider": edge_weight_provider,
    }

    solver = _construct_with_supported_keywords(
        DiffeqSolver,
        positional=(),
        keywords=solver_keywords,
    )

    if not isinstance(solver, nn.Module):
        raise TypeError("DiffeqSolver must inherit torch.nn.Module")

    solver = solver.to(device)

    if hasattr(solver, "set_physical_graph"):
        solver.set_physical_graph(edge_index)

    if hasattr(solver, "set_edge_weight_provider"):
        try:
            solver.set_edge_weight_provider(
                edge_weight_provider,
                mode=edge_weight_mode,
            )
        except TypeError:
            solver.set_edge_weight_provider(edge_weight_provider)

    # Auditable protocol metadata.
    solver.model_type = config.model_name
    solver.uses_graph = True
    solver.physical_edge_index = edge_index
    solver.edge_weight_mode = edge_weight_mode
    solver.edge_weight_provider = edge_weight_provider

    ode_function.model_type = config.model_name
    ode_function.physical_edge_index = edge_index
    ode_function.edge_weight_mode = edge_weight_mode
    # The trainable transport module is owned and registered by
    # DiffeqSolver. GraphODEFunc receives only the temporary per-solve
    # cache query callable inside DiffeqSolver.forward().
    ode_function.edge_weight_provider = None

    return solver


def _verify_solver_protocol(
    solver: DiffeqSolver,
    model_name: str,
) -> None:
    if model_name not in {"lgode", "atode"}:
        raise ValueError(
            "_verify_solver_protocol supports only lgode and atode"
        )

    expected_mode = "ones" if model_name == "lgode" else "transport"

    actual_model_type = getattr(solver, "model_type", model_name)
    if actual_model_type != model_name:
        raise RuntimeError(
            f"{model_name} solver has model_type "
            f"{actual_model_type!r}"
        )

    if not bool(getattr(solver, "uses_graph", True)):
        raise RuntimeError(
            f"{model_name} must use graph-based dynamics"
        )

    actual_mode = getattr(solver, "edge_weight_mode", None)
    if actual_mode != expected_mode:
        raise RuntimeError(
            f"{model_name} solver must use edge-weight mode "
            f"{expected_mode!r}; got {actual_mode!r}"
        )

    provider = getattr(solver, "edge_weight_provider", None)

    if model_name == "lgode" and provider is not None:
        raise RuntimeError(
            "LG-ODE must use fixed unit physical-edge weights and must not "
            "contain a transport provider"
        )

    if model_name == "atode" and provider is None:
        raise RuntimeError(
            "AT-ODE must contain a transport edge-weight provider"
        )


def _attach_model_metadata(
    model: nn.Module,
    config: PowerGridModelConfig,
    edge_index: Tensor,
) -> nn.Module:
    """Attach non-parameter protocol metadata."""

    model.model_name = config.model_name
    model.model_type = config.model_name
    model.powergrid_task = config.task
    model.powergrid_config = config
    model.powergrid_config_dict = asdict(config)

    # Physical edges are attached only to graph-based models.
    if config.model_name in {"lgode", "atode"}:
        if "powergrid_edge_index" in model._buffers:
            model._buffers["powergrid_edge_index"] = edge_index.clone()
        else:
            model.register_buffer(
                "powergrid_edge_index",
                edge_index.clone(),
                persistent=True,
            )

    return model


def _build_persistence(
    config: PowerGridModelConfig,
    device: torch.device,
) -> PersistenceBaseline:
    model = PersistenceBaseline(task=config.task)
    return model.to(device)


def _build_independent_latent_ode(
    config: PowerGridModelConfig,
    device: torch.device,
) -> IndependentLatentODE:
    prior = _standard_normal_prior(
        config.latent_dim,
        device,
    )

    model = IndependentLatentODE(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        recognition_dim=config.recognition_dim,
        ode_hidden_dim=config.ode_hidden_dim,
        encoder_layers=config.encoder_layers,
        ode_layers=config.ode_layers,
        augment_dim=config.augmentation_dim,
        solver=config.solver,
        rtol=config.rtol,
        atol=config.atol,
        dropout=config.dropout,
        z0_prior=prior,
        obsrv_std=config.observation_std,
        device=device,
    )

    return model.to(device)


def _build_graph_latent_ode(
    config: PowerGridModelConfig,
    args: Any,
    edge_index: Tensor,
    device: torch.device,
) -> LatentGraphODE:
    if config.model_name not in {"lgode", "atode"}:
        raise ValueError(
            "_build_graph_latent_ode supports only lgode and atode"
        )

    solver_args = _solver_args_namespace(
        args,
        config,
    )

    edge_weight_mode = (
        "ones" if config.model_name == "lgode" else "transport"
    )

    # ------------------------------------------------------------------
    # P0 controlled-initialization order
    # ------------------------------------------------------------------
    # Every shared parameterized component is constructed before AT-ODE's
    # transport provider. This ordering must remain identical for LG-ODE and
    # AT-ODE:
    #
    #   1. shared GTrans encoder
    #   2. shared linear decoder
    #   3. shared NRI ODE network
    #   4. shared GraphODEFunc wrapper
    #   5. AT-ODE-only transport provider
    #   6. solver/context wiring
    #
    # GraphODEFunc is included before the provider because controlled or
    # legacy implementations may initialize the supplied ODE network in the
    # GraphODEFunc constructor.
    # ------------------------------------------------------------------
    encoder = _make_gtrans_encoder(config).to(device)
    decoder = _make_decoder(config).to(device)
    ode_network = _make_graph_ode_network(config).to(device)

    ode_function = _make_graph_ode_function(
        config=config,
        ode_network=ode_network,
        edge_index=edge_index,
        device=device,
        edge_weight_mode=edge_weight_mode,
    )

    # Only now may AT-ODE consume RNG state for extra transport parameters.
    if config.model_name == "atode":
        edge_weight_provider = _make_transport_provider(
            config,
            edge_index,
            device,
        )
    else:
        edge_weight_provider = None

    solver = _make_graph_solver(
        config=config,
        solver_args=solver_args,
        ode_function=ode_function,
        edge_index=edge_index,
        device=device,
        edge_weight_mode=edge_weight_mode,
        edge_weight_provider=edge_weight_provider,
    )

    prior = _standard_normal_prior(
        config.latent_dim,
        device,
    )

    model = _construct_with_supported_keywords(
        LatentGraphODE,
        positional=(),
        keywords={
            "input_dim": config.input_dim,
            "latent_dim": config.latent_dim,
            "encoder_z0": encoder,
            "decoder": decoder,
            "diffeq_solver": solver,
            "z0_prior": prior,
            "device": device,
            "obsrv_std": config.observation_std,
            "model_type": config.model_name,
        },
    )

    if not isinstance(model, nn.Module):
        raise TypeError(
            "LatentGraphODE must inherit torch.nn.Module"
        )

    model = model.to(device)

    # Expose graph/transport modules for diagnostics and tests.
    model.generative_ode_function = ode_function
    model.edge_weight_mode = edge_weight_mode
    model.edge_weight_provider = edge_weight_provider
    model.model_type = config.model_name

    _verify_solver_protocol(
        solver,
        config.model_name,
    )

    return model


def build_powergrid_lgode_model(
    model_name: str,
    input_dim: int,
    num_nodes: int,
    edge_index: Tensor,
    args: Any,
    device: Union[str, torch.device],
) -> nn.Module:
    """
    Build one power-grid experiment model.

    Parameters
    ----------
    model_name:
        ``persistence``, ``latentode``, ``lgode`` or ``atode``.
    input_dim:
        Number of normalized bus-state features.
    num_nodes:
        Number of buses.
    edge_index:
        Physical grid edges with shape [2, E] or [E, 2].
    args:
        Namespace or mapping containing model hyperparameters.
    device:
        PyTorch device.

    Notes
    -----
    For paper runs comparing LG-ODE with AT-ODE, prefer
    ``build_lgode_atode_protocol_pair``. It constructs both models and
    automatically validates the controlled-ablation protocol.
    """

    normalized_name = _normalize_model_name(model_name)
    resolved_device = torch.device(device)

    config = _resolve_config(
        model_name=normalized_name,
        input_dim=int(input_dim),
        num_nodes=int(num_nodes),
        args=args,
    )

    physical_edge_index = _validate_edge_index(
        edge_index,
        num_nodes=config.num_nodes,
    ).to(resolved_device)

    with _deterministic_initialization(
        config.seed,
        resolved_device,
    ):
        if normalized_name == "persistence":
            model = _build_persistence(
                config,
                resolved_device,
            )
        elif normalized_name == "latentode":
            model = _build_independent_latent_ode(
                config,
                resolved_device,
            )
        else:
            model = _build_graph_latent_ode(
                config,
                args,
                physical_edge_index,
                resolved_device,
            )

    return _attach_model_metadata(
        model,
        config,
        physical_edge_index,
    )


def build_lgode_atode_protocol_pair(
    input_dim: int,
    num_nodes: int,
    edge_index: Tensor,
    args: Any,
    device: Union[str, torch.device],
    *,
    check_initial_values: bool = True,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tuple[LatentGraphODE, LatentGraphODE]:
    """
    Build and validate the controlled LG-ODE/AT-ODE model pair.

    This is the recommended entry point for paper experiments. Both models
    are constructed independently from the same configured seed. The shared
    initialization and architecture are checked automatically before either
    model is returned.

    Returns
    -------
    (lgode, atode)
        The validated LG-ODE and AT-ODE models.
    """

    lgode = build_powergrid_lgode_model(
        model_name="lgode",
        input_dim=input_dim,
        num_nodes=num_nodes,
        edge_index=edge_index,
        args=args,
        device=device,
    )

    atode = build_powergrid_lgode_model(
        model_name="atode",
        input_dim=input_dim,
        num_nodes=num_nodes,
        edge_index=edge_index,
        args=args,
        device=device,
    )

    if not isinstance(lgode, LatentGraphODE):
        raise TypeError(
            "LG-ODE factory result is not a LatentGraphODE"
        )

    if not isinstance(atode, LatentGraphODE):
        raise TypeError(
            "AT-ODE factory result is not a LatentGraphODE"
        )

    # Mandatory startup validation for the paired paper protocol.
    assert_lgode_atode_protocol_match(
        lgode,
        atode,
        check_initial_values=check_initial_values,
        rtol=rtol,
        atol=atol,
    )

    return lgode, atode


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Count trainable scalar parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(
    model: nn.Module,
) -> int:
    """Count all scalar parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def shared_graph_state_dict(
    model: nn.Module,
) -> Dict[str, Tensor]:
    """
    Extract state entries shared by LG-ODE and AT-ODE.

    AT-ODE-only transport-provider parameters and transport-only state are
    excluded.
    """

    excluded_fragments = (
        "edge_weight_provider",
        "attention_transport",
        "transport_provider",
        "transport.",
    )

    result: Dict[str, Tensor] = {}

    for name, value in model.state_dict().items():
        if any(
            fragment in name
            for fragment in excluded_fragments
        ):
            continue

        result[name] = value

    return result


def assert_lgode_atode_protocol_match(
    lgode: nn.Module,
    atode: nn.Module,
    *,
    check_initial_values: bool = True,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """
    Assert that LG-ODE and AT-ODE differ only by transport edge weighting.
    """

    lgode_config = getattr(
        lgode,
        "powergrid_config",
        None,
    )
    atode_config = getattr(
        atode,
        "powergrid_config",
        None,
    )

    if not isinstance(
        lgode_config,
        PowerGridModelConfig,
    ):
        raise TypeError(
            "lgode was not built by build_powergrid_lgode_model"
        )

    if not isinstance(
        atode_config,
        PowerGridModelConfig,
    ):
        raise TypeError(
            "atode was not built by build_powergrid_lgode_model"
        )

    if lgode_config.model_name != "lgode":
        raise ValueError(
            "First model must be LG-ODE; got "
            f"{lgode_config.model_name}"
        )

    if atode_config.model_name != "atode":
        raise ValueError(
            "Second model must be AT-ODE; got "
            f"{atode_config.model_name}"
        )

    lgode_signature = (
        lgode_config.common_graph_model_signature()
    )
    atode_signature = (
        atode_config.common_graph_model_signature()
    )

    if lgode_signature != atode_signature:
        all_keys = sorted(
            set(lgode_signature).union(atode_signature)
        )

        differing = {
            key: (
                lgode_signature.get(key),
                atode_signature.get(key),
            )
            for key in all_keys
            if lgode_signature.get(key)
            != atode_signature.get(key)
        }

        raise AssertionError(
            "LG-ODE and AT-ODE common configurations differ: "
            f"{differing}"
        )

    if getattr(lgode, "model_type", None) != "lgode":
        raise AssertionError(
            "LG-ODE model_type must be 'lgode'"
        )

    if getattr(atode, "model_type", None) != "atode":
        raise AssertionError(
            "AT-ODE model_type must be 'atode'"
        )

    if getattr(lgode, "edge_weight_mode", None) != "ones":
        raise AssertionError(
            "LG-ODE must use unit edge weights"
        )

    if getattr(atode, "edge_weight_mode", None) != "transport":
        raise AssertionError(
            "AT-ODE must use transport edge weights"
        )

    if getattr(lgode, "edge_weight_provider", None) is not None:
        raise AssertionError(
            "LG-ODE unexpectedly has a transport provider"
        )

    if getattr(atode, "edge_weight_provider", None) is None:
        raise AssertionError(
            "AT-ODE does not have a transport provider"
        )

    lgode_solver = getattr(
        lgode,
        "diffeq_solver",
        None,
    )
    atode_solver = getattr(
        atode,
        "diffeq_solver",
        None,
    )

    if lgode_solver is None:
        raise AssertionError(
            "LG-ODE does not expose diffeq_solver"
        )

    if atode_solver is None:
        raise AssertionError(
            "AT-ODE does not expose diffeq_solver"
        )

    _verify_solver_protocol(
        lgode_solver,
        "lgode",
    )
    _verify_solver_protocol(
        atode_solver,
        "atode",
    )

    lgode_edges = getattr(
        lgode,
        "powergrid_edge_index",
        None,
    )
    atode_edges = getattr(
        atode,
        "powergrid_edge_index",
        None,
    )

    if lgode_edges is None or atode_edges is None:
        raise AssertionError(
            "Both graph models must expose powergrid_edge_index"
        )

    if not torch.equal(
        lgode_edges,
        atode_edges,
    ):
        raise AssertionError(
            "LG-ODE and AT-ODE use different physical edge indices"
        )

    if not check_initial_values:
        return

    lgode_state = shared_graph_state_dict(lgode)
    atode_state = shared_graph_state_dict(atode)

    lgode_names = set(lgode_state)
    atode_names = set(atode_state)

    missing_from_atode = sorted(
        lgode_names - atode_names
    )
    missing_from_lgode = sorted(
        atode_names - lgode_names
    )

    if missing_from_atode or missing_from_lgode:
        raise AssertionError(
            "LG-ODE and AT-ODE shared state dictionaries have different "
            "entries. "
            f"Missing from AT-ODE: {missing_from_atode[:10]}; "
            f"missing from LG-ODE: {missing_from_lgode[:10]}"
        )

    common_names = sorted(
        lgode_names.intersection(atode_names)
    )

    if not common_names:
        raise AssertionError(
            "LG-ODE and AT-ODE have no common state-dict entries"
        )

    mismatches = []

    for name in common_names:
        left = lgode_state[name]
        right = atode_state[name]

        if left.shape != right.shape:
            mismatches.append(
                f"{name}: shape {tuple(left.shape)} != "
                f"{tuple(right.shape)}"
            )
            continue

        if left.dtype != right.dtype:
            mismatches.append(
                f"{name}: dtype {left.dtype} != {right.dtype}"
            )
            continue

        if left.is_floating_point() or left.is_complex():
            equal = torch.allclose(
                left,
                right,
                rtol=rtol,
                atol=atol,
            )
        else:
            equal = torch.equal(
                left,
                right,
            )

        if not equal:
            mismatches.append(name)

    if mismatches:
        preview = mismatches[:10]
        suffix = "..." if len(mismatches) > 10 else ""

        raise AssertionError(
            "LG-ODE and AT-ODE shared initial parameters differ. "
            "This invalidates the controlled transport ablation. "
            f"Mismatches: {preview}{suffix}"
        )


__all__ = [
    "PowerGridModelConfig",
    "SUPPORTED_POWERGRID_MODELS",
    "assert_lgode_atode_protocol_match",
    "build_lgode_atode_protocol_pair",
    "build_powergrid_lgode_model",
    "count_total_parameters",
    "count_trainable_parameters",
    "shared_graph_state_dict",
]
