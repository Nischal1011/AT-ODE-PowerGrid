"""
Solver-safe causal attention transport for LG-ODE.

The module constructs time-dependent weights for directed physical edges.
Weights are precomputed on a deterministic time grid before the ODE solve.
The ODE vector field then obtains weights through linear interpolation.

Nothing is mutated during an ODE function evaluation. This is important when
using adaptive ODE solvers, adjoint gradients, rejected solver steps, or
repeated evaluations at the same time.

Overview
--------
For every directed physical edge j -> i:

1. An attention score is inferred from the encoded initial latent states.
2. The score is placed into an observation-age distribution.
3. The age distribution is transported toward older age bins as forecast
   time advances.
4. An exponential temporal kernel reads the transported evidence.
5. Active incoming edges are normalized to have mean weight one.

The normalization makes the scale comparable with an LG-ODE control whose
active physical-edge weights are all one.

Tensor conventions
------------------
S
    Number of latent trajectory samples.

B
    Batch size.

N
    Number of physical nodes.

E
    Number of directed candidate edges.

D
    Latent dimension.

K
    Number of age bins.

T
    Number of transport-grid times.

Initial latent state
--------------------
Accepted layouts:

    [B, N, D]
    [S, B * N, D]
    [S, B, N, D]

Internally these become:

    [S * B, N, D]

Latest observation times
------------------------
Accepted layouts:

    [B, N]
    [B * N]
    [S, B, N]
    [S * B, N]

Physical edge mask
------------------
Accepted layouts:

    [B, E]
    [S * B, E]

The mask must be positive on active physical edges and zero elsewhere.

Transport cache
---------------
mu_grid:
    [T, S * B, E, K]

edge_weight_grid:
    [T, S * B, E, 1]

The cache can be queried at arbitrary scalar ODE times through linear
interpolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def _inverse_softplus(value: float) -> float:
    """Return x such that softplus(x) is approximately value."""

    if value <= 0.0:
        raise ValueError(
            "The inverse-softplus input must be strictly positive."
        )

    value_tensor = torch.tensor(
        float(value),
        dtype=torch.float64,
    )

    result = (
        value_tensor
        + torch.log(
            -torch.expm1(-value_tensor)
        )
    )

    return float(result.item())


def _require_finite(
    tensor: Tensor,
    name: str,
) -> None:
    """Raise an informative exception if a tensor is not finite."""

    if not torch.isfinite(tensor).all():
        finite_fraction = float(
            torch.isfinite(tensor)
            .to(torch.float32)
            .mean()
            .detach()
            .cpu()
        )

        raise FloatingPointError(
            f"{name} contains nonfinite values. "
            f"Finite fraction: {finite_fraction:.6f}."
        )


def _as_scalar_time(
    value: Tensor | float,
    reference: Tensor,
) -> Tensor:
    """Convert a Python value or one-element tensor into a scalar tensor."""

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                "Transport-cache queries require a scalar time. "
                f"Received shape {tuple(value.shape)}."
            )

        return value.reshape(()).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    return torch.tensor(
        float(value),
        device=reference.device,
        dtype=reference.dtype,
    )


# ---------------------------------------------------------------------------
# Immutable cache queried by the ODE vector field
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionTransportCache:
    """
    Precomputed solver-safe transport trajectory.

    Attributes
    ----------
    time_grid
        Strictly increasing one-dimensional tensor [T].

    mu_grid
        Transported age-bin evidence [T, effective_batch, E, K].

    edge_weight_grid
        Normalized physical-edge weights [T, effective_batch, E, 1].

    physical_edge_mask
        Boolean mask [effective_batch, E].

    diagnostics
        Tensor diagnostics computed when the cache was created.
    """

    time_grid: Tensor
    mu_grid: Tensor
    edge_weight_grid: Tensor
    physical_edge_mask: Tensor
    diagnostics: Dict[str, Tensor]

    def edge_weights_at(
        self,
        time_value: Tensor | float,
    ) -> Tensor:
        """
        Return linearly interpolated edge weights.

        Parameters
        ----------
        time_value
            Scalar ODE time.

        Returns
        -------
        Tensor
            Edge weights with shape [effective_batch, E, 1].
        """

        time_value = _as_scalar_time(
            time_value,
            self.time_grid,
        )

        if self.time_grid.numel() == 1:
            return self.edge_weight_grid[0]

        # Clamp outside-grid evaluations to the nearest endpoint. Adaptive
        # solvers may evaluate at a value differing from an endpoint by tiny
        # floating-point error.
        clamped_time = torch.clamp(
            time_value,
            min=self.time_grid[0],
            max=self.time_grid[-1],
        )

        upper_index = torch.searchsorted(
            self.time_grid,
            clamped_time,
            right=False,
        )

        upper_index = torch.clamp(
            upper_index,
            min=1,
            max=self.time_grid.numel() - 1,
        )

        lower_index = upper_index - 1

        lower_time = self.time_grid[lower_index]
        upper_time = self.time_grid[upper_index]

        denominator = torch.clamp(
            upper_time - lower_time,
            min=torch.finfo(self.time_grid.dtype).eps,
        )

        interpolation = (
            clamped_time - lower_time
        ) / denominator

        lower_weights = self.edge_weight_grid[lower_index]
        upper_weights = self.edge_weight_grid[upper_index]

        result = (
            lower_weights
            + interpolation
            * (upper_weights - lower_weights)
        )

        # Interpolation between zero-valued inactive edges remains zero, but
        # apply the mask explicitly to make the invariant obvious.
        result = result * self.physical_edge_mask.unsqueeze(-1).to(
            dtype=result.dtype
        )

        return result

    def age_evidence_at(
        self,
        time_value: Tensor | float,
    ) -> Tensor:
        """
        Return linearly interpolated age-bin evidence.

        Returns
        -------
        Tensor
            Age-bin evidence with shape [effective_batch, E, K].
        """

        time_value = _as_scalar_time(
            time_value,
            self.time_grid,
        )

        if self.time_grid.numel() == 1:
            return self.mu_grid[0]

        clamped_time = torch.clamp(
            time_value,
            min=self.time_grid[0],
            max=self.time_grid[-1],
        )

        upper_index = torch.searchsorted(
            self.time_grid,
            clamped_time,
            right=False,
        )

        upper_index = torch.clamp(
            upper_index,
            min=1,
            max=self.time_grid.numel() - 1,
        )

        lower_index = upper_index - 1

        lower_time = self.time_grid[lower_index]
        upper_time = self.time_grid[upper_index]

        denominator = torch.clamp(
            upper_time - lower_time,
            min=torch.finfo(self.time_grid.dtype).eps,
        )

        interpolation = (
            clamped_time - lower_time
        ) / denominator

        lower_mu = self.mu_grid[lower_index]
        upper_mu = self.mu_grid[upper_index]

        result = (
            lower_mu
            + interpolation
            * (upper_mu - lower_mu)
        )

        result = result * self.physical_edge_mask[
            ...,
            None,
        ].to(dtype=result.dtype)

        return result

    def detached_diagnostics(self) -> Dict[str, float]:
        """Return JSON-compatible scalar diagnostics."""

        output: Dict[str, float] = {}

        for name, value in self.diagnostics.items():
            if not torch.is_tensor(value):
                output[name] = float(value)
                continue

            detached = value.detach().cpu()

            if detached.numel() != 1:
                raise ValueError(
                    f"Diagnostic {name!r} is not scalar: "
                    f"shape={tuple(detached.shape)}."
                )

            output[name] = float(detached.item())

        return output


# ---------------------------------------------------------------------------
# Causal attention transport
# ---------------------------------------------------------------------------

class SolverSafeAttentionTransport(nn.Module):
    """
    Construct deterministic time-dependent physical-edge weights.

    Parameters
    ----------
    latent_dim
        Dimension of every node latent state.

    edge_index
        Directed candidate edges with shape [2, E]. For edge e,
        edge_index[0, e] is the sender and edge_index[1, e] is the receiver.

    num_nodes
        Number of physical nodes.

    num_bins
        Number of observation-age bins.

    max_age
        Largest represented normalized age. Evidence older than max_age is
        retained in the final bin instead of being discarded.

    hidden_dim
        Hidden dimension of the edge evidence network.

    attention_dim
        Per-head attention dimension.

    num_heads
        Number of attention heads.

    initial_speed
        Initial positive speed at which evidence advances through age.

    initial_decay
        Initial positive exponential temporal-decay rate.

    learnable_speed
        Whether transport speed is trainable.

    learnable_decay
        Whether temporal decay is trainable.

    dropout
        Dropout applied to projected queries and keys during training.

    epsilon
        Positive numerical stabilization constant.
    """

    def __init__(
        self,
        latent_dim: int,
        edge_index: Tensor,
        num_nodes: int,
        num_bins: int = 16,
        max_age: float = 4.0,
        hidden_dim: int = 64,
        attention_dim: int = 16,
        num_heads: int = 4,
        initial_speed: float = 1.0,
        initial_decay: float = 1.0,
        learnable_speed: bool = True,
        learnable_decay: bool = True,
        dropout: float = 0.0,
        epsilon: float = 1.0e-8,
    ):
        super().__init__()

        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")

        if num_nodes < 1:
            raise ValueError("num_nodes must be positive.")

        if num_bins < 2:
            raise ValueError("num_bins must be at least two.")

        if max_age <= 0.0:
            raise ValueError("max_age must be positive.")

        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")

        if attention_dim < 1:
            raise ValueError("attention_dim must be positive.")

        if num_heads < 1:
            raise ValueError("num_heads must be positive.")

        if initial_speed <= 0.0:
            raise ValueError("initial_speed must be positive.")

        if initial_decay <= 0.0:
            raise ValueError("initial_decay must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        edge_index = torch.as_tensor(
            edge_index,
            dtype=torch.long,
        )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E]. "
                f"Received {tuple(edge_index.shape)}."
            )

        if edge_index.shape[1] < 1:
            raise ValueError(
                "edge_index must contain at least one directed edge."
            )

        if int(edge_index.min()) < 0:
            raise ValueError("edge_index contains a negative node index.")

        if int(edge_index.max()) >= num_nodes:
            raise ValueError(
                "edge_index contains a node index outside num_nodes."
            )

        if torch.any(edge_index[0] == edge_index[1]):
            raise ValueError(
                "Self-edges should not be supplied to attention transport."
            )

        self.latent_dim = int(latent_dim)
        self.num_nodes = int(num_nodes)
        self.num_bins = int(num_bins)
        self.max_age = float(max_age)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.epsilon = float(epsilon)

        self.register_buffer(
            "edge_index",
            edge_index.contiguous(),
            persistent=True,
        )

        age_centers = torch.linspace(
            0.0,
            self.max_age,
            self.num_bins,
            dtype=torch.float32,
        )

        self.register_buffer(
            "age_centers",
            age_centers,
            persistent=True,
        )

        self.query_projection = nn.Linear(
            self.latent_dim,
            self.num_heads * self.attention_dim,
            bias=False,
        )

        self.key_projection = nn.Linear(
            self.latent_dim,
            self.num_heads * self.attention_dim,
            bias=False,
        )

        # The MLP supplements dot-product attention with relative-state
        # evidence. It receives sender, receiver, signed difference, and
        # absolute difference.
        self.edge_evidence_network = nn.Sequential(
            nn.Linear(
                4 * self.latent_dim,
                self.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_dim,
                self.num_heads,
            ),
        )

        self.head_bias = nn.Parameter(
            torch.zeros(self.num_heads)
        )

        self.dropout = nn.Dropout(dropout)

        speed_raw = torch.tensor(
            _inverse_softplus(initial_speed),
            dtype=torch.float32,
        )

        decay_raw = torch.tensor(
            _inverse_softplus(initial_decay),
            dtype=torch.float32,
        )

        self.raw_transport_speed = nn.Parameter(
            speed_raw,
            requires_grad=learnable_speed,
        )

        self.raw_decay_rate = nn.Parameter(
            decay_raw,
            requires_grad=learnable_decay,
        )

        self.reset_parameters()

        # reset_parameters initializes linear layers only. Restore requested
        # positive scalar initializations afterward.
        with torch.no_grad():
            self.raw_transport_speed.fill_(
                _inverse_softplus(initial_speed)
            )
            self.raw_decay_rate.fill_(
                _inverse_softplus(initial_decay)
            )

    @property
    def num_edges(self) -> int:
        """Number of directed candidate edges."""

        return int(self.edge_index.shape[1])

    def positive_transport_speed(self) -> Tensor:
        """Return the strictly positive transport speed."""

        return (
            F.softplus(self.raw_transport_speed)
            + self.epsilon
        )

    def positive_decay_rate(self) -> Tensor:
        """Return the strictly positive temporal decay rate."""

        return (
            F.softplus(self.raw_decay_rate)
            + self.epsilon
        )

    def reset_parameters(self) -> None:
        """Initialize trainable projections and evidence networks."""

        nn.init.xavier_uniform_(
            self.query_projection.weight
        )

        nn.init.xavier_uniform_(
            self.key_projection.weight
        )

        for module in self.edge_evidence_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.head_bias)

    # ------------------------------------------------------------------
    # Shape normalization
    # ------------------------------------------------------------------

    def _reshape_initial_state(
        self,
        z0: Tensor,
    ) -> Tuple[Tensor, int, int]:
        """
        Convert accepted z0 layouts to [effective_batch, N, D].

        Returns
        -------
        reshaped
            Tensor [S * B, N, D].

        sample_count
            Number of latent samples S.

        batch_size
            Original trajectory batch size B.
        """

        if z0.ndim == 4:
            sample_count, batch_size, nodes, latent_dim = z0.shape

            if nodes != self.num_nodes:
                raise ValueError(
                    f"z0 contains {nodes} nodes, expected "
                    f"{self.num_nodes}."
                )

            if latent_dim != self.latent_dim:
                raise ValueError(
                    f"z0 latent dimension is {latent_dim}, expected "
                    f"{self.latent_dim}."
                )

            reshaped = z0.reshape(
                sample_count * batch_size,
                nodes,
                latent_dim,
            )

            return (
                reshaped,
                int(sample_count),
                int(batch_size),
            )

        if z0.ndim != 3:
            raise ValueError(
                "z0 must have shape [B,N,D], [S,B*N,D], or "
                f"[S,B,N,D]. Received {tuple(z0.shape)}."
            )

        first, second, latent_dim = z0.shape

        if latent_dim != self.latent_dim:
            raise ValueError(
                f"z0 latent dimension is {latent_dim}, expected "
                f"{self.latent_dim}."
            )

        # Layout [B, N, D].
        if second == self.num_nodes:
            return (
                z0,
                1,
                int(first),
            )

        # Layout [S, B*N, D].
        if second % self.num_nodes != 0:
            raise ValueError(
                f"The flattened node dimension {second} is not divisible "
                f"by num_nodes={self.num_nodes}."
            )

        sample_count = int(first)
        batch_size = int(second // self.num_nodes)

        reshaped = z0.reshape(
            sample_count,
            batch_size,
            self.num_nodes,
            self.latent_dim,
        ).reshape(
            sample_count * batch_size,
            self.num_nodes,
            self.latent_dim,
        )

        return (
            reshaped,
            sample_count,
            batch_size,
        )

    def _reshape_latest_observation_time(
        self,
        latest_observation_time: Tensor,
        sample_count: int,
        batch_size: int,
        reference: Tensor,
    ) -> Tensor:
        """Convert observation times to [S*B, N]."""

        latest = torch.as_tensor(
            latest_observation_time,
            device=reference.device,
            dtype=reference.dtype,
        )

        effective_batch = sample_count * batch_size

        if latest.ndim == 1:
            if latest.numel() == self.num_nodes:
                latest = latest.reshape(
                    1,
                    self.num_nodes,
                )
            elif latest.numel() == batch_size * self.num_nodes:
                latest = latest.reshape(
                    batch_size,
                    self.num_nodes,
                )
            elif latest.numel() == (
                effective_batch * self.num_nodes
            ):
                latest = latest.reshape(
                    effective_batch,
                    self.num_nodes,
                )
            else:
                raise ValueError(
                    "Cannot infer latest_observation_time layout from "
                    f"shape {tuple(latest.shape)}."
                )

        elif latest.ndim == 2:
            if latest.shape[1] != self.num_nodes:
                raise ValueError(
                    "latest_observation_time must end with num_nodes. "
                    f"Received {tuple(latest.shape)}."
                )

        elif latest.ndim == 3:
            expected = (
                sample_count,
                batch_size,
                self.num_nodes,
            )

            if tuple(latest.shape) != expected:
                raise ValueError(
                    "Three-dimensional latest_observation_time must have "
                    f"shape {expected}, received {tuple(latest.shape)}."
                )

            latest = latest.reshape(
                effective_batch,
                self.num_nodes,
            )

        else:
            raise ValueError(
                "latest_observation_time must have one, two, or three "
                f"dimensions. Received {tuple(latest.shape)}."
            )

        rows = int(latest.shape[0])

        if rows == effective_batch:
            result = latest
        elif rows == batch_size:
            result = latest.repeat(
                sample_count,
                1,
            )
        elif rows == 1:
            result = latest.repeat(
                effective_batch,
                1,
            )
        else:
            raise ValueError(
                f"latest_observation_time has {rows} rows; expected 1, "
                f"{batch_size}, or {effective_batch}."
            )

        _require_finite(
            result,
            "latest_observation_time",
        )

        return result

    def _reshape_physical_edge_mask(
        self,
        physical_edge_mask: Optional[Tensor],
        sample_count: int,
        batch_size: int,
        reference: Tensor,
    ) -> Tensor:
        """Convert the physical-edge mask to [S*B, E]."""

        effective_batch = sample_count * batch_size

        if physical_edge_mask is None:
            return torch.ones(
                effective_batch,
                self.num_edges,
                device=reference.device,
                dtype=torch.bool,
            )

        mask = torch.as_tensor(
            physical_edge_mask,
            device=reference.device,
        )

        if mask.ndim == 1:
            if mask.numel() != self.num_edges:
                raise ValueError(
                    "One-dimensional physical_edge_mask must contain "
                    f"{self.num_edges} entries, received {mask.numel()}."
                )

            mask = mask.reshape(
                1,
                self.num_edges,
            )

        elif mask.ndim == 2:
            if mask.shape[1] != self.num_edges:
                raise ValueError(
                    "physical_edge_mask must end with num_edges. "
                    f"Received {tuple(mask.shape)}."
                )

        elif mask.ndim == 3:
            expected = (
                sample_count,
                batch_size,
                self.num_edges,
            )

            if tuple(mask.shape) != expected:
                raise ValueError(
                    "Three-dimensional physical_edge_mask must have "
                    f"shape {expected}, received {tuple(mask.shape)}."
                )

            mask = mask.reshape(
                effective_batch,
                self.num_edges,
            )

        else:
            raise ValueError(
                "physical_edge_mask must have one, two, or three "
                f"dimensions. Received {tuple(mask.shape)}."
            )

        rows = int(mask.shape[0])

        if rows == effective_batch:
            result = mask
        elif rows == batch_size:
            result = mask.repeat(
                sample_count,
                1,
            )
        elif rows == 1:
            result = mask.repeat(
                effective_batch,
                1,
            )
        else:
            raise ValueError(
                f"physical_edge_mask has {rows} rows; expected 1, "
                f"{batch_size}, or {effective_batch}."
            )

        return result > 0

    # ------------------------------------------------------------------
    # Attention evidence
    # ------------------------------------------------------------------

    def _edge_attention_evidence(
        self,
        z0: Tensor,
        physical_edge_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Infer positive evidence for every directed physical edge.

        Parameters
        ----------
        z0
            Initial latent states [effective_batch, N, D].

        physical_edge_mask
            Active physical edges [effective_batch, E].

        Returns
        -------
        evidence
            Positive scalar evidence [effective_batch, E].

        head_evidence
            Positive evidence per attention head
            [effective_batch, E, H].
        """

        sender_index = self.edge_index[0]
        receiver_index = self.edge_index[1]

        sender_state = z0[:, sender_index, :]
        receiver_state = z0[:, receiver_index, :]

        query = self.query_projection(
            receiver_state
        ).reshape(
            z0.shape[0],
            self.num_edges,
            self.num_heads,
            self.attention_dim,
        )

        key = self.key_projection(
            sender_state
        ).reshape(
            z0.shape[0],
            self.num_edges,
            self.num_heads,
            self.attention_dim,
        )

        query = self.dropout(query)
        key = self.dropout(key)

        dot_score = (
            query * key
        ).sum(dim=-1) / math.sqrt(
            float(self.attention_dim)
        )

        relative_features = torch.cat(
            [
                sender_state,
                receiver_state,
                receiver_state - sender_state,
                torch.abs(receiver_state - sender_state),
            ],
            dim=-1,
        )

        relational_score = self.edge_evidence_network(
            relative_features
        )

        logits = (
            dot_score
            + relational_score
            + self.head_bias.view(1, 1, -1)
        )

        head_evidence = (
            F.softplus(logits)
            + self.epsilon
        )

        head_evidence = head_evidence * physical_edge_mask[
            ...,
            None,
        ].to(dtype=head_evidence.dtype)

        evidence = head_evidence.mean(dim=-1)

        _require_finite(
            evidence,
            "edge attention evidence",
        )

        return evidence, head_evidence

    # ------------------------------------------------------------------
    # Age transport
    # ------------------------------------------------------------------

    def _deposit_in_age_bins(
        self,
        evidence: Tensor,
        age: Tensor,
        physical_edge_mask: Tensor,
    ) -> Tensor:
        """
        Deposit edge evidence into neighboring age bins.

        Linear interpolation between bins makes the age transport
        differentiable with respect to transport speed, except at the usual
        piecewise-linear boundaries.

        Evidence older than max_age is retained in the final bin.
        """

        if evidence.shape != age.shape:
            raise ValueError(
                "Evidence and age shapes must match. "
                f"Received {tuple(evidence.shape)} and "
                f"{tuple(age.shape)}."
            )

        spacing = self.max_age / float(
            self.num_bins - 1
        )

        bounded_age = torch.clamp(
            age,
            min=0.0,
            max=self.max_age,
        )

        continuous_index = bounded_age / spacing

        lower_index = torch.floor(
            continuous_index
        ).to(torch.long)

        upper_index = torch.clamp(
            lower_index + 1,
            max=self.num_bins - 1,
        )

        upper_fraction = (
            continuous_index
            - lower_index.to(continuous_index.dtype)
        )

        # At the final bin lower_index == upper_index. Assign all mass to
        # that bin rather than adding it twice.
        same_bin = lower_index == upper_index

        upper_fraction = torch.where(
            same_bin,
            torch.zeros_like(upper_fraction),
            upper_fraction,
        )

        lower_fraction = 1.0 - upper_fraction

        effective_batch, edges = evidence.shape

        mu = torch.zeros(
            effective_batch,
            edges,
            self.num_bins,
            device=evidence.device,
            dtype=evidence.dtype,
        )

        mu.scatter_add_(
            dim=-1,
            index=lower_index.unsqueeze(-1),
            src=(
                evidence * lower_fraction
            ).unsqueeze(-1),
        )

        mu.scatter_add_(
            dim=-1,
            index=upper_index.unsqueeze(-1),
            src=(
                evidence * upper_fraction
            ).unsqueeze(-1),
        )

        mu = mu * physical_edge_mask[
            ...,
            None,
        ].to(dtype=mu.dtype)

        return mu

    def _normalize_incoming_edges(
        self,
        raw_edge_weight: Tensor,
        physical_edge_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Scale active incoming weights using the initial-time incoming mean.

        For every receiver node:

            sum(normalized incoming weights at t0) = active incoming degree

        This keeps the message scale comparable to an LG-ODE control using
        unit physical-edge weights without cancelling later temporal decay.

        Parameters
        ----------
        raw_edge_weight
            Positive edge weights [T, effective_batch, E].

        physical_edge_mask
            Active physical edges [effective_batch, E].

        Returns
        -------
        normalized
            Normalized edge weights [T, effective_batch, E].

        fallback_fraction
            Fraction of receiver-batch groups that required the static
            unit-weight fallback at the initial time.
        """

        if raw_edge_weight.ndim != 3:
            raise ValueError(
                "raw_edge_weight must have shape [T,B,E]. "
                f"Received {tuple(raw_edge_weight.shape)}."
            )

        time_count, effective_batch, edges = raw_edge_weight.shape

        if edges != self.num_edges:
            raise ValueError(
                f"raw_edge_weight contains {edges} edges, expected "
                f"{self.num_edges}."
            )

        if physical_edge_mask.shape != (
            effective_batch,
            self.num_edges,
        ):
            raise ValueError(
                "physical_edge_mask shape is incompatible with "
                "raw_edge_weight."
            )

        receiver_index = self.edge_index[1]

        mask_float = physical_edge_mask.to(
            dtype=raw_edge_weight.dtype
        )

        active_raw = raw_edge_weight * mask_float.unsqueeze(0)

        initial_incoming_sum = torch.zeros(
            effective_batch,
            self.num_nodes,
            device=raw_edge_weight.device,
            dtype=raw_edge_weight.dtype,
        )

        initial_incoming_sum.index_add_(
            dim=1,
            index=receiver_index,
            source=active_raw[0],
        )

        incoming_degree = torch.zeros(
            effective_batch,
            self.num_nodes,
            device=raw_edge_weight.device,
            dtype=raw_edge_weight.dtype,
        )

        incoming_degree.index_add_(
            dim=1,
            index=receiver_index,
            source=mask_float,
        )

        denominator = initial_incoming_sum[
            :,
            receiver_index,
        ].unsqueeze(0)

        degree_per_edge = incoming_degree[
            :,
            receiver_index,
        ].unsqueeze(0)

        valid_denominator = (
            denominator > self.epsilon
        ) & (
            degree_per_edge > 0.0
        )

        normalized = torch.where(
            valid_denominator,
            active_raw
            * degree_per_edge
            / torch.clamp(
                denominator,
                min=self.epsilon,
            ),
            mask_float.unsqueeze(0).expand(
                time_count,
                -1,
                -1,
            ),
        )

        normalized = normalized * mask_float.unsqueeze(0)

        receiver_has_edges = incoming_degree > 0.0

        valid_receiver_sum = initial_incoming_sum > self.epsilon

        fallback_receiver = (
            receiver_has_edges & (~valid_receiver_sum)
        )

        denominator_count = torch.clamp(
            receiver_has_edges.sum(),
            min=1,
        )

        fallback_fraction = (
            fallback_receiver.sum().to(raw_edge_weight.dtype)
            / denominator_count.to(raw_edge_weight.dtype)
        )

        return normalized, fallback_fraction

    # ------------------------------------------------------------------
    # Public cache construction
    # ------------------------------------------------------------------

    def build_cache(
        self,
        z0: Tensor,
        latest_observation_time: Tensor,
        time_grid: Tensor,
        physical_edge_mask: Optional[Tensor] = None,
    ) -> AttentionTransportCache:
        """
        Construct a deterministic transport cache for one ODE solve.

        Parameters
        ----------
        z0
            Initial latent states. Accepted layouts are documented at the
            top of this module.

        latest_observation_time
            Latest observed normalized time for every physical node.

        time_grid
            Strictly increasing scalar times at which transport is
            precomputed. Include the ODE initial time and every requested
            prediction time.

        physical_edge_mask
            Optional active-edge mask. When omitted, every edge in
            edge_index is treated as active.

        Returns
        -------
        AttentionTransportCache
            Immutable cache queried by the ODE vector field.
        """

        z0_reshaped, sample_count, batch_size = (
            self._reshape_initial_state(z0)
        )

        _require_finite(
            z0_reshaped,
            "z0",
        )

        time_grid = torch.as_tensor(
            time_grid,
            device=z0_reshaped.device,
            dtype=z0_reshaped.dtype,
        ).flatten()

        if time_grid.numel() < 1:
            raise ValueError(
                "time_grid must contain at least one time."
            )

        _require_finite(
            time_grid,
            "time_grid",
        )

        if time_grid.numel() > 1:
            differences = torch.diff(time_grid)

            if not torch.all(differences > 0.0):
                raise ValueError(
                    "time_grid must be strictly increasing."
                )

        latest = self._reshape_latest_observation_time(
            latest_observation_time=latest_observation_time,
            sample_count=sample_count,
            batch_size=batch_size,
            reference=z0_reshaped,
        )

        mask = self._reshape_physical_edge_mask(
            physical_edge_mask=physical_edge_mask,
            sample_count=sample_count,
            batch_size=batch_size,
            reference=z0_reshaped,
        )

        evidence, head_evidence = self._edge_attention_evidence(
            z0=z0_reshaped,
            physical_edge_mask=mask,
        )

        sender_index = self.edge_index[0]

        latest_sender_time = latest[
            :,
            sender_index,
        ]

        start_time = time_grid[0]

        initial_age = torch.clamp(
            start_time - latest_sender_time,
            min=0.0,
        )

        speed = self.positive_transport_speed()
        decay_rate = self.positive_decay_rate()

        elapsed = torch.clamp(
            time_grid - start_time,
            min=0.0,
        )

        age_grid = (
            initial_age.unsqueeze(0)
            + speed
            * elapsed.view(-1, 1, 1)
        )

        age_grid = age_grid.expand(
            -1,
            evidence.shape[0],
            -1,
        )

        mu_values = []

        for time_index in range(time_grid.numel()):
            mu_values.append(
                self._deposit_in_age_bins(
                    evidence=evidence,
                    age=age_grid[time_index],
                    physical_edge_mask=mask,
                )
            )

        mu_grid = torch.stack(
            mu_values,
            dim=0,
        )

        age_centers = self.age_centers.to(
            device=z0_reshaped.device,
            dtype=z0_reshaped.dtype,
        )

        temporal_kernel = torch.exp(
            -decay_rate * age_centers
        )

        raw_edge_weight = (
            mu_grid
            * temporal_kernel.view(
                1,
                1,
                1,
                self.num_bins,
            )
        ).sum(dim=-1)

        normalized_edge_weight, fallback_fraction = (
            self._normalize_incoming_edges(
                raw_edge_weight=raw_edge_weight,
                physical_edge_mask=mask,
            )
        )

        edge_weight_grid = normalized_edge_weight.unsqueeze(
            -1
        )

        _require_finite(
            mu_grid,
            "mu_grid",
        )

        _require_finite(
            edge_weight_grid,
            "edge_weight_grid",
        )

        active_mask_float = mask.to(
            dtype=z0_reshaped.dtype
        )

        expected_mass = evidence * active_mask_float
        transported_mass = mu_grid.sum(dim=-1)

        mass_error = torch.abs(
            transported_mass
            - expected_mass.unsqueeze(0)
        )

        active_mass_error = mass_error * active_mask_float.unsqueeze(0)

        max_mass_drift = active_mass_error.max()

        active_count = torch.clamp(
            active_mask_float.sum(),
            min=1.0,
        )

        initial_age_mean = (
            initial_age * active_mask_float
        ).sum() / active_count

        final_age = age_grid[-1]

        final_age_mean = (
            final_age * active_mask_float
        ).sum() / active_count

        evidence_mean = (
            evidence * active_mask_float
        ).sum() / active_count

        active_weights = edge_weight_grid[
            ...,
            0,
        ]

        active_weight_count = torch.clamp(
            active_mask_float.sum()
            * time_grid.numel(),
            min=1.0,
        )

        edge_weight_mean = (
            active_weights
            * active_mask_float.unsqueeze(0)
        ).sum() / active_weight_count

        large_value = torch.full_like(
            active_weights,
            torch.inf,
        )

        edge_weight_min = torch.where(
            mask.unsqueeze(0),
            active_weights,
            large_value,
        ).min()

        edge_weight_min = torch.where(
            torch.isfinite(edge_weight_min),
            edge_weight_min,
            torch.zeros_like(edge_weight_min),
        )

        edge_weight_max = (
            active_weights
            * active_mask_float.unsqueeze(0)
        ).max()

        initial_active_weights = active_weights[0]
        final_active_weights = active_weights[-1]
        endpoint_absolute_change = torch.abs(
            final_active_weights - initial_active_weights
        ) * active_mask_float
        mean_absolute_endpoint_change = (
            endpoint_absolute_change.sum() / active_count
        )
        changed_active_edges = (
            ~torch.isclose(
                initial_active_weights,
                final_active_weights,
                rtol=1.0e-5,
                atol=1.0e-7,
            )
        ).to(z0_reshaped.dtype) * active_mask_float
        changed_active_edge_fraction = (
            changed_active_edges.sum() / active_count
        )
        initial_active_edge_weight_mean = (
            initial_active_weights * active_mask_float
        ).sum() / active_count
        final_active_edge_weight_mean = (
            final_active_weights * active_mask_float
        ).sum() / active_count

        represented_age_fraction = (
            (
                age_grid < self.max_age
            ).to(z0_reshaped.dtype)
            * active_mask_float.unsqueeze(0)
        ).sum() / active_weight_count

        active_edge_fraction = mask.to(
            z0_reshaped.dtype
        ).mean()

        diagnostics: Dict[str, Tensor] = {
            "transport_speed": speed,
            "decay_rate": decay_rate,
            "initial_evidence_mean": evidence_mean,
            "initial_age_mean": initial_age_mean,
            "final_age_mean": final_age_mean,
            "edge_weight_mean": edge_weight_mean,
            "edge_weight_min": edge_weight_min,
            "edge_weight_max": edge_weight_max,
            "mean_absolute_first_last_edge_weight_change": (
                mean_absolute_endpoint_change
            ),
            "changed_active_edge_fraction": changed_active_edge_fraction,
            "initial_active_edge_weight_mean": (
                initial_active_edge_weight_mean
            ),
            "final_active_edge_weight_mean": final_active_edge_weight_mean,
            "active_edge_fraction": active_edge_fraction,
            "represented_age_fraction": represented_age_fraction,
            "fallback_fraction": fallback_fraction,
            "maximum_mass_drift": max_mass_drift,
            "head_evidence_mean": head_evidence.mean(),
            "transport_grid_points": torch.tensor(
                float(time_grid.numel()),
                device=z0_reshaped.device,
                dtype=z0_reshaped.dtype,
            ),
            "transport_max_age": torch.tensor(
                self.max_age,
                device=z0_reshaped.device,
                dtype=z0_reshaped.dtype,
            ),
        }

        return AttentionTransportCache(
            time_grid=time_grid,
            mu_grid=mu_grid,
            edge_weight_grid=edge_weight_grid,
            physical_edge_mask=mask,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        z0: Tensor,
        latest_observation_time: Tensor,
        time_grid: Tensor,
        physical_edge_mask: Optional[Tensor] = None,
    ) -> AttentionTransportCache:
        """Alias for build_cache."""

        return self.build_cache(
            z0=z0,
            latest_observation_time=latest_observation_time,
            time_grid=time_grid,
            physical_edge_mask=physical_edge_mask,
        )


# ---------------------------------------------------------------------------
# Standalone diagnostic test
# ---------------------------------------------------------------------------

def _self_test() -> Dict[str, float]:
    """Run deterministic shape, mass, interpolation, and gradient tests."""

    torch.manual_seed(17)

    sample_count = 2
    batch_size = 3
    num_nodes = 4
    latent_dim = 8

    # Complete directed graph without self-edges.
    senders = []
    receivers = []

    for receiver in range(num_nodes):
        for sender in range(num_nodes):
            if sender == receiver:
                continue

            senders.append(sender)
            receivers.append(receiver)

    edge_index = torch.tensor(
        [senders, receivers],
        dtype=torch.long,
    )

    num_edges = edge_index.shape[1]

    module = SolverSafeAttentionTransport(
        latent_dim=latent_dim,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_bins=8,
        max_age=4.0,
        hidden_dim=16,
        attention_dim=4,
        num_heads=2,
        initial_speed=1.0,
        initial_decay=0.7,
        learnable_speed=True,
        learnable_decay=True,
        dropout=0.0,
    )

    z0 = torch.randn(
        sample_count,
        batch_size * num_nodes,
        latent_dim,
        requires_grad=True,
    )

    latest_observation_time = torch.tensor(
        [
            [-0.5, -0.2, -0.8, -0.1],
            [-0.3, -0.7, -0.4, -0.6],
            [-0.9, -0.2, -0.5, -0.3],
        ],
        dtype=torch.float32,
    )

    time_grid = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=torch.float32,
    )

    physical_edge_mask = torch.ones(
        batch_size,
        num_edges,
        dtype=torch.bool,
    )

    # Remove one physical edge from one trajectory.
    physical_edge_mask[0, 0] = False

    cache = module.build_cache(
        z0=z0,
        latest_observation_time=latest_observation_time,
        time_grid=time_grid,
        physical_edge_mask=physical_edge_mask,
    )

    expected_effective_batch = (
        sample_count * batch_size
    )

    assert cache.mu_grid.shape == (
        time_grid.numel(),
        expected_effective_batch,
        num_edges,
        module.num_bins,
    )

    assert cache.edge_weight_grid.shape == (
        time_grid.numel(),
        expected_effective_batch,
        num_edges,
        1,
    )

    weights_a = cache.edge_weights_at(0.375)
    weights_b = cache.edge_weights_at(0.375)

    assert torch.equal(
        weights_a,
        weights_b,
    ), "Repeated solver queries must be deterministic."

    repeated_mask = physical_edge_mask.repeat(
        sample_count,
        1,
    )

    assert torch.all(
        weights_a[
            ~repeated_mask
        ] == 0.0
    ), "Inactive physical edges must remain zero."

    assert torch.isfinite(
        cache.mu_grid
    ).all()

    assert torch.isfinite(
        cache.edge_weight_grid
    ).all()

    max_mass_drift = cache.diagnostics[
        "maximum_mass_drift"
    ]

    assert float(max_mass_drift.detach()) < 1.0e-5, (
        "Age-bin transport should preserve edge evidence mass."
    )

    mean_weight_change = cache.diagnostics[
        "mean_absolute_first_last_edge_weight_change"
    ]
    changed_edge_fraction = cache.diagnostics[
        "changed_active_edge_fraction"
    ]
    assert float(mean_weight_change.detach()) > 1.0e-7, (
        "AT-ODE weights are numerically constant over a nonzero interval."
    )
    assert float(changed_edge_fraction.detach()) > 0.0, (
        "No active AT-ODE edge changed over a nonzero interval."
    )

    loss = (
        cache.edge_weight_grid.square().mean()
        + cache.mu_grid.square().mean()
    )

    loss.backward()

    assert z0.grad is not None
    assert torch.isfinite(z0.grad).all()

    trainable_parameters = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    for parameter in trainable_parameters:
        assert parameter.grad is not None, (
            "Every trainable transport parameter should receive a gradient "
            "in the self-test."
        )

        assert torch.isfinite(parameter.grad).all()

    diagnostics = cache.detached_diagnostics()

    diagnostics["loss"] = float(
        loss.detach().cpu()
    )

    return diagnostics


if __name__ == "__main__":
    self_test_diagnostics = _self_test()

    print("attention_transport.py self-test: PASS")

    for diagnostic_name in sorted(self_test_diagnostics):
        print(
            f"{diagnostic_name}: "
            f"{self_test_diagnostics[diagnostic_name]:.8f}"
        )
