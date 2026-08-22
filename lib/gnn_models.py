# lib/gnn_models.py

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

import lib.utils as utils


SUPPORTED_CONVOLUTION_TYPES = {
    "GTrans",
    "NRI",
}

SUPPORTED_AGGREGATIONS = {
    "add",
    "attention",
}

SUPPORTED_EDGE_WEIGHT_MODES = {
    "ones",
    "transport",
    # Optional ablation modes. Their weights must still be supplied by the
    # solver/provider; this module does not implement age evolution.
    "instant",
    "age",
}


class TemporalEncoding(nn.Module):
    """Sinusoidal encoding of continuous observation times."""

    def __init__(self, d_hid):
        super(TemporalEncoding, self).__init__()

        if d_hid < 1:
            raise ValueError(
                f"d_hid must be positive; got {d_hid}"
            )

        self.d_hid = int(d_hid)

        div_term = torch.tensor(
            [
                1.0
                / np.power(
                    10000,
                    2 * (hidden_index // 2) / self.d_hid,
                )
                for hidden_index in range(self.d_hid)
            ],
            dtype=torch.float32,
        ).reshape(1, -1)

        self.div_term = nn.Parameter(
            div_term,
            requires_grad=False,
        )

    def forward(self, t):
        """
        Parameters
        ----------
        t:
            Continuous timestamps with shape [num_events] or [num_events, 1].

        Returns
        -------
        Tensor
            Temporal encodings with shape [num_events, d_hid].
        """

        t = t.reshape(-1, 1)
        t = t * 200.0

        position_term = torch.matmul(
            t,
            self.div_term.to(
                device=t.device,
                dtype=t.dtype,
            ),
        )

        # Avoid modifying the result of matmul in place. This is safer for
        # autograd when timestamps require gradients.
        encoded = torch.empty_like(position_term)
        encoded[:, 0::2] = torch.sin(
            position_term[:, 0::2]
        )
        encoded[:, 1::2] = torch.cos(
            position_term[:, 1::2]
        )

        return encoded


class GTrans(MessagePassing):
    """
    Original irregular-observation graph transformer.

    This encoder is shared unchanged between LG-ODE and AT-ODE. Transport
    weighting is deliberately not implemented here; it belongs only to the
    NRI interaction layers inside the generative ODE.
    """

    def __init__(
        self,
        n_heads=2,
        d_input=6,
        d_k=6,
        dropout=0.1,
        **kwargs,
    ):
        super(GTrans, self).__init__(
            aggr="add",
            **kwargs,
        )

        if n_heads < 1:
            raise ValueError(
                f"n_heads must be positive; got {n_heads}"
            )
        if d_input < 1:
            raise ValueError(
                f"d_input must be positive; got {d_input}"
            )
        if d_k < 1:
            raise ValueError(
                f"d_k must be positive; got {d_k}"
            )
        if d_k % n_heads != 0:
            raise ValueError(
                "d_k must be divisible by n_heads; "
                f"got d_k={d_k}, n_heads={n_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1); got {dropout}"
            )

        self.n_heads = int(n_heads)
        self.dropout = nn.Dropout(dropout)

        self.d_input = int(d_input)
        self.d_k = int(d_k) // self.n_heads
        self.d_q = int(d_k) // self.n_heads
        self.d_e = int(d_k) // self.n_heads
        self.d_sqrt = math.sqrt(self.d_k)

        self.w_k_list_same = nn.ModuleList(
            [
                nn.Linear(
                    self.d_input,
                    self.d_k,
                    bias=True,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.w_k_list_diff = nn.ModuleList(
            [
                nn.Linear(
                    self.d_input,
                    self.d_k,
                    bias=True,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.w_q_list = nn.ModuleList(
            [
                nn.Linear(
                    self.d_input,
                    self.d_q,
                    bias=True,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.w_v_list_same = nn.ModuleList(
            [
                nn.Linear(
                    self.d_input,
                    self.d_e,
                    bias=True,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.w_v_list_diff = nn.ModuleList(
            [
                nn.Linear(
                    self.d_input,
                    self.d_k,
                    bias=True,
                )
                for _ in range(self.n_heads)
            ]
        )

        self.w_transfer = nn.ModuleList(
            [
                nn.Linear(
                    self.d_input + 1,
                    self.d_k,
                    bias=True,
                )
                for _ in range(self.n_heads)
            ]
        )

        utils.init_network_weights(self.w_k_list_same)
        utils.init_network_weights(self.w_k_list_diff)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list_same)
        utils.init_network_weights(self.w_v_list_diff)
        utils.init_network_weights(self.w_transfer)

        self.temporal_net = TemporalEncoding(self.d_input)
        self.layer_norm = nn.LayerNorm(self.d_input)

    def forward(
        self,
        x,
        edge_index,
        edge_value,
        time_nodes,
        edge_same,
    ):
        del time_nodes

        if edge_index is None:
            raise ValueError(
                "GTrans requires edge_index"
            )
        if edge_value is None:
            raise ValueError(
                "GTrans requires temporal edge attributes"
            )
        if edge_same is None:
            raise ValueError(
                "GTrans requires edge_same"
            )

        residual = x
        x = self.layer_norm(x)

        return self.propagate(
            edge_index,
            x=x,
            edges_temporal=edge_value,
            edge_same=edge_same,
            residual=residual,
        )

    def message(
        self,
        x_j,
        x_i,
        edge_index_i,
        edges_temporal,
        edge_same,
    ):
        """
        Construct temporal messages.

        x_j:
            Sender event features [num_edges, d].
        x_i:
            Receiver event features [num_edges, d].
        edge_index_i:
            Receiver event indices [num_edges].
        edges_temporal:
            Relative event times [num_edges].
        edge_same:
            One for same-object temporal edges and zero for different-object
            temporal edges.
        """

        messages = []
        edge_same = edge_same.reshape(-1, 1).to(
            device=x_j.device,
            dtype=x_j.dtype,
        )
        edge_temporal_column = edges_temporal.reshape(
            -1, 1
        ).to(
            device=x_j.device,
            dtype=x_j.dtype,
        )

        edge_temporal_encoding = self.temporal_net(
            edge_temporal_column
        )

        for head_index in range(self.n_heads):
            key_same = self.w_k_list_same[head_index]
            key_different = self.w_k_list_diff[head_index]
            query = self.w_q_list[head_index]
            value_same = self.w_v_list_same[head_index]
            value_different = self.w_v_list_diff[head_index]
            transfer = self.w_transfer[head_index]

            transferred_sender = F.gelu(
                transfer(
                    torch.cat(
                        (x_j, edge_temporal_column),
                        dim=1,
                    )
                )
            ) + edge_temporal_encoding

            attention = self.each_head_attention(
                transferred_sender,
                key_same,
                key_different,
                query,
                x_i,
                edge_same,
            )
            attention = attention / self.d_sqrt
            normalized_attention = softmax(
                attention,
                edge_index_i,
            )

            sender_same = (
                edge_same
                * value_same(transferred_sender)
            )
            sender_different = (
                (1.0 - edge_same)
                * value_different(transferred_sender)
            )
            sender = sender_same + sender_different

            messages.append(
                normalized_attention * sender
            )

        return torch.cat(messages, dim=1)

    def each_head_attention(
        self,
        x_j_transfer,
        w_k_same,
        w_k_diff,
        w_q,
        x_i,
        edge_same,
    ):
        receiver_query = w_q(x_i)

        sender_same = (
            edge_same * w_k_same(x_j_transfer)
        )
        sender_different = (
            (1.0 - edge_same)
            * w_k_diff(x_j_transfer)
        )
        sender_key = sender_same + sender_different

        attention = torch.bmm(
            sender_key.unsqueeze(1),
            receiver_query.unsqueeze(2),
        )

        return attention.squeeze(1)

    def update(self, aggr_out, residual):
        x_new = residual + F.gelu(aggr_out)
        return self.dropout(x_new)

    def __repr__(self):
        return self.__class__.__name__


class NRIConv(nn.Module):
    """
    NRI interaction layer with physical-edge and transport-weight support.

    Runtime graph fields
    --------------------
    rel_type:
        One-hot candidate-edge types [B, E, edge_types]. Edge type zero means
        no physical line and edge type one means an existing physical line.

    rel_rec:
        Candidate-edge receiver matrix [E, N].

    rel_send:
        Candidate-edge sender matrix [E, N].

    edge_weight_mode:
        ``ones`` for fixed LG-ODE edges or ``transport`` for AT-ODE weights.
        Optional ``instant`` and ``age`` values are accepted for controlled
        ablations and follow the same external-weight path as ``transport``.

    edge_weight:
        External weights [B, E, 1] for transport/ablation modes.

    physical_edge_mask:
        Computed from ``rel_type[..., 1:2]`` on every forward pass.

    latest_observation_time:
        Optional per-bus observation timing supplied by the solver. It is
        retained for diagnostics but is not interpreted inside NRIConv.

    current_time:
        Current ODE solver time, retained for diagnostics. Transport
        trajectories and interpolation are computed outside this module.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0,
        skip_first=False,
        edge_weight_mode="ones",
        normalize_incoming_weights=True,
        normalization_eps=1e-8,
    ):
        super(NRIConv, self).__init__()

        if in_channels < 1:
            raise ValueError(
                "in_channels must be positive"
            )
        if out_channels < 1:
            raise ValueError(
                "out_channels must be positive"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1); got {dropout}"
            )
        if edge_weight_mode not in SUPPORTED_EDGE_WEIGHT_MODES:
            raise ValueError(
                "Unsupported edge-weight mode: "
                f"{edge_weight_mode!r}. Supported modes are "
                f"{sorted(SUPPORTED_EDGE_WEIGHT_MODES)}"
            )
        if normalization_eps <= 0.0:
            raise ValueError(
                "normalization_eps must be positive"
            )

        self.edge_types = 2
        self.msg_fc1 = nn.ModuleList(
            [
                nn.Linear(
                    2 * in_channels,
                    out_channels,
                )
                for _ in range(self.edge_types)
            ]
        )
        self.msg_fc2 = nn.ModuleList(
            [
                nn.Linear(
                    out_channels,
                    out_channels,
                )
                for _ in range(self.edge_types)
            ]
        )

        self.msg_out_shape = int(out_channels)
        self.skip_first_edge_type = bool(skip_first)

        self.out_fc1 = nn.Linear(
            in_channels + out_channels,
            out_channels,
        )
        self.out_fc2 = nn.Linear(
            out_channels,
            out_channels,
        )
        self.dropout = nn.Dropout(dropout)

        self.normalize_incoming_weights = bool(
            normalize_incoming_weights
        )
        self.normalization_eps = float(
            normalization_eps
        )

        # Graph structure supplied by GraphODEFunc.set_graph().
        self.rel_type: Optional[Tensor] = None
        self.rel_rec: Optional[Tensor] = None
        self.rel_send: Optional[Tensor] = None

        # Runtime transport context.
        self.edge_weight_mode = edge_weight_mode
        self.edge_weight: Optional[Tensor] = None
        self.physical_edge_mask: Optional[Tensor] = None
        self.latest_observation_time: Optional[Tensor] = None
        self.current_time: Optional[Tensor] = None

        # Last normalized effective weights are exposed for diagnostics. They
        # are not a recurrent state and are replaced on every call.
        self.last_effective_edge_weight: Optional[Tensor] = None

    def set_edge_weight_mode(self, mode: str) -> None:
        if mode not in SUPPORTED_EDGE_WEIGHT_MODES:
            raise ValueError(
                f"Unsupported edge-weight mode: {mode!r}. "
                f"Supported modes are "
                f"{sorted(SUPPORTED_EDGE_WEIGHT_MODES)}"
            )
        self.edge_weight_mode = mode

    def set_edge_runtime(
        self,
        *,
        edge_weight_mode: Optional[str] = None,
        edge_weight: Optional[Tensor] = None,
        latest_observation_time: Optional[Tensor] = None,
        current_time: Optional[Tensor] = None,
    ) -> None:
        """
        Set solver-provided edge context without transforming transport state.

        Calling this method does not perform age-bin evolution or interpolate
        a transport trajectory. Those operations belong to the external
        transport provider and ODE solver.
        """

        if edge_weight_mode is not None:
            self.set_edge_weight_mode(edge_weight_mode)

        self.edge_weight = edge_weight
        self.latest_observation_time = (
            latest_observation_time
        )
        self.current_time = current_time

    def clear_edge_runtime(self) -> None:
        """Clear optional external runtime values."""

        self.edge_weight = None
        self.latest_observation_time = None
        self.current_time = None
        self.physical_edge_mask = None
        self.last_effective_edge_weight = None

    def _validate_graph_inputs(
        self,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if inputs.ndim != 3:
            raise ValueError(
                "NRIConv inputs must have shape [B, N, F]; "
                f"got {tuple(inputs.shape)}"
            )

        if self.rel_type is None:
            raise RuntimeError(
                "NRIConv.rel_type has not been set"
            )
        if self.rel_rec is None:
            raise RuntimeError(
                "NRIConv.rel_rec has not been set"
            )
        if self.rel_send is None:
            raise RuntimeError(
                "NRIConv.rel_send has not been set"
            )

        rel_type = self.rel_type
        rel_rec = self.rel_rec
        rel_send = self.rel_send

        if rel_type.ndim != 3:
            raise ValueError(
                "rel_type must have shape [B, E, K]; "
                f"got {tuple(rel_type.shape)}"
            )
        if rel_rec.ndim != 2:
            raise ValueError(
                "rel_rec must have shape [E, N]; "
                f"got {tuple(rel_rec.shape)}"
            )
        if rel_send.ndim != 2:
            raise ValueError(
                "rel_send must have shape [E, N]; "
                f"got {tuple(rel_send.shape)}"
            )

        batch_size, num_nodes, _ = inputs.shape
        candidate_edges = rel_rec.shape[0]

        if rel_type.shape[0] != batch_size:
            raise ValueError(
                "rel_type batch size does not match messages: "
                f"{rel_type.shape[0]} != {batch_size}"
            )
        if rel_type.shape[1] != candidate_edges:
            raise ValueError(
                "rel_type candidate-edge count does not match rel_rec: "
                f"{rel_type.shape[1]} != {candidate_edges}"
            )
        if rel_type.shape[2] < 2:
            raise ValueError(
                "rel_type must contain at least two edge types because "
                "rel_type[..., 1] identifies physical lines"
            )
        if rel_rec.shape != rel_send.shape:
            raise ValueError(
                "rel_rec and rel_send must have identical shapes; "
                f"got {tuple(rel_rec.shape)} and "
                f"{tuple(rel_send.shape)}"
            )
        if rel_rec.shape[1] != num_nodes:
            raise ValueError(
                "Relation matrices do not match the node count: "
                f"{rel_rec.shape[1]} != {num_nodes}"
            )

        tensors = {
            "rel_type": rel_type,
            "rel_rec": rel_rec,
            "rel_send": rel_send,
        }
        for name, value in tensors.items():
            if value.device != inputs.device:
                raise ValueError(
                    f"{name} is on {value.device}, but inputs are on "
                    f"{inputs.device}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"{name} contains non-finite values"
                )

        return rel_type, rel_rec, rel_send

    def _validate_external_edge_weight(
        self,
        edge_weight: Tensor,
        *,
        batch_size: int,
        candidate_edges: int,
        device: torch.device,
    ) -> Tensor:
        if not isinstance(edge_weight, Tensor):
            raise TypeError(
                "External edge_weight must be a torch.Tensor"
            )

        expected_shape = (
            batch_size,
            candidate_edges,
            1,
        )

        if tuple(edge_weight.shape) != expected_shape:
            raise ValueError(
                "External edge_weight must have exact shape [B, E, 1]; "
                f"expected {expected_shape}, got "
                f"{tuple(edge_weight.shape)}"
            )

        if edge_weight.device != device:
            raise ValueError(
                "External edge_weight must be on the same device as "
                f"messages; got {edge_weight.device} and {device}"
            )

        if not (
            edge_weight.is_floating_point()
            or edge_weight.is_complex()
        ):
            raise TypeError(
                "External edge_weight must have a floating-point dtype; "
                f"got {edge_weight.dtype}"
            )

        if edge_weight.is_complex():
            raise TypeError(
                "External edge_weight must be real-valued"
            )

        if not torch.isfinite(edge_weight).all():
            raise ValueError(
                "External edge_weight contains NaN or infinity"
            )

        if torch.any(edge_weight < 0):
            minimum = float(
                edge_weight.detach().min().item()
            )
            raise ValueError(
                "External edge_weight must be nonnegative; "
                f"minimum value is {minimum}"
            )

        return edge_weight

    def _raw_edge_weight(
        self,
        inputs: Tensor,
        physical_edge_mask: Tensor,
    ) -> Tensor:
        batch_size = inputs.shape[0]
        candidate_edges = physical_edge_mask.shape[1]

        mode = self.edge_weight_mode

        if mode not in SUPPORTED_EDGE_WEIGHT_MODES:
            raise ValueError(
                f"Unsupported edge-weight mode: {mode!r}. "
                f"Supported modes are "
                f"{sorted(SUPPORTED_EDGE_WEIGHT_MODES)}"
            )

        if mode == "ones":
            # Original LG-ODE weighting on actual physical edges.
            raw_weight = torch.ones(
                (
                    batch_size,
                    candidate_edges,
                    1,
                ),
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            if self.edge_weight is None:
                raise RuntimeError(
                    f"edge_weight_mode={mode!r} requires externally "
                    "supplied edge_weight"
                )

            raw_weight = self._validate_external_edge_weight(
                self.edge_weight,
                batch_size=batch_size,
                candidate_edges=candidate_edges,
                device=inputs.device,
            )

            if raw_weight.dtype != inputs.dtype:
                # A dtype conversion is differentiable and does not alter the
                # required shape. Device mismatches remain explicit errors.
                raw_weight = raw_weight.to(
                    dtype=inputs.dtype
                )

        # Nonphysical candidate pairs cannot transport evidence. This masking
        # is applied even if a provider emitted nonzero values for such pairs.
        return raw_weight * physical_edge_mask

    def _normalize_physical_incoming_weights(
        self,
        raw_weight: Tensor,
        physical_edge_mask: Tensor,
        rel_rec: Tensor,
    ) -> Tensor:
        """
        Normalize active weights to mean one per receiving node.

        Only physical incoming edges contribute to the denominator. A receiver
        with no physical incoming edges has zero edge contribution. If all raw
        physical weights for a receiver are zero, they remain zero.
        """

        if not self.normalize_incoming_weights:
            return raw_weight * physical_edge_mask

        batch_size, candidate_edges, _ = raw_weight.shape

        receiver_matrix = rel_rec.to(
            device=raw_weight.device,
            dtype=raw_weight.dtype,
        )

        physical = physical_edge_mask.squeeze(-1)
        raw = raw_weight.squeeze(-1)

        # [B, E] @ [E, N] -> [B, N]
        incoming_physical_count = torch.matmul(
            physical,
            receiver_matrix,
        )
        incoming_weight_sum = torch.matmul(
            raw,
            receiver_matrix,
        )

        incoming_mean = torch.where(
            incoming_physical_count > 0,
            incoming_weight_sum
            / incoming_physical_count.clamp_min(1.0),
            torch.zeros_like(incoming_weight_sum),
        )

        # Map receiver-level means back to candidate edges:
        # [B, N] @ [N, E] -> [B, E].
        edge_incoming_mean = torch.matmul(
            incoming_mean,
            receiver_matrix.transpose(0, 1),
        )

        valid_denominator = (
            edge_incoming_mean > self.normalization_eps
        )

        normalized = torch.where(
            valid_denominator,
            raw
            / edge_incoming_mean.clamp_min(
                self.normalization_eps
            ),
            torch.zeros_like(raw),
        )

        normalized = (
            normalized.unsqueeze(-1)
            * physical_edge_mask
        )

        if tuple(normalized.shape) != (
            batch_size,
            candidate_edges,
            1,
        ):
            raise RuntimeError(
                "Internal incoming-edge normalization produced an "
                f"invalid shape: {tuple(normalized.shape)}"
            )

        return normalized

    def effective_edge_weight(
        self,
        inputs: Tensor,
        rel_type: Tensor,
        rel_rec: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Return the physical mask and normalized effective edge weights.
        """

        physical_edge_mask = rel_type[..., 1:2].to(
            device=inputs.device,
            dtype=inputs.dtype,
        )

        if tuple(physical_edge_mask.shape) != (
            inputs.shape[0],
            rel_rec.shape[0],
            1,
        ):
            raise ValueError(
                "Physical edge mask has an invalid shape: "
                f"{tuple(physical_edge_mask.shape)}"
            )

        if not torch.isfinite(physical_edge_mask).all():
            raise ValueError(
                "physical_edge_mask contains non-finite values"
            )
        if torch.any(physical_edge_mask < 0):
            raise ValueError(
                "physical_edge_mask contains negative values"
            )

        raw_weight = self._raw_edge_weight(
            inputs,
            physical_edge_mask,
        )
        normalized_weight = (
            self._normalize_physical_incoming_weights(
                raw_weight,
                physical_edge_mask,
                rel_rec,
            )
        )

        if not torch.isfinite(normalized_weight).all():
            raise RuntimeError(
                "Normalized edge weights contain NaN or infinity"
            )
        if torch.any(normalized_weight < 0):
            raise RuntimeError(
                "Normalized edge weights contain negative values"
            )

        # Runtime diagnostic fields. These do not participate in transport
        # evolution and are replaced rather than accumulated on every call.
        self.physical_edge_mask = physical_edge_mask
        self.last_effective_edge_weight = normalized_weight

        return physical_edge_mask, normalized_weight

    def forward(self, inputs, pred_steps=1):
        """
        Parameters
        ----------
        inputs:
            Node states [B, N, F].
        pred_steps:
            Retained for compatibility with the original NRI decoder.

        Returns
        -------
        Tensor
            Updated node states [B, N, out_channels].
        """

        del pred_steps

        rel_type, rel_rec, rel_send = (
            self._validate_graph_inputs(inputs)
        )

        # Node-to-edge conversion.
        receivers = torch.matmul(
            rel_rec,
            inputs,
        )
        senders = torch.matmul(
            rel_send,
            inputs,
        )
        pre_msg = torch.cat(
            [senders, receivers],
            dim=-1,
        )

        all_msgs = inputs.new_zeros(
            (
                pre_msg.shape[0],
                pre_msg.shape[1],
                self.msg_out_shape,
            )
        )

        start_idx = (
            1 if self.skip_first_edge_type else 0
        )

        for edge_type_index in range(
            start_idx,
            len(self.msg_fc2),
        ):
            msg = F.relu(
                self.msg_fc1[edge_type_index](
                    pre_msg
                )
            )
            msg = self.dropout(msg)
            msg = F.relu(
                self.msg_fc2[edge_type_index](msg)
            )

            msg = (
                msg
                * rel_type[
                    :,
                    :,
                    edge_type_index : edge_type_index + 1,
                ]
            )
            all_msgs = all_msgs + msg

        physical_edge_mask, edge_weight = (
            self.effective_edge_weight(
                inputs,
                rel_type,
                rel_rec,
            )
        )

        # Required message path:
        #
        # edge-type MLP -> physical mask -> transport weight -> aggregation
        #
        # Nonphysical complete-graph pairs remain exactly zero regardless of
        # their edge-type-zero MLP output or provider-produced weight.
        all_msgs = all_msgs * physical_edge_mask
        all_msgs = all_msgs * edge_weight

        # Edge-to-node receiver aggregation.
        agg_msgs = (
            all_msgs.transpose(-2, -1)
            .matmul(rel_rec)
            .transpose(-2, -1)
        )

        augmented_inputs = torch.cat(
            [inputs, agg_msgs],
            dim=-1,
        )

        pred = self.dropout(
            F.relu(
                self.out_fc1(augmented_inputs)
            )
        )
        pred = self.dropout(
            F.relu(
                self.out_fc2(pred)
            )
        )

        return inputs + pred


class GeneralConv(nn.Module):
    """Wrapper around supported encoder and generative convolutions."""

    def __init__(
        self,
        conv_name,
        in_hid,
        out_hid,
        n_heads,
        dropout,
    ):
        super(GeneralConv, self).__init__()

        if conv_name not in SUPPORTED_CONVOLUTION_TYPES:
            raise ValueError(
                f"Unsupported convolution type: {conv_name}"
            )

        self.conv_name = conv_name

        if self.conv_name == "GTrans":
            self.base_conv = GTrans(
                n_heads,
                in_hid,
                out_hid,
                dropout,
            )
        elif self.conv_name == "NRI":
            self.base_conv = NRIConv(
                in_hid,
                out_hid,
                dropout,
            )
        else:  # Defensive guard if the supported set changes.
            raise ValueError(
                f"Unsupported convolution type: {conv_name}"
            )

    def forward(
        self,
        x,
        edge_index,
        edge_time,
        x_time,
        edge_same,
    ):
        if self.conv_name == "GTrans":
            return self.base_conv(
                x,
                edge_index,
                edge_time,
                x_time,
                edge_same,
            )

        if self.conv_name == "NRI":
            return self.base_conv(x)

        raise ValueError(
            f"Unsupported convolution type: {self.conv_name}"
        )


class GNN(nn.Module):
    """Multi-layer GTrans encoder or NRI ODE network."""

    def __init__(
        self,
        in_dim,
        n_hid,
        out_dim,
        n_heads,
        n_layers,
        dropout=0.2,
        conv_name="GTrans",
        aggregate="add",
    ):
        super(GNN, self).__init__()

        if in_dim < 1:
            raise ValueError(
                f"in_dim must be positive; got {in_dim}"
            )
        if n_hid < 1:
            raise ValueError(
                f"n_hid must be positive; got {n_hid}"
            )
        if out_dim < 1:
            raise ValueError(
                f"out_dim must be positive; got {out_dim}"
            )
        if n_layers < 1:
            raise ValueError(
                f"n_layers must be positive; got {n_layers}"
            )
        if conv_name not in SUPPORTED_CONVOLUTION_TYPES:
            raise ValueError(
                f"Unsupported convolution type: {conv_name}"
            )
        if aggregate not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation: {aggregate!r}. "
                f"Supported aggregations are "
                f"{sorted(SUPPORTED_AGGREGATIONS)}"
            )

        # Attention aggregation applies only to temporal encoder graphs.
        if conv_name != "GTrans" and aggregate == "attention":
            raise ValueError(
                "attention aggregation is supported only for GTrans"
            )

        self.gcs = nn.ModuleList()
        self.in_dim = int(in_dim)
        self.n_hid = int(n_hid)
        self.out_dim = int(out_dim)
        self.conv_name = conv_name
        self.aggregate = aggregate

        self.drop = nn.Dropout(dropout)
        self.adapt_ws = nn.Linear(
            in_dim,
            n_hid,
        )
        self.sequence_w = nn.Linear(
            n_hid,
            n_hid,
        )
        self.out_w_ode = nn.Linear(
            n_hid,
            out_dim,
        )
        self.out_w_encoder = nn.Linear(
            n_hid,
            out_dim * 2,
        )

        utils.init_network_weights(self.adapt_ws)
        utils.init_network_weights(self.sequence_w)
        utils.init_network_weights(self.out_w_ode)
        utils.init_network_weights(self.out_w_encoder)

        self.layer_norm = nn.LayerNorm(n_hid)

        for _ in range(n_layers):
            self.gcs.append(
                GeneralConv(
                    conv_name,
                    n_hid,
                    n_hid,
                    n_heads,
                    dropout,
                )
            )

        if conv_name == "GTrans":
            self.temporal_net = TemporalEncoding(
                n_hid
            )
            self.w_transfer = nn.Linear(
                n_hid + 1,
                n_hid,
                bias=True,
            )
            utils.init_network_weights(
                self.w_transfer
            )

    def set_graph(
        self,
        rel_type: Tensor,
        rel_rec: Tensor,
        rel_send: Tensor,
        edge_types: Optional[int] = None,
    ) -> None:
        """
        Set NRI candidate-edge structure on all generative layers.

        This is a convenience method for GraphODEFunc. GTrans models reject it
        because temporal encoder graphs are passed directly to ``forward``.
        """

        if self.conv_name != "NRI":
            raise RuntimeError(
                "set_graph is valid only for an NRI GNN"
            )

        if edge_types is not None and int(edge_types) < 2:
            raise ValueError(
                "NRI physical-edge masking requires at least two edge types"
            )

        for layer in self.gcs:
            convolution = layer.base_conv
            convolution.rel_type = rel_type
            convolution.rel_rec = rel_rec
            convolution.rel_send = rel_send

            if edge_types is not None:
                convolution.edge_types = int(
                    edge_types
                )

    def set_edge_runtime(
        self,
        *,
        edge_weight_mode: Optional[str] = None,
        edge_weight: Optional[Tensor] = None,
        latest_observation_time: Optional[Tensor] = None,
        current_time: Optional[Tensor] = None,
    ) -> None:
        """
        Propagate solver-provided edge weights and timing to all NRI layers.
        """

        if self.conv_name != "NRI":
            raise RuntimeError(
                "set_edge_runtime is valid only for an NRI GNN"
            )

        if (
            edge_weight_mode is not None
            and edge_weight_mode
            not in SUPPORTED_EDGE_WEIGHT_MODES
        ):
            raise ValueError(
                "Unsupported edge-weight mode: "
                f"{edge_weight_mode!r}. Supported modes are "
                f"{sorted(SUPPORTED_EDGE_WEIGHT_MODES)}"
            )

        for layer in self.gcs:
            layer.base_conv.set_edge_runtime(
                edge_weight_mode=edge_weight_mode,
                edge_weight=edge_weight,
                latest_observation_time=(
                    latest_observation_time
                ),
                current_time=current_time,
            )

    def clear_edge_runtime(self) -> None:
        if self.conv_name != "NRI":
            return

        for layer in self.gcs:
            layer.base_conv.clear_edge_runtime()

    def forward(
        self,
        x,
        edge_time=None,
        edge_index=None,
        x_time=None,
        edge_same=None,
        batch=None,
        batch_y=None,
    ):
        h_0 = F.relu(self.adapt_ws(x))
        h_t = self.drop(h_0)
        h_t = self.layer_norm(h_t)

        for graph_convolution in self.gcs:
            h_t = graph_convolution(
                h_t,
                edge_index,
                edge_time,
                x_time,
                edge_same,
            )

        if batch is not None:
            if self.conv_name != "GTrans":
                raise ValueError(
                    "Batched object aggregation is supported only for "
                    "the GTrans recognition encoder"
                )
            if batch_y is None:
                raise ValueError(
                    "batch_y is required for GTrans object aggregation"
                )
            if x_time is None:
                raise ValueError(
                    "x_time is required for GTrans object aggregation"
                )

            batch_new = self.rewrite_batch(
                batch,
                batch_y,
            )

            if self.aggregate == "add":
                # Retain the original implementation's mean pooling behavior.
                h_ball = global_mean_pool(
                    h_t,
                    batch_new,
                )

            elif self.aggregate == "attention":
                x_time_column = x_time.reshape(
                    -1, 1
                ).to(
                    device=h_t.device,
                    dtype=h_t.dtype,
                )

                h_t = F.gelu(
                    self.w_transfer(
                        torch.cat(
                            (
                                h_t,
                                x_time_column,
                            ),
                            dim=1,
                        )
                    )
                ) + self.temporal_net(
                    x_time_column
                )

                attention_vector = F.relu(
                    self.sequence_w(
                        global_mean_pool(
                            h_t,
                            batch_new,
                        )
                    )
                )

                attention_vector_expanded = (
                    self.attention_expand(
                        attention_vector,
                        batch,
                        batch_y,
                    )
                )

                attention_nodes = torch.sigmoid(
                    torch.bmm(
                        attention_vector_expanded.unsqueeze(
                            1
                        ),
                        h_t.unsqueeze(2),
                    )
                    .squeeze(-1)
                    .reshape(-1, 1)
                )

                nodes_attention = (
                    attention_nodes * h_t
                )
                h_ball = global_mean_pool(
                    nodes_attention,
                    batch_new,
                )

            else:
                raise ValueError(
                    f"Unsupported aggregation: {self.aggregate!r}"
                )

            h_out = self.out_w_encoder(h_ball)

            # Shared positive posterior scale construction for all neural
            # models using this encoder.
            mean, raw_std = self.split_mean_mu(
                h_out
            )
            std = F.softplus(raw_std) + 1e-5
            return mean, std

        if self.conv_name != "NRI":
            raise ValueError(
                "GTrans requires batch and batch_y for encoder aggregation"
            )

        h_out = self.out_w_ode(h_t)
        return h_out

    def rewrite_batch(
        self,
        batch,
        batch_y,
    ):
        """
        Reassign event nodes to consecutive trajectory-bus groups.
        """

        if batch.ndim != 1:
            raise ValueError(
                "batch must be one-dimensional"
            )

        counts = torch.as_tensor(
            batch_y,
            device=batch.device,
            dtype=torch.long,
        ).reshape(-1)

        if torch.any(counts < 0):
            raise ValueError(
                "batch_y cannot contain negative event counts"
            )

        if int(counts.sum().item()) != int(
            batch.numel()
        ):
            raise ValueError(
                "sum(batch_y) must equal the number of event nodes; "
                f"got {int(counts.sum().item())} and "
                f"{int(batch.numel())}"
            )

        groups = torch.arange(
            counts.numel(),
            device=batch.device,
            dtype=batch.dtype,
        )

        return torch.repeat_interleave(
            groups,
            counts,
        )

    def attention_expand(
        self,
        attention_ball,
        batch,
        batch_y,
    ):
        """
        Expand one object-level attention vector to its event nodes.
        """

        if attention_ball.ndim != 2:
            raise ValueError(
                "attention_ball must have shape [objects, hidden_dim]"
            )

        counts = torch.as_tensor(
            batch_y,
            device=attention_ball.device,
            dtype=torch.long,
        ).reshape(-1)

        if counts.numel() != attention_ball.shape[0]:
            raise ValueError(
                "batch_y must contain one count per attention vector; "
                f"got {counts.numel()} counts and "
                f"{attention_ball.shape[0]} vectors"
            )

        if int(counts.sum().item()) != int(
            batch.numel()
        ):
            raise ValueError(
                "sum(batch_y) must equal the number of event nodes"
            )

        return torch.repeat_interleave(
            attention_ball,
            counts,
            dim=0,
        )

    def split_mean_mu(self, h):
        if h.shape[-1] % 2 != 0:
            raise ValueError(
                "Encoder output dimension must be even to form mean and "
                f"scale tensors; got {h.shape[-1]}"
            )

        last_dim = h.shape[-1] // 2
        return (
            h[..., :last_dim],
            h[..., last_dim:],
        )


__all__ = [
    "GeneralConv",
    "GNN",
    "GTrans",
    "NRIConv",
    "SUPPORTED_AGGREGATIONS",
    "SUPPORTED_CONVOLUTION_TYPES",
    "SUPPORTED_EDGE_WEIGHT_MODES",
    "TemporalEncoding",
]
