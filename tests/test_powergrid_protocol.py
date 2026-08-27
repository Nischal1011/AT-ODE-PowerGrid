from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from lib.attention_transport import SolverSafeAttentionTransport
from lib.gnn_models import NRIConv
from lib.latent_ode import LatentGraphODE
from lib.powergrid_baselines import NodeWiseODEFunc
from lib.powergrid_model_factory import (
    assert_lgode_atode_protocol_match,
    build_lgode_atode_protocol_pair,
)
from lib.simbench_lgode_data import (
    SimBenchArchive,
    SimBenchLGODEDataset,
    fit_training_normalization,
)
from run_powergrid_lgode import build_candidate_graph, stable_clip_grad_norm


def relation_matrices(edge_index: torch.Tensor, nodes: int):
    return (
        F.one_hot(edge_index[1], num_classes=nodes).float(),
        F.one_hot(edge_index[0], num_classes=nodes).float(),
    )


class PowerGridProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )

    def test_candidate_graph_modes(self) -> None:
        sparse_edges, sparse_labels = build_candidate_graph(
            3, self.edge_index, "physical_sparse"
        )
        self.assertTrue(torch.equal(sparse_edges, self.edge_index))
        self.assertTrue(torch.all(sparse_labels == 1))

        dense_edges, dense_labels = build_candidate_graph(
            3, self.edge_index, "all_pairs_nri"
        )
        self.assertEqual(tuple(dense_edges.shape), (2, 6))
        self.assertEqual(int(dense_labels.sum()), 4)
        self.assertTrue(torch.all(dense_edges[0] != dense_edges[1]))

    def test_graph_changes_nri_and_receives_gradients(self) -> None:
        layer = NRIConv(2, 2, dropout=0.0, edge_weight_mode="ones")
        rel_rec, rel_send = relation_matrices(self.edge_index, 3)
        layer.rel_rec = rel_rec
        layer.rel_send = rel_send
        inputs = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            requires_grad=True,
        )

        layer.rel_type = F.one_hot(
            torch.ones((1, 4), dtype=torch.long), num_classes=2
        ).float()
        physical_output = layer(inputs)
        physical_output.square().sum().backward()
        graph_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in layer.named_parameters()
            if "msg_" in name and parameter.grad is not None
        )
        self.assertGreater(graph_gradient, 0.0)

        layer.rel_type = F.one_hot(
            torch.zeros((1, 4), dtype=torch.long), num_classes=2
        ).float()
        no_edge_output = layer(inputs.detach())
        self.assertFalse(torch.allclose(physical_output, no_edge_output))

    def test_independent_dynamics_are_graph_free(self) -> None:
        function = NodeWiseODEFunc(2, 4, 2)
        state = torch.randn(1, 3, 2)
        first = function(torch.tensor(0.0), state)
        permuted_graph = self.edge_index.flip(1)
        self.assertFalse(torch.equal(permuted_graph, self.edge_index))
        second = function(torch.tensor(0.0), state)
        self.assertTrue(torch.equal(first, second))

    def test_posterior_mean_and_sampling_policy(self) -> None:
        mean = torch.randn(4, 3)
        std = torch.full_like(mean, 0.2)
        deterministic, sampled = LatentGraphODE._construct_initial_state(
            object(), mean, std, 2, False
        )
        self.assertFalse(sampled)
        self.assertTrue(torch.equal(deterministic[0], deterministic[1]))
        stochastic_a, _ = LatentGraphODE._construct_initial_state(
            object(), mean, std, 2, True
        )
        stochastic_b, _ = LatentGraphODE._construct_initial_state(
            object(), mean, std, 2, True
        )
        self.assertFalse(torch.equal(stochastic_a, stochastic_b))

    def test_split_windows_and_task_masks(self) -> None:
        archive = SimBenchArchive(
            path=None,
            bus_state=torch.randn(60, 3, 2),
            timestamps_hours=torch.arange(60, dtype=torch.float64),
            bus_indices=torch.arange(3),
            edge_index=self.edge_index,
            edge_type=torch.ones(4, dtype=torch.long),
            train_end=36,
            validation_end=48,
            bus_feature_names=("a", "b"),
            metadata={},
        )
        normalization = fit_training_normalization(archive)
        datasets = [
            SimBenchLGODEDataset(
                archive,
                split,
                "interpolation",
                0.4,
                normalization=normalization,
                trajectory_length=6,
                stride=3,
                mask_seed=5,
            )
            for split in ("train", "validation", "test")
        ]
        for dataset in datasets:
            split_start, split_stop = dataset.split_bounds
            for record in dataset.windows:
                self.assertGreaterEqual(record.start, split_start)
                self.assertLessEqual(record.stop, split_stop)
            sample = dataset[0]
            observed = sample["encoder_observation_mask"].unsqueeze(-1)
            withheld = sample["interpolation_withheld_mask"]
            self.assertFalse(torch.any(observed & withheld))
            self.assertTrue(torch.equal(withheld, sample["training_loss_mask"]))

    def test_transport_changes_and_backpropagates(self) -> None:
        module = SolverSafeAttentionTransport(
            latent_dim=4,
            edge_index=self.edge_index,
            num_nodes=3,
            num_bins=8,
            max_age=4.0,
            hidden_dim=8,
            attention_dim=2,
            num_heads=2,
            initial_speed=1.0,
            initial_decay=1.0,
            dropout=0.0,
        )
        z0 = torch.randn(1, 3, 4, requires_grad=True)
        cache = module(
            z0,
            latest_observation_time=torch.tensor([[-0.5, -0.2, -0.8]]),
            time_grid=torch.tensor([0.0, 0.5, 1.0]),
        )
        self.assertAlmostEqual(
            float(
                cache.diagnostics["initial_active_edge_weight_mean"]
                .detach()
            ),
            1.0,
            places=5,
        )
        self.assertGreater(
            float(
                cache.diagnostics["changed_active_edge_fraction"].detach()
            ),
            0.0,
        )
        cache.edge_weight_grid.square().mean().backward()
        self.assertTrue(torch.isfinite(z0.grad).all())
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and bool(torch.any(parameter.grad != 0))
                for parameter in module.parameters()
                if parameter.requires_grad
            )
        )

    def test_controlled_initialization_and_ode_dropout(self) -> None:
        args = SimpleNamespace(
            task="extrapolation",
            latent_dim=4,
            recognition_dim=8,
            ode_hidden_dim=8,
            augmentation_dim=0,
            encoder_layers=1,
            ode_layers=1,
            attention_heads=1,
            edge_types=2,
            dropout=0.2,
            ode_dropout=0.0,
            solver="rk4",
            rtol=1e-3,
            atol=1e-4,
            observation_std=0.01,
            seed=3,
            transport_bins=8,
            transport_max_age=4.0,
            transport_hidden_dim=8,
            transport_attention_dim=2,
            transport_heads=2,
            transport_speed=1.0,
            transport_decay=1.0,
        )
        lgode, atode = build_lgode_atode_protocol_pair(
            2, 3, self.edge_index, args, "cpu"
        )
        assert_lgode_atode_protocol_match(lgode, atode)
        encoder_dropout = [
            module.p
            for module in lgode.encoder_z0.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        ode_dropout = [
            module.p
            for module in lgode.generative_ode_function.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        self.assertIn(0.2, encoder_dropout)
        self.assertTrue(all(value == 0.0 for value in ode_dropout))

    def test_stable_gradient_norm_does_not_overflow(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.full_like(parameter, 1.0e30)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        norm = stable_clip_grad_norm([parameter], 10.0)
        self.assertTrue(torch.isfinite(norm))
        self.assertGreater(float(norm), 1.0e30)
        clipped_norm = torch.linalg.vector_norm(
            parameter.grad.double()
        )
        self.assertLessEqual(float(clipped_norm), 10.000001)


if __name__ == "__main__":
    unittest.main()