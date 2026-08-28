from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lib.ieee39_transient_data import (
    _mask_for_scenario,
    build_ieee39_dataloaders,
    load_ieee39_archive,
)
from lib.powergrid_baselines import PersistenceBaseline
from scripts.preprocess_ieee39_transient import complete_directed_graph


PROCESSED = Path(
    "data/ieee39_transient/processed/ieee39_transient_v1.npz"
)


def collate_one(sample):
    from lib.simbench_lgode_data import collate_powergrid_lgode

    return collate_powergrid_lgode([sample])


class IEEE39TransientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.archive = load_ieee39_archive(PROCESSED)

    def test_processed_schema_and_generator_order(self) -> None:
        self.assertEqual(
            tuple(self.archive.scenario_state.shape), (12852, 60, 10, 5)
        )
        self.assertEqual(
            self.archive.generator_names,
            tuple(f"G{index:02d}" for index in range(1, 11)),
        )
        self.assertEqual(tuple(self.archive.timestamps_seconds.shape), (12852, 60))
        self.assertTrue(torch.isfinite(self.archive.scenario_state).all())

    def test_scenario_splits_are_disjoint(self) -> None:
        train = set(self.archive.split_indices["train"].tolist())
        validation = set(self.archive.split_indices["validation"].tolist())
        test = set(self.archive.split_indices["test"].tolist())
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(len(train | validation | test), 12852)

    def test_training_only_normalization(self) -> None:
        training = self.archive.scenario_state[
            self.archive.split_indices["train"]
        ].double()
        expected_mean = training.mean(dim=(0, 1, 2))
        expected_std = training.std(dim=(0, 1, 2), unbiased=False)
        self.assertTrue(torch.allclose(expected_mean, self.archive.normalization_mean))
        self.assertTrue(torch.allclose(expected_std, self.archive.normalization_std))

    def test_complete_graph_is_deterministic(self) -> None:
        first = complete_directed_graph(10)
        second = complete_directed_graph(10)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first[:, 0]))
        self.assertEqual(first.shape, (2, 90))
        self.assertFalse(np.any(first[0] == first[1]))
        self.assertEqual(len(set(map(tuple, first.T.tolist()))), 90)
        self.assertEqual(
            hashlib.sha256(first.tobytes()).hexdigest(),
            self.archive.metadata["graph_sha256"],
        )

    def test_masks_are_asynchronous_and_reproducible(self) -> None:
        first = _mask_for_scenario(4, 10, 60, 0.2, 9, "interpolation")
        second = _mask_for_scenario(4, 10, 60, 0.2, 9, "interpolation")
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(any(not np.array_equal(first[0], first[node]) for node in range(1, 10)))
        self.assertTrue(np.all(first.sum(axis=1) >= 2))

        loaders_a = build_ieee39_dataloaders(
            PROCESSED, task="interpolation", observed_fraction=0.2,
            batch_size=4, seed=1, mask_seed=11, scale="smoke",
        )
        loaders_b = build_ieee39_dataloaders(
            PROCESSED, task="interpolation", observed_fraction=0.2,
            batch_size=4, seed=99, mask_seed=11, scale="smoke",
        )
        self.assertTrue(
            torch.equal(loaders_a.train.dataset.masks, loaders_b.train.dataset.masks)
        )

    def test_interpolation_withheld_mask(self) -> None:
        loaders = build_ieee39_dataloaders(
            PROCESSED, task="interpolation", observed_fraction=0.2,
            batch_size=4, seed=1, mask_seed=1, scale="smoke",
        )
        batch = next(iter(loaders.train))
        observed = batch.encoder_observation_mask.unsqueeze(-1).expand_as(
            batch.target_values
        )
        self.assertFalse(torch.any(observed & batch.interpolation_withheld_mask))
        self.assertTrue(
            torch.equal(batch.training_loss_mask, batch.interpolation_withheld_mask)
        )

    def test_extrapolation_has_no_future_encoder_values(self) -> None:
        loaders = build_ieee39_dataloaders(
            PROCESSED, task="extrapolation", observed_fraction=0.8,
            batch_size=4, seed=1, mask_seed=1, scale="smoke",
        )
        batch = next(iter(loaders.train))
        self.assertEqual(batch.encoder_observation_mask.shape[-1], 30)
        self.assertEqual(batch.target_values.shape[2], 30)
        self.assertLessEqual(float(batch.encoder_graph.pos.max()), 0.0)
        self.assertGreater(float(batch.target_times.min()), 0.0)
        self.assertTrue(torch.all(batch.extrapolation_future_mask))

        dataset = loaders.validation.dataset
        scenario_index = dataset.windows[0].scenario_index
        before = dataset[0]
        before_prediction = PersistenceBaseline("extrapolation").predict(
            collate_one(before)
        )
        original = dataset.archive.scenario_state[
            scenario_index, 45, 0, 0
        ].item()
        dataset.archive.scenario_state[
            scenario_index, 45, 0, 0
        ] = original + 1000.0
        try:
            after = dataset[0]
            after_prediction = PersistenceBaseline("extrapolation").predict(
                collate_one(after)
            )
        finally:
            dataset.archive.scenario_state[
                scenario_index, 45, 0, 0
            ] = original
        self.assertTrue(torch.equal(before["encoder_graph"].x, after["encoder_graph"].x))
        self.assertTrue(torch.equal(before_prediction, after_prediction))
        self.assertFalse(torch.equal(before["target_values"], after["target_values"]))

    def test_processed_cache_hash_invalidation(self) -> None:
        metadata_path = PROCESSED.with_name("ieee39_transient_v1_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["processed_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "metadata.json"
            invalid.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ieee39_archive(PROCESSED, invalid)

    def test_persistence_uses_latest_observed_context(self) -> None:
        loaders = build_ieee39_dataloaders(
            PROCESSED, task="extrapolation", observed_fraction=0.2,
            batch_size=2, seed=1, mask_seed=1, scale="smoke",
        )
        batch = next(iter(loaders.validation))
        prediction = PersistenceBaseline("extrapolation").predict(batch)
        counts = batch.encoder_graph.y.reshape(2, 10)
        pointer = batch.encoder_graph.ptr
        for scenario in range(2):
            offset = int(pointer[scenario])
            for node in range(10):
                count = int(counts[scenario, node])
                latest = batch.encoder_graph.x[offset + count - 1]
                self.assertTrue(torch.equal(prediction[scenario, node, 0], latest))
                offset += count


if __name__ == "__main__":
    unittest.main()