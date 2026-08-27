from __future__ import annotations

import unittest

from scripts.publication_powergrid_report import (
    COMPARISONS,
    MODELS,
    TASKS,
    build_statistics,
    holm_adjust,
    primary_metrics,
    validate_factorial,
)


class PublicationReportingTests(unittest.TestCase):
    def make_records(self):
        records = []
        for task_index, task in enumerate(TASKS):
            for model_index, model in enumerate(MODELS):
                for seed in range(1, 6):
                    value = (
                        0.1
                        + 0.01 * task_index
                        + 0.005 * model_index
                        + seed * 1e-4
                        + model_index * seed * 1e-5
                    )
                    records.append(
                        {
                            "task": task,
                            "model": model,
                            "seed": seed,
                            "test": {
                                "normalized_mse_full": value,
                                "normalized_mae_full": value / 2,
                                "normalized_mse_unobserved": value + 0.02,
                                "normalized_mae_unobserved": value / 2 + 0.01,
                            },
                            "trainable_parameter_count": 100 + model_index,
                            "training_time_seconds": 10.0 + seed,
                            "_primary_mse": (
                                value + 0.02 if task == "interpolation" else value
                            ),
                            "_primary_mae": (
                                value / 2 + 0.01
                                if task == "interpolation"
                                else value / 2
                            ),
                        }
                    )
        return records

    def test_primary_metric_selection(self) -> None:
        interpolation = self.make_records()[0]
        mse, mae = primary_metrics(interpolation)
        self.assertEqual(mse, interpolation["test"]["normalized_mse_unobserved"])
        self.assertEqual(mae, interpolation["test"]["normalized_mae_unobserved"])

    def test_exact_factorial_and_statistics(self) -> None:
        records = self.make_records()
        validate_factorial(records, [1, 2, 3, 4, 5])
        summaries, comparisons = build_statistics(records)
        self.assertEqual(len(summaries), len(TASKS) * len(MODELS))
        self.assertEqual(len(comparisons), len(TASKS) * len(COMPARISONS))
        self.assertTrue(all(0.0 <= row["holm_p"] <= 1.0 for row in comparisons))

        with self.assertRaises(ValueError):
            validate_factorial(records[:-1], [1, 2, 3, 4, 5])

    def test_holm_adjustment(self) -> None:
        adjusted = holm_adjust([0.01, 0.03, 0.02])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertGreaterEqual(adjusted[0], 0.01)


if __name__ == "__main__":
    unittest.main()