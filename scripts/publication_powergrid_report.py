#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


MODELS = ("persistence", "latentode", "lgode", "atode")
TASKS = ("interpolation", "extrapolation")
COMPARISONS = (
    ("lgode", "persistence"),
    ("lgode", "latentode"),
    ("atode", "persistence"),
    ("atode", "latentode"),
    ("atode", "lgode"),
)


def primary_metrics(record: Dict[str, Any]) -> Tuple[float, float]:
    test = record["test"]
    if record["task"] == "interpolation":
        return (
            float(test["normalized_mse_unobserved"]),
            float(test["normalized_mae_unobserved"]),
        )
    return (
        float(test["normalized_mse_full"]),
        float(test["normalized_mae_full"]),
    )


def load_records(input_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(input_dir.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        mse, mae = primary_metrics(data)
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(f"Nonfinite primary metric in {path}")
        data["_path"] = str(path)
        data["_primary_mse"] = mse
        data["_primary_mae"] = mae
        records.append(data)
    return records


def validate_factorial(records: Sequence[Dict[str, Any]], seeds: Sequence[int]) -> None:
    expected = {
        (task, model, seed)
        for task in TASKS
        for model in MODELS
        for seed in seeds
    }
    actual = {
        (record["task"], record["model"], int(record["seed"]))
        for record in records
    }
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"Expected exactly {len(expected)} unique primary runs; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def mean_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else float("nan")
    if array.size > 1:
        half_width = float(
            stats.t.ppf(0.975, array.size - 1) * std / np.sqrt(array.size)
        )
        return mean, std, mean - half_width, mean + half_width
    return mean, std, float("nan"), float("nan")


def bootstrap_mean_ci(values: np.ndarray, seed: int = 12345) -> Tuple[float, float]:
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(10000, values.size), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    adjusted = np.full(len(values), np.nan, dtype=float)
    if finite_indices.size == 0:
        return adjusted.tolist()
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_statistics(records: Sequence[Dict[str, Any]]):
    grouped = defaultdict(list)
    indexed = {}
    for record in records:
        key = (record["task"], record["model"])
        grouped[key].append(record)
        indexed[(record["task"], record["model"], int(record["seed"]))] = record

    summaries = []
    for task in TASKS:
        for model in MODELS:
            group = grouped[(task, model)]
            mse = [item["_primary_mse"] for item in group]
            mae = [item["_primary_mae"] for item in group]
            mse_stats = mean_ci(mse)
            mae_stats = mean_ci(mae)
            summaries.append(
                {
                    "task": task,
                    "observed_fraction": group[0]["observed_fraction"],
                    "model": model,
                    "seeds": len(group),
                    "primary_mse_mean": mse_stats[0],
                    "primary_mse_std": mse_stats[1],
                    "primary_mse_ci95_low": mse_stats[2],
                    "primary_mse_ci95_high": mse_stats[3],
                    "primary_mae_mean": mae_stats[0],
                    "primary_mae_std": mae_stats[1],
                    "primary_mae_ci95_low": mae_stats[2],
                    "primary_mae_ci95_high": mae_stats[3],
                    "trainable_parameters": group[0]["trainable_parameter_count"],
                    "training_time_mean": float(
                        np.mean([item["training_time_seconds"] for item in group])
                    ),
                }
            )

    comparisons = []
    for task in TASKS:
        seeds = sorted(
            int(record["seed"]) for record in grouped[(task, "persistence")]
        )
        for candidate, reference in COMPARISONS:
            candidate_values = np.asarray(
                [indexed[(task, candidate, seed)]["_primary_mse"] for seed in seeds]
            )
            reference_values = np.asarray(
                [indexed[(task, reference, seed)]["_primary_mse"] for seed in seeds]
            )
            differences = candidate_values - reference_values
            difference_std = (
                differences.std(ddof=1)
                if differences.size > 1
                else float("nan")
            )
            effect_size = (
                float(differences.mean() / difference_std)
                if difference_std > 0
                else float("nan")
            )
            if differences.size > 1:
                t_p = float(
                    stats.ttest_rel(candidate_values, reference_values).pvalue
                )
                try:
                    wilcoxon_p = float(
                        stats.wilcoxon(candidate_values, reference_values).pvalue
                    )
                except ValueError:
                    wilcoxon_p = float("nan")
            else:
                t_p = float("nan")
                wilcoxon_p = float("nan")
            bootstrap_low, bootstrap_high = bootstrap_mean_ci(differences)
            comparisons.append(
                {
                    "task": task,
                    "observed_fraction": indexed[
                        (task, candidate, seeds[0])
                    ]["observed_fraction"],
                    "candidate": candidate,
                    "reference": reference,
                    "seeds": len(seeds),
                    "paired_differences": json.dumps(differences.tolist()),
                    "mean_difference": float(differences.mean()),
                    "percent_improvement": float(
                        100.0 * (reference_values.mean() - candidate_values.mean())
                        / reference_values.mean()
                    ),
                    "paired_effect_size_dz": effect_size,
                    "paired_t_p": t_p,
                    "wilcoxon_p": wilcoxon_p,
                    "bootstrap_mean_difference_ci95_low": bootstrap_low,
                    "bootstrap_mean_difference_ci95_high": bootstrap_high,
                    "holm_p": 0.0,
                    "assumptions": (
                        "Paired t-test assumes approximately normal paired differences; "
                        "Wilcoxon and paired bootstrap are robustness checks."
                    ),
                }
            )
    adjusted = holm_adjust([row["paired_t_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_p"] = value
    return summaries, comparisons


def metric_structure(record: Dict[str, Any], dimension: str) -> Dict[str, Any]:
    suffix = "unobserved" if record["task"] == "interpolation" else "full"
    return record["test"][f"normalized_per_{dimension}_{suffix}"]


def plot_curves(records: Sequence[Dict[str, Any]], output: Path) -> None:
    colors = dict(zip(MODELS, ("#555555", "#277da1", "#43aa8b", "#f3722c")))
    for task in TASKS:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        for model in MODELS:
            histories = [
                record["training"]["history"]
                for record in records
                if record["task"] == task and record["model"] == model
                and record["training"]["history"]
            ]
            if not histories:
                continue
            minimum = min(len(history) for history in histories)
            epochs = np.arange(1, minimum + 1)
            objective = np.mean(
                [[row["training"]["loss"] for row in history[:minimum]] for history in histories],
                axis=0,
            )
            mse_key = (
                "normalized_mse_unobserved"
                if task == "interpolation"
                else "normalized_mse_full"
            )
            mae_key = mse_key.replace("mse", "mae")
            validation_mse = np.mean(
                [[row["validation"][mse_key] for row in history[:minimum]] for history in histories],
                axis=0,
            )
            validation_mae = np.mean(
                [[row["validation"][mae_key] for row in history[:minimum]] for history in histories],
                axis=0,
            )
            axes[0].plot(epochs, objective, label=model, color=colors[model])
            axes[1].plot(epochs, validation_mse, label=model, color=colors[model])
            axes[2].plot(epochs, validation_mae, label=model, color=colors[model])
        for axis, title in zip(axes, ("Training objective", "Validation primary MSE", "Validation primary MAE")):
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
        axes[0].legend()
        figure.tight_layout()
        figure.savefig(output / f"{task}_training_validation.pdf")
        plt.close(figure)


def plot_structured(records: Sequence[Dict[str, Any]], output: Path) -> None:
    for dimension in ("horizon", "node"):
        for metric in ("mse", "mae"):
            for task in TASKS:
                figure, axis = plt.subplots(figsize=(8, 4))
                for model in MODELS:
                    arrays = [
                        metric_structure(record, dimension)[metric]
                        for record in records
                        if record["task"] == task and record["model"] == model
                    ]
                    axis.plot(np.mean(arrays, axis=0), label=model)
                axis.set_xlabel(
                    "Forecast horizon" if dimension == "horizon" else "Node index"
                )
                axis.set_ylabel(f"Primary normalized {metric.upper()}")
                axis.legend()
                axis.grid(alpha=0.25)
                figure.tight_layout()
                figure.savefig(output / f"{task}_{dimension}_{metric}.pdf")
                plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, task in zip(axes, TASKS):
        feature_names = list(
            metric_structure(
                next(record for record in records if record["task"] == task),
                "feature",
            )
        )
        positions = np.arange(len(feature_names))
        width = 0.18
        for model_index, model in enumerate(MODELS):
            values = []
            for feature in feature_names:
                values.append(
                    np.mean([
                        metric_structure(record, "feature")[feature]["mae"]
                        for record in records
                        if record["task"] == task and record["model"] == model
                    ])
                )
            axis.bar(positions + model_index * width, values, width, label=model)
        axis.set_xticks(positions + 1.5 * width, feature_names, rotation=30)
        axis.set_title(task)
        axis.set_ylabel("Primary normalized MAE")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output / "per_feature_mae.pdf")
    plt.close(figure)


def plot_comparisons(records: Sequence[Dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, (candidate, reference) in zip(axes.flat, COMPARISONS):
        for task, marker in zip(TASKS, ("o", "s")):
            candidate_values = [
                record["_primary_mse"] for record in records
                if record["task"] == task and record["model"] == candidate
            ]
            reference_values = [
                record["_primary_mse"] for record in records
                if record["task"] == task and record["model"] == reference
            ]
            axis.scatter(reference_values, candidate_values, label=task, marker=marker)
        limits = axis.get_xlim()
        axis.plot(limits, limits, color="black", linewidth=1)
        axis.set_title(f"{candidate} vs {reference}")
        axis.set_xlabel(reference)
        axis.set_ylabel(candidate)
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    figure.tight_layout()
    figure.savefig(output / "paired_seed_primary_mse.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for task, marker in zip(TASKS, ("o", "s")):
        task_records = [record for record in records if record["task"] == task]
        axis.scatter(
            [record["trainable_parameter_count"] for record in task_records],
            [record["_primary_mse"] for record in task_records],
            label=task,
            marker=marker,
            alpha=0.75,
        )
    axis.set_xlabel("Trainable parameters")
    axis.set_ylabel("Primary normalized MSE")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "parameters_vs_primary_mse.pdf")
    plt.close(figure)

    atode = [record for record in records if record["model"] == "atode"]
    if atode:
        figure, axis = plt.subplots(figsize=(8, 4))
        labels = [f"{record['task']}-s{record['seed']}" for record in atode]
        changes = [
            float(
                record.get("diagnostics", {})
                .get("test", {})
                .get("transport", {})
                .get("mean_absolute_first_last_edge_weight_change", float("nan"))
            )
            for record in atode
        ]
        axis.bar(labels, changes)
        axis.set_ylabel("Mean |first-last edge weight|")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(output / "atode_transport_change.pdf")
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    records = load_records(args.input_dir, args.pattern)
    validate_factorial(records, args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, comparisons = build_statistics(records)
    write_csv(args.output_dir / "publication_summary.csv", summaries)
    write_csv(args.output_dir / "publication_paired_comparisons.csv", comparisons)
    plot_curves(records, args.output_dir)
    plot_structured(records, args.output_dir)
    plot_comparisons(records, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())