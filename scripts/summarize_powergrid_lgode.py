#!/usr/bin/env python3
"""
Aggregate power-grid LG-ODE experiment result JSON files.

Outputs
-------
results.csv
    One standardized row per experiment result JSON.

summary.csv
    Aggregated by:
        task
        observed_fraction
        model

    Reports:
        number of seeds
        mean normalized test MSE
        sample standard deviation of normalized test MSE
        mean normalized test MAE
        sample standard deviation of normalized test MAE
        mean best-validation epoch
        mean training time
        parameter count

paired_comparisons.csv
    Seed-paired comparisons:

        AT-ODE versus LG-ODE
        LG-ODE versus Latent ODE
        AT-ODE versus Latent ODE

    The relative improvement for model A versus model B is:

        100 * (MSE_B - MSE_A) / MSE_B

    Therefore, the main AT-ODE transport improvement is:

        100 * (MSE_LGODE - MSE_ATODE) / MSE_LGODE

Example
-------
python scripts/summarize_powergrid_lgode.py \\
    --input-dir results/powergrid_lgode \\
    --output-dir results/powergrid_lgode/summary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "persistence",
    "latentode",
    "lgode",
    "atode",
]

TASK_ORDER = [
    "interpolation",
    "extrapolation",
]

OBSERVATION_FRACTION_ORDER = [
    0.4,
    0.6,
    0.8,
]

PAIRED_COMPARISONS = [
    ("atode", "lgode", "AT-ODE versus LG-ODE"),
    ("lgode", "latentode", "LG-ODE versus Latent ODE"),
    ("atode", "latentode", "AT-ODE versus Latent ODE"),
]


@dataclass(frozen=True)
class ParsedResult:
    """One standardized experiment result."""

    source_file: str
    simbench_code: str
    task: str
    observed_fraction: float
    model: str
    seed: int
    mask_seed: int
    normalized_test_mse: float
    normalized_test_mae: float
    best_validation_epoch: int
    best_validation_mse: float
    training_time_seconds: float
    parameter_count: int
    trainable_parameter_count: int
    trajectory_length: Optional[int]
    context_length: Optional[int]
    forecast_length: Optional[int]
    stride: Optional[int]
    batch_size: Optional[int]
    alias: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "simbench_code": self.simbench_code,
            "task": self.task,
            "observed_fraction": self.observed_fraction,
            "model": self.model,
            "seed": self.seed,
            "mask_seed": self.mask_seed,
            "normalized_test_mse": self.normalized_test_mse,
            "normalized_test_mae": self.normalized_test_mae,
            "best_validation_epoch": self.best_validation_epoch,
            "best_validation_mse": self.best_validation_mse,
            "training_time_seconds": self.training_time_seconds,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "trajectory_length": self.trajectory_length,
            "context_length": self.context_length,
            "forecast_length": self.forecast_length,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "alias": self.alias,
        }


def _normalize_key(key: str) -> str:
    return (
        str(key)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _flatten_mapping(
    value: Mapping[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Flatten a nested JSON mapping using normalized dotted keys.

    For example:

        {"test": {"normalized_mse": 0.1}}

    becomes:

        {"test.normalized_mse": 0.1}
    """

    flattened: Dict[str, Any] = {}

    for raw_key, item in value.items():
        key = _normalize_key(raw_key)
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(item, Mapping):
            flattened.update(
                _flatten_mapping(item, prefix=full_key)
            )
        else:
            flattened[full_key] = item

    return flattened


def _candidate_value(
    flattened: Mapping[str, Any],
    candidates: Sequence[str],
    *,
    default: Any = None,
    required: bool = False,
    field_name: Optional[str] = None,
) -> Any:
    """Return the first non-null value matching a candidate key."""

    normalized_candidates = [
        ".".join(_normalize_key(part) for part in key.split("."))
        for key in candidates
    ]

    for key in normalized_candidates:
        if key in flattened and flattened[key] is not None:
            return flattened[key]

    # Also permit a candidate to match a unique nested suffix. This supports
    # result structures such as "results.test.normalized_mse" without making
    # the parser dependent on one exact top-level schema.
    for candidate in normalized_candidates:
        suffix = f".{candidate}"
        matches = [
            value
            for key, value in flattened.items()
            if key.endswith(suffix) and value is not None
        ]

        if len(matches) == 1:
            return matches[0]

    if required:
        label = field_name or "/".join(candidates)
        raise KeyError(f"Missing required result field: {label}")

    return default


def _to_float(
    value: Any,
    field_name: str,
    *,
    finite: bool = True,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric, not boolean"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be numeric; got {value!r}"
        ) from exc

    if finite and not math.isfinite(result):
        raise ValueError(
            f"{field_name} must be finite; got {result}"
        )

    return result


def _to_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be an integer, not boolean"
        )

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be an integer; got {value!r}"
        ) from exc

    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"{field_name} must be an integer; got {value!r}"
        )

    return int(numeric)


def _optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _to_int(value, field_name)


def _normalize_model_name(value: Any) -> str:
    model = _normalize_key(str(value))

    aliases = {
        "copy_persistence": "persistence",
        "copypersistence": "persistence",
        "persistencebaseline": "persistence",
        "persistence_baseline": "persistence",
        "latent_ode": "latentode",
        "independent_latent_ode": "latentode",
        "independentlatentode": "latentode",
        "lg_ode": "lgode",
        "at_ode": "atode",
        "attention_transport_ode": "atode",
        "attentiontransportode": "atode",
    }
    model = aliases.get(model, model)

    if model not in MODEL_ORDER:
        raise ValueError(
            f"Unknown model {value!r}; expected one of {MODEL_ORDER}"
        )

    return model


def _display_model_name(model: str) -> str:
    names = {
        "persistence": "Persistence",
        "latentode": "Latent ODE",
        "lgode": "LG-ODE",
        "atode": "AT-ODE",
    }
    return names.get(model, model)


def _normalize_task_name(value: Any) -> str:
    task = _normalize_key(str(value))

    aliases = {
        "interp": "interpolation",
        "interpolate": "interpolation",
        "extrap": "extrapolation",
        "forecast": "extrapolation",
        "forecasting": "extrapolation",
    }
    task = aliases.get(task, task)

    if task not in TASK_ORDER:
        raise ValueError(
            f"Unknown task {value!r}; expected one of {TASK_ORDER}"
        )

    return task


def _normalize_observed_fraction(value: Any) -> float:
    fraction = _to_float(value, "observed_fraction")

    # Permit command-line-style percentages in JSON while writing canonical
    # fractions to the output.
    if fraction in {40.0, 60.0, 80.0}:
        fraction /= 100.0

    for allowed in OBSERVATION_FRACTION_ORDER:
        if math.isclose(
            fraction,
            allowed,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return allowed

    raise ValueError(
        "observed_fraction must be 0.4, 0.6 or 0.8; "
        f"got {fraction}"
    )


def _parse_result_json(
    path: Path,
    input_directory: Path,
) -> ParsedResult:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file {path}") from exc

    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Result JSON must contain an object: {path}"
        )

    flattened = _flatten_mapping(raw)

    model = _normalize_model_name(
        _candidate_value(
            flattened,
            [
                "model",
                "model_name",
                "config.model",
                "config.model_name",
                "args.model",
                "arguments.model",
                "experiment.model",
            ],
            required=True,
            field_name="model",
        )
    )

    task = _normalize_task_name(
        _candidate_value(
            flattened,
            [
                "task",
                "config.task",
                "args.task",
                "arguments.task",
                "experiment.task",
            ],
            required=True,
            field_name="task",
        )
    )

    observed_fraction = _normalize_observed_fraction(
        _candidate_value(
            flattened,
            [
                "observed_fraction",
                "observation_fraction",
                "sample_percent",
                "config.observed_fraction",
                "config.observation_fraction",
                "args.observed_fraction",
                "arguments.observed_fraction",
                "experiment.observed_fraction",
            ],
            required=True,
            field_name="observed_fraction",
        )
    )

    seed = _to_int(
        _candidate_value(
            flattened,
            [
                "seed",
                "random_seed",
                "config.seed",
                "config.random_seed",
                "args.seed",
                "args.random_seed",
                "arguments.seed",
                "experiment.seed",
            ],
            required=True,
            field_name="seed",
        ),
        "seed",
    )

    mask_seed_raw = _candidate_value(
        flattened,
        [
            "mask_seed",
            "config.mask_seed",
            "args.mask_seed",
            "arguments.mask_seed",
            "experiment.mask_seed",
        ],
        default=seed,
    )
    mask_seed = _to_int(mask_seed_raw, "mask_seed")

    normalized_test_mse = _to_float(
        _candidate_value(
            flattened,
            [
                "test.normalized_mse",
                "test.normalized_test_mse",
                "test.metrics.normalized_mse",
                "test.metrics.mse",
                "metrics.test.normalized_mse",
                "metrics.test.mse",
                "test_mse",
                "normalized_test_mse",
                "test_mse_normalized",
                "results.test.normalized_mse",
                "results.test.mse",
            ],
            required=True,
            field_name="normalized test MSE",
        ),
        "normalized_test_mse",
    )

    normalized_test_mae = _to_float(
        _candidate_value(
            flattened,
            [
                "test.normalized_mae",
                "test.normalized_test_mae",
                "test.metrics.normalized_mae",
                "test.metrics.mae",
                "metrics.test.normalized_mae",
                "metrics.test.mae",
                "test_mae",
                "normalized_test_mae",
                "test_mae_normalized",
                "results.test.normalized_mae",
                "results.test.mae",
            ],
            required=True,
            field_name="normalized test MAE",
        ),
        "normalized_test_mae",
    )

    best_validation_epoch = _to_int(
        _candidate_value(
            flattened,
            [
                "best_validation_epoch",
                "best_val_epoch",
                "best_epoch",
                "validation.best_epoch",
                "validation.epoch",
                "metrics.best_validation_epoch",
                "training.best_validation_epoch",
                "training.best_epoch",
                "checkpoint.best_epoch",
            ],
            required=True,
            field_name="best-validation epoch",
        ),
        "best_validation_epoch",
    )

    best_validation_mse_raw = _candidate_value(
        flattened,
        [
            "best_validation_mse",
            "best_val_mse",
            "validation.best_mse",
            "validation.normalized_mse",
            "metrics.best_validation_mse",
            "training.best_validation_mse",
            "checkpoint.validation_mse",
        ],
        default=float("nan"),
    )
    best_validation_mse = _to_float(
        best_validation_mse_raw,
        "best_validation_mse",
        finite=False,
    )

    training_time_seconds = _to_float(
        _candidate_value(
            flattened,
            [
                "training_time_seconds",
                "train_time_seconds",
                "training_seconds",
                "elapsed_training_seconds",
                "runtime.training_seconds",
                "timing.training_seconds",
                "training.time_seconds",
                "training.elapsed_seconds",
            ],
            required=True,
            field_name="training time",
        ),
        "training_time_seconds",
    )

    parameter_count = _to_int(
        _candidate_value(
            flattened,
            [
                "parameter_count",
                "num_parameters",
                "number_of_parameters",
                "total_parameters",
                "model.parameter_count",
                "model.num_parameters",
                "model.total_parameters",
                "metrics.parameter_count",
            ],
            required=True,
            field_name="parameter count",
        ),
        "parameter_count",
    )

    trainable_parameter_count_raw = _candidate_value(
        flattened,
        [
            "trainable_parameter_count",
            "trainable_parameters",
            "num_trainable_parameters",
            "model.trainable_parameter_count",
            "model.trainable_parameters",
        ],
        default=parameter_count,
    )
    trainable_parameter_count = _to_int(
        trainable_parameter_count_raw,
        "trainable_parameter_count",
    )

    simbench_code = str(
        _candidate_value(
            flattened,
            [
                "simbench_code",
                "grid",
                "grid_code",
                "dataset.simbench_code",
                "config.simbench_code",
                "args.simbench_code",
                "arguments.simbench_code",
                "experiment.simbench_code",
            ],
            default="unknown",
        )
    )

    trajectory_length = _optional_int(
        _candidate_value(
            flattened,
            [
                "trajectory_length",
                "config.trajectory_length",
                "args.trajectory_length",
                "arguments.trajectory_length",
            ],
            default=None,
        ),
        "trajectory_length",
    )

    context_length = _optional_int(
        _candidate_value(
            flattened,
            [
                "context_length",
                "config.context_length",
                "args.context_length",
                "arguments.context_length",
            ],
            default=None,
        ),
        "context_length",
    )

    forecast_length = _optional_int(
        _candidate_value(
            flattened,
            [
                "forecast_length",
                "config.forecast_length",
                "args.forecast_length",
                "arguments.forecast_length",
            ],
            default=None,
        ),
        "forecast_length",
    )

    stride = _optional_int(
        _candidate_value(
            flattened,
            [
                "stride",
                "config.stride",
                "args.stride",
                "arguments.stride",
            ],
            default=None,
        ),
        "stride",
    )

    batch_size = _optional_int(
        _candidate_value(
            flattened,
            [
                "batch_size",
                "config.batch_size",
                "args.batch_size",
                "arguments.batch_size",
                "training.batch_size",
            ],
            default=None,
        ),
        "batch_size",
    )

    alias = str(
        _candidate_value(
            flattened,
            [
                "alias",
                "config.alias",
                "args.alias",
                "arguments.alias",
                "experiment.alias",
            ],
            default="",
        )
    )

    if normalized_test_mse < 0.0:
        raise ValueError(
            f"normalized_test_mse cannot be negative: {path}"
        )
    if normalized_test_mae < 0.0:
        raise ValueError(
            f"normalized_test_mae cannot be negative: {path}"
        )
    if best_validation_epoch < 0:
        raise ValueError(
            f"best_validation_epoch cannot be negative: {path}"
        )
    if training_time_seconds < 0.0:
        raise ValueError(
            f"training_time_seconds cannot be negative: {path}"
        )
    if parameter_count < 0:
        raise ValueError(
            f"parameter_count cannot be negative: {path}"
        )
    if trainable_parameter_count < 0:
        raise ValueError(
            f"trainable_parameter_count cannot be negative: {path}"
        )

    try:
        source_file = str(path.relative_to(input_directory))
    except ValueError:
        source_file = str(path)

    return ParsedResult(
        source_file=source_file,
        simbench_code=simbench_code,
        task=task,
        observed_fraction=observed_fraction,
        model=model,
        seed=seed,
        mask_seed=mask_seed,
        normalized_test_mse=normalized_test_mse,
        normalized_test_mae=normalized_test_mae,
        best_validation_epoch=best_validation_epoch,
        best_validation_mse=best_validation_mse,
        training_time_seconds=training_time_seconds,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        trajectory_length=trajectory_length,
        context_length=context_length,
        forecast_length=forecast_length,
        stride=stride,
        batch_size=batch_size,
        alias=alias,
    )


def _find_json_files(
    input_directory: Path,
    pattern: str,
    output_directory: Path,
) -> List[Path]:
    files = []

    for path in input_directory.glob(pattern):
        if not path.is_file() or path.suffix.lower() != ".json":
            continue

        # Avoid parsing JSON files inside an output directory nested under the
        # input directory if a caller uses a broad pattern.
        try:
            path.resolve().relative_to(output_directory.resolve())
            inside_output = True
        except ValueError:
            inside_output = False

        if inside_output and output_directory != input_directory:
            continue

        files.append(path)

    return sorted(files)


def load_results(
    input_directory: Path,
    pattern: str,
    output_directory: Path,
    *,
    strict: bool,
) -> pd.DataFrame:
    files = _find_json_files(
        input_directory,
        pattern,
        output_directory,
    )

    if not files:
        raise FileNotFoundError(
            f"No JSON files matched {pattern!r} under {input_directory}"
        )

    parsed: List[ParsedResult] = []
    skipped: List[Tuple[Path, str]] = []

    for path in files:
        try:
            parsed.append(
                _parse_result_json(path, input_directory)
            )
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise RuntimeError(
                    f"Could not parse result file {path}: {exc}"
                ) from exc
            skipped.append((path, str(exc)))

    for path, reason in skipped:
        warnings.warn(
            f"Skipping non-result or malformed JSON {path}: {reason}",
            stacklevel=2,
        )

    if not parsed:
        raise RuntimeError(
            "No valid power-grid result JSON files were found"
        )

    frame = pd.DataFrame([item.as_dict() for item in parsed])

    frame["task"] = pd.Categorical(
        frame["task"],
        categories=TASK_ORDER,
        ordered=True,
    )
    frame["observed_fraction"] = pd.Categorical(
        frame["observed_fraction"],
        categories=OBSERVATION_FRACTION_ORDER,
        ordered=True,
    )
    frame["model"] = pd.Categorical(
        frame["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    frame = frame.sort_values(
        [
            "task",
            "observed_fraction",
            "model",
            "seed",
            "mask_seed",
            "source_file",
        ],
        kind="stable",
    ).reset_index(drop=True)

    # Convert categorical columns back to ordinary values for portable CSVs.
    frame["task"] = frame["task"].astype(str)
    frame["observed_fraction"] = frame[
        "observed_fraction"
    ].astype(float)
    frame["model"] = frame["model"].astype(str)

    return frame


def _check_duplicate_runs(
    results: pd.DataFrame,
    *,
    allow_duplicates: bool,
) -> None:
    """
    Detect ambiguous repeated runs.

    The primary experiment identity follows the frozen protocol:
    grid, task, fraction, model, seed and mask seed.
    """

    identity = [
        "simbench_code",
        "task",
        "observed_fraction",
        "model",
        "seed",
        "mask_seed",
    ]

    duplicate_mask = results.duplicated(
        identity,
        keep=False,
    )

    if not duplicate_mask.any():
        return

    duplicate_rows = results.loc[
        duplicate_mask,
        identity + ["source_file"],
    ].sort_values(identity)

    message = (
        "Multiple result files have the same experiment identity:\n"
        + duplicate_rows.to_string(index=False)
    )

    if allow_duplicates:
        warnings.warn(
            message
            + "\nDuplicates will remain in results.csv and will be averaged "
              "within each seed before paired comparisons.",
            stacklevel=2,
        )
    else:
        raise ValueError(
            message
            + "\nRemove duplicate runs or use --allow-duplicates."
        )


def _sample_standard_deviation(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) < 2:
        return float("nan")

    return float(values.std(ddof=1))


def _unique_parameter_count(series: pd.Series) -> Any:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().astype(np.int64)

    unique = np.sort(values.unique())

    if unique.size == 0:
        return np.nan

    if unique.size == 1:
        return int(unique[0])

    # A changed count within one task/fraction/model group normally indicates
    # an accidental architecture mismatch. Preserve that information rather
    # than silently selecting one count.
    return "|".join(str(int(value)) for value in unique)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Build the required task/fraction/model summary."""

    group_columns = [
        "task",
        "observed_fraction",
        "model",
    ]

    rows: List[Dict[str, Any]] = []

    for keys, group in results.groupby(
        group_columns,
        sort=False,
        observed=True,
    ):
        task, observed_fraction, model = keys

        rows.append(
            {
                "task": task,
                "observed_fraction": float(observed_fraction),
                "model": model,
                "model_display_name": _display_model_name(model),
                "number_of_seeds": int(group["seed"].nunique()),
                "number_of_runs": int(len(group)),
                "mean_normalized_mse": float(
                    group["normalized_test_mse"].mean()
                ),
                "std_normalized_mse": _sample_standard_deviation(
                    group["normalized_test_mse"]
                ),
                "mean_normalized_mae": float(
                    group["normalized_test_mae"].mean()
                ),
                "std_normalized_mae": _sample_standard_deviation(
                    group["normalized_test_mae"]
                ),
                "mean_best_validation_epoch": float(
                    group["best_validation_epoch"].mean()
                ),
                "std_best_validation_epoch": (
                    _sample_standard_deviation(
                        group["best_validation_epoch"]
                    )
                ),
                "mean_best_validation_mse": float(
                    group["best_validation_mse"].mean()
                ),
                "mean_training_time_seconds": float(
                    group["training_time_seconds"].mean()
                ),
                "std_training_time_seconds": (
                    _sample_standard_deviation(
                        group["training_time_seconds"]
                    )
                ),
                "parameter_count": _unique_parameter_count(
                    group["parameter_count"]
                ),
                "trainable_parameter_count": (
                    _unique_parameter_count(
                        group["trainable_parameter_count"]
                    )
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary["task"] = pd.Categorical(
        summary["task"],
        categories=TASK_ORDER,
        ordered=True,
    )
    summary["observed_fraction"] = pd.Categorical(
        summary["observed_fraction"],
        categories=OBSERVATION_FRACTION_ORDER,
        ordered=True,
    )
    summary["model"] = pd.Categorical(
        summary["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    summary = summary.sort_values(
        ["task", "observed_fraction", "model"],
        kind="stable",
    ).reset_index(drop=True)

    summary["task"] = summary["task"].astype(str)
    summary["observed_fraction"] = summary[
        "observed_fraction"
    ].astype(float)
    summary["model"] = summary["model"].astype(str)

    return summary


def _collapse_runs_for_pairing(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one row per paired identity and model.

    Normally there is exactly one row. If --allow-duplicates was used, repeated
    runs are averaged before pairing so they cannot give one seed extra weight.
    """

    grouping = [
        "simbench_code",
        "task",
        "observed_fraction",
        "seed",
        "mask_seed",
        "model",
    ]

    collapsed = (
        results.groupby(
            grouping,
            as_index=False,
            sort=False,
            observed=True,
        )
        .agg(
            normalized_test_mse=(
                "normalized_test_mse",
                "mean",
            ),
            normalized_test_mae=(
                "normalized_test_mae",
                "mean",
            ),
            duplicate_run_count=("source_file", "size"),
        )
    )

    return collapsed


def _safe_relative_improvement(
    challenger: pd.Series,
    baseline: pd.Series,
) -> pd.Series:
    """
    Compute 100 * (baseline - challenger) / baseline.

    A zero baseline gives NaN rather than an infinite or misleading percentage.
    """

    challenger = challenger.astype(float)
    baseline = baseline.astype(float)

    denominator = baseline.where(baseline != 0.0, np.nan)
    return 100.0 * (baseline - challenger) / denominator


def build_paired_comparisons(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build seed-paired model comparisons.

    Pairing keys include grid, task, observation fraction, seed and mask seed,
    ensuring that compared models use the same experiment realization.
    """

    collapsed = _collapse_runs_for_pairing(results)

    pairing_keys = [
        "simbench_code",
        "task",
        "observed_fraction",
        "seed",
        "mask_seed",
    ]

    rows: List[Dict[str, Any]] = []

    group_columns = [
        "task",
        "observed_fraction",
    ]

    for group_key, task_fraction_group in collapsed.groupby(
        group_columns,
        sort=False,
        observed=True,
    ):
        task, observed_fraction = group_key

        for model_a, model_b, comparison_name in PAIRED_COMPARISONS:
            left = task_fraction_group[
                task_fraction_group["model"] == model_a
            ][
                pairing_keys
                + [
                    "normalized_test_mse",
                    "normalized_test_mae",
                ]
            ].rename(
                columns={
                    "normalized_test_mse": "mse_a",
                    "normalized_test_mae": "mae_a",
                }
            )

            right = task_fraction_group[
                task_fraction_group["model"] == model_b
            ][
                pairing_keys
                + [
                    "normalized_test_mse",
                    "normalized_test_mae",
                ]
            ].rename(
                columns={
                    "normalized_test_mse": "mse_b",
                    "normalized_test_mae": "mae_b",
                }
            )

            paired = left.merge(
                right,
                on=pairing_keys,
                how="inner",
                validate="one_to_one",
            )

            if paired.empty:
                rows.append(
                    {
                        "task": task,
                        "observed_fraction": float(
                            observed_fraction
                        ),
                        "comparison": comparison_name,
                        "model_a": model_a,
                        "model_a_display_name": (
                            _display_model_name(model_a)
                        ),
                        "model_b": model_b,
                        "model_b_display_name": (
                            _display_model_name(model_b)
                        ),
                        "number_of_pairs": 0,
                        "mean_normalized_mse_model_a": np.nan,
                        "mean_normalized_mse_model_b": np.nan,
                        "mean_mse_improvement_model_a": np.nan,
                        "std_mse_improvement_model_a": np.nan,
                        "mean_relative_mse_improvement_percent": np.nan,
                        "std_relative_mse_improvement_percent": np.nan,
                        "median_relative_mse_improvement_percent": np.nan,
                        "mean_normalized_mae_model_a": np.nan,
                        "mean_normalized_mae_model_b": np.nan,
                        "mean_mae_improvement_model_a": np.nan,
                        "std_mae_improvement_model_a": np.nan,
                        "mean_relative_mae_improvement_percent": np.nan,
                        "win_rate_mse_model_a": np.nan,
                        "tie_rate_mse": np.nan,
                    }
                )
                continue

            # Positive differences mean model A improved upon model B.
            mse_improvement = paired["mse_b"] - paired["mse_a"]
            mae_improvement = paired["mae_b"] - paired["mae_a"]

            relative_mse_improvement = _safe_relative_improvement(
                challenger=paired["mse_a"],
                baseline=paired["mse_b"],
            )
            relative_mae_improvement = _safe_relative_improvement(
                challenger=paired["mae_a"],
                baseline=paired["mae_b"],
            )

            tolerance = 1e-12
            wins = mse_improvement > tolerance
            ties = mse_improvement.abs() <= tolerance

            rows.append(
                {
                    "task": task,
                    "observed_fraction": float(observed_fraction),
                    "comparison": comparison_name,
                    "model_a": model_a,
                    "model_a_display_name": (
                        _display_model_name(model_a)
                    ),
                    "model_b": model_b,
                    "model_b_display_name": (
                        _display_model_name(model_b)
                    ),
                    "number_of_pairs": int(len(paired)),
                    "mean_normalized_mse_model_a": float(
                        paired["mse_a"].mean()
                    ),
                    "mean_normalized_mse_model_b": float(
                        paired["mse_b"].mean()
                    ),
                    "mean_mse_improvement_model_a": float(
                        mse_improvement.mean()
                    ),
                    "std_mse_improvement_model_a": (
                        _sample_standard_deviation(mse_improvement)
                    ),
                    "mean_relative_mse_improvement_percent": float(
                        relative_mse_improvement.mean()
                    ),
                    "std_relative_mse_improvement_percent": (
                        _sample_standard_deviation(
                            relative_mse_improvement
                        )
                    ),
                    "median_relative_mse_improvement_percent": float(
                        relative_mse_improvement.median()
                    ),
                    "mean_normalized_mae_model_a": float(
                        paired["mae_a"].mean()
                    ),
                    "mean_normalized_mae_model_b": float(
                        paired["mae_b"].mean()
                    ),
                    "mean_mae_improvement_model_a": float(
                        mae_improvement.mean()
                    ),
                    "std_mae_improvement_model_a": (
                        _sample_standard_deviation(mae_improvement)
                    ),
                    "mean_relative_mae_improvement_percent": float(
                        relative_mae_improvement.mean()
                    ),
                    "win_rate_mse_model_a": float(wins.mean()),
                    "tie_rate_mse": float(ties.mean()),
                }
            )

    paired_comparisons = pd.DataFrame(rows)

    comparison_order = [
        name for _, _, name in PAIRED_COMPARISONS
    ]

    paired_comparisons["task"] = pd.Categorical(
        paired_comparisons["task"],
        categories=TASK_ORDER,
        ordered=True,
    )
    paired_comparisons["observed_fraction"] = pd.Categorical(
        paired_comparisons["observed_fraction"],
        categories=OBSERVATION_FRACTION_ORDER,
        ordered=True,
    )
    paired_comparisons["comparison"] = pd.Categorical(
        paired_comparisons["comparison"],
        categories=comparison_order,
        ordered=True,
    )

    paired_comparisons = paired_comparisons.sort_values(
        [
            "task",
            "observed_fraction",
            "comparison",
        ],
        kind="stable",
    ).reset_index(drop=True)

    paired_comparisons["task"] = paired_comparisons[
        "task"
    ].astype(str)
    paired_comparisons["observed_fraction"] = paired_comparisons[
        "observed_fraction"
    ].astype(float)
    paired_comparisons["comparison"] = paired_comparisons[
        "comparison"
    ].astype(str)

    return paired_comparisons


def _atomic_write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write a CSV atomically to avoid partially written summaries."""

    path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(file_descriptor)

    temporary_path = Path(temporary_name)

    try:
        frame.to_csv(
            temporary_path,
            index=False,
            float_format="%.10g",
        )
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _print_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    output_directory: Path,
) -> None:
    number_of_files = len(results)
    number_of_seeds = results["seed"].nunique()

    print(
        f"Aggregated {number_of_files} result files "
        f"across {number_of_seeds} unique seeds."
    )
    print(f"Wrote: {output_directory / 'results.csv'}")
    print(f"Wrote: {output_directory / 'summary.csv'}")
    print(
        f"Wrote: {output_directory / 'paired_comparisons.csv'}"
    )

    transport = paired[
        paired["comparison"] == "AT-ODE versus LG-ODE"
    ]

    if not transport.empty:
        print("\nAT-ODE transport improvement over LG-ODE:")
        for row in transport.itertuples(index=False):
            value = row.mean_relative_mse_improvement_percent
            pairs = row.number_of_pairs

            if pd.isna(value):
                text = "not available"
            else:
                text = f"{value:.3f}%"

            print(
                f"  {row.task}, "
                f"{100.0 * row.observed_fraction:.0f}% observed: "
                f"{text} ({pairs} paired seeds)"
            )

    print("\nSummary:")
    printable_columns = [
        "task",
        "observed_fraction",
        "model_display_name",
        "number_of_seeds",
        "mean_normalized_mse",
        "std_normalized_mse",
        "mean_normalized_mae",
    ]
    print(
        summary[printable_columns].to_string(
            index=False,
            na_rep="NA",
        )
    )


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate SimBench LG-ODE result JSON files into results.csv, "
            "summary.csv and paired_comparisons.csv."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing experiment result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for output CSV files. Defaults to --input-dir."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.json",
        help=(
            "Glob pattern relative to --input-dir. "
            "Default: **/*.json"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on every malformed or unrelated JSON file instead of "
            "warning and skipping it."
        ),
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help=(
            "Allow repeated grid/task/fraction/model/seed/mask-seed runs. "
            "They remain in results.csv and are averaged within each seed "
            "for paired comparisons."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the terminal summary.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    input_directory = args.input_dir.expanduser().resolve()
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_directory
    )

    if not input_directory.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {input_directory}"
        )

    results = load_results(
        input_directory=input_directory,
        pattern=args.pattern,
        output_directory=output_directory,
        strict=args.strict,
    )

    _check_duplicate_runs(
        results,
        allow_duplicates=args.allow_duplicates,
    )

    summary = build_summary(results)
    paired_comparisons = build_paired_comparisons(results)

    _atomic_write_csv(
        results,
        output_directory / "results.csv",
    )
    _atomic_write_csv(
        summary,
        output_directory / "summary.csv",
    )
    _atomic_write_csv(
        paired_comparisons,
        output_directory / "paired_comparisons.csv",
    )

    if not args.quiet:
        _print_report(
            results,
            summary,
            paired_comparisons,
            output_directory,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
