#!/usr/bin/env python3
# scripts/plot_powergrid_lgode.py

"""
Generate publication figures for the SimBench LG-ODE experiments.

The script reads saved CSV summaries and saved prediction artifacts. It never
contains manually entered experiment values.

Generated figures
-----------------
Required:
    interpolation_mse.pdf
    extrapolation_mse.pdf
    paired_atode_vs_lgode.pdf
    representative_voltage_trajectory.pdf
    horizon_extrapolation_error.pdf

Optional, when transport weights are present:
    transport_weights.pdf

Expected summary inputs
-----------------------
The summary CSV should contain one row per task/model/observation-fraction
combination. Common column aliases are accepted, including:

    task
    model or model_name
    observed_fraction or observation_fraction
    normalized_test_mse_mean, test_mse_mean, or mse_mean
    normalized_test_mse_std, test_mse_std, or mse_std

The paired-comparison CSV should contain AT-ODE versus LG-ODE comparisons.
If it is unavailable, paired improvements are calculated from results.csv
using rows paired by task, observation fraction, and seed.

Prediction artifact format
--------------------------
NPZ is the recommended format. Each prediction artifact should contain:

    target_values:
        [B, N, T, F], [N, T, F], or [T, F]

    predictions or mean_prediction:
        [S, B, N, T, F], [B, N, T, F], [N, T, F], or [T, F]

    target_times:
        [T], [B, T], [B, N, T], or another shape ending in T

Optional fields:

    target_mask:
        Same shape as target_values, or broadcast-compatible.

    observed_event_mask:
        [B, N, T], [N, T], or [T]. Used only to mark observations.

    feature_names:
        Feature labels. The script selects a voltage feature automatically
        when possible.

    model or model_name
    task
    observed_fraction or observation_fraction
    seed
    trajectory_id
    value_space

    transport_weights:
        A transport-weight trajectory. Several common layouts are accepted,
        including [T, E], [B, T, E], and [T, B, E, 1].

    transport_times:
        Optional transport time coordinates.

Prediction files may also be referenced by a ``prediction_path`` column in
results.csv. Relative paths are resolved relative to results.csv and then
relative to --input-dir.

Examples
--------
    python scripts/plot_powergrid_lgode.py \\
        --input-dir results/powergrid_lgode/summary

    python scripts/plot_powergrid_lgode.py \\
        --input-dir results/powergrid_lgode \\
        --summary-csv summary/summary.csv \\
        --paired-comparisons-csv summary/paired_comparisons.csv \\
        --results-csv summary/results.csv \\
        --output-dir figures

    python scripts/plot_powergrid_lgode.py \\
        --input-dir results/powergrid_lgode \\
        --prediction-file runs/atode_extrapolation_0.6_seed0_predictions.npz \\
        --trajectory-model atode \\
        --trajectory-task extrapolation \\
        --trajectory-fraction 0.6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "persistence",
    "latentode",
    "lgode",
    "atode",
)

MODEL_LABELS = {
    "persistence": "Persistence",
    "latentode": "Latent ODE",
    "lgode": "LG-ODE",
    "atode": "AT-ODE",
}

MODEL_COLORS = {
    "persistence": "#7f7f7f",
    "latentode": "#1f77b4",
    "lgode": "#d62728",
    "atode": "#2ca02c",
}

MODEL_MARKERS = {
    "persistence": "o",
    "latentode": "s",
    "lgode": "^",
    "atode": "D",
}

MODEL_LINESTYLES = {
    "persistence": "--",
    "latentode": "-.",
    "lgode": "-",
    "atode": "-",
}

TASK_ORDER = (
    "interpolation",
    "extrapolation",
)

OBSERVATION_FRACTIONS = (
    0.4,
    0.6,
    0.8,
)

_EPS = 1e-12


@dataclass
class PredictionArtifact:
    path: Path
    model: Optional[str]
    task: Optional[str]
    observed_fraction: Optional[float]
    seed: Optional[int]
    target_values: np.ndarray
    mean_prediction: np.ndarray
    target_times: np.ndarray
    target_mask: np.ndarray
    observed_event_mask: Optional[np.ndarray]
    feature_names: List[str]
    trajectory_id: Optional[np.ndarray]
    value_space: str
    transport_weights: Optional[np.ndarray]
    transport_times: Optional[np.ndarray]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _normalize_model(value: Any) -> Optional[str]:
    if value is None:
        return None

    normalized = str(value).strip().lower()
    normalized = {
        "copy-persistence": "persistence",
        "copy_persistence": "persistence",
        "latent-ode": "latentode",
        "latent_ode": "latentode",
        "lg-ode": "lgode",
        "lg_ode": "lgode",
        "at-ode": "atode",
        "at_ode": "atode",
    }.get(normalized, normalized)

    return normalized if normalized in MODEL_ORDER else None


def _normalize_task(value: Any) -> Optional[str]:
    if value is None:
        return None

    normalized = str(value).strip().lower()
    normalized = {
        "interp": "interpolation",
        "interpolate": "interpolation",
        "extrap": "extrapolation",
        "forecast": "extrapolation",
        "forecasting": "extrapolation",
    }.get(normalized, normalized)

    return normalized if normalized in TASK_ORDER else None


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return value.reshape(-1)[0].item()

    if isinstance(value, np.generic):
        return value.item()

    return value


def _float_or_none(value: Any) -> Optional[float]:
    value = _scalar(value)

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        result = float(text)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> Optional[int]:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(number)


def _first_present(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    default: Any = None,
) -> Any:
    for name in names:
        if name in mapping:
            value = mapping[name]
            if value is not None:
                return value
    return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_input_path(
    input_dir: Path,
    explicit: Optional[str],
    default_names: Sequence[str],
    *,
    required: bool,
) -> Optional[Path]:
    candidates: List[Path] = []

    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.is_absolute():
            candidates.append(explicit_path)
        else:
            candidates.append(input_dir / explicit_path)
            candidates.append(explicit_path)

    for name in default_names:
        candidates.append(input_dir / name)
        candidates.append(input_dir / "summary" / name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    if required:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not locate required input. Checked: {checked}"
        )

    return None


def _atomic_save_figure(
    figure: plt.Figure,
    output_path: Path,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix or ".pdf"
    file_format = suffix.lstrip(".").lower()

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=suffix,
        dir=str(output_path.parent),
    )
    os.close(fd)

    temporary_path = Path(temporary_name)

    try:
        figure.savefig(
            temporary_path,
            format=file_format,
            dpi=dpi,
            bbox_inches="tight",
        )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

        plt.close(figure)


def _metric_column(
    rows: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    *,
    required: bool,
) -> Optional[str]:
    if not rows:
        if required:
            raise ValueError("The CSV contains no rows.")
        return None

    columns = set(rows[0].keys())

    for alias in aliases:
        if alias in columns:
            return alias

    if required:
        raise ValueError(
            "Missing required metric column. Expected one of: "
            + ", ".join(aliases)
        )

    return None


def _canonical_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    model_column = _metric_column(
        rows,
        ("model", "model_name"),
        required=True,
    )
    task_column = _metric_column(
        rows,
        ("task",),
        required=True,
    )
    fraction_column = _metric_column(
        rows,
        (
            "observed_fraction",
            "observation_fraction",
            "obs_fraction",
            "fraction",
        ),
        required=True,
    )
    mean_column = _metric_column(
        rows,
        (
            "normalized_test_mse_mean",
            "test_mse_mean",
            "mse_mean",
            "mean_normalized_test_mse",
            "mean_test_mse",
            "normalized_test_mse",
            "test_mse",
            "mse",
        ),
        required=True,
    )
    std_column = _metric_column(
        rows,
        (
            "normalized_test_mse_std",
            "test_mse_std",
            "mse_std",
            "std_normalized_test_mse",
            "std_test_mse",
        ),
        required=False,
    )

    canonical = []

    for row in rows:
        model = _normalize_model(row.get(model_column))
        task = _normalize_task(row.get(task_column))
        fraction = _float_or_none(row.get(fraction_column))
        mean = _float_or_none(row.get(mean_column))
        std = _float_or_none(row.get(std_column)) if std_column else 0.0

        if model is None or task is None or fraction is None or mean is None:
            continue

        if mean < 0:
            raise ValueError(
                f"Negative MSE found for {task}/{model}/{fraction}: {mean}"
            )

        if std is None:
            std = 0.0

        if std < 0:
            raise ValueError(
                f"Negative MSE standard deviation found: {std}"
            )

        canonical.append(
            {
                "model": model,
                "task": task,
                "observed_fraction": fraction,
                "mse_mean": mean,
                "mse_std": std,
            }
        )

    if not canonical:
        raise ValueError(
            "No usable rows were found in the summary CSV."
        )

    return canonical


def create_mse_figure(
    summary_rows: Sequence[Mapping[str, Any]],
    task: str,
    output_path: Path,
    *,
    dpi: int,
    allow_incomplete: bool,
) -> None:
    task = _normalize_task(task)
    if task is None:
        raise ValueError(f"Unsupported task: {task!r}")

    rows = [
        row
        for row in summary_rows
        if row["task"] == task
    ]

    if not rows:
        raise ValueError(
            f"No summary rows are available for task {task!r}."
        )

    figure, axis = plt.subplots(figsize=(4.9, 3.25))

    plotted = 0

    for model in MODEL_ORDER:
        model_rows = [
            row
            for row in rows
            if row["model"] == model
        ]
        model_rows.sort(key=lambda row: row["observed_fraction"])

        if not model_rows:
            if allow_incomplete:
                continue
            raise ValueError(
                f"Summary CSV has no {task} rows for {model}."
            )

        x = np.asarray(
            [row["observed_fraction"] for row in model_rows],
            dtype=np.float64,
        )
        y = np.asarray(
            [row["mse_mean"] for row in model_rows],
            dtype=np.float64,
        )
        yerr = np.asarray(
            [row["mse_std"] for row in model_rows],
            dtype=np.float64,
        )

        if not allow_incomplete:
            available = {
                round(float(value), 8)
                for value in x
            }
            expected = {
                round(float(value), 8)
                for value in OBSERVATION_FRACTIONS
            }
            missing = sorted(expected - available)

            if missing:
                raise ValueError(
                    f"Missing {task}/{model} observation fractions: "
                    f"{missing}"
                )

        axis.errorbar(
            x,
            y,
            yerr=yerr,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linestyle=MODEL_LINESTYLES[model],
            capsize=2.5,
            capthick=0.8,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError(f"No model data were plotted for {task}.")

    axis.set_xlabel("Observed fraction")
    axis.set_ylabel("Normalized test MSE")
    axis.set_title(f"{task.capitalize()} performance")
    axis.set_xticks(OBSERVATION_FRACTIONS)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)

    figure.tight_layout()
    _atomic_save_figure(figure, output_path, dpi)


def _paired_rows_from_csv(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    canonical = []

    for row in rows:
        task = _normalize_task(
            _first_present(row, ("task",))
        )
        fraction = _float_or_none(
            _first_present(
                row,
                (
                    "observed_fraction",
                    "observation_fraction",
                    "obs_fraction",
                    "fraction",
                ),
            )
        )

        model_a = _normalize_model(
            _first_present(
                row,
                ("model_a", "model", "challenger_model"),
            )
        )
        model_b = _normalize_model(
            _first_present(
                row,
                ("model_b", "baseline_model", "reference_model"),
            )
        )

        improvement = _float_or_none(
            _first_present(
                row,
                (
                    "relative_improvement_percent_mean",
                    "mean_relative_improvement_percent",
                    "relative_improvement_percent",
                    "improvement_percent_mean",
                    "improvement_percent",
                    "transport_gain_percent",
                ),
            )
        )
        improvement_std = _float_or_none(
            _first_present(
                row,
                (
                    "relative_improvement_percent_std",
                    "std_relative_improvement_percent",
                    "improvement_percent_std",
                    "transport_gain_percent_std",
                ),
                0.0,
            )
        )

        if task is None or fraction is None or improvement is None:
            continue

        if model_a == "atode" and model_b == "lgode":
            signed_improvement = improvement
        elif model_a == "lgode" and model_b == "atode":
            signed_improvement = -improvement
        else:
            comparison = str(
                _first_present(
                    row,
                    ("comparison", "pair", "label"),
                    "",
                )
            ).lower()

            if "at" in comparison and "lg" in comparison:
                signed_improvement = improvement
            else:
                continue

        canonical.append(
            {
                "task": task,
                "observed_fraction": fraction,
                "improvement_mean": signed_improvement,
                "improvement_std": improvement_std or 0.0,
            }
        )

    return canonical


def _paired_rows_from_results(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float, int], Dict[str, float]] = {}

    for row in rows:
        model = _normalize_model(
            _first_present(row, ("model", "model_name"))
        )
        task = _normalize_task(
            _first_present(row, ("task",))
        )
        fraction = _float_or_none(
            _first_present(
                row,
                (
                    "observed_fraction",
                    "observation_fraction",
                    "obs_fraction",
                    "fraction",
                ),
            )
        )
        seed = _int_or_none(
            _first_present(row, ("seed", "random_seed"))
        )
        mse = _float_or_none(
            _first_present(
                row,
                (
                    "normalized_test_mse",
                    "test_mse",
                    "mse",
                ),
            )
        )

        if (
            model not in {"lgode", "atode"}
            or task is None
            or fraction is None
            or seed is None
            or mse is None
        ):
            continue

        grouped.setdefault(
            (task, fraction, seed),
            {},
        )[model] = mse

    improvements: Dict[Tuple[str, float], List[float]] = {}

    for (task, fraction, _seed), values in grouped.items():
        if "lgode" not in values or "atode" not in values:
            continue

        denominator = values["lgode"]
        if abs(denominator) <= _EPS:
            continue

        gain = 100.0 * (
            denominator - values["atode"]
        ) / denominator

        improvements.setdefault(
            (task, fraction),
            [],
        ).append(gain)

    result = []

    for (task, fraction), values in sorted(improvements.items()):
        array = np.asarray(values, dtype=np.float64)
        result.append(
            {
                "task": task,
                "observed_fraction": fraction,
                "improvement_mean": float(np.mean(array)),
                "improvement_std": (
                    float(np.std(array, ddof=1))
                    if array.size > 1
                    else 0.0
                ),
            }
        )

    return result


def create_paired_improvement_figure(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    dpi: int,
    allow_incomplete: bool,
) -> None:
    if not rows:
        raise ValueError(
            "No AT-ODE versus LG-ODE paired comparisons are available."
        )

    figure, axis = plt.subplots(figsize=(4.9, 3.25))

    task_styles = {
        "interpolation": {
            "color": "#9467bd",
            "marker": "o",
            "linestyle": "-",
        },
        "extrapolation": {
            "color": "#ff7f0e",
            "marker": "s",
            "linestyle": "--",
        },
    }

    plotted = 0

    for task in TASK_ORDER:
        task_rows = [
            row
            for row in rows
            if row["task"] == task
        ]
        task_rows.sort(key=lambda row: row["observed_fraction"])

        if not task_rows:
            if allow_incomplete:
                continue
            raise ValueError(
                f"No paired AT-ODE/LG-ODE rows found for {task}."
            )

        x = np.asarray(
            [row["observed_fraction"] for row in task_rows],
            dtype=np.float64,
        )
        y = np.asarray(
            [row["improvement_mean"] for row in task_rows],
            dtype=np.float64,
        )
        yerr = np.asarray(
            [row["improvement_std"] for row in task_rows],
            dtype=np.float64,
        )

        style = task_styles[task]

        axis.errorbar(
            x,
            y,
            yerr=yerr,
            label=task.capitalize(),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=2.5,
            capthick=0.8,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError(
            "No paired AT-ODE/LG-ODE rows could be plotted."
        )

    axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.65,
    )
    axis.set_xlabel("Observed fraction")
    axis.set_ylabel("AT-ODE improvement over LG-ODE (%)")
    axis.set_title("Paired transport improvement")
    axis.set_xticks(OBSERVATION_FRACTIONS)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False)

    figure.tight_layout()
    _atomic_save_figure(figure, output_path, dpi)


def _decode_feature_names(value: Any, feature_count: int) -> List[str]:
    if value is None:
        return [f"feature_{index}" for index in range(feature_count)]

    array = np.asarray(value).reshape(-1)
    names = []

    for item in array:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(_scalar(item)))

    if len(names) != feature_count:
        return [f"feature_{index}" for index in range(feature_count)]

    return names


def _metadata_from_filename(path: Path) -> Dict[str, Any]:
    name = path.stem.lower()

    model = None
    for candidate in MODEL_ORDER:
        variants = {
            candidate,
            candidate.replace("ode", "-ode"),
            candidate.replace("ode", "_ode"),
        }
        if any(variant in name for variant in variants):
            model = candidate
            break

    task = None
    if "interpolation" in name or "interp" in name:
        task = "interpolation"
    elif (
        "extrapolation" in name
        or "extrap" in name
        or "forecast" in name
    ):
        task = "extrapolation"

    fraction = None
    fraction_match = re.search(
        r"(?:fraction|frac|obs)[_-]?([01](?:\.\d+)?)",
        name,
    )
    if fraction_match:
        fraction = _float_or_none(fraction_match.group(1))

    seed = None
    seed_match = re.search(r"seed[_-]?(\d+)", name)
    if seed_match:
        seed = int(seed_match.group(1))

    return {
        "model": model,
        "task": task,
        "observed_fraction": fraction,
        "seed": seed,
    }


def _load_npz_mapping(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key])
            for key in archive.files
        }


def _load_json_mapping(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"Prediction JSON must contain an object: {path}"
        )

    return value


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    # Optional tensor support without importing torch into plotting runs that
    # only use NPZ/JSON artifacts.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def _load_prediction_mapping(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".npz":
        return _load_npz_mapping(path)

    if suffix == ".json":
        return _load_json_mapping(path)

    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                f"PyTorch is required to read {path}"
            ) from exc

        try:
            value = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            value = torch.load(
                path,
                map_location="cpu",
            )

        if not isinstance(value, dict):
            raise ValueError(
                f"Torch prediction file must contain a dictionary: {path}"
            )

        return {
            str(key): (
                item.detach().cpu().numpy()
                if hasattr(item, "detach")
                else item
            )
            for key, item in value.items()
        }

    raise ValueError(
        f"Unsupported prediction format {suffix!r}: {path}"
    )


def _canonicalize_values(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target)
    prediction = np.asarray(prediction)

    # Prediction may include a leading Monte Carlo sample dimension.
    if prediction.ndim == target.ndim + 1:
        prediction = np.mean(prediction, axis=0)

    if target.ndim == 2:
        # [T, F] -> [1, 1, T, F]
        target = target[None, None, :, :]
    elif target.ndim == 3:
        # [N, T, F] -> [1, N, T, F]
        target = target[None, :, :, :]
    elif target.ndim != 4:
        raise ValueError(
            "target_values must have shape [B,N,T,F], [N,T,F], "
            f"or [T,F]; got {tuple(target.shape)}"
        )

    if prediction.ndim == 2:
        prediction = prediction[None, None, :, :]
    elif prediction.ndim == 3:
        prediction = prediction[None, :, :, :]
    elif prediction.ndim != 4:
        raise ValueError(
            "Mean predictions must have shape [B,N,T,F], [N,T,F], "
            f"or [T,F]; got {tuple(prediction.shape)}"
        )

    if target.shape != prediction.shape:
        raise ValueError(
            "target_values and mean prediction shapes differ after "
            f"canonicalization: {target.shape} versus {prediction.shape}"
        )

    if not np.all(np.isfinite(target)):
        raise ValueError("target_values contains non-finite values.")

    if not np.all(np.isfinite(prediction)):
        raise ValueError("predictions contains non-finite values.")

    return (
        target.astype(np.float64, copy=False),
        prediction.astype(np.float64, copy=False),
    )


def _canonicalize_target_mask(
    value: Any,
    target_shape: Tuple[int, ...],
) -> np.ndarray:
    if value is None:
        return np.ones(target_shape, dtype=np.float64)

    mask = _to_numpy(value)

    if mask.ndim == 2 and len(target_shape) == 4:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3 and len(target_shape) == 4:
        mask = mask[None, :, :, :]

    try:
        mask = np.broadcast_to(mask, target_shape)
    except ValueError as exc:
        raise ValueError(
            f"target_mask shape {mask.shape} is not broadcast-compatible "
            f"with target shape {target_shape}."
        ) from exc

    mask = mask.astype(np.float64, copy=False)

    if not np.all(np.isfinite(mask)):
        raise ValueError("target_mask contains non-finite values.")

    if np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("target_mask values must lie within [0, 1].")

    if not np.any(mask > 0.0):
        raise ValueError("target_mask contains no evaluated values.")

    return mask


def _canonicalize_times(
    value: Any,
    target_shape: Tuple[int, int, int, int],
) -> np.ndarray:
    batch_size, num_nodes, num_times, _ = target_shape

    if value is None:
        return np.arange(num_times, dtype=np.float64)

    times = np.asarray(value, dtype=np.float64)

    if not np.all(np.isfinite(times)):
        raise ValueError("target_times contains non-finite values.")

    if times.ndim == 1:
        selected = times
    elif times.shape[-1] == num_times:
        selected = times.reshape(-1, num_times)[0]
    elif times.size == num_times:
        selected = times.reshape(num_times)
    else:
        raise ValueError(
            f"Could not interpret target_times shape {times.shape}; "
            f"expected a dimension of length {num_times}."
        )

    if selected.size != num_times:
        raise ValueError(
            f"target_times length is {selected.size}, expected {num_times}."
        )

    if np.any(np.diff(selected) < 0):
        raise ValueError("target_times must be non-decreasing.")

    return selected


def load_prediction_artifact(path: Path) -> PredictionArtifact:
    path = path.resolve()
    mapping = _load_prediction_mapping(path)
    filename_metadata = _metadata_from_filename(path)

    target_value = _first_present(
        mapping,
        (
            "target_values",
            "targets",
            "truth",
            "target",
            "y_true",
        ),
    )
    prediction_value = _first_present(
        mapping,
        (
            "mean_prediction",
            "mean_predictions",
            "predictions",
            "prediction",
            "pred_y",
            "y_pred",
        ),
    )

    if target_value is None:
        raise ValueError(
            f"Prediction artifact lacks target_values: {path}"
        )

    if prediction_value is None:
        raise ValueError(
            f"Prediction artifact lacks predictions: {path}"
        )

    target_values, mean_prediction = _canonicalize_values(
        _to_numpy(target_value),
        _to_numpy(prediction_value),
    )

    target_mask = _canonicalize_target_mask(
        _first_present(
            mapping,
            ("target_mask", "mask", "evaluation_mask"),
        ),
        target_values.shape,
    )

    target_times = _canonicalize_times(
        _first_present(
            mapping,
            ("target_times", "times", "time"),
        ),
        target_values.shape,
    )

    observed_mask_value = _first_present(
        mapping,
        (
            "observed_event_mask",
            "observation_mask",
            "observed_mask",
        ),
    )
    observed_event_mask = (
        None
        if observed_mask_value is None
        else _to_numpy(observed_mask_value)
    )

    feature_names = _decode_feature_names(
        _first_present(
            mapping,
            ("feature_names", "features", "channel_names"),
        ),
        target_values.shape[-1],
    )

    model = _normalize_model(
        _scalar(
            _first_present(
                mapping,
                ("model", "model_name"),
                filename_metadata["model"],
            )
        )
    )
    task = _normalize_task(
        _scalar(
            _first_present(
                mapping,
                ("task",),
                filename_metadata["task"],
            )
        )
    )
    observed_fraction = _float_or_none(
        _first_present(
            mapping,
            (
                "observed_fraction",
                "observation_fraction",
                "obs_fraction",
            ),
            filename_metadata["observed_fraction"],
        )
    )
    seed = _int_or_none(
        _first_present(
            mapping,
            ("seed", "random_seed"),
            filename_metadata["seed"],
        )
    )

    value_space = str(
        _scalar(
            _first_present(
                mapping,
                ("value_space", "prediction_space"),
                "normalized",
            )
        )
    ).strip()

    transport_weights_value = _first_present(
        mapping,
        (
            "transport_weights",
            "edge_weight_trajectory",
            "edge_weights",
        ),
    )
    transport_times_value = _first_present(
        mapping,
        (
            "transport_times",
            "edge_weight_times",
        ),
    )

    return PredictionArtifact(
        path=path,
        model=model,
        task=task,
        observed_fraction=observed_fraction,
        seed=seed,
        target_values=target_values,
        mean_prediction=mean_prediction,
        target_times=target_times,
        target_mask=target_mask,
        observed_event_mask=observed_event_mask,
        feature_names=feature_names,
        trajectory_id=(
            None
            if "trajectory_id" not in mapping
            else _to_numpy(mapping["trajectory_id"])
        ),
        value_space=value_space,
        transport_weights=(
            None
            if transport_weights_value is None
            else _to_numpy(transport_weights_value)
        ),
        transport_times=(
            None
            if transport_times_value is None
            else _to_numpy(transport_times_value)
        ),
    )


def _prediction_paths_from_results(
    results_csv: Optional[Path],
    input_dir: Path,
) -> List[Path]:
    if results_csv is None:
        return []

    rows = _read_csv(results_csv)
    paths = []

    for row in rows:
        raw_path = _first_present(
            row,
            (
                "prediction_path",
                "predictions_path",
                "prediction_file",
                "predictions_file",
                "artifact_path",
            ),
        )

        if not raw_path:
            continue

        candidate = Path(str(raw_path))

        candidates = (
            candidate,
            results_csv.parent / candidate,
            input_dir / candidate,
        )

        for resolved in candidates:
            if resolved.is_file():
                paths.append(resolved.resolve())
                break

    return paths


def discover_prediction_paths(
    input_dir: Path,
    results_csv: Optional[Path],
    explicit_paths: Sequence[str],
) -> List[Path]:
    paths: List[Path] = []

    for value in explicit_paths:
        candidate = Path(value)

        if not candidate.is_absolute():
            input_candidate = input_dir / candidate
            if input_candidate.is_file():
                candidate = input_candidate

        if not candidate.is_file():
            raise FileNotFoundError(
                f"Prediction file does not exist: {candidate}"
            )

        paths.append(candidate.resolve())

    paths.extend(
        _prediction_paths_from_results(
            results_csv,
            input_dir,
        )
    )

    patterns = (
        "**/*prediction*.npz",
        "**/*predictions*.npz",
        "**/*prediction*.json",
        "**/*predictions*.json",
        "**/*prediction*.pt",
        "**/*predictions*.pt",
        "**/*prediction*.pth",
        "**/*predictions*.pth",
    )

    for pattern in patterns:
        paths.extend(path.resolve() for path in input_dir.glob(pattern))

    unique = []
    seen = set()

    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


def load_prediction_artifacts(
    paths: Sequence[Path],
    *,
    quiet: bool,
) -> List[PredictionArtifact]:
    artifacts = []

    for path in paths:
        try:
            artifact = load_prediction_artifact(path)
        except Exception as exc:
            if not quiet:
                print(
                    f"Warning: skipping prediction artifact {path}: {exc}"
                )
            continue

        artifacts.append(artifact)

    return artifacts


def _artifact_score(
    artifact: PredictionArtifact,
    *,
    model: Optional[str],
    task: Optional[str],
    fraction: Optional[float],
    seed: Optional[int],
) -> Tuple[int, str]:
    score = 0

    if model is not None:
        score += 8 if artifact.model == model else -8

    if task is not None:
        score += 4 if artifact.task == task else -4

    if fraction is not None:
        if artifact.observed_fraction is not None and math.isclose(
            artifact.observed_fraction,
            fraction,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            score += 2
        else:
            score -= 2

    if seed is not None:
        score += 1 if artifact.seed == seed else -1

    return score, str(artifact.path)


def select_prediction_artifact(
    artifacts: Sequence[PredictionArtifact],
    *,
    model: Optional[str],
    task: Optional[str],
    fraction: Optional[float],
    seed: Optional[int],
) -> PredictionArtifact:
    if not artifacts:
        raise ValueError("No usable prediction artifacts were found.")

    ranked = sorted(
        artifacts,
        key=lambda artifact: _artifact_score(
            artifact,
            model=model,
            task=task,
            fraction=fraction,
            seed=seed,
        ),
        reverse=True,
    )

    selected = ranked[0]
    score, _ = _artifact_score(
        selected,
        model=model,
        task=task,
        fraction=fraction,
        seed=seed,
    )

    if score < 0:
        raise ValueError(
            "No prediction artifact matches the requested model/task/"
            "fraction/seed selection."
        )

    return selected


def _choose_feature_index(
    artifact: PredictionArtifact,
    requested: Optional[int],
) -> int:
    feature_count = artifact.target_values.shape[-1]

    if requested is not None:
        if requested < 0 or requested >= feature_count:
            raise ValueError(
                f"feature-index {requested} is outside [0, "
                f"{feature_count - 1}]."
            )
        return requested

    voltage_tokens = (
        "voltage",
        "vm",
        "v_mag",
        "voltage_magnitude",
        "pu",
    )

    for index, name in enumerate(artifact.feature_names):
        normalized = name.lower().replace(" ", "_")
        if any(token in normalized for token in voltage_tokens):
            return index

    return 0


def _select_observed_mask(
    artifact: PredictionArtifact,
    batch_index: int,
    bus_index: int,
    feature_index: int,
) -> Optional[np.ndarray]:
    if artifact.observed_event_mask is None:
        return None

    mask = np.asarray(artifact.observed_event_mask)
    num_times = artifact.target_values.shape[2]

    if mask.ndim == 4:
        selected = mask[
            min(batch_index, mask.shape[0] - 1),
            min(bus_index, mask.shape[1] - 1),
            :,
            min(feature_index, mask.shape[3] - 1),
        ]
    elif mask.ndim == 3:
        selected = mask[
            min(batch_index, mask.shape[0] - 1),
            min(bus_index, mask.shape[1] - 1),
            :,
        ]
    elif mask.ndim == 2:
        selected = mask[
            min(bus_index, mask.shape[0] - 1),
            :,
        ]
    elif mask.ndim == 1:
        selected = mask
    else:
        return None

    selected = np.asarray(selected).reshape(-1)

    if selected.size != num_times:
        return None

    return selected > 0


def create_representative_trajectory_figure(
    artifact: PredictionArtifact,
    output_path: Path,
    *,
    batch_index: int,
    bus_index: int,
    feature_index: Optional[int],
    dpi: int,
) -> None:
    batch_size, num_nodes, _, _ = artifact.target_values.shape

    if batch_index < 0 or batch_index >= batch_size:
        raise ValueError(
            f"batch-index {batch_index} is outside [0, {batch_size - 1}]."
        )

    if bus_index < 0 or bus_index >= num_nodes:
        raise ValueError(
            f"bus-index {bus_index} is outside [0, {num_nodes - 1}]."
        )

    selected_feature = _choose_feature_index(
        artifact,
        feature_index,
    )

    times = artifact.target_times
    truth = artifact.target_values[
        batch_index,
        bus_index,
        :,
        selected_feature,
    ]
    prediction = artifact.mean_prediction[
        batch_index,
        bus_index,
        :,
        selected_feature,
    ]
    evaluation_mask = artifact.target_mask[
        batch_index,
        bus_index,
        :,
        selected_feature,
    ] > 0

    figure, axis = plt.subplots(figsize=(6.4, 3.25))

    axis.plot(
        times,
        truth,
        color="black",
        linewidth=1.7,
        label="Ground truth",
    )
    axis.plot(
        times,
        prediction,
        color=MODEL_COLORS.get(artifact.model or "", "#2ca02c"),
        linewidth=1.8,
        label=MODEL_LABELS.get(
            artifact.model or "",
            artifact.model or "Prediction",
        ),
    )

    observed_mask = _select_observed_mask(
        artifact,
        batch_index,
        bus_index,
        selected_feature,
    )

    if observed_mask is not None and np.any(observed_mask):
        axis.scatter(
            times[observed_mask],
            truth[observed_mask],
            s=16,
            color="#1f77b4",
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
            label="Observed event",
        )

    if np.any(evaluation_mask):
        first_eval = int(np.flatnonzero(evaluation_mask)[0])
        if first_eval > 0:
            boundary = 0.5 * (
                times[first_eval - 1] + times[first_eval]
            )
            axis.axvline(
                boundary,
                color="#666666",
                linestyle=":",
                linewidth=1.0,
                label="Evaluation boundary",
            )

    feature_label = artifact.feature_names[selected_feature]
    model_label = MODEL_LABELS.get(
        artifact.model or "",
        artifact.model or "Model",
    )

    axis.set_xlabel("Time")
    axis.set_ylabel(
        f"{feature_label} ({artifact.value_space} space)"
    )
    axis.set_title(
        f"Representative voltage trajectory: {model_label}, "
        f"bus {bus_index}"
    )
    axis.grid(True, alpha=0.22, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)

    figure.tight_layout()
    _atomic_save_figure(figure, output_path, dpi)


def _horizon_mse(
    artifact: PredictionArtifact,
) -> np.ndarray:
    squared_error = (
        artifact.mean_prediction - artifact.target_values
    ) ** 2

    weighted_error = squared_error * artifact.target_mask

    # Reduce batch, node, and feature dimensions; preserve time.
    numerator = np.sum(weighted_error, axis=(0, 1, 3))
    denominator = np.sum(
        artifact.target_mask,
        axis=(0, 1, 3),
    )

    result = np.full(
        squared_error.shape[2],
        np.nan,
        dtype=np.float64,
    )

    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]

    return result


def _select_horizon_artifacts(
    artifacts: Sequence[PredictionArtifact],
    *,
    fraction: Optional[float],
    seed: Optional[int],
) -> List[PredictionArtifact]:
    selected = []

    for model in MODEL_ORDER:
        candidates = [
            artifact
            for artifact in artifacts
            if artifact.model == model
            and artifact.task == "extrapolation"
        ]

        if fraction is not None:
            exact = [
                artifact
                for artifact in candidates
                if artifact.observed_fraction is not None
                and math.isclose(
                    artifact.observed_fraction,
                    fraction,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ]
            if exact:
                candidates = exact

        if seed is not None:
            exact = [
                artifact
                for artifact in candidates
                if artifact.seed == seed
            ]
            if exact:
                candidates = exact

        if candidates:
            candidates.sort(key=lambda item: str(item.path))
            selected.append(candidates[0])

    return selected


def create_horizon_error_figure(
    artifacts: Sequence[PredictionArtifact],
    output_path: Path,
    *,
    fraction: Optional[float],
    seed: Optional[int],
    dpi: int,
    allow_incomplete: bool,
) -> None:
    selected = _select_horizon_artifacts(
        artifacts,
        fraction=fraction,
        seed=seed,
    )

    if not selected:
        raise ValueError(
            "No extrapolation prediction artifacts are available for the "
            "horizon-wise error figure."
        )

    present_models = {
        artifact.model
        for artifact in selected
    }

    if (
        not allow_incomplete
        and not {"lgode", "atode"}.issubset(present_models)
    ):
        raise ValueError(
            "Horizon-wise comparison requires both LG-ODE and AT-ODE "
            "prediction artifacts."
        )

    figure, axis = plt.subplots(figsize=(5.4, 3.25))

    for artifact in selected:
        error = _horizon_mse(artifact)
        times = artifact.target_times

        valid = np.isfinite(error)

        if not np.any(valid):
            continue

        model = artifact.model or "unknown"

        axis.plot(
            times[valid],
            error[valid],
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, None),
            marker=MODEL_MARKERS.get(model, "o"),
            linestyle=MODEL_LINESTYLES.get(model, "-"),
            markevery=max(1, int(np.sum(valid) // 8)),
        )

    axis.set_xlabel("Forecast horizon / target time")
    axis.set_ylabel("Masked MSE")
    axis.set_title("Horizon-wise extrapolation error")
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False)

    figure.tight_layout()
    _atomic_save_figure(figure, output_path, dpi)


def _canonicalize_transport_weights(
    artifact: PredictionArtifact,
) -> Tuple[np.ndarray, np.ndarray]:
    if artifact.transport_weights is None:
        raise ValueError(
            f"No transport weights are available in {artifact.path}."
        )

    weights = np.asarray(
        artifact.transport_weights,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(weights)):
        raise ValueError("Transport weights contain non-finite values.")

    if np.any(weights < 0):
        raise ValueError("Transport weights must be non-negative.")

    weights = np.squeeze(weights)

    if weights.ndim == 1:
        weights = weights[:, None]
    elif weights.ndim == 2:
        pass
    elif weights.ndim == 3:
        # Prefer a dimension matching the target/transport time count.
        expected_times = artifact.target_times.size
        possible_time_axes = [
            axis
            for axis, size in enumerate(weights.shape)
            if size == expected_times
        ]

        time_axis = possible_time_axes[0] if possible_time_axes else 0
        weights = np.moveaxis(weights, time_axis, 0)

        # Average any batch/sample dimensions, preserving the final edge axis.
        if weights.ndim == 3:
            weights = np.mean(weights, axis=1)
    else:
        # Move a likely time axis first and flatten all remaining dimensions
        # into edge-like channels.
        expected_times = artifact.target_times.size
        possible_time_axes = [
            axis
            for axis, size in enumerate(weights.shape)
            if size == expected_times
        ]
        time_axis = possible_time_axes[0] if possible_time_axes else 0
        weights = np.moveaxis(weights, time_axis, 0)
        weights = weights.reshape(weights.shape[0], -1)

    if weights.ndim != 2:
        raise ValueError(
            "Could not canonicalize transport weights to [T, E]; got "
            f"{weights.shape}."
        )

    if artifact.transport_times is not None:
        times = np.asarray(
            artifact.transport_times,
            dtype=np.float64,
        ).reshape(-1)
    elif weights.shape[0] == artifact.target_times.size:
        times = artifact.target_times
    else:
        times = np.arange(weights.shape[0], dtype=np.float64)

    if times.size != weights.shape[0]:
        raise ValueError(
            "transport_times length does not match the transport-weight "
            f"trajectory: {times.size} versus {weights.shape[0]}."
        )

    return weights, times


def create_transport_weight_figure(
    artifact: PredictionArtifact,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    weights, times = _canonicalize_transport_weights(artifact)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.2, 4.4),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex=True,
    )

    heatmap_axis, summary_axis = axes

    image = heatmap_axis.imshow(
        weights.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(
            float(times[0]),
            float(times[-1]) if times.size > 1 else float(times[0] + 1),
            -0.5,
            weights.shape[1] - 0.5,
        ),
        cmap="viridis",
    )

    heatmap_axis.set_ylabel("Physical edge")
    heatmap_axis.set_title("AT-ODE transport weights")

    colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        pad=0.015,
    )
    colorbar.set_label("Normalized edge weight")

    summary_axis.plot(
        times,
        np.mean(weights, axis=1),
        color=MODEL_COLORS["atode"],
        label="Mean",
    )

    if weights.shape[1] > 1:
        summary_axis.fill_between(
            times,
            np.percentile(weights, 10, axis=1),
            np.percentile(weights, 90, axis=1),
            color=MODEL_COLORS["atode"],
            alpha=0.2,
            linewidth=0,
            label="10–90 percentile",
        )

    summary_axis.set_xlabel("Time")
    summary_axis.set_ylabel("Weight")
    summary_axis.grid(True, alpha=0.25, linewidth=0.6)
    summary_axis.legend(frameon=False, ncol=2)

    figure.tight_layout()
    _atomic_save_figure(figure, output_path, dpi)


def generate_figures(args: argparse.Namespace) -> List[Path]:
    _configure_matplotlib()

    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = _resolve_input_path(
        input_dir,
        args.summary_csv,
        ("summary.csv",),
        required=True,
    )
    paired_csv = _resolve_input_path(
        input_dir,
        args.paired_comparisons_csv,
        ("paired_comparisons.csv",),
        required=False,
    )
    results_csv = _resolve_input_path(
        input_dir,
        args.results_csv,
        ("results.csv",),
        required=False,
    )

    summary_rows = _canonical_summary_rows(
        _read_csv(summary_csv)
    )

    generated: List[Path] = []

    for task in TASK_ORDER:
        output_path = output_dir / f"{task}_mse.pdf"
        create_mse_figure(
            summary_rows,
            task,
            output_path,
            dpi=args.dpi,
            allow_incomplete=args.allow_incomplete,
        )
        generated.append(output_path)

    paired_rows: List[Dict[str, Any]] = []

    if paired_csv is not None:
        paired_rows = _paired_rows_from_csv(
            _read_csv(paired_csv)
        )

    if not paired_rows and results_csv is not None:
        paired_rows = _paired_rows_from_results(
            _read_csv(results_csv)
        )

    paired_output = output_dir / "paired_atode_vs_lgode.pdf"
    create_paired_improvement_figure(
        paired_rows,
        paired_output,
        dpi=args.dpi,
        allow_incomplete=args.allow_incomplete,
    )
    generated.append(paired_output)

    prediction_paths = discover_prediction_paths(
        input_dir,
        results_csv,
        args.prediction_file,
    )
    artifacts = load_prediction_artifacts(
        prediction_paths,
        quiet=args.quiet,
    )

    if not artifacts:
        message = (
            "No usable prediction artifacts were found. Prediction files "
            "are required for the representative trajectory and horizon-wise "
            "error figures."
        )

        if args.allow_incomplete:
            if not args.quiet:
                print(f"Warning: {message}")
            return generated

        raise FileNotFoundError(message)

    trajectory_model = _normalize_model(args.trajectory_model)
    trajectory_task = _normalize_task(args.trajectory_task)

    trajectory_artifact = select_prediction_artifact(
        artifacts,
        model=trajectory_model,
        task=trajectory_task,
        fraction=args.trajectory_fraction,
        seed=args.trajectory_seed,
    )

    trajectory_output = (
        output_dir / "representative_voltage_trajectory.pdf"
    )
    create_representative_trajectory_figure(
        trajectory_artifact,
        trajectory_output,
        batch_index=args.batch_index,
        bus_index=args.bus_index,
        feature_index=args.feature_index,
        dpi=args.dpi,
    )
    generated.append(trajectory_output)

    horizon_output = output_dir / "horizon_extrapolation_error.pdf"
    try:
        create_horizon_error_figure(
            artifacts,
            horizon_output,
            fraction=args.horizon_fraction,
            seed=args.horizon_seed,
            dpi=args.dpi,
            allow_incomplete=args.allow_incomplete,
        )
    except (ValueError, FileNotFoundError) as exc:
        if not args.allow_incomplete:
            raise
        if not args.quiet:
            print(f"Warning: skipping horizon figure: {exc}")
    else:
        generated.append(horizon_output)

    transport_candidates = [
        artifact
        for artifact in artifacts
        if artifact.model == "atode"
        and artifact.transport_weights is not None
    ]

    if transport_candidates:
        transport_artifact = select_prediction_artifact(
            transport_candidates,
            model="atode",
            task=trajectory_task,
            fraction=args.trajectory_fraction,
            seed=args.trajectory_seed,
        )

        transport_output = output_dir / "transport_weights.pdf"
        create_transport_weight_figure(
            transport_artifact,
            transport_output,
            dpi=args.dpi,
        )
        generated.append(transport_output)
    elif args.require_transport:
        raise FileNotFoundError(
            "No AT-ODE prediction artifact contains transport weights."
        )
    elif not args.quiet:
        print(
            "Note: no transport-weight trajectory was found; "
            "transport_weights.pdf was not generated."
        )

    return generated


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication figures from saved SimBench LG-ODE "
            "summary and prediction artifacts."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Experiment result directory containing summary CSVs and/or "
            "saved prediction artifacts."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=(
            "Path to summary.csv. Relative paths are resolved against "
            "--input-dir."
        ),
    )
    parser.add_argument(
        "--paired-comparisons-csv",
        default=None,
        help=(
            "Path to paired_comparisons.csv. If omitted or unavailable, "
            "paired improvements are computed from results.csv."
        ),
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help=(
            "Path to results.csv. It may contain prediction_path entries "
            "and per-seed metrics."
        ),
    )
    parser.add_argument(
        "--prediction-file",
        action="append",
        default=[],
        help=(
            "Explicit saved prediction artifact. May be supplied multiple "
            "times. NPZ is recommended; JSON and tensor-only PT/PTH "
            "dictionaries are also supported."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Figure output directory. Defaults to <input-dir>/figures."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization DPI used when saving figures.",
    )
    parser.add_argument(
        "--trajectory-model",
        default="atode",
        choices=MODEL_ORDER,
        help="Model used for the representative trajectory.",
    )
    parser.add_argument(
        "--trajectory-task",
        default="extrapolation",
        choices=TASK_ORDER,
        help="Task used for the representative trajectory.",
    )
    parser.add_argument(
        "--trajectory-fraction",
        type=float,
        default=0.6,
        help=(
            "Preferred observed fraction for the representative trajectory."
        ),
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=None,
        help="Preferred seed for the representative trajectory.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch trajectory index for the representative plot.",
    )
    parser.add_argument(
        "--bus-index",
        type=int,
        default=0,
        help="Bus index for the representative plot.",
    )
    parser.add_argument(
        "--feature-index",
        type=int,
        default=None,
        help=(
            "Voltage feature index. If omitted, the script searches feature "
            "names for a voltage channel and otherwise uses feature zero."
        ),
    )
    parser.add_argument(
        "--horizon-fraction",
        type=float,
        default=0.6,
        help="Preferred observed fraction for horizon-wise error.",
    )
    parser.add_argument(
        "--horizon-seed",
        type=int,
        default=None,
        help="Preferred seed for horizon-wise error.",
    )
    parser.add_argument(
        "--require-transport",
        action="store_true",
        help=(
            "Fail if no saved AT-ODE transport-weight trajectory is found."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Generate available figures even when some models, fractions, "
            "or prediction artifacts are missing."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress and non-fatal warning messages.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    if not 0.0 <= args.trajectory_fraction <= 1.0:
        parser.error("--trajectory-fraction must lie within [0, 1]")

    if not 0.0 <= args.horizon_fraction <= 1.0:
        parser.error("--horizon-fraction must lie within [0, 1]")

    if args.batch_index < 0:
        parser.error("--batch-index must be non-negative")

    if args.bus_index < 0:
        parser.error("--bus-index must be non-negative")

    if args.feature_index is not None and args.feature_index < 0:
        parser.error("--feature-index must be non-negative")

    generated = generate_figures(args)

    if not args.quiet:
        print(f"Generated {len(generated)} figure(s):")
        for path in generated:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
