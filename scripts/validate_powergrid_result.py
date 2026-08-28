#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict


EXPECTED_CONFIG: Dict[str, Any] = {
    "observed_fraction": 0.4,
    "batch_size": 8,
    "stride": 24,
    "niters": 200,
    "patience": 25,
    "min_delta": 0.0,
    "lr": 5e-4,
    "lr_patience": 8,
    "lr_factor": 0.5,
    "weight_decay": 0.0,
    "optimizer": "adam",
    "gradient_clip": 10.0,
    "kl_coef": 1.0,
    "kl_warmup_epochs": 10,
    "train_samples": 1,
    "eval_samples": 1,
    "latent_dim": 16,
    "recognition_dim": 64,
    "ode_hidden_dim": 128,
    "augmentation_dim": 0,
    "encoder_layers": 2,
    "ode_layers": 1,
    "attention_heads": 1,
    "edge_types": 2,
    "dropout": 0.2,
    "ode_dropout": 0.0,
    "observation_std": 0.01,
    "solver": "rk4",
    "rtol": 1e-3,
    "atol": 1e-4,
    "transport_bins": 32,
    "transport_max_age": 4.0,
    "transport_hidden_dim": 64,
    "transport_attention_dim": 16,
    "transport_heads": 4,
    "transport_speed": 1.0,
    "transport_decay": 1.0,
    "deterministic": True,
    "num_workers": 0,
    "eval_seed": 12345,
    "graph_mode": "physical_sparse",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equal(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
    return left == right


def validate(args: argparse.Namespace) -> None:
    result = json.loads(args.result.read_text(encoding="utf-8"))
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    expected_identity = {
        "model": args.model,
        "task": args.task,
        "seed": args.seed,
        "mask_seed": args.seed,
        "eval_seed": 12345,
        "batch_size": args.batch_size,
        "stride": (
            1 if args.dataset_format == "ieee39_transient" else 24
        ),
        "observed_fraction": args.observed_fraction,
        "git_commit": current_commit,
        "git_dirty": current_dirty,
    }
    for key, expected in expected_identity.items():
        if key not in result or not equal(result[key], expected):
            raise ValueError(
                f"{args.result}: {key}={result.get(key)!r}, expected {expected!r}"
            )

    config = result.get("config", {})
    expected_config = dict(EXPECTED_CONFIG)
    if args.dataset_format == "ieee39_transient":
        expected_config.update(
            {
                "dataset_format": "ieee39_transient",
                "scenario_scale": args.run_mode,
                "graph_mode": "complete_directed_generator_candidate",
                "observed_fraction": args.observed_fraction,
                "stride": 1,
                "niters": {
                    "smoke": 1,
                    "development": 20,
                    "publication": 150,
                }[args.run_mode],
                "validation_interval": (
                    1 if args.run_mode == "smoke" else 2
                ),
                "patience": {
                    "smoke": 1,
                    "development": 5,
                    "publication": 15,
                }[args.run_mode],
                "lr_patience": 5,
                "batch_size": args.batch_size,
                "transport_max_age": 0.6,
            }
        )
    for key, expected in expected_config.items():
        if key not in config or not equal(config[key], expected):
            raise ValueError(
                f"{args.result}: config.{key}={config.get(key)!r}, "
                f"expected {expected!r}"
            )
    expected_lengths = (
        {"trajectory_length": 60}
        if args.task == "interpolation"
        else {"context_length": 30, "forecast_length": 30}
    ) if args.dataset_format == "ieee39_transient" else (
        {"trajectory_length": 24}
        if args.task == "interpolation"
        else {"context_length": 12, "forecast_length": 12}
    )
    for key, expected in expected_lengths.items():
        if not equal(result.get(key), expected):
            raise ValueError(f"{args.result}: invalid {key}")

    dataset = result.get("dataset", {})
    if dataset.get("sha256") != sha256(args.data):
        raise ValueError(f"{args.result}: dataset hash mismatch")
    for key in ("window_hashes", "observation_mask_hashes", "edge_index_sha256"):
        if not dataset.get(key):
            raise ValueError(f"{args.result}: missing dataset.{key}")
    if args.dataset_format == "ieee39_transient":
        source_path = Path(dataset.get("source_path", ""))
        if not source_path.is_file():
            raise ValueError(f"{args.result}: IEEE39 source path is unavailable")
        if dataset.get("source_sha256") != sha256(source_path):
            raise ValueError(f"{args.result}: IEEE39 source hash mismatch")
        if dataset.get("processed_sha256") != sha256(args.data):
            raise ValueError(f"{args.result}: IEEE39 cache hash mismatch")
        if result.get("protocol", {}).get("graph_mode") != (
            "complete_directed_generator_candidate"
        ):
            raise ValueError(f"{args.result}: wrong IEEE39 graph mode")
        if dataset.get("mask_schema_version") != 2:
            raise ValueError(f"{args.result}: wrong IEEE39 mask schema")
        expected_counts = {
            "smoke": {"train": 16, "validation": 8, "test": 8},
            "development": {"train": 512, "validation": 128, "test": 128},
            "publication": {"train": 8996, "validation": 1927, "test": 1929},
        }[args.run_mode]
        if dataset.get("window_counts") != expected_counts:
            raise ValueError(f"{args.result}: wrong IEEE39 scenario counts")

    fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    for key in ("window_hashes", "observation_mask_hashes", "edge_index_hash"):
        result_key = "edge_index_sha256" if key == "edge_index_hash" else key
        if dataset.get(result_key) != fingerprint.get(key):
            raise ValueError(f"{args.result}: protocol fingerprint mismatch for {key}")

    test = result.get("test", {})
    metric_key = (
        "normalized_mse_unobserved"
        if args.task == "interpolation"
        else "normalized_mse_full"
    )
    mae_key = metric_key.replace("mse", "mae")
    for key in (metric_key, mae_key):
        value = float(test.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"{args.result}: nonfinite test.{key}")

    expected_selection = (
        "normalized_mse_unobserved"
        if args.task == "interpolation"
        else "normalized_mse_full"
    )
    if result.get("checkpoint_selection_metric") != expected_selection:
        raise ValueError(f"{args.result}: wrong checkpoint selection metric")
    if args.model in {"lgode", "atode"}:
        expected_edges = (
            90 if args.dataset_format == "ieee39_transient" else 194
        )
        if dataset.get("physical_directed_edge_count") != expected_edges:
            raise ValueError(
                f"{args.result}: expected {expected_edges} graph edges"
            )
        if dataset.get("candidate_directed_edge_count") != expected_edges:
            raise ValueError(
                f"{args.result}: expected {expected_edges} candidates"
            )
        if not result.get("shared_initialization_sha256"):
            raise ValueError(f"{args.result}: missing shared initialization hash")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--model", choices=("persistence", "latentode", "lgode", "atode"), required=True)
    parser.add_argument("--task", choices=("interpolation", "extrapolation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("simbench", "ieee39_transient"),
        default="simbench",
    )
    parser.add_argument(
        "--run-mode",
        choices=("smoke", "development", "publication"),
        default="publication",
    )
    parser.add_argument("--observed-fraction", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    validate(args)
    print(f"Validated completed result: {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())