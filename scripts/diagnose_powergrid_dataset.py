#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from lib.powergrid_baselines import PersistenceBaseline
from lib.simbench_lgode_data import build_simbench_dataloaders


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if denominator <= 0:
        return float("nan")
    return float((left * right).sum() / denominator)


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    loaders = build_simbench_dataloaders(
        data_path=args.data_path,
        simbench_code=None,
        task="extrapolation",
        observed_fraction=args.observed_fraction,
        batch_size=args.batch_size,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        stride=args.stride,
        seed=args.seed,
        mask_seed=args.seed,
        num_workers=0,
    )
    archive = loaders.archive
    model = PersistenceBaseline("extrapolation")

    squared_by_horizon = None
    absolute_by_horizon = None
    count_by_horizon = None
    absolute_by_node = None
    count_by_node = None
    absolute_by_feature = None
    count_by_feature = None

    for batch in loaders.test:
        prediction = model.predict(batch)
        error = prediction - batch.target_values
        mask = batch.extrapolation_future_mask.double()
        squared = error.square().double() * mask
        absolute = error.abs().double() * mask
        if squared_by_horizon is None:
            squared_by_horizon = torch.zeros(error.shape[2], dtype=torch.float64)
            absolute_by_horizon = torch.zeros_like(squared_by_horizon)
            count_by_horizon = torch.zeros_like(squared_by_horizon)
            absolute_by_node = torch.zeros(error.shape[1], dtype=torch.float64)
            count_by_node = torch.zeros_like(absolute_by_node)
            absolute_by_feature = torch.zeros(error.shape[3], dtype=torch.float64)
            count_by_feature = torch.zeros_like(absolute_by_feature)
        squared_by_horizon += squared.sum(dim=(0, 1, 3))
        absolute_by_horizon += absolute.sum(dim=(0, 1, 3))
        count_by_horizon += mask.sum(dim=(0, 1, 3))
        absolute_by_node += absolute.sum(dim=(0, 2, 3))
        count_by_node += mask.sum(dim=(0, 2, 3))
        absolute_by_feature += absolute.sum(dim=(0, 1, 2))
        count_by_feature += mask.sum(dim=(0, 1, 2))

    if squared_by_horizon is None:
        raise RuntimeError("Test loader produced no diagnostic batches.")

    normalized = loaders.normalization.normalize(archive.bus_state)
    train = normalized[: archive.train_end]
    temporal_change = torch.abs(train[1:] - train[:-1]).flatten().double()
    normalized_variance = train.double().var(dim=(0, 1), unbiased=False)

    receiver_degree = torch.bincount(
        archive.edge_index[1], minlength=archive.num_nodes
    )
    node_mae = absolute_by_node / count_by_node.clamp_min(1.0)
    degree_error = {
        str(int(degree)): float(node_mae[receiver_degree == degree].mean())
        for degree in torch.unique(receiver_degree)
    }

    source = archive.edge_index[0]
    receiver = archive.edge_index[1]
    neighbor_sum = torch.zeros_like(train[:-1])
    neighbor_count = torch.zeros(
        archive.num_nodes, dtype=train.dtype
    )
    neighbor_count.index_add_(0, receiver, torch.ones_like(receiver, dtype=train.dtype))
    for time_index in range(train.shape[0] - 1):
        neighbor_sum[time_index].index_add_(0, receiver, train[time_index, source])
    neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1.0).view(1, -1, 1)

    report = {
        "split": "test",
        "persistence_normalized_mse_by_horizon": (
            squared_by_horizon / count_by_horizon.clamp_min(1.0)
        ).tolist(),
        "persistence_normalized_mae_by_horizon": (
            absolute_by_horizon / count_by_horizon.clamp_min(1.0)
        ).tolist(),
        "temporal_change_mean": float(temporal_change.mean()),
        "temporal_change_quantiles": {
            str(quantile): float(torch.quantile(temporal_change, quantile))
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "normalized_variance_by_feature": {
            name: float(normalized_variance[index])
            for index, name in enumerate(archive.bus_feature_names)
        },
        "persistence_normalized_mae_by_node": node_mae.tolist(),
        "persistence_normalized_mae_by_feature": {
            name: float(
                absolute_by_feature[index]
                / count_by_feature[index].clamp_min(1.0)
            )
            for index, name in enumerate(archive.bus_feature_names)
        },
        "persistence_mae_by_receiver_degree": degree_error,
        "own_history_predictive_correlation": correlation(
            train[:-1], train[1:]
        ),
        "neighbor_history_predictive_correlation": correlation(
            neighbor_mean, train[1:]
        ),
        "note": "Read-only diagnostics; not used for model selection.",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--observed-fraction", type=float, default=0.4)
    parser.add_argument("--context-length", type=int, default=12)
    parser.add_argument("--forecast-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    report = diagnose(args)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())