#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lib.ieee39_transient_data import (
    build_ieee39_dataloaders,
    load_ieee39_archive,
    observation_statistics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "data/ieee39_transient/processed/ieee39_transient_v1.npz"
        ),
    )
    parser.add_argument(
        "--scale",
        choices=("smoke", "development", "publication"),
        default="smoke",
    )
    parser.add_argument("--mask-seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    archive = load_ieee39_archive(args.data_path)
    report = {
        "scenario_shape": list(archive.scenario_state.shape),
        "timestamp_shape": list(archive.timestamps_seconds.shape),
        "timestamp_min_seconds": float(archive.timestamps_seconds.min()),
        "timestamp_max_seconds": float(archive.timestamps_seconds.max()),
        "timestamp_spacing_seconds": 0.01,
        "generator_order": list(archive.generator_names),
        "feature_order": list(archive.bus_feature_names),
        "feature_units": list(archive.feature_units),
        "label_counts": {
            str(label): int((archive.labels == label).sum())
            for label in (0, 1)
        },
        "split_counts": {
            split: int(indices.numel())
            for split, indices in archive.split_indices.items()
        },
        "split_disjoint": len(
            torch.unique(torch.cat(tuple(archive.split_indices.values())))
        ) == archive.num_scenarios,
        "feature_variance": {
            name: float(
                archive.scenario_state[..., index].double().var(
                    unbiased=False
                )
            )
            for index, name in enumerate(archive.bus_feature_names)
        },
        "graph_mode": archive.metadata["graph_mode"],
        "graph_edge_count": int(archive.edge_index.shape[1]),
        "graph_sha256": archive.metadata["graph_sha256"],
        "mapping_limitation": archive.metadata["mapping_limitation"],
        "mask_conditions": {},
    }
    for task in ("interpolation", "extrapolation"):
        for fraction in (0.2, 0.4, 0.6, 0.8):
            loaders = build_ieee39_dataloaders(
                args.data_path,
                task=task,
                observed_fraction=fraction,
                batch_size=64,
                seed=args.mask_seed,
                mask_seed=args.mask_seed,
                scale=args.scale,
            )
            report["mask_conditions"][f"{task}_obs{int(100*fraction)}"] = (
                observation_statistics(loaders.train.dataset)
            )
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