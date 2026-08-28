#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from lib.ieee39_transient_data import build_ieee39_dataloaders
from lib.powergrid_model_factory import build_powergrid_lgode_model
from run_powergrid_lgode import build_candidate_graph, train_one_epoch


def model_args(model: str, nodes: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        model_name=model,
        model_type=model,
        task="extrapolation",
        num_nodes=nodes,
        latent_dim=16,
        recognition_dim=64,
        ode_hidden_dim=128,
        augmentation_dim=0,
        encoder_layers=2,
        ode_layers=1,
        attention_heads=1,
        edge_types=2,
        dropout=0.2,
        ode_dropout=0.0,
        observation_std=0.01,
        solver="rk4",
        rtol=1e-3,
        atol=1e-4,
        seed=1,
        transport_bins=32,
        transport_max_age=0.6,
        transport_hidden_dim=64,
        transport_attention_dim=16,
        transport_heads=4,
        transport_speed=1.0,
        transport_decay=1.0,
        graph_mode="complete_directed_generator_candidate",
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--observed-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    rows = []
    for batch_size in (16, 32, 64):
        loaders = build_ieee39_dataloaders(
            args.data_path,
            task="extrapolation",
            observed_fraction=args.observed_fraction,
            batch_size=batch_size,
            seed=1,
            mask_seed=1,
            scale="development",
        )
        edge_index, labels = build_candidate_graph(
            loaders.archive.num_nodes,
            loaders.archive.edge_index,
            "complete_directed_generator_candidate",
        )
        batch = next(iter(loaders.train))
        for model_name in ("latentode", "lgode", "atode"):
            model = build_powergrid_lgode_model(
                model_name,
                loaders.archive.input_dim,
                loaders.archive.num_nodes,
                edge_index,
                model_args(model_name, loaders.archive.num_nodes),
                device,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            metrics = train_one_epoch(
                model,
                model_name,
                [batch],
                labels,
                optimizer,
                device,
                n_traj_samples=1,
                kl_coef=0.1,
                gradient_clip=10.0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "model": model_name,
                    "batch_size": batch_size,
                    "batch_seconds": elapsed,
                    "development_epoch_seconds_estimate": (
                        elapsed * math.ceil(512 / batch_size)
                    ),
                    "peak_cuda_memory_bytes": (
                        int(torch.cuda.max_memory_allocated())
                        if device.type == "cuda"
                        else None
                    ),
                    "loss": metrics["loss"],
                }
            )
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    text = json.dumps(rows, indent=2, allow_nan=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())