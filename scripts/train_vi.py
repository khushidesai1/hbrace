#!/usr/bin/env python3
"""Entry point for testing the VI skeleton with synthetic data."""
from __future__ import annotations

import argparse
import torch

from hbrace.config import load_config
from hbrace.data.synthetic_data import SimulatedDataGenerator
from hbrace.training import run_vi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=8,
        help="Number of synthetic patients to simulate for the dry-run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to place tensors on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(config.vi.seed)
    device = torch.device(args.device)
    generator = SimulatedDataGenerator.from_model_config(
        model_config=config.model,
        n_patients=args.num_patients,
        seed=config.vi.seed,
    )
    batch, _ = generator.generate_batch(device=device, return_simulation=True)
    run_vi(batch, config)


if __name__ == "__main__":
    main()
