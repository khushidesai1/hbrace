"""Configuration dataclasses tying the math in the proposal to runnable code."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    """Hyper-parameters for the hierarchical T-cell model."""

    n_subtypes: int
    n_cell_types: int
    n_genes: int
    latent_dim: int = 4  # dimension for latent z_i treatment effect
    delta_dim: int = 4  # dimension for simplex shift δ controlling interventions
    subtype_concentration: float = 2.0  # Gamma(shape, rate) ~ (2, 0.1) in proposal
    subtype_rate: float = 0.1
    nb_dispersion_prior: float = 2.0
    nb_dispersion_rate: float = 1.0


@dataclass
class VIConfig:
    """Settings for variational inference."""

    learning_rate: float = 5e-3
    num_steps: int = 5_000
    log_interval: int = 100
    guide: str = "auto_delta"
    seed: int = 0


@dataclass
class ExperimentConfig:
    model: ModelConfig
    vi: VIConfig


def _dict_to_config(obj: Dict[str, Any]) -> ExperimentConfig:
    model_cfg = ModelConfig(**obj["model"])
    vi_cfg = VIConfig(**obj["vi"])
    return ExperimentConfig(model=model_cfg, vi=vi_cfg)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config into the strongly-typed dataclasses."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _dict_to_config(raw)
