"""Configuration dataclasses tying the math in the proposal to runnable code."""
from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional


@dataclass
class ModelConfig:
    """Hyper-parameters for the hierarchical T-cell model."""
    n_subtypes: int
    n_cell_types: int
    n_genes: int
    z_dim: int = 4  # dimension for latent z_i treatment effect
    u_dim: int = 5  # dimension for latent u_i confounder
    beta_t_l1_strength: float = 0.0  # strength of L1 sparsity penalty on beta_t
    beta_t_gate_alpha: float = 0.5  # Beta prior alpha for beta_t gate (smaller favors sparsity)
    beta_t_gate_beta: float = 2.0   # Beta prior beta for beta_t gate (larger favors sparsity)
    subtype_concentration: float = 2.0  # Gamma(shape, rate) ~ (2, 0.1) in proposal
    subtype_rate: float = 0.1
    nb_dispersion_prior: float = 2.0
    nb_dispersion_rate: float = 1.0


@dataclass
class VIConfig:
    """Settings for variational inference."""
    early_stopping_patience: float = 5
    learning_rate: float = 5e-3
    num_epochs: int = 100
    log_interval: int = 100
    guide: str = "auto_delta"
    guide_rank: Optional[int] = None
    seed: int = 0


@dataclass
class DataConfig:
    """Configuration for the data."""
    num_patients: int = 128
    test_fraction: float = 0.25
    batch_size: int = 8
    seed: int = 0
    device: str = "cpu"
    beta_t_active_frac: float = 0.1  # fraction of genes truly predictive in synthetic data
    beta_t_active_scale: float = 0.5  # std for active gene coefficients in synthetic data
    beta_t_inactive_loc: float = 0.0  # mean for inactive gene coefficients in synthetic data
    beta_t_inactive_scale: float = 0.0  # std for inactive gene coefficients in synthetic data
    response_base_rate: float = 0.3  # target baseline response rate in synthetic data
    logit_std_target: float = 1.0  # target std for logits before sigmoid in synthetic data


def _dict_to_config(obj: Dict[str, Any]) -> Tuple[ModelConfig, VIConfig, DataConfig]:
    model_config = ModelConfig(**obj["model"])
    vi_config = VIConfig(**obj["vi"])
    data_config = DataConfig(**obj["data"])
    run_name = obj["run_name"]
    return run_name, model_config, vi_config, data_config


def load_config(path: str | Path) -> Tuple[ModelConfig, VIConfig, DataConfig]:
    """Load a YAML config into the strongly-typed dataclasses."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _dict_to_config(raw)
