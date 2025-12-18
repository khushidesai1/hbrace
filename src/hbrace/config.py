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
    gene_sparsity: bool = True  # if True, use Laplace prior on beta_t; if False, use Normal(0, 2.0)
    composition_model: str = "linear"  # "linear" or "poe" (product of experts)
    beta_t_laplace_scale: float = 1.0  # scale of Laplace prior on beta_t (only used if gene_sparsity=True)
    beta0_loc: float = -0.85  # prior mean for intercept (approx logit of base rate 0.3)
    beta0_scale: float = 1.0  # prior std for intercept
    logit_scale: float = 0.5  # scaling for response linear predictor
    gamma_scale: float = 1.0  # std for gamma prior
    beta_s_scale: float = 1.0  # std for beta_s prior
    head_input_scale: float = 0.1  # scaling applied to q_t_mean and u in the head
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
    delta_warmup_epochs: int = 0  # optional warm-up with AutoDelta before switching to main guide
    delta_warmup_lr: Optional[float] = None  # optional LR for warm-up; defaults to learning_rate


@dataclass
class DataConfig:
    """Configuration for the data."""
    num_patients: int = 128
    test_fraction: float = 0.25
    batch_size: int = 8
    seed: int = 0
    device: str = "cpu"
    gene_sparsity: bool = True  # if True, use sparse beta_t; if False, use Normal(0, 2.0) for all genes
    composition_model: str = "linear"  # "linear" or "poe" (product of experts)
    beta_t_active_frac: float = 0.1  # fraction of genes truly predictive in synthetic data (only used if gene_sparsity=True)
    beta_t_active_scale: float = 1.0  # base scale for beta_t coefficients in synthetic data (only used if gene_sparsity=True)
    response_base_rate: float = 0.3  # target baseline response rate in synthetic data
    logit_scale: float = 0.5  # scaling for response linear predictor in synthetic data
    beta0_loc: float = -0.85  # mean for intercept in synthetic data
    beta0_scale: float = 1.0  # std for intercept in synthetic data
    gamma_scale: float = 1.0  # std for gamma in synthetic data
    beta_s_scale: float = 1.0  # std for beta_s in synthetic data
    head_input_scale: float = 0.1  # scaling applied to q_t_mean and u in synthetic data


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
