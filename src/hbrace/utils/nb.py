"""Helpers for parameterizing Negative Binomial distributions."""
from __future__ import annotations

import torch


def nb_logits(mean: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
    """Convert mean/dispersion parameterization to logits used by Pyro."""

    mean = mean.clamp_min(1e-5)
    total_count = total_count.clamp_min(1e-5)
    probs = mean / (mean + total_count)
    return torch.log(probs) - torch.log1p(-probs)
