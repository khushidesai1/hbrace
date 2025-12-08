"""Factories for VI guides."""
from __future__ import annotations

from typing import Callable

from pyro.infer.autoguide import AutoDelta, AutoLowRankMultivariateNormal, AutoNormal

from hbrace.config import ModelConfig


def build_guide(model: Callable, config: ModelConfig, strategy: str) -> Callable:
    """Return an AutoGuide tailored to the chosen strategy."""

    strategy = strategy.lower()
    if strategy == "auto_delta":
        return AutoDelta(model)
    if strategy == "auto_normal":
        return AutoNormal(model)
    if strategy == "auto_lowrank":
        return AutoLowRankMultivariateNormal(model, rank=min(5, config.n_cell_types))
    raise ValueError(f"Unknown guide strategy: {strategy}")
