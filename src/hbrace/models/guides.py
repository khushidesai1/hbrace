from __future__ import annotations
from typing import Callable, Optional

from pyro import poutine
from pyro.infer.autoguide import AutoDelta, AutoLowRankMultivariateNormal, AutoNormal
from hbrace.config import ModelConfig


def build_guide(model: Callable, config: ModelConfig, strategy: str, rank: Optional[int] = None) -> Callable:
    """
    Return an AutoGuide tailored to the chosen strategy.

    Args:
        model: The model to guide.
        config: The model configuration.
        strategy: The guide strategy to use.

    Returns:
        An AutoGuide tailored to the chosen strategy.
    """

    strategy = strategy.lower()
    # Hide discrete site y from the guide so autoguides only parameterize continuous latents.
    model_for_guide = poutine.block(model, hide=["y"])
    if strategy == "auto_delta":
        return AutoDelta(model_for_guide)
    if strategy == "auto_normal":
        return AutoNormal(model_for_guide)
    if strategy == "auto_lowrank":
        guide_rank = rank if rank is not None else min(5, config.n_cell_types)
        return AutoLowRankMultivariateNormal(model_for_guide, rank=guide_rank)
    raise ValueError(f"Unknown guide strategy: {strategy}")
