"""Reusable VI loop for future experiments."""
from __future__ import annotations

import functools
from typing import Dict, List

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from hbrace.config import ExperimentConfig
from hbrace.data import PatientBatch
from hbrace.guides import build_guide
from hbrace.models import hierarchical_model


def run_vi(batch: PatientBatch, config: ExperimentConfig) -> Dict[str, List[float]]:
    """Run SVI and return the ELBO trace."""

    pyro.clear_param_store()
    model_fn = functools.partial(hierarchical_model, config=config.model)
    guide = build_guide(model_fn, config.model, config.vi.guide)
    optimizer = ClippedAdam({"lr": config.vi.learning_rate})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    history: List[float] = []
    for step in range(config.vi.num_steps):
        loss = svi.step(batch)
        if step % config.vi.log_interval == 0:
            print(f"step={step:05d} loss={loss:.2f}")
        history.append(loss)

    return {"elbo": history}
