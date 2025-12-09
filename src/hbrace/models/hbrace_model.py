from __future__ import annotations

from typing import Dict, List, Optional

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from hbrace.config import ExperimentConfig
from hbrace.data import PatientBatch
from hbrace.guides import build_guide
from .hierarchical import hierarchical_model


class HBRACEModel:
    """HBRACE model wrapper around the hierarchical Pyro model."""

    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initialize the HBRACEModel.

        Args:
            config: The ExperimentConfig to use for the model.
        """
        self.config = config

    def train(
        self,
        batch: PatientBatch,
        num_steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        log_interval: Optional[int] = None,
        guide: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Run stochastic variational inference on the provided batch.

        Args:
            batch: PatientBatch with counts/responses/subtypes.
            num_steps: Override for number of SVI steps.
            learning_rate: Override for optimizer learning rate.
            log_interval: How often to print loss (0 disables).
            guide: Guide strategy name override.
            seed: RNG seed for reproducibility.
        """

        pyro.clear_param_store()
        pyro.set_rng_seed(seed if seed is not None else self.config.vi.seed)

        guide_name = guide if guide is not None else self.config.vi.guide
        guide_fn = build_guide(hierarchical_model, self.config.model, guide_name)
        lr = learning_rate if learning_rate is not None else self.config.vi.learning_rate
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(hierarchical_model, guide_fn, optimizer, loss=Trace_ELBO())

        steps = num_steps if num_steps is not None else self.config.vi.num_steps
        interval = self.config.vi.log_interval if log_interval is None else log_interval

        history: List[float] = []
        for step in range(steps):
            loss = svi.step(batch)
            if interval and step % interval == 0:
                print(f"step={step:05d} loss={loss:.2f}")
            history.append(loss)

        return {"elbo": history}
