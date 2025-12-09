from __future__ import annotations

import functools
from typing import Dict, List, Optional

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from hbrace.config import ExperimentConfig
from hbrace.guides import build_guide
from hbrace.patient_data import PatientBatch
from .hierarchical import hierarchical_model

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover - fallback if tqdm is missing
    trange = None


class HBRACEModel:
    """HBRACE model wrapper around the hierarchical Pyro model."""

    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initialize the HBRACEModel.

        Args:
            config: The ExperimentConfig to use for the model.
        """
        self.config = config
        self.model_fn = functools.partial(hierarchical_model, config=config.model)

    def train(
        self,
        batch: PatientBatch,
        num_steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        log_interval: Optional[int] = None,
        guide: Optional[str] = None,
        seed: Optional[int] = None,
        progress: bool = True,
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
            progress: Whether to show a tqdm progress bar when available.
        """

        pyro.clear_param_store()
        pyro.set_rng_seed(seed if seed is not None else self.config.vi.seed)

        guide_name = guide if guide is not None else self.config.vi.guide
        guide_fn = build_guide(self.model_fn, self.config.model, guide_name)
        lr = learning_rate if learning_rate is not None else self.config.vi.learning_rate
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model_fn, guide_fn, optimizer, loss=Trace_ELBO())

        steps = num_steps if num_steps is not None else self.config.vi.num_steps
        interval = self.config.vi.log_interval if log_interval is None else log_interval

        use_tqdm = progress and trange is not None
        iterator = trange(steps) if use_tqdm else range(steps)

        history: List[float] = []
        for step in iterator:
            loss = svi.step(batch)
            if use_tqdm:
                iterator.set_postfix(loss=f"{loss:.2f}")
            if interval and step % interval == 0:
                msg = f"step={step:05d} loss={loss:.2f}"
                if use_tqdm:
                    iterator.write(msg)
                else:
                    print(msg)
            history.append(loss)

        return {"elbo": history}
