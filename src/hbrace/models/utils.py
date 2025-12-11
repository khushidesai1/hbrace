from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch
from pyro import poutine


def nb_logits(mean: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
    """Convert mean/dispersion parameterization to logits used by Pyro."""

    mean = mean.clamp_min(1e-5)
    total_count = total_count.clamp_min(1e-5)
    probs = mean / (mean + total_count)
    return torch.log(probs) - torch.log1p(-probs)


def predictive_log_likelihood(
    model: Callable, guide: Callable, dataloader: Iterable, num_samples: int = 32
) -> float:
    """
    Monte Carlo estimate of the average predictive log likelihood per patient.

    Args:
        model: Pyro model callable.
        guide: Trained guide callable.
        dataloader: Iterable of PatientBatch objects.
        num_samples: Number of posterior samples to average over.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    log_weights = []
    total_obs = None

    with torch.no_grad():
        for _ in range(num_samples):
            total_log_prob = 0.0
            obs_count = 0
            for batch in dataloader:
                obs_count += batch.responses.shape[0]
                guide_trace = poutine.trace(guide).get_trace(batch)
                model_trace = poutine.trace(
                    poutine.replay(model, trace=guide_trace)
                ).get_trace(batch)
                total_log_prob += model_trace.log_prob_sum() - guide_trace.log_prob_sum()

            log_weights.append(total_log_prob)
            if total_obs is None:
                total_obs = obs_count

    log_weights_tensor = torch.stack([torch.as_tensor(w) for w in log_weights])
    log_normalizer = torch.log(torch.tensor(float(num_samples), device=log_weights_tensor.device))
    log_z = torch.logsumexp(log_weights_tensor - log_normalizer, dim=0)

    return log_z.item() / max(total_obs or 1, 1)


class EarlyStopping:
    """
    Keeps track of when the loss does not improve after a given patience.
    Useful to stop training when the validation loss does not improve anymore.
    Modified from https://github.com/azizilab/decipher/blob/main/decipher/tools/utils.py#L4.

    Args:
        patience: How long to wait after the last validation loss improvement.

    Returns:
        True if the training should stop.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.validation_loss_min = np.inf

    def __call__(self, validation_loss):
        """Returns True if the training should stop."""
        if validation_loss < self.validation_loss_min:
            self.validation_loss_min = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def has_stopped(self):
        """Returns True if the stopping condition has been met."""
        return self.early_stop
