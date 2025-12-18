from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np
import torch
from pyro import poutine
from pyro.infer import Predictive
from sklearn.metrics import average_precision_score


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
                    poutine.condition(poutine.replay(model, trace=guide_trace), data={"y": None})
                ).get_trace(batch)
                total_log_prob += model_trace.log_prob_sum() - guide_trace.log_prob_sum()

            log_weights.append(total_log_prob)
            if total_obs is None:
                total_obs = obs_count

    log_weights_tensor = torch.stack([torch.as_tensor(w) for w in log_weights])
    log_normalizer = torch.log(torch.tensor(float(num_samples), device=log_weights_tensor.device))
    log_z = torch.logsumexp(log_weights_tensor - log_normalizer, dim=0)

    return log_z.item() / max(total_obs or 1, 1)


def auprc_for_responses(
    model: Callable,
    guide: Callable,
    dataloader: Iterable,
    num_samples: int,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AUPRC for response predictions using posterior predictive probabilities.
    Args:
        model: Pyro model callable.
        guide: Trained guide callable.
        dataloader: Iterable of PatientBatch objects.
        num_samples: Number of posterior samples to average over.
        device: Device to run the model on.
    Returns:
        AUPRC, y_true, y_score.
    """
    y_true_list = []
    y_score_list = []
    for batch in dataloader:
        batch = batch.to(device)
        predictive = Predictive(
            model,
            guide=guide,
            num_samples=num_samples,
            return_sites=("logit_y",),
            parallel=False,
        )
        # Drop labels so predictive samples are truly conditional on latents only.
        batch_no_y = batch.to(device)
        batch_no_y = batch_no_y.__class__(
            pre_counts=batch_no_y.pre_counts,
            on_counts=batch_no_y.on_counts,
            cell_type_proportions=batch_no_y.cell_type_proportions,
            responses=None,
            subtype_ids=batch_no_y.subtype_ids,
        )
        samples = predictive(batch_no_y)
        probs = torch.sigmoid(samples["logit_y"])  # (S, B, ...)
        probs_mean = probs.mean(dim=0).reshape(-1)
        y_obs = batch.responses.reshape(-1)
        y_true_list.append(y_obs.cpu().numpy())
        y_score_list.append(probs_mean.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)
    return average_precision_score(y_true, y_score), y_true, y_score

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
        # Ignore non-finite metrics to avoid tripping the counter spuriously.
        if not np.isfinite(validation_loss):
            return False
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
