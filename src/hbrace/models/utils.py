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
        samples = predictive(batch)
        probs = torch.sigmoid(samples["logit_y"])  # (S, B, ...)
        probs_mean = probs.mean(dim=0).reshape(-1)
        y_obs = batch.responses.reshape(-1)
        y_true_list.append(y_obs.cpu().numpy())
        y_score_list.append(probs_mean.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)
    return average_precision_score(y_true, y_score), y_true, y_score


def posterior_predictive_check(
    model: Callable,
    guide: Callable,
    dataloader: Iterable,
    num_samples: int = 500,
    device: torch.device = torch.device("cpu"),
    target: str = "counts",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute posterior predictive p-value using chi-squared discrepancy.

    Following Gelman, Meng & Stern (1996) methodology:
    1. Draw posterior samples θ^(s) from guide
    2. For each sample, compute T(y_obs, θ) and T(y_rep, θ)
    3. p-value = proportion where T(y_rep) >= T(y_obs)

    Args:
        model: Pyro model callable.
        guide: Trained guide callable.
        dataloader: Iterable of PatientBatch objects.
        num_samples: Number of posterior samples.
        device: Device to run computations on.
        target: What to compute discrepancy on - "responses", "counts", or "both"

    Returns:
        p_value: Posterior predictive p-value (good models have p ~ 0.5)
        T_obs_all: Chi-squared discrepancies for observed data (per sample)
        T_rep_all: Chi-squared discrepancies for replicated data (per sample)
    """
    # Collect all batches first
    all_batches = [batch.to(device) for batch in dataloader]

    T_obs_total = np.zeros(num_samples)
    T_rep_total = np.zeros(num_samples)

    for batch in all_batches:
        # Get predictive samples for this batch
        return_sites = []
        if target in ("responses", "both"):
            return_sites.extend(["logit_y", "y"])
        if target in ("counts", "both"):
            return_sites.extend(["f_p", "f_t", "mu_t", "phi_t", "log_mu_p", "phi_p"])

        predictive = Predictive(
            model,
            guide=guide,
            num_samples=num_samples,
            return_sites=return_sites,
            parallel=False,
        )
        samples = predictive(batch)

        if target in ("responses", "both"):
            y_obs = batch.responses  # (B,)
            logits = samples["logit_y"]  # (S, B)
            p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            y_rep = samples["y"]
            var_p = p * (1 - p)
            T_obs_batch = ((y_obs.unsqueeze(0) - p) ** 2 / var_p).sum(dim=-1)
            T_rep_batch = ((y_rep - p) ** 2 / var_p).sum(dim=-1)
            T_obs_total += T_obs_batch.cpu().numpy()
            T_rep_total += T_rep_batch.cpu().numpy()

        if target in ("counts", "both"):
            # Chi-squared on count data using Pearson residuals
            # For pre-treatment counts
            pre_obs = batch.pre_counts  # (B, C, G)
            f_p_rep = samples["f_p"]  # (S, B, C, G)
            log_mu_p = samples["log_mu_p"]  # (S, B, C, G)
            mu_p = torch.exp(log_mu_p)
            phi_p = samples["phi_p"]  # (S, B, C, G)
            # Variance of NB is mu + mu^2/phi
            var_p_counts = mu_p + mu_p**2 / phi_p.clamp_min(1e-6)
            var_p_counts = var_p_counts.clamp_min(1e-6)

            T_obs_pre = ((pre_obs.unsqueeze(0) - mu_p) ** 2 / var_p_counts).sum(dim=(-1, -2, -3))
            T_rep_pre = ((f_p_rep - mu_p) ** 2 / var_p_counts).sum(dim=(-1, -2, -3))

            # For on-treatment counts
            on_obs = batch.on_counts  # (B, C, G)
            f_t_rep = samples["f_t"]  # (S, B, C, G)
            mu_t = samples["mu_t"]  # (S, B, C, G)
            phi_t = samples["phi_t"]  # (S, B, C, G)
            var_t_counts = mu_t + mu_t**2 / phi_t.clamp_min(1e-6)
            var_t_counts = var_t_counts.clamp_min(1e-6)

            T_obs_on = ((on_obs.unsqueeze(0) - mu_t) ** 2 / var_t_counts).sum(dim=(-1, -2, -3))
            T_rep_on = ((f_t_rep - mu_t) ** 2 / var_t_counts).sum(dim=(-1, -2, -3))

            T_obs_total += (T_obs_pre + T_obs_on).cpu().numpy()
            T_rep_total += (T_rep_pre + T_rep_on).cpu().numpy()

    # p-value = proportion where T(y_rep) >= T(y_obs)
    p_value = np.mean(T_rep_total >= T_obs_total)

    return p_value, T_obs_total, T_rep_total


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
