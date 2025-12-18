"""Posterior Predictive Checks for HBRACE model."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Callable

import numpy as np
import torch
from pyro.infer import Predictive
from scipy.stats import pearsonr, spearmanr, chi2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from hbrace.config import ModelConfig, VIConfig
from hbrace.models import HBRACEModel
from hbrace.models.guides import build_guide
from hbrace.patient_data import SimulatedData
from hbrace.patient_data.types import PatientBatch


def sample_q_t(
    model: HBRACEModel,
    batch: PatientBatch,
    num_samples: int = 1000,
    return_all: bool = False,
    batch_size: int = 8,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Sample q_t (composition-weighted on-treatment expression) from posterior.

    Args:
        model: Trained HBRACEModel instance.
        batch: PatientBatch to condition on.
        num_samples: Number of posterior samples to draw.
        return_all: If True, return all sampled variables; if False, only q_t_mean.
        batch_size: Batch size for sampling (should match training batch_size for AutoGuides).

    Returns:
        If return_all=False: q_t samples of shape (num_samples, N, G)
        If return_all=True: Dictionary with all sampled sites including:
            - q_t_mean: (num_samples, N, G)
            - pi_t: (num_samples, N, C)
            - mu_t: (num_samples, N, C, G)
            - z: (num_samples, N, d_z)
            - u: (num_samples, N, r_u)
            - logit_y: (num_samples, N)
    """
    N = batch.pre_counts.shape[0]

    # If batch size matches or is larger, process directly
    if N <= batch_size:
        # Create a version of the batch without responses for true posterior sampling
        batch_no_y = PatientBatch(
            pre_counts=batch.pre_counts,
            on_counts=batch.on_counts,
            cell_type_proportions=batch.cell_type_proportions,
            responses=None,
            subtype_ids=batch.subtype_ids,
        )

        # Use Predictive to sample from posterior
        predictive = Predictive(
            model.model_fn,
            guide=model.guide_fn,
            num_samples=num_samples,
            return_sites=(
                "q_t_mean", "pi_t", "mu_t", "z", "u", "logit_y", "eta_t"
            ) if return_all else ("q_t_mean",),
            parallel=False,
        )

        with torch.no_grad():
            samples = predictive(batch_no_y)

        if return_all:
            return samples
        return samples["q_t_mean"]

    # Otherwise, process in batches and concatenate
    all_samples = [] if not return_all else None
    all_samples_dict = {} if return_all else None

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)

        # Create mini-batch
        mini_batch = PatientBatch(
            pre_counts=batch.pre_counts[start_idx:end_idx],
            on_counts=batch.on_counts[start_idx:end_idx],
            cell_type_proportions=batch.cell_type_proportions,
            responses=None,
            subtype_ids=batch.subtype_ids[start_idx:end_idx],
        )

        # Sample from this mini-batch
        predictive = Predictive(
            model.model_fn,
            guide=model.guide_fn,
            num_samples=num_samples,
            return_sites=(
                "q_t_mean", "pi_t", "mu_t", "z", "u", "logit_y", "eta_t"
            ) if return_all else ("q_t_mean",),
            parallel=False,
        )

        with torch.no_grad():
            mini_samples = predictive(mini_batch)

        if return_all:
            # Accumulate all sites
            for site_name, site_values in mini_samples.items():
                if site_name not in all_samples_dict:
                    all_samples_dict[site_name] = []
                all_samples_dict[site_name].append(site_values)
        else:
            all_samples.append(mini_samples["q_t_mean"])

    # Concatenate along patient dimension
    if return_all:
        concatenated = {}
        for site_name, site_list in all_samples_dict.items():
            concatenated[site_name] = torch.cat(site_list, dim=1)  # dim=1 is patient dimension
        return concatenated
    else:
        return torch.cat(all_samples, dim=1)  # dim=1 is patient dimension


def compute_observed_q_t(sim_data: SimulatedData) -> np.ndarray:
    """Compute observed q_t from observed data.

    In the HBRACE model, q_t is the composition-weighted on-treatment expression:
        q_t[i,g] = Σ_c pi_t[i,c] * mu_t[i,c,g]

    Since:
    - f_t[i,c,g] are observed mean counts per cell type (≈ mu_t[i,c,g])
    - pi_t[i,c] are observed from cell type assignments

    We can compute observed q_t as:
        q_t_obs[i,g] = Σ_c pi_t[i,c] * f_t[i,c,g]

    Args:
        sim_data: SimulatedData object with observed counts and proportions.

    Returns:
        Observed q_t of shape (N, G), computed from actual data.
    """
    pi_t = sim_data.pi_t  # (N, C) - observed cell type proportions
    f_t = sim_data.on_counts  # (N, C, G) - observed mean counts ≈ mu_t

    # q_t = sum_c pi_t[:, c] * f_t[:, c, :]
    q_t_observed = np.einsum("nc,ncg->ng", pi_t, f_t)
    return q_t_observed
    

def chi_squared_statistic(
    observed: np.ndarray,
    expected: np.ndarray,
    variance: Optional[np.ndarray] = None,
) -> float:
    """Compute chi-squared discrepancy statistic.

    Args:
        observed: Observed data, shape (N, G) or flattened.
        expected: Expected values under model, shape (N, G) or flattened.
        variance: Variance of each observation. If None, uses expected as variance.

    Returns:
        Chi-squared statistic value.
    """
    observed_flat = observed.flatten()
    expected_flat = expected.flatten()

    if variance is None:
        # For Poisson-like data, variance = mean
        variance_flat = np.maximum(expected_flat, 1e-6)
    else:
        variance_flat = variance.flatten()
        variance_flat = np.maximum(variance_flat, 1e-6)

    chi_sq = np.sum((observed_flat - expected_flat) ** 2 / variance_flat)
    return chi_sq


def bayesian_p_value(
    observed_statistic: float,
    replicated_statistics: np.ndarray,
) -> float:
    """Compute Bayesian p-value.

    The Bayesian p-value is P(T(y_rep) >= T(y_obs) | y_obs), where T is a test
    statistic, y_obs is observed data, and y_rep is replicated data from posterior.

    Args:
        observed_statistic: Test statistic computed on observed data.
        replicated_statistics: Test statistics computed on posterior replicated data,
            shape (num_samples,).

    Returns:
        Bayesian p-value in [0, 1]. Values near 0 or 1 indicate poor model fit.
    """
    p_value = np.mean(replicated_statistics >= observed_statistic)
    return p_value


def posterior_predictive_check_q_t(
    model: HBRACEModel,
    sim_data: SimulatedData,
    indices: Optional[np.ndarray] = None,
    num_samples: int = 500,
    device: str = "cpu",
    statistic_fn: Optional[Callable] = None,
    batch_size: int = 8,
) -> Dict[str, any]:
    """Perform posterior predictive check with chi-squared discrepancy on q_t.

    Following Gelman et al. (1996), this computes a Bayesian p-value by:
    1. Computing test statistic T(y_obs) on observed data
    2. Drawing replicated data y_rep from posterior predictive
    3. Computing T(y_rep) for each replicate
    4. p-value = P(T(y_rep) >= T(y_obs))

    Args:
        model: Trained HBRACEModel.
        sim_data: SimulatedData with ground truth.
        indices: Patient indices to evaluate.
        num_samples: Number of posterior predictive samples.
        device: Device for computation.
        statistic_fn: Custom test statistic function. If None, uses chi-squared.

    Returns:
        Dictionary containing:
            - observed_statistic: T(q_t_obs)
            - replicated_statistics: T(q_t_rep) for each sample, shape (num_samples,)
            - p_value: Bayesian p-value
            - q_t_observed: Observed q_t (computed from data)
            - q_t_replicated: Replicated q_t samples, shape (num_samples, N, G)
    """
    # Get observed q_t (computed from data)
    if indices is None:
        q_t_observed = compute_observed_q_t(sim_data)
    else:
        sim_data_subset = SimulatedData(
            config=sim_data.config,
            pre_counts=sim_data.pre_counts[indices],
            on_counts=sim_data.on_counts[indices],
            responses=sim_data.responses[indices],
            subtype_ids=sim_data.subtype_ids[indices],
            pi_p=sim_data.pi_p[indices],
            pi_t=sim_data.pi_t[indices],
            pre_cell_types=[sim_data.pre_cell_types[i] for i in indices] if sim_data.pre_cell_types else None,
            post_cell_types=[sim_data.post_cell_types[i] for i in indices] if sim_data.post_cell_types else None,
            extra_params={k: v[indices] if isinstance(v, np.ndarray) and v.shape[0] == len(sim_data.responses)
                         else v for k, v in sim_data.extra_params.items()} if sim_data.extra_params else None,
        )
        q_t_observed = compute_observed_q_t(sim_data_subset)

    # Sample q_t from posterior predictive
    batch = sim_data.to_patient_batch(device=device, indices=indices)
    q_t_replicated = sample_q_t(model, batch, num_samples=num_samples, batch_size=batch_size)  # (S, N, G)
    q_t_replicated_np = q_t_replicated.cpu().numpy()

    # Define default statistic function (chi-squared)
    if statistic_fn is None:
        def statistic_fn(q_t_rep):
            # Chi-squared: sum((observed - replicated)^2 / variance)
            # For continuous data, use empirical variance from replicates
            q_t_mean = q_t_rep.mean(axis=0) if q_t_rep.ndim == 3 else q_t_rep
            variance = np.var(q_t_replicated_np, axis=0).mean()  # pooled variance
            return chi_squared_statistic(q_t_observed, q_t_mean, variance)

    # Compute observed statistic
    # For "observed", use mean of first few samples as point estimate
    observed_statistic = statistic_fn(q_t_replicated_np[:10].mean(axis=0))

    # Compute replicated statistics
    replicated_statistics = np.zeros(num_samples)
    for s in range(num_samples):
        replicated_statistics[s] = statistic_fn(q_t_replicated_np[s])

    # Compute p-value
    p_value = bayesian_p_value(observed_statistic, replicated_statistics)

    return {
        "observed_statistic": observed_statistic,
        "replicated_statistics": replicated_statistics,
        "p_value": p_value,
        "q_t_observed": q_t_observed,
        "q_t_replicated": q_t_replicated_np,
    }
