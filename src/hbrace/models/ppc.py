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


def compute_true_q_t(sim_data: SimulatedData) -> np.ndarray:
    """Compute true q_t from simulated data parameters.

    Args:
        sim_data: SimulatedData object with extra_params containing true values.

    Returns:
        True q_t of shape (N, G).
    """
    if sim_data.extra_params is None or "mu_t" not in sim_data.extra_params:
        raise ValueError(
            "SimulatedData must have extra_params with 'mu_t' and 'pi_t' "
            "to compute true q_t. Regenerate data with save=True."
        )

    pi_t = sim_data.pi_t  # (N, C)
    mu_t = sim_data.extra_params["mu_t"]  # (N, C, G)

    # q_t = sum_c pi_t[:, c] * mu_t[:, c, :]
    q_t_true = np.einsum("nc,ncg->ng", pi_t, mu_t)
    return q_t_true


def ppc_metrics(
    predicted: torch.Tensor | np.ndarray,
    true: torch.Tensor | np.ndarray,
    per_patient: bool = False,
) -> Dict[str, float | np.ndarray]:
    """Compute reconstruction metrics between predicted and true q_t.

    Args:
        predicted: Predicted q_t values, shape (num_samples, N, G) or (N, G).
        true: True q_t values, shape (N, G).
        per_patient: If True, compute per-patient metrics; if False, compute global.

    Returns:
        Dictionary with metrics:
            - mse: Mean squared error
            - mae: Mean absolute error
            - r2: R-squared score
            - pearson_r: Pearson correlation coefficient
            - spearman_r: Spearman correlation coefficient
            If per_patient=True, values are arrays of shape (N,).
    """
    # Convert to numpy and compute mean over samples if needed
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()

    if predicted.ndim == 3:  # (num_samples, N, G)
        predicted_mean = predicted.mean(axis=0)  # (N, G)
    else:  # (N, G)
        predicted_mean = predicted

    if per_patient:
        # Compute metrics per patient
        N = predicted_mean.shape[0]
        metrics = {
            "mse": np.zeros(N),
            "mae": np.zeros(N),
            "r2": np.zeros(N),
            "pearson_r": np.zeros(N),
            "spearman_r": np.zeros(N),
        }

        for i in range(N):
            pred_i = predicted_mean[i]  # (G,)
            true_i = true[i]  # (G,)

            metrics["mse"][i] = mean_squared_error(true_i, pred_i)
            metrics["mae"][i] = mean_absolute_error(true_i, pred_i)
            metrics["r2"][i] = r2_score(true_i, pred_i)
            metrics["pearson_r"][i] = pearsonr(true_i, pred_i)[0]
            metrics["spearman_r"][i] = spearmanr(true_i, pred_i)[0]
    else:
        # Compute global metrics (flatten all values)
        pred_flat = predicted_mean.flatten()
        true_flat = true.flatten()

        metrics = {
            "mse": mean_squared_error(true_flat, pred_flat),
            "mae": mean_absolute_error(true_flat, pred_flat),
            "r2": r2_score(true_flat, pred_flat),
            "pearson_r": pearsonr(true_flat, pred_flat)[0],
            "spearman_r": spearmanr(true_flat, pred_flat)[0],
        }

    return metrics


def evaluate_q_t_reconstruction(
    model: HBRACEModel,
    sim_data: SimulatedData,
    indices: Optional[np.ndarray] = None,
    num_samples: int = 1000,
    device: str = "cpu",
    batch_size: int = 8,
) -> Tuple[Dict[str, float], Dict[str, float], torch.Tensor, np.ndarray]:
    """Evaluate q_t reconstruction via posterior predictive checks.

    Args:
        model: Trained HBRACEModel instance.
        sim_data: SimulatedData object with ground truth.
        indices: Optional patient indices to evaluate (e.g., validation set).
        num_samples: Number of posterior samples.
        device: Device to run evaluation on.

    Returns:
        Tuple of:
            - global_metrics: Dict with global reconstruction metrics
            - per_patient_metrics: Dict with per-patient metrics (arrays)
            - q_t_predicted: Posterior samples of q_t, shape (num_samples, N, G)
            - q_t_true: True q_t values, shape (N, G)
    """
    # Create batch from data
    batch = sim_data.to_patient_batch(device=device, indices=indices)

    # Sample q_t from posterior
    q_t_predicted = sample_q_t(model, batch, num_samples=num_samples, batch_size=batch_size)

    # Compute true q_t
    if indices is None:
        q_t_true = compute_true_q_t(sim_data)
    else:
        # Compute for subset of patients
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
        q_t_true = compute_true_q_t(sim_data_subset)

    # Compute metrics
    global_metrics = ppc_metrics(q_t_predicted, q_t_true, per_patient=False)
    per_patient_metrics = ppc_metrics(q_t_predicted, q_t_true, per_patient=True)

    return global_metrics, per_patient_metrics, q_t_predicted, q_t_true


def sample_responses(
    model: HBRACEModel,
    batch: PatientBatch,
    num_samples: int = 1000,
    batch_size: int = 8,
) -> torch.Tensor:
    """Sample response predictions from posterior.

    Args:
        model: Trained HBRACEModel instance.
        batch: PatientBatch to condition on.
        num_samples: Number of posterior samples.
        batch_size: Batch size for sampling (should match training batch_size).

    Returns:
        Response probabilities of shape (num_samples, N).
    """
    N = batch.pre_counts.shape[0]

    # If batch size matches or is smaller, process directly
    if N <= batch_size:
        # Create batch without responses
        batch_no_y = PatientBatch(
            pre_counts=batch.pre_counts,
            on_counts=batch.on_counts,
            cell_type_proportions=batch.cell_type_proportions,
            responses=None,
            subtype_ids=batch.subtype_ids,
        )

        predictive = Predictive(
            model.model_fn,
            guide=model.guide_fn,
            num_samples=num_samples,
            return_sites=("logit_y",),
            parallel=False,
        )

        with torch.no_grad():
            samples = predictive(batch_no_y)

        # Convert logits to probabilities
        probs = torch.sigmoid(samples["logit_y"])
        return probs

    # Otherwise, process in batches
    all_probs = []

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)

        mini_batch = PatientBatch(
            pre_counts=batch.pre_counts[start_idx:end_idx],
            on_counts=batch.on_counts[start_idx:end_idx],
            cell_type_proportions=batch.cell_type_proportions,
            responses=None,
            subtype_ids=batch.subtype_ids[start_idx:end_idx],
        )

        predictive = Predictive(
            model.model_fn,
            guide=model.guide_fn,
            num_samples=num_samples,
            return_sites=("logit_y",),
            parallel=False,
        )

        with torch.no_grad():
            mini_samples = predictive(mini_batch)

        probs = torch.sigmoid(mini_samples["logit_y"])
        all_probs.append(probs)

    return torch.cat(all_probs, dim=1)  # dim=1 is patient dimension


def evaluate_response_calibration(
    model: HBRACEModel,
    sim_data: SimulatedData,
    indices: Optional[np.ndarray] = None,
    num_samples: int = 1000,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Evaluate response prediction calibration.

    Args:
        model: Trained HBRACEModel instance.
        sim_data: SimulatedData object with ground truth.
        indices: Optional patient indices to evaluate.
        num_samples: Number of posterior samples.
        device: Device to run evaluation on.

    Returns:
        Dictionary with:
            - predicted_probs: Predicted response probabilities, shape (N,)
            - true_probs: True response probabilities from data generation, shape (N,)
            - observed_responses: Observed binary responses, shape (N,)
    """
    batch = sim_data.to_patient_batch(device=device, indices=indices)

    # Sample response probabilities
    probs_samples = sample_responses(model, batch, num_samples=num_samples)
    predicted_probs = probs_samples.mean(dim=0).cpu().numpy()

    # Get observed responses
    observed_responses = batch.responses.cpu().numpy()

    # Get true probabilities if available in extra_params
    true_probs = None
    if sim_data.extra_params is not None:
        # Need to recompute from scratch since it's not stored
        # This requires regenerating the data, which we'll skip for now
        pass

    return {
        "predicted_probs": predicted_probs,
        "observed_responses": observed_responses,
        "true_probs": true_probs,
    }


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


def compute_test_statistic(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
) -> float:
    """Compute a test statistic on data.

    Args:
        data: Data array.
        statistic_fn: Function that takes data and returns a scalar statistic.

    Returns:
        Test statistic value.
    """
    return statistic_fn(data)


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
            - observed_statistic: T(y_obs)
            - replicated_statistics: T(y_rep) for each sample, shape (num_samples,)
            - p_value: Bayesian p-value
            - q_t_observed: Observed q_t (from true parameters)
            - q_t_replicated: Replicated q_t samples, shape (num_samples, N, G)
    """
    # Get true q_t as "observed"
    if indices is None:
        q_t_observed = compute_true_q_t(sim_data)
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
        q_t_observed = compute_true_q_t(sim_data_subset)

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


def posterior_predictive_check_summary_stats(
    model: HBRACEModel,
    sim_data: SimulatedData,
    indices: Optional[np.ndarray] = None,
    num_samples: int = 500,
    device: str = "cpu",
    batch_size: int = 8,
) -> Dict[str, any]:
    """Perform PPC on summary statistics (mean, median, variance per patient/gene).

    This generates the types of plots shown in the Gelman paper where you compare
    observed vs replicated summary statistics.

    Args:
        model: Trained HBRACEModel.
        sim_data: SimulatedData with ground truth.
        indices: Patient indices to evaluate.
        num_samples: Number of posterior predictive samples.
        device: Device for computation.

    Returns:
        Dictionary with observed and replicated summary statistics.
    """
    # Get observed q_t
    if indices is None:
        q_t_observed = compute_true_q_t(sim_data)
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
        q_t_observed = compute_true_q_t(sim_data_subset)

    # Sample q_t from posterior predictive
    batch = sim_data.to_patient_batch(device=device, indices=indices)
    q_t_replicated = sample_q_t(model, batch, num_samples=num_samples, batch_size=batch_size).cpu().numpy()  # (S, N, G)

    # Compute summary statistics
    # Per-patient statistics
    observed_patient_mean = q_t_observed.mean(axis=1)  # (N,)
    observed_patient_median = np.median(q_t_observed, axis=1)  # (N,)
    observed_patient_var = q_t_observed.var(axis=1)  # (N,)

    replicated_patient_mean = q_t_replicated.mean(axis=2)  # (S, N)
    replicated_patient_median = np.median(q_t_replicated, axis=2)  # (S, N)
    replicated_patient_var = q_t_replicated.var(axis=2)  # (S, N)

    # Per-gene statistics
    observed_gene_mean = q_t_observed.mean(axis=0)  # (G,)
    observed_gene_median = np.median(q_t_observed, axis=0)  # (G,)
    observed_gene_var = q_t_observed.var(axis=0)  # (G,)

    replicated_gene_mean = q_t_replicated.mean(axis=1)  # (S, G)
    replicated_gene_median = np.median(q_t_replicated, axis=1)  # (S, G)
    replicated_gene_var = q_t_replicated.var(axis=1)  # (S, G)

    return {
        "observed": {
            "patient_mean": observed_patient_mean,
            "patient_median": observed_patient_median,
            "patient_var": observed_patient_var,
            "gene_mean": observed_gene_mean,
            "gene_median": observed_gene_median,
            "gene_var": observed_gene_var,
        },
        "replicated": {
            "patient_mean": replicated_patient_mean,
            "patient_median": replicated_patient_median,
            "patient_var": replicated_patient_var,
            "gene_mean": replicated_gene_mean,
            "gene_median": replicated_gene_median,
            "gene_var": replicated_gene_var,
        },
    }
