"""
Causal inference evaluation for HBRACE model.

Implements the causal estimands defined in Section 3.3 of the proposal:
- On-treatment composition interventions: ACEt(δt)
- Pre-treatment composition interventions: ACEp(δp)
- Validation against ground truth from synthetic data
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from hbrace.patient_data.types import SimulatedData
from hbrace.patient_data.utils import clr, inv_clr, softmax


class CausalInferenceEvaluator:
    """
    Evaluates causal estimands using posterior samples from the HBRACE model.

    This class implements the causal inference framework from the proposal,
    allowing interventions on cell type compositions and estimation of causal
    effects on patient response.
    """

    def __init__(
        self,
        posterior_samples: Dict[str, torch.Tensor],
        simulated_data: SimulatedData,
        config: Any,  # ModelConfig
        device: str = "cpu",
    ):
        """
        Initialize the CausalInferenceEvaluator.

        Args:
            posterior_samples: Dictionary of posterior samples from trained model.
                Expected keys: 'z', 'u', 'pi_p', 'pi_t', 'mu_t', 'beta_t', 'gamma',
                               'beta_s', 'beta0', 'T', 'W', 'V' (if PoE), etc.
            simulated_data: SimulatedData object containing ground truth.
            config: ModelConfig with model hyperparameters.
            device: Device for tensor computations.
        """
        self.posterior = posterior_samples
        self.sim_data = simulated_data
        self.config = config
        self.device = torch.device(device)

        # Extract ground truth parameters from synthetic data
        self.ground_truth = simulated_data.extra_params if simulated_data.extra_params else {}

        # Move posterior samples to device if needed
        self._move_to_device()

        # Compute number of posterior samples
        self.n_posterior_samples = self._get_n_samples()

        # Store patient data
        self.n_patients = simulated_data.pre_counts.shape[0]
        self.n_cell_types = config.n_cell_types
        self.n_genes = config.n_genes

    def _move_to_device(self) -> None:
        """Move posterior samples to the specified device."""
        for key, value in self.posterior.items():
            if isinstance(value, torch.Tensor):
                self.posterior[key] = value.to(self.device)

    def _get_n_samples(self) -> int:
        """Get the number of posterior samples."""
        # Assume all posterior samples have the same number of samples
        for key, value in self.posterior.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value.shape[0]
        return 1

    def intervene_on_treatment_composition(
        self,
        delta_t: np.ndarray,
        cell_type_idx: Optional[int] = None,
        patient_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply intervention on on-treatment composition as defined in Section 3.2.

        Intervention: π̃_t(δt) = softmax(log π_t + δt)

        Args:
            delta_t: Perturbation vector.
                - If cell_type_idx is provided, this is a scalar or (n_patients,) array
                - Otherwise, this should be (n_cell_types,) or (n_patients, n_cell_types)
            cell_type_idx: If provided, only perturb this specific cell type.
            patient_indices: Subset of patients to intervene on. If None, use all patients.

        Returns:
            Intervened compositions π̃_t of shape (n_posterior_samples, n_patients, n_cell_types)
        """
        if patient_indices is None:
            patient_indices = np.arange(self.n_patients)

        n_patients_subset = len(patient_indices)

        # Get posterior samples of pi_t: shape (n_samples, n_patients, C)
        pi_t = self.posterior['pi_t'].cpu().numpy()  # (S, N, C)
        pi_t_subset = pi_t[:, patient_indices, :]  # (S, n_patients_subset, C)

        # Create delta vector
        if cell_type_idx is not None:
            # Perturb only specific cell type
            delta_vec = np.zeros((n_patients_subset, self.n_cell_types))
            if np.isscalar(delta_t):
                delta_vec[:, cell_type_idx] = delta_t
            else:
                delta_vec[:, cell_type_idx] = delta_t
        else:
            # Perturb all cell types
            if delta_t.ndim == 1:
                # Broadcast to all patients
                delta_vec = np.broadcast_to(delta_t[None, :], (n_patients_subset, self.n_cell_types))
            else:
                delta_vec = delta_t

        # Apply intervention: π̃(δ) = softmax(log π + δ)
        # Shape: (n_samples, n_patients_subset, C)
        pi_t_intervened = np.zeros_like(pi_t_subset)
        for s in tqdm(range(self.n_posterior_samples), desc="Intervening on compositions", leave=False):
            log_pi_t = np.log(np.clip(pi_t_subset[s], 1e-12, 1.0))
            pi_t_intervened[s] = softmax(log_pi_t + delta_vec, axis=-1)

        return pi_t_intervened

    def intervene_on_pretreatment_composition(
        self,
        delta_p: np.ndarray,
        cell_type_idx: Optional[int] = None,
        patient_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply intervention on pre-treatment composition and propagate through model.

        Intervention: π̃_p(δp) = softmax(log π_p + δp)
        Then propagate: η_p(δp) -> η_t(δp) -> π_t(δp)

        Args:
            delta_p: Perturbation vector (see intervene_on_treatment_composition).
            cell_type_idx: If provided, only perturb this specific cell type.
            patient_indices: Subset of patients to intervene on.

        Returns:
            Tuple of:
                - Intervened pre-treatment compositions π̃_p: (n_samples, n_patients, C)
                - Resulting on-treatment compositions π_t(δp): (n_samples, n_patients, C)
        """
        if patient_indices is None:
            patient_indices = np.arange(self.n_patients)

        n_patients_subset = len(patient_indices)

        # Get posterior samples
        pi_p = self.posterior['pi_p'].cpu().numpy()[:, patient_indices, :]  # (S, N, C)
        z = self.posterior['z'].cpu().numpy()[:, patient_indices, :]  # (S, N, d_z)
        T = self.posterior['T'].cpu().numpy()  # (S, C, C)
        W = self.posterior['W'].cpu().numpy()  # (S, C, d_z)
        epsilon_samples = self.posterior.get('epsilon')  # (S, N, C) or None

        if epsilon_samples is not None:
            epsilon = epsilon_samples.cpu().numpy()[:, patient_indices, :]
        else:
            epsilon = np.zeros((self.n_posterior_samples, n_patients_subset, self.n_cell_types))

        # Handle PoE model
        if self.config.composition_model == "poe":
            u = self.posterior['u'].cpu().numpy()[:, patient_indices, :]  # (S, N, r_u)
            V = self.posterior.get('V')  # (S, C, r_u) or None
            if V is not None:
                V = V.cpu().numpy()

        # Create delta vector
        if cell_type_idx is not None:
            delta_vec = np.zeros((n_patients_subset, self.n_cell_types))
            if np.isscalar(delta_p):
                delta_vec[:, cell_type_idx] = delta_p
            else:
                delta_vec[:, cell_type_idx] = delta_p
        else:
            if delta_p.ndim == 1:
                delta_vec = np.broadcast_to(delta_p[None, :], (n_patients_subset, self.n_cell_types))
            else:
                delta_vec = delta_p

        # Apply intervention on pi_p
        pi_p_intervened = np.zeros_like(pi_p)
        pi_t_result = np.zeros_like(pi_p)

        for s in tqdm(range(self.n_posterior_samples), desc="Propagating pre-treatment intervention", leave=False):
            # Intervene on pre-treatment composition
            log_pi_p = np.log(np.clip(pi_p[s], 1e-12, 1.0))
            pi_p_intervened[s] = softmax(log_pi_p + delta_vec, axis=-1)

            # Propagate through composition transformation
            eta_p_intervened = clr(pi_p_intervened[s])  # (N, C)

            if self.config.composition_model == "linear":
                # Linear model: η_t = η_p + z @ (T @ W)^T + ε
                TW = T[s] @ W[s].T  # (C, C) @ (C, d_z)^T = (C, d_z)
                eta_shift = z[s] @ TW.T  # (N, d_z) @ (d_z, C) = (N, C)
                eta_t = eta_p_intervened + eta_shift + epsilon[s]
                pi_t_result[s] = inv_clr(eta_t)

            elif self.config.composition_model == "poe":
                # Product of Experts model
                # Expert 1: treatment effect
                TW = T[s] @ W[s].T
                eta_z = z[s] @ TW.T
                g_z = inv_clr(eta_z)

                # Expert 2: confounder effect
                eta_u = u[s] @ V[s].T  # (N, r_u) @ (r_u, C) = (N, C)
                h_u = inv_clr(eta_u)

                # Product of experts
                pi_t_unnorm = pi_p_intervened[s] * g_z * h_u
                pi_t_clean = pi_t_unnorm / pi_t_unnorm.sum(axis=-1, keepdims=True)

                # Add noise in CLR space
                eta_t_clean = clr(pi_t_clean)
                eta_t = eta_t_clean + epsilon[s]
                pi_t_result[s] = inv_clr(eta_t)

        return pi_p_intervened, pi_t_result

    def predict_response(
        self,
        pi_t: np.ndarray,
        patient_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict patient response given on-treatment composition.

        Model: y ~ Bernoulli(p) where σ(p) = β_0 + β_t^T q_t + γ^T u + β_s

        Args:
            pi_t: On-treatment compositions of shape (n_samples, n_patients, C)
            patient_indices: Subset of patients. If None, use all patients.

        Returns:
            Response probabilities of shape (n_samples, n_patients)
        """
        if patient_indices is None:
            patient_indices = np.arange(self.n_patients)

        n_patients_subset = len(patient_indices)

        # Get posterior samples
        mu_t = self.posterior['mu_t'].cpu().numpy()[:, patient_indices, :, :]  # (S, N, C, G)
        beta_t = self.posterior['beta_t'].cpu().numpy()  # (S, G)

        # Ensure beta_t has correct shape (S, G) by squeezing out any singleton dims
        while beta_t.ndim > 2 and beta_t.shape[1] == 1:
            beta_t = np.squeeze(beta_t, axis=1)

        gamma = self.posterior['gamma'].cpu().numpy()  # (S, r_u)
        beta_s = self.posterior['beta_s'].cpu().numpy()  # (S, n_subtypes)
        beta0 = self.posterior['beta0'].cpu().numpy()  # (S,) or scalar
        u = self.posterior['u'].cpu().numpy()[:, patient_indices, :]  # (S, N, r_u)

        # Get subtype IDs
        subtype_ids = self.sim_data.subtype_ids[patient_indices]

        # Compute q_t_mean = sum_c π_t[c] * μ_t[c]
        # pi_t: (S, N, C), mu_t: (S, N, C, G) -> q_t_mean: (S, N, G)
        q_t_mean = np.einsum('snc,sncg->sng', pi_t, mu_t)

        # Scale inputs
        q_t_head = q_t_mean * self.config.head_input_scale
        u_head = u * self.config.head_input_scale

        # Compute linear predictor
        # (S, N, G) * (S, 1, G) -> (S, N)
        beta_t_expanded = beta_t[:, None, :]
        term1 = (q_t_head * beta_t_expanded).sum(axis=-1)

        # (S, N, r_u) * (S, 1, r_u) -> (S, N)
        gamma_expanded = gamma[:, None, :]
        term2 = (u_head * gamma_expanded).sum(axis=-1)

        # Subtype-specific effects
        beta_s_indexed = beta_s[:, subtype_ids]  # (S, N)

        # Combine terms
        if beta0.ndim == 0:
            beta0_expanded = beta0
        else:
            beta0_expanded = beta0[:, None]

        linear = self.config.logit_scale * (term1 + term2 + beta_s_indexed)
        logit_y = beta0_expanded + linear

        # Convert to probability
        p_y = 1.0 / (1.0 + np.exp(-logit_y))

        return p_y

    def estimate_ACE_on_treatment(
        self,
        delta_t: float,
        cell_type_idx: int,
        subtype: Optional[int] = None,
        patient_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Estimate ACEt(δt) as defined in Section 3.3 of the proposal.

        ACEt(δt) = E[Y_i(δt) - Y_i(0)]

        Args:
            delta_t: Perturbation magnitude for the specified cell type.
            cell_type_idx: Which cell type to perturb.
            subtype: If provided, condition on this subtype (subtype-specific ACE).
            patient_indices: Subset of patients to evaluate on.

        Returns:
            Dictionary with:
                - 'ace_mean': Point estimate of ACE
                - 'ace_std': Standard error of ACE
                - 'ace_ci_lower': Lower bound of 95% credible interval
                - 'ace_ci_upper': Upper bound of 95% credible interval
        """
        # Filter by subtype if specified
        if patient_indices is None:
            if subtype is not None:
                patient_indices = np.where(self.sim_data.subtype_ids == subtype)[0]
            else:
                patient_indices = np.arange(self.n_patients)

        # Factual (δ=0): use observed pi_t
        pi_t_factual = self.posterior['pi_t'].cpu().numpy()[:, patient_indices, :]
        p_y_factual = self.predict_response(pi_t_factual, patient_indices)

        # Counterfactual (δ=δt): intervene on composition
        pi_t_counterfactual = self.intervene_on_treatment_composition(
            delta_t, cell_type_idx, patient_indices
        )
        p_y_counterfactual = self.predict_response(pi_t_counterfactual, patient_indices)

        # Compute ITE for each posterior sample and patient
        ite = p_y_counterfactual - p_y_factual  # (n_samples, n_patients)

        # Average across patients to get ACE for each posterior sample
        ace_samples = ite.mean(axis=1)  # (n_samples,)

        # Compute statistics
        ace_mean = ace_samples.mean()
        ace_std = ace_samples.std()
        ace_ci_lower = np.percentile(ace_samples, 2.5)
        ace_ci_upper = np.percentile(ace_samples, 97.5)

        return {
            'ace_mean': ace_mean,
            'ace_std': ace_std,
            'ace_ci_lower': ace_ci_lower,
            'ace_ci_upper': ace_ci_upper,
            'ace_samples': ace_samples,
            'ite': ite,
        }

    def estimate_ACE_pre_treatment(
        self,
        delta_p: float,
        cell_type_idx: int,
        subtype: Optional[int] = None,
        patient_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Estimate ACEp(δp) as defined in Section 3.3 of the proposal.

        ACEp(δp) = E[Y_i^p(δp) - Y_i^p(0)]

        This propagates the pre-treatment intervention through the full causal chain:
        π_p(δp) -> η_p(δp) -> η_t(δp) -> π_t(δp) -> q_t(δp) -> y(δp)

        Args:
            delta_p: Perturbation magnitude for the specified cell type.
            cell_type_idx: Which cell type to perturb.
            subtype: If provided, condition on this subtype.
            patient_indices: Subset of patients to evaluate on.

        Returns:
            Dictionary with ACE statistics (see estimate_ACE_on_treatment).
        """
        # Filter by subtype if specified
        if patient_indices is None:
            if subtype is not None:
                patient_indices = np.where(self.sim_data.subtype_ids == subtype)[0]
            else:
                patient_indices = np.arange(self.n_patients)

        # Factual (δ=0): use observed pi_t
        pi_t_factual = self.posterior['pi_t'].cpu().numpy()[:, patient_indices, :]
        p_y_factual = self.predict_response(pi_t_factual, patient_indices)

        # Counterfactual (δ=δp): intervene on pre-treatment and propagate
        _, pi_t_counterfactual = self.intervene_on_pretreatment_composition(
            delta_p, cell_type_idx, patient_indices
        )
        p_y_counterfactual = self.predict_response(pi_t_counterfactual, patient_indices)

        # Compute ITE and ACE
        ite = p_y_counterfactual - p_y_factual
        ace_samples = ite.mean(axis=1)

        return {
            'ace_mean': ace_samples.mean(),
            'ace_std': ace_samples.std(),
            'ace_ci_lower': np.percentile(ace_samples, 2.5),
            'ace_ci_upper': np.percentile(ace_samples, 97.5),
            'ace_samples': ace_samples,
            'ite': ite,
        }

    def generate_ground_truth_ACE_on_treatment(
        self,
        delta_t: float,
        cell_type_idx: int,
        subtype: Optional[int] = None,
        n_samples: int = 1000,
    ) -> Dict[str, float]:
        """
        Generate ground truth ACEt(δt) using the synthetic data generator parameters.

        This uses the true parameters (beta_t, gamma, T, W, etc.) from extra_params
        to compute the true causal effect.

        Args:
            delta_t: Perturbation magnitude.
            cell_type_idx: Which cell type to perturb.
            subtype: If provided, condition on this subtype.
            n_samples: Number of Monte Carlo samples for expectation.

        Returns:
            Dictionary with ground truth ACE statistics.
        """
        if not self.ground_truth:
            raise ValueError("Ground truth parameters not available in simulated_data.extra_params")

        # Filter by subtype if specified
        if subtype is not None:
            patient_mask = self.sim_data.subtype_ids == subtype
        else:
            patient_mask = np.ones(self.n_patients, dtype=bool)

        patient_indices = np.where(patient_mask)[0]
        n_patients_subset = len(patient_indices)

        # Extract ground truth parameters
        beta_t_true = self.ground_truth['beta_t']  # (G,)
        gamma_true = self.ground_truth['gamma']  # (r_u,)
        beta_s_true = self.ground_truth['beta_s']  # (n_subtypes,)
        beta0_true = self.ground_truth['beta0']  # scalar

        # Get ground truth latent variables for these patients
        z_true = self.ground_truth['z'][patient_indices]  # (N, d_z)
        u_true = self.ground_truth['u'][patient_indices]  # (N, r_u)
        pi_t_true = self.sim_data.pi_t[patient_indices]  # (N, C)
        mu_t_true = self.ground_truth.get('mu_t')  # Might need to recompute

        if mu_t_true is not None:
            mu_t_true = mu_t_true[patient_indices]  # (N, C, G)
        else:
            # Recompute mu_t from mu_p and Delta if available
            # This is a simplified version; full computation may be needed
            raise NotImplementedError("mu_t reconstruction not yet implemented")

        subtype_ids = self.sim_data.subtype_ids[patient_indices]

        # Factual: compute response with observed pi_t
        q_t_factual = np.einsum('nc,ncg->ng', pi_t_true, mu_t_true)  # (N, G)
        q_t_head_factual = q_t_factual * self.config.head_input_scale
        u_head = u_true * self.config.head_input_scale

        linear_factual = self.config.logit_scale * (
            (q_t_head_factual * beta_t_true).sum(axis=-1) +
            (u_head * gamma_true).sum(axis=-1) +
            beta_s_true[subtype_ids]
        )
        logit_y_factual = beta0_true + linear_factual
        p_y_factual = 1.0 / (1.0 + np.exp(-logit_y_factual))

        # Counterfactual: intervene on pi_t
        delta_vec = np.zeros((n_patients_subset, self.n_cell_types))
        delta_vec[:, cell_type_idx] = delta_t

        log_pi_t = np.log(np.clip(pi_t_true, 1e-12, 1.0))
        pi_t_intervened = softmax(log_pi_t + delta_vec, axis=-1)

        q_t_counterfactual = np.einsum('nc,ncg->ng', pi_t_intervened, mu_t_true)
        q_t_head_counterfactual = q_t_counterfactual * self.config.head_input_scale

        linear_counterfactual = self.config.logit_scale * (
            (q_t_head_counterfactual * beta_t_true).sum(axis=-1) +
            (u_head * gamma_true).sum(axis=-1) +
            beta_s_true[subtype_ids]
        )
        logit_y_counterfactual = beta0_true + linear_counterfactual
        p_y_counterfactual = 1.0 / (1.0 + np.exp(-logit_y_counterfactual))

        # Compute ITE and ACE
        ite_true = p_y_counterfactual - p_y_factual
        ace_true = ite_true.mean()

        return {
            'ace_true': ace_true,
            'ite_true': ite_true,
        }

    def validate_against_ground_truth(
        self,
        delta_grid: np.ndarray,
        cell_type_idx: int,
        subtype: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate estimated ACE against ground truth across a grid of interventions.

        Args:
            delta_grid: Array of intervention magnitudes to test.
            cell_type_idx: Which cell type to perturb.
            subtype: If provided, condition on this subtype.

        Returns:
            Dictionary with validation metrics:
                - 'delta_grid': The intervention grid
                - 'ace_estimated': Estimated ACE for each delta
                - 'ace_true': True ACE for each delta
                - 'bias': Bias for each delta
                - 'mse': Mean squared error
                - 'coverage': Proportion of 95% CIs containing truth
                - 'correlation': Correlation between estimated and true ACE
        """
        n_deltas = len(delta_grid)

        ace_estimated = np.zeros(n_deltas)
        ace_true = np.zeros(n_deltas)
        ace_ci_lower = np.zeros(n_deltas)
        ace_ci_upper = np.zeros(n_deltas)

        for i, delta in enumerate(tqdm(delta_grid, desc="Validating interventions")):
            # Estimated ACE
            result_est = self.estimate_ACE_on_treatment(
                delta, cell_type_idx, subtype
            )
            ace_estimated[i] = result_est['ace_mean']
            ace_ci_lower[i] = result_est['ace_ci_lower']
            ace_ci_upper[i] = result_est['ace_ci_upper']

            # Ground truth ACE
            try:
                result_true = self.generate_ground_truth_ACE_on_treatment(
                    delta, cell_type_idx, subtype
                )
                ace_true[i] = result_true['ace_true']
            except (ValueError, NotImplementedError):
                # Ground truth not available
                ace_true[i] = np.nan

        # Compute metrics
        bias = ace_estimated - ace_true
        mse = np.nanmean(bias ** 2)

        # Coverage: proportion of times 95% CI contains truth
        coverage = np.nanmean(
            (ace_true >= ace_ci_lower) & (ace_true <= ace_ci_upper)
        )

        # Correlation
        valid_mask = ~np.isnan(ace_true)
        if valid_mask.sum() > 1:
            correlation = np.corrcoef(ace_estimated[valid_mask], ace_true[valid_mask])[0, 1]
        else:
            correlation = np.nan

        return {
            'delta_grid': delta_grid,
            'ace_estimated': ace_estimated,
            'ace_true': ace_true,
            'ace_ci_lower': ace_ci_lower,
            'ace_ci_upper': ace_ci_upper,
            'bias': bias,
            'mse': mse,
            'coverage': coverage,
            'correlation': correlation,
        }

    def check_exclusion_restriction(
        self,
        n_permutations: int = 100,
    ) -> Dict[str, float]:
        """
        Test Assumption 4: Exclusion restriction y ⊥ z | (q_t, s).

        Tests whether z has a direct effect on y beyond its effect through q_t.

        Args:
            n_permutations: Number of permutation tests.

        Returns:
            Dictionary with test results:
                - 'correlation': Correlation between z and residuals
                - 'p_value': P-value from permutation test
                - 'passes': Whether the assumption appears to hold
        """
        # Get posterior samples
        z = self.posterior['z'].cpu().numpy()  # (S, N, d_z)

        # Compute residuals: y - E[y | q_t, s]
        pi_t = self.posterior['pi_t'].cpu().numpy()
        p_y = self.predict_response(pi_t)  # (S, N)

        y_obs = self.sim_data.responses  # (N,)

        # For each posterior sample, compute correlation between z and residuals
        correlations = []
        for s in range(self.n_posterior_samples):
            residuals = y_obs - p_y[s]

            # Compute correlation between each dimension of z and residuals
            for d in range(z.shape[-1]):
                corr = np.corrcoef(z[s, :, d], residuals)[0, 1]
                correlations.append(np.abs(corr))

        observed_corr = np.mean(correlations)

        # Permutation test
        null_correlations = []
        rng = np.random.default_rng(42)

        for _ in tqdm(range(n_permutations), desc="Permutation test"):
            # Permute z
            z_perm = z.copy()
            for s in range(self.n_posterior_samples):
                perm_idx = rng.permutation(self.n_patients)
                z_perm[s] = z_perm[s, perm_idx, :]

            # Compute correlation with permuted z
            perm_corrs = []
            for s in range(self.n_posterior_samples):
                residuals = y_obs - p_y[s]
                for d in range(z_perm.shape[-1]):
                    corr = np.corrcoef(z_perm[s, :, d], residuals)[0, 1]
                    perm_corrs.append(np.abs(corr))

            null_correlations.append(np.mean(perm_corrs))

        # P-value
        p_value = np.mean(np.array(null_correlations) >= observed_corr)

        return {
            'correlation': observed_corr,
            'p_value': p_value,
            'passes': p_value > 0.05,  # Conventional threshold
        }

    def check_no_direct_effect_pretreatment(
        self,
        n_bootstrap: int = 100,
    ) -> Dict[str, float]:
        """
        Test Assumption 6: No direct effect of q_p on y given q_t.

        Tests whether pre-treatment composition affects response only through
        on-treatment composition.

        Args:
            n_bootstrap: Number of bootstrap samples.

        Returns:
            Dictionary with test results.
        """
        # Get posterior samples
        pi_p = self.posterior['pi_p'].cpu().numpy()  # (S, N, C)
        pi_t = self.posterior['pi_t'].cpu().numpy()  # (S, N, C)

        y_obs = self.sim_data.responses

        # For each posterior sample, regress y on q_p controlling for q_t
        # If assumption holds, q_p should have no additional predictive power

        direct_effects = []

        n_samples_to_test = min(self.n_posterior_samples, 10)  # Sample subset for speed
        for s in tqdm(range(n_samples_to_test), desc="Testing pre-treatment direct effect"):
            # Simple test: correlation between pi_p and residuals after accounting for pi_t
            p_y = self.predict_response(pi_t[s:s+1])  # (1, N)
            residuals = y_obs - p_y[0]

            # Compute correlation between each cell type in pi_p and residuals
            for c in range(self.n_cell_types):
                corr = np.corrcoef(pi_p[s, :, c], residuals)[0, 1]
                direct_effects.append(np.abs(corr))

        mean_direct_effect = np.mean(direct_effects)

        return {
            'mean_direct_effect': mean_direct_effect,
            'passes': mean_direct_effect < 0.1,  # Threshold for "small" effect
        }

    def compute_dose_response_curve(
        self,
        cell_type_idx: int,
        delta_range: Tuple[float, float] = (-2.0, 2.0),
        n_points: int = 20,
        subtype: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute dose-response curve: ACE as a function of intervention magnitude.

        Args:
            cell_type_idx: Which cell type to perturb.
            delta_range: Range of intervention magnitudes (min, max).
            n_points: Number of points on the curve.
            subtype: If provided, condition on this subtype.

        Returns:
            Dictionary with:
                - 'delta': Intervention magnitudes
                - 'ace_mean': Mean ACE at each magnitude
                - 'ace_ci_lower': Lower 95% CI
                - 'ace_ci_upper': Upper 95% CI
        """
        delta_grid = np.linspace(delta_range[0], delta_range[1], n_points)

        ace_mean = np.zeros(n_points)
        ace_ci_lower = np.zeros(n_points)
        ace_ci_upper = np.zeros(n_points)

        for i, delta in enumerate(tqdm(delta_grid, desc="Computing dose-response")):
            result = self.estimate_ACE_on_treatment(delta, cell_type_idx, subtype)
            ace_mean[i] = result['ace_mean']
            ace_ci_lower[i] = result['ace_ci_lower']
            ace_ci_upper[i] = result['ace_ci_upper']

        return {
            'delta': delta_grid,
            'ace_mean': ace_mean,
            'ace_ci_lower': ace_ci_lower,
            'ace_ci_upper': ace_ci_upper,
        }
