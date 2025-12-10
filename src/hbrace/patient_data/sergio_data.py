from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from hbrace.config import ModelConfig
from hbrace.patient_data.types import PatientBatch
from hbrace.patient_data.utils import sample_nb, clr, inv_clr


class SemiSyntheticDataGenerator:
    """
    Generate semi-synthetic immunotherapy data with known ground truth.

    This generator creates data following the hierarchical causal model structure,
    allowing validation with known treatment effects (z_i).
    """

    def __init__(
        self,
        n_patients: int,
        n_genes: int = 30,
        n_cell_types: int = 8,
        seed: int | None = None,
    ):
        self.n_patients = n_patients
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(self.seed)

    def generate_batch(
        self,
        model_config: ModelConfig,
        device: torch.device | str = "cpu",
    ) -> Tuple[PatientBatch, Dict[str, Any]]:
        """
        Generate a batch of patients with known ground truth causal effects.

        Returns:
            batch: PatientBatch with pre/on counts, responses, subtypes
            ground_truth: Dict with true latent variables (z, u, pi_p, pi_t, etc.)
        """
        N = self.n_patients
        C = self.n_cell_types
        G = self.n_genes
        S = model_config.n_subtypes
        d = model_config.latent_dim

        # Generate ground truth latent variables
        z_true = self.rng.normal(0, 1, size=(N, d))
        u_true = self.rng.normal(0, 1, size=(N, d))

        # Subtype assignments
        subtype_ids = self.rng.integers(0, S, size=N)

        # Subtype-specific cell type priors
        theta = np.zeros((S, C))
        for s in range(S):
            bump = np.zeros(C)
            bump[s % C] = 3.0
            theta[s] = self.rng.dirichlet(2.0 * np.ones(C) + bump)

        # Pre-treatment cell type proportions
        tau = self.rng.gamma(2.0, 0.1, size=N)
        pi_p = np.zeros((N, C))
        for i in range(N):
            pi_p[i] = self.rng.dirichlet(tau[i] * theta[subtype_ids[i]])

        # On-treatment proportions (shifted by treatment effect z)
        W_P = self.rng.normal(0, 0.3, size=(C, d))

        # Simple transition matrix T (sparse)
        adjacency = self.rng.random((C, C)) < 0.3
        np.fill_diagonal(adjacency, 0)
        lambda_h = np.abs(self.rng.normal(0, 0.2))
        lambda_l = np.abs(self.rng.normal(0, 0.05))
        scale_T = np.where(adjacency, lambda_l, lambda_h)
        T = self.rng.choice([-1.0, 1.0], size=(C, C)) * self.rng.exponential(scale=scale_T)

        eta_p = clr(pi_p)
        eps = self.rng.normal(0, 0.1, size=(N, C))
        eta_t = eta_p + (z_true @ (T @ W_P).T) + eps
        pi_t = inv_clr(eta_t)

        # Base NB parameters for pre-treatment
        log_mu_p = self.rng.normal(1.5, 0.8, size=(C, G))
        mu_p = np.exp(log_mu_p)
        phi_p = np.exp(self.rng.normal(1.0, 0.5, size=C))

        # On-treatment NB parameters (shifted by z)
        Delta = self.rng.normal(0, 0.5, size=(C, G, d))
        tau_c = np.abs(self.rng.normal(0, 0.5, size=C)) + 1e-3
        delta_ic = self.rng.normal(0, tau_c[None, :], size=(N, C))

        dot_Dz = np.tensordot(z_true, Delta, axes=([1], [2]))  # (N, C, G)
        mu_t = np.exp(log_mu_p[None, :, :] + dot_Dz)
        phi_t = phi_p[None, :] * np.exp(delta_ic)

        # Generate aggregated counts (simulating many cells per type)
        pre_counts = np.zeros((N, C, G), dtype=np.int64)
        on_counts = np.zeros((N, C, G), dtype=np.int64)

        for i in range(N):
            for c in range(C):
                # Sample counts for this cell type
                pre_counts[i, c] = sample_nb(mu_p[c], np.repeat(phi_p[c], G), self.rng)
                on_counts[i, c] = sample_nb(mu_t[i, c], np.repeat(phi_t[i, c], G), self.rng)

        # Generate patient responses (clonal expansion)
        beta0 = self.rng.normal(0, 0.5)
        beta_t = self.rng.normal(0, 0.5, size=C)
        gamma = self.rng.normal(0, 0.5, size=d)
        beta_s = self.rng.normal(0, 0.5, size=S)

        linear = (
            beta0
            + (pi_t * beta_t[None, :]).sum(axis=1)
            + (u_true * gamma[None, :]).sum(axis=1)
            + beta_s[subtype_ids]
        )
        prob = 1.0 / (1.0 + np.exp(-linear))
        responses = self.rng.binomial(1, prob, size=N).astype(np.float32)

        # Convert to PatientBatch
        batch = PatientBatch(
            pre_counts=torch.tensor(pre_counts, dtype=torch.float32, device=device),
            on_counts=torch.tensor(on_counts, dtype=torch.float32, device=device),
            responses=torch.tensor(responses, dtype=torch.float32, device=device),
            subtype_ids=torch.tensor(subtype_ids, dtype=torch.long, device=device),
        )

        # Ground truth for validation
        ground_truth = {
            "z": z_true,
            "u": u_true,
            "pi_p": pi_p,
            "pi_t": pi_t,
            "theta": theta,
            "tau": tau,
            "T": T,
            "W_P": W_P,
            "Delta": Delta,
            "mu_p": mu_p,
            "phi_p": phi_p,
            "mu_t": mu_t,
            "phi_t": phi_t,
            "beta0": beta0,
            "beta_t": beta_t,
            "gamma": gamma,
            "beta_s": beta_s,
            "responses": responses,
        }

        return batch, ground_truth
