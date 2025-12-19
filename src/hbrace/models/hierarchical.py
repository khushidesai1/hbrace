from __future__ import annotations

import torch
from pyro import deterministic, factor, param, plate, sample
from pyro.distributions import (
    Beta,
    Bernoulli,
    Dirichlet,
    Gamma,
    HalfCauchy,
    Laplace,
    NegativeBinomial,
    Normal,
)
from torch.distributions import constraints

from hbrace.config import ModelConfig
from hbrace.patient_data import PatientBatch
from .utils import nb_logits


def _clr(pi: torch.Tensor) -> torch.Tensor:
    pi = torch.clamp(pi, min=1e-8)
    log_pi = torch.log(pi)
    return log_pi - log_pi.mean(dim=-1, keepdim=True)


def _inv_clr(eta: torch.Tensor) -> torch.Tensor:
    ex = torch.exp(eta)
    return ex / ex.sum(dim=-1, keepdim=True)


def hierarchical_model(batch: PatientBatch, config: ModelConfig) -> None:
    """
    Collapsed flat model following the proposal; patient dimension is batched via params.
    """
    
    device = batch.pre_counts.device
    n_patients = batch.pre_counts.shape[0]

    C = config.n_cell_types
    G = config.n_genes
    d_z = config.z_dim
    r_u = config.u_dim

    # Mixing weights for pre-treatment cell distributions.
    theta = param(
        "theta",
        batch.cell_type_proportions,
        constraint=constraints.simplex,
    )
    theta_expanded = theta[batch.subtype_ids]

    # Mean and dispersion shifts for on-treatment distributions.
    if config.gene_sparsity:
        # Sparsity version: 5x wider prior (mean=5.0, up from 1.0)
        Delta_std = sample(
            "Delta_std",
            Gamma(
                torch.full((C, G, d_z), 2.0, device=device),
                torch.full((C, G, d_z), 0.4, device=device),  # rate=0.4, mean=5.0 (5x increase)
            ).to_event(3),
        )
    else:
        # Original version: looser prior
        Delta_std = sample(
            "Delta_std",
            Gamma(
                torch.full((C, G, d_z), 2.0, device=device),
                torch.full((C, G, d_z), 2.0, device=device),  # rate=2.0, mean=1.0
            ).to_event(3),
        )
    Delta = sample("Delta", Normal(torch.zeros((C, G, d_z), device=device), Delta_std).to_event(3))
    delta_std = sample(
        "delta_std",
        Gamma(
            torch.full((C,), 2.0, device=device),  # shape
            torch.full((C,), 4.0, device=device),  # rate -> mean 0.5
        ).to_event(1),
    )

    # Cell-type proportion shifts for on-treatment mixture weights.
    W = sample(
        "W",
        Normal(
            torch.zeros((C, d_z), device=device),
            torch.full((C, d_z), 10.0, device=device),  # 5x wider: 10.0 (up from 2.0)
        ).to_event(2),
    )
    epsilon_std = sample(
        "epsilon_std",
        Gamma(
            torch.full((C,), 2.0, device=device),  # shape
            torch.full((C,), 2.0, device=device),  # rate=2.0 -> mean 1.0 (5x wider, up from 0.2)
        ).to_event(1),
    )

    lambda_T = sample(
        "lambda_T",
        Beta(torch.tensor(3.0, device=device), torch.tensor(4.0, device=device)),  # Moderate mean ~0.43
    )
    T = sample(
        "T",
        Laplace(torch.zeros((C, C), device=device), lambda_T).to_event(2),
    )

    beta0 = sample(
        "beta0",
        Normal(
            torch.tensor(config.beta0_loc, device=device),
            torch.tensor(config.beta0_scale, device=device),
        ),
    )
    if config.gene_sparsity:
        # Hierarchical horseshoe-like prior for sparsity with heavy tails
        # β_t[g] ~ Normal(0, τ * λ_g)
        # λ_g ~ HalfCauchy(1)  (gene-specific scale)
        # τ   ~ HalfCauchy(1)  (global scale)
        tau = sample("beta_t_tau", HalfCauchy(torch.ones((), device=device)))
        lambda_g = sample(
            "beta_t_lambda_g",
            HalfCauchy(torch.ones((G,), device=device)).to_event(1),
        )
        beta_t = sample(
            "beta_t",
            Normal(
                torch.zeros((G,), device=device),
                tau * lambda_g,
            ).to_event(1),
        )
    else:
        # Original version: Normal prior (no sparsity)
        beta_t = sample(
            "beta_t",
            Normal(
                torch.zeros((G,), device=device),
                torch.full((G,), 2.0, device=device),
            ).to_event(1),
        )
    gamma = sample(
        "gamma",
        Normal(torch.zeros((r_u,), device=device), torch.full((r_u,), config.gamma_scale, device=device)).to_event(1),
    )
    beta_s = sample(
        "beta_s",
        Normal(
            torch.zeros((config.n_subtypes,), device=device),
            torch.full((config.n_subtypes,), config.beta_s_scale, device=device),
        ).to_event(1),
    )

    # Sample V matrix for PoE model (must be outside patient plate!)
    if config.composition_model == "poe":
        if config.gene_sparsity:
            V = sample(
                "V",
                Normal(
                    torch.zeros((C, r_u), device=device),
                    torch.full((C, r_u), 1.0, device=device),  # Moderate increase from 0.5 for stability
                ).to_event(2),
            )
        else:
            V_std = sample(
                "V_std",
                Gamma(
                    torch.full((C, r_u), 2.0, device=device),
                    torch.full((C, r_u), 2.0, device=device),  # rate=2.0, mean=1.0
                ).to_event(2),
            )
            V = sample(
                "V",
                Normal(torch.zeros((C, r_u), device=device), V_std).to_event(2),
            )

    # Patient-level plate.
    with plate("patients", n_patients):
        tau_i_p = sample(
            "tau_i_p",
            Gamma(torch.tensor(2.0, device=device), torch.tensor(1.0, device=device)),  # rate=1.0 -> mean=2.0 (was rate=5.0 -> mean=0.4)
        )
        pi_p = sample(
            "pi_p",
            Dirichlet(tau_i_p.unsqueeze(-1) * theta_expanded.clamp_min(1e-6)),
        )

        if config.gene_sparsity:
            # Sparsity version: tighter prior
            log_mu_p = sample(
                "log_mu_p",
                Normal(
                    torch.full((C, G), 1.0, device=device),
                    torch.full((C, G), 0.5, device=device),
                ).to_event(2),
            )
        else:
            # Original version: looser prior
            log_mu_p = sample(
                "log_mu_p",
                Normal(
                    torch.full((C, G), 1.5, device=device),
                    torch.full((C, G), 0.8, device=device),
                ).to_event(2),
            )
        mu_p = torch.exp(log_mu_p)

        phi_p_std = sample(
            "phi_p_std",
            Gamma(
                torch.full((C, G), 2.0, device=device),
                torch.full((C, G), 2.0, device=device),
            ).to_event(2),
        )
        phi_p = sample(
            "phi_p",
            Gamma(
                phi_p_std,
                torch.ones_like(phi_p_std),
            ).to_event(2),
        )

        # Scale dispersion by number of cells summed
        m_pre = batch.pre_ncells  # (N, C)
        m_pre_expand = m_pre.unsqueeze(-1)  # (N, C, 1)
        phi_p_sum = phi_p * m_pre_expand.clamp_min(1.0)  # (N, C, G)

        logits_p = nb_logits(mu_p, phi_p)
        f_p = sample(
            "f_p",
            NegativeBinomial(
                total_count=phi_p_sum,
                logits=logits_p,
            ).to_event(2),
            obs=batch.pre_counts,
        )

        z = sample(
            "z",
            Normal(torch.zeros(d_z, device=device), torch.ones(d_z, device=device)).to_event(1),
        )
        # Allow predictive sample dimensions to broadcast through all latent sites.
        log_mu_t_i = log_mu_p + torch.einsum("...nd,...cgd->...ncg", z, Delta)
        mu_t_i = deterministic("mu_t", torch.exp(log_mu_t_i))
        delta = sample(
            "delta",
            Normal(torch.zeros(C, device=device), delta_std).to_event(1),
        )
        delta_expanded = delta.unsqueeze(-1).expand(phi_p.shape)
        phi_t = deterministic("phi_t", phi_p * torch.exp(delta_expanded))

        epsilon = sample(
            "epsilon",
            Normal(torch.zeros(C, device=device), epsilon_std).to_event(1),
        )

        # Sample u early (needed for PoE composition model)
        u = sample(
            "u",
            Normal(torch.zeros(r_u, device=device), torch.ones(r_u, device=device)).to_event(1),
        )

        eta_p = _clr(pi_p)

        # Composition shift: linear vs. product of experts (PoE)
        if config.composition_model == "linear":
            # Linear model: eta^t = eta^p + z @ (T @ W)^T + epsilon
            TW = torch.einsum("...ct,...td->...cd", T, W)
            eta_shift = torch.einsum("...nd,...cd->...nc", z, TW)
            eta_t = deterministic("eta_t", eta_p + eta_shift + epsilon)
            pi_t = deterministic("pi_t", _inv_clr(eta_t))
        elif config.composition_model == "poe":
            # Product of Experts model
            # V was sampled outside the patient plate

            # Expert 1: treatment effect
            TW = torch.einsum("...ct,...td->...cd", T, W)
            eta_z = torch.einsum("...nd,...cd->...nc", z, TW)
            g_z = _inv_clr(eta_z)  # treatment composition

            # Expert 2: confounder effect
            # u is (N, r_u), V is (C, r_u), we want (N, C)
            # Transpose V: (C, r_u) -> (r_u, C)
            V_transposed = V.transpose(-2, -1)  # (r_u, C)
            eta_u = torch.matmul(u, V_transposed)  # (N, r_u) @ (r_u, C) = (N, C)
            h_u = _inv_clr(eta_u)  # confounder composition

            # Product of experts (multiply in probability space)
            # All should be (N, C)
            pi_t_unnorm = pi_p * g_z * h_u
            pi_t_clean = pi_t_unnorm / pi_t_unnorm.sum(dim=-1, keepdim=True)

            # Add noise in CLR space
            eta_t_clean = _clr(pi_t_clean)
            eta_t = deterministic("eta_t", eta_t_clean + epsilon)
            pi_t = deterministic("pi_t", _inv_clr(eta_t))
        else:
            raise ValueError(f"Unknown composition_model: {config.composition_model}")

        # Scale on-treatment dispersion by number of cells summed
        m_on = batch.on_ncells  # (N, C)
        m_on_expand = m_on.unsqueeze(-1)  # (N, C, 1)
        phi_t_sum = phi_t * m_on_expand.clamp_min(1.0)  # (N, C, G)

        logits_t = nb_logits(mu_t_i, phi_t)
        f_t = sample(
            "f_t",
            NegativeBinomial(
                total_count=phi_t_sum,
                logits=logits_t,
            ).to_event(2),
            obs=batch.on_counts,
        )
        q_t_mean = deterministic("q_t_mean", (pi_t.unsqueeze(-1) * mu_t_i).sum(dim=-2))
        q_t_head = q_t_mean * config.head_input_scale
        u_head = u * config.head_input_scale
        subtype_ids_ohe = torch.nn.functional.one_hot(batch.subtype_ids, num_classes=config.n_subtypes)

        # Compute gene and confounder contributions separately
        s_q = (q_t_head * beta_t).sum(dim=-1)  # Gene contribution (N,)
        s_u = (u_head * gamma).sum(dim=-1)     # Confounder contribution (N,)
        s_s = (beta_s * subtype_ids_ohe).sum(dim=-1)  # Subtype contribution (N,)

        # Orthogonality constraint: penalize correlation between gene and confounder effects
        if config.lambda_orth > 0:
            factor("orth_q_u", -config.lambda_orth * torch.mean(s_q * s_u))

        linear = config.logit_scale * (s_q + s_u + s_s)
        logit_y = deterministic("logit_y", beta0 + linear)
        sample(
            "y",
            Bernoulli(logits=logit_y),
            obs=batch.responses,
        )
