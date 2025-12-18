from __future__ import annotations

import torch
from pyro import deterministic, factor, param, plate, sample
from pyro.distributions import (
    Beta,
    Bernoulli,
    Dirichlet,
    Gamma,
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
        # Sparsity version: tighter prior
        Delta_std = sample(
            "Delta_std",
            Gamma(
                torch.full((C, G, d_z), 2.0, device=device),
                torch.full((C, G, d_z), 5.0, device=device),  # rate=5.0, mean=0.4
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
    if config.gene_sparsity:
        # Sparsity version: simplified fixed scale
        W = sample(
            "W",
            Normal(
                torch.zeros((C, d_z), device=device),
                torch.full((C, d_z), 0.5, device=device),
            ).to_event(2),
        )
    else:
        # Original version: hierarchical prior
        W_std = sample(
            "W_std",
            Gamma(
                torch.full((C, d_z), 2.0, device=device),
                torch.full((C, d_z), 4.0, device=device),  # rate=4.0, mean=0.5
            ).to_event(2),
        )
        W = sample(
            "W",
            Normal(torch.zeros((C, d_z), device=device), W_std).to_event(2),
        )
    epsilon_std = sample(
        "epsilon_std",
        Gamma(
            torch.full((C,), 2.0, device=device),  # shape
            torch.full((C,), 20.0, device=device),  # rate -> mean 0.1
        ).to_event(1),
    )

    lambda_T = sample(
        "lambda_T",
        Beta(torch.tensor(2.0, device=device), torch.tensor(5.0, device=device)),
    )
    T = sample(
        "T",
        Laplace(torch.zeros((C, C), device=device), lambda_T).to_event(2),
    )
    # # Enforce zero diagonal to mirror synthetic generator.
    # T = deterministic("T_masked", T * (1.0 - torch.eye(C, device=device)))

    beta0 = sample(
        "beta0",
        Normal(
            torch.tensor(config.beta0_loc, device=device),
            torch.tensor(config.beta0_scale, device=device),
        ),
    )
    if config.gene_sparsity:
        # Sparsity version: Laplace prior for sparsity
        beta_t = sample(
            "beta_t",
            Laplace(
                torch.zeros((G,), device=device),
                torch.full((G,), config.beta_t_laplace_scale, device=device),
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

    # Patient-level plate.
    with plate("patients", n_patients):
        tau_i_p = sample(
            "tau_i_p",
            Gamma(torch.tensor(2.0, device=device), torch.tensor(5.0, device=device)),
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

        logits_p = nb_logits(mu_p, phi_p)
        f_p = sample(
            "f_p",
            NegativeBinomial(
                total_count=phi_p,
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

        eta_p = _clr(pi_p)
        # Composition shift: z @ (T @ W)^T to match synthetic generation ordering.
        TW = torch.einsum("...ct,...td->...cd", T, W)
        eta_shift = torch.einsum("...nd,...cd->...nc", z, TW)
        eta_t = deterministic("eta_t", eta_p + eta_shift + epsilon)
        pi_t = deterministic("pi_t", _inv_clr(eta_t))

        logits_t = nb_logits(mu_t_i, phi_t)
        f_t = sample(
            "f_t",
            NegativeBinomial(
                total_count=phi_t,
                logits=logits_t,
            ).to_event(2),
            obs=batch.on_counts,
        )
        q_t_mean = deterministic("q_t_mean", (pi_t.unsqueeze(-1) * mu_t_i).sum(dim=-2))
        q_t_head = q_t_mean * config.head_input_scale

        u = sample(
            "u",
            Normal(torch.zeros(r_u, device=device), torch.ones(r_u, device=device)).to_event(1),
        )
        u_head = u * config.head_input_scale
        subtype_ids_ohe = torch.nn.functional.one_hot(batch.subtype_ids, num_classes=config.n_subtypes)
        linear = config.logit_scale * (
            (q_t_head * beta_t).sum(dim=-1)
            + (u_head * gamma).sum(dim=-1)
            + (beta_s * subtype_ids_ohe).sum(dim=-1)
        )
        logit_y = deterministic("logit_y", beta0 + linear)
        sample(
            "y",
            Bernoulli(logits=logit_y),
            obs=batch.responses,
        )
