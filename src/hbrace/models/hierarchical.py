from __future__ import annotations

import torch
from pyro import deterministic, param, plate, sample
from pyro.distributions import (
    Bernoulli,
    Categorical,
    Dirichlet,
    Gamma,
    HalfNormal,
    Laplace,
    NegativeBinomial,
    Normal,
)
from torch.distributions import constraints

from hbrace.config import ModelConfig
from hbrace.patient_data import PatientBatch
from hbrace.utils import nb_logits


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
    n_patients = batch.responses.shape[0]
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

    # Pre-treatment NB dispersion prior (cell-type level).
    phi_p_std = sample(
        "phi_p_std",
        Gamma(
            torch.full((C,), config.nb_dispersion_prior, device=device),
            torch.full((C,), config.nb_dispersion_rate, device=device),
        ).to_event(1),
    )

    # Mean and dispersion shifts for on-treatment distributions.
    Delta_std = sample(
        "Delta_std",
        HalfNormal(torch.full((C, G, d_z), 0.5, device=device)).to_event(3),
    )
    Delta = sample("Delta", Normal(torch.zeros((C, G, d_z), device=device), Delta_std).to_event(3))
    delta_std = sample(
        "delta_std",
        HalfNormal(torch.full((C,), 0.5, device=device)).to_event(1),
    )

    # Cell-type proportion shifts for on-treatment mixture weights.
    W_std = sample(
        "W_std",
        HalfNormal(torch.full((C, d_z), 0.5, device=device)).to_event(2),
    )
    W = sample(
        "W",
        Normal(torch.zeros((C, d_z), device=device), W_std).to_event(2),
    )
    epsilon_std = sample(
        "epsilon_std",
        HalfNormal(torch.full((C,), 0.1, device=device)).to_event(1),
    )

    lambda_h = sample("lambda_h", HalfNormal(torch.tensor(0.2, device=device)))
    lambda_l = sample("lambda_l", HalfNormal(torch.tensor(0.05, device=device)))
    scale_T = torch.where(batch.transition_prior > 0, lambda_l, lambda_h)
    T = sample(
        "T",
        Laplace(torch.zeros((C, C), device=device), scale_T).to_event(2),
    )
    T = T - torch.diag(torch.diag(T))  # zero diagonal transitions

    beta0 = sample(
        "beta0",
        Normal(torch.tensor(0.0, device=device), torch.tensor(0.5, device=device)),
    )
    beta_t = sample(
        "beta_t",
        Normal(torch.zeros((C,), device=device), torch.full((C,), 0.5, device=device)).to_event(1),
    )
    gamma = sample(
        "gamma",
        Normal(torch.zeros((r_u,), device=device), torch.full((r_u,), 0.5, device=device)).to_event(1),
    )
    beta_s = sample(
        "beta_s",
        Normal(torch.zeros((config.n_subtypes,), device=device), torch.full((config.n_subtypes,), 0.5, device=device)).to_event(1),
    )

    # Patient-level plate.
    with plate("patients", n_patients):
        tau_i_p = sample(
            "tau_i_p",
            Gamma(torch.tensor(2.0, device=device), torch.tensor(0.1, device=device)),
        )
        pi_p = sample(
            "pi_p",
            Dirichlet(tau_i_p.unsqueeze(-1) * theta_expanded.clamp_min(1e-6)),
        )

        log_mu_p = sample(
            "log_mu_p",
            Normal(
                torch.full((C, G), 1.5, device=device),
                torch.full((C, G), 0.8, device=device),
            ).to_event(2),
        )
        mu_p = torch.exp(log_mu_p)

        phi_p = sample(
            "phi_p",
            Gamma(
                phi_p_std,
                torch.ones_like(phi_p_std),
            ).to_event(1),
        )

        phi_p_exp = phi_p.unsqueeze(-1).expand(n_patients, C, G)
        logits_p = nb_logits(mu_p, phi_p_exp)
        f_p = sample(
            "f_p",
            NegativeBinomial(
                total_count=phi_p_exp,
                logits=logits_p,
            ).to_event(2),
            obs=batch.pre_counts,
        )

        z = sample(
            "z",
            Normal(torch.zeros(d_z, device=device), torch.ones(d_z, device=device)).to_event(1),
        )

        log_mu_t_i = log_mu_p + torch.einsum("nd,cgd->ncg", z, Delta)
        mu_t_i = deterministic("mu_t", torch.exp(log_mu_t_i))

        delta = sample(
            "delta",
            Normal(torch.zeros(C, device=device), delta_std).to_event(1),
        )
        phi_t_i = deterministic("phi_t", phi_p * torch.exp(delta))

        epsilon = sample(
            "epsilon",
            Normal(torch.zeros(C, device=device), epsilon_std).to_event(1),
        )

        eta_p = _clr(pi_p)
        eta_t = deterministic("eta_t", eta_p + (z @ W.t()) @ T.T + epsilon)
        pi_t = deterministic("pi_t", _inv_clr(eta_t))

        phi_t_exp = phi_t_i.unsqueeze(-1).expand(n_patients, C, G)
        logits_t = nb_logits(mu_t_i, phi_t_exp)
        f_t = sample(
            "f_t",
            NegativeBinomial(
                total_count=phi_t_exp,
                logits=logits_t,
            ).to_event(2),
            obs=batch.on_counts,
        )

        u = sample(
            "u",
            Normal(torch.zeros(r_u, device=device), torch.ones(r_u, device=device)).to_event(1),
        )
        logit_y = deterministic(
            "logit_y",
            beta0
            + (pi_t * beta_t).sum(dim=-1)
            + (u * gamma).sum(dim=-1)
            + beta_s[batch.subtype_ids],
        )
        sample(
            "y",
            Bernoulli(logits=logit_y),
            obs=batch.responses,
        )
