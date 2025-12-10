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
from torch.distributions import Independent, MixtureSameFamily, constraints

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
    tau_i_p = sample(
        "tau_i_p",
        Gamma(torch.full((n_patients, 1), 2, device=device), torch.full((n_patients, 1), 0.1, device=device)).to_event(1),
    )
    theta_expanded = theta[batch.subtype_ids]
    pi_p = sample(
        "pi_p",
        Dirichlet(tau_i_p * theta_expanded).to_event(1),
    )

    # Pre-treatment NB distribution.
    log_mu_p = sample(
        "log_mu_p",
        Normal(
            torch.full((C, G), 1.5, device=device),
            torch.full((C, G), 0.8, device=device),
        ).to_event(2),
    )
    phi_p = sample(
        "phi_p",
        Gamma(
            torch.full((C,), config.nb_dispersion_prior, device=device),
            torch.full((C,), config.nb_dispersion_rate, device=device),
        ).to_event(1),
    )
    mu_p = torch.exp(log_mu_p)
    
    mu_p_batch  = mu_p.unsqueeze(0).expand(n_patients, -1, -1)
    phi_p_batch = phi_p.view(1, C, 1).expand(n_patients, -1, G)
    logits_p = nb_logits(mu_p_batch, phi_p_batch)

    f_p = sample(
        "f_p",
        NegativeBinomial(
            total_count=phi_p_batch,
            logits=logits_p,
        ).to_event(1),
        obs=batch.pre_counts,
    )
    
    q_p = MixtureSameFamily(
        Categorical(pi_p),
        Independent(
            NegativeBinomial(
                total_count=phi_p_batch,
                logits=logits_p,
            ),
            1,
        ),
    )

    # Mean and dispersion shifts for on-treatment distributions.
    Delta_std = sample(
        "Delta_std",
        HalfNormal(torch.full((C, G, d_z), 0.5, device=device)).to_event(3),
    )
    Delta = sample(
        "Delta",
        Normal(torch.zeros((C, G, d_z), device=device), Delta_std).to_event(3),
    )
    z = sample(
        "z",
        Normal(
            torch.zeros((n_patients, d_z), device=device),
            torch.ones((n_patients, d_z), device=device),
        ).to_event(2),
    )
    log_mu_t = deterministic(
        "log_mu_t",
        log_mu_p[None, :, :] + torch.einsum("id,cgd->icg", z, Delta),
    )
    mu_t = deterministic("mu_t", torch.exp(log_mu_t))
    delta_std = sample(
        "delta_std",
        HalfNormal(torch.full((C,), 0.5, device=device)).to_event(1),
    )
    delta = sample(
        "delta",
        Normal(
            torch.zeros((n_patients, C), device=device),
            delta_std.expand(n_patients, C),
        ).to_event(2),
    )
    phi_t = deterministic("phi_t", phi_p[None, :] * torch.exp(delta))

    # Cell-type proportion shifts for on-treatment mixture weights.
    W_std = sample(
        "W_std",
        HalfNormal(torch.tensor(config.z_dim, device=device)).expand([C, d_z]).to_event(2),
    )
    W = sample(
        "W",
        Normal(torch.zeros((C, d_z), device=device), W_std).to_event(2),
    )
    epsilon_std = sample(
        "epsilon_std",
        HalfNormal(torch.full((C,), 0.1, device=device)).to_event(1),
    )
    epsilon = sample(
        "epsilon",
        Normal(
            torch.zeros((n_patients, C), device=device),
            epsilon_std.expand(n_patients, C),
        ).to_event(2),
    )

    lambda_h = sample("lambda_h", HalfNormal(torch.tensor(0.2, device=device)))
    lambda_l = sample("lambda_l", HalfNormal(torch.tensor(0.05, device=device)))
    scale_T = torch.where(batch.transition_prior > 0, lambda_l, lambda_h)
    T = sample(
        "T",
        Laplace(torch.zeros((C, C), device=device), scale_T).to_event(2),
    )
    T = T - torch.diag(torch.diag(T))  # zero diagonal transitions

    eta_p = _clr(pi_p)
    eta_t = deterministic(
        "eta_t",
        eta_p + (z @ W.t()) @ T.T + epsilon,
    )
    pi_t = deterministic("pi_t", _inv_clr(eta_t))

    # On-treatment NB distribution and mixture.
    phi_t_batch = phi_t.unsqueeze(-1).expand(n_patients, C, G)
    logits_t = nb_logits(mu_t, phi_t_batch)

    f_t = sample(
        "f_t",
        NegativeBinomial(
            total_count=phi_t_batch,
            logits=logits_t,
        ).to_event(1),
        obs=batch.on_counts,
    )

    q_t = MixtureSameFamily(
        Categorical(pi_t),
        Independent(
            NegativeBinomial(
                total_count=phi_t_batch,
                logits=logits_t,
            ),
            1,
        ),
    )

    # Confounder variable and patient repsonse.
    u = sample(

    )
    beta_0 = sample(

    )
    beta_t = sample(

    )
    gamma_u = sample(

    )
    beta_s = sample(

    )
    y_prob = deterministic(

    )
    y = sample(
        
    )
