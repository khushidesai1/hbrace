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
    delta_dim = config.delta_dim

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
    logits_p = nb_logits(mu_p, phi_p[:, None])
    base_nb = NegativeBinomial(
        total_count=phi_p[:, None],
        logits=logits_p,
    )
    base_nb_batch = base_nb.expand((n_patients, C, G))
    f_p = sample(
        "f_p",
        base_nb_batch.to_event(3),
        obs=batch.pre_counts,
    )
    q_p = sample(
        "q_p",
        MixtureSameFamily(
            Categorical(pi_p),
            Independent(base_nb_batch, 1),
        ),
    )
