from __future__ import annotations

import pyro
import torch
from pyro import deterministic, param, sample
from pyro.distributions import Categorical, Dirichlet, Gamma, NegativeBinomial, Normal
from torch.distributions import constraints

from hbrace.config import ModelConfig
from hbrace.patient_data import PatientBatch
from hbrace.utils import nb_logits


def _simplex(name: str, shape: torch.Size, value: float = 1.0) -> torch.Tensor:
    """Helper to create a simplex tensor."""
    tensor = torch.full(shape, value, dtype=torch.float32)
    tensor = tensor / tensor.sum(-1, keepdim=True)
    return param(name, tensor, constraint=constraints.simplex)


def hierarchical_model(batch: PatientBatch, config: ModelConfig) -> None:
    """Hierarchical Pyro model implementation.

    Args:
        batch: PatientBatch with counts/responses/subtypes.
        config: The ModelConfig to use for the model.
    """
    device = batch.pre_counts.device
    n_patients = batch.responses.shape[0]
    n_cell_types = config.n_cell_types
    n_genes = config.n_genes

    theta = _simplex("theta", torch.Size((config.n_subtypes, n_cell_types)))

    with pyro.plate("patients", n_patients, dim=-1):
        tau = sample(
            "tau",
            Gamma(config.subtype_concentration, config.subtype_rate)
        )

        subtype = sample(
            "s",
            Categorical(logits=torch.zeros(config.n_subtypes, device=device)),
            obs=batch.subtype_ids.long(),
        )
        subtype_prior = theta[subtype]

        pi_p = sample(
            "pi_p",
            Dirichlet(tau.unsqueeze(-1) * subtype_prior),
        )

        mix_shift_basis = param(
            "mix_shift_basis",
            0.01 * torch.randn(config.delta_dim, n_cell_types, device=device),
        )
        delta = sample(
            "delta",
            Normal(0.0, 1.0).expand([config.delta_dim]).to_event(1),
        )
        shift_logits = delta @ mix_shift_basis
        pi_t = torch.softmax(torch.log(pi_p + 1e-8) + shift_logits, dim=-1)
        deterministic("pi_t", pi_t)

        latent_z = sample(
            "z",
            Normal(0.0, 1.0).expand([config.latent_dim]).to_event(1),
        )
        latent_u = sample(
            "u",
            Normal(0.0, 1.0).expand([config.latent_dim]).to_event(1),
        )

        base_pre_rate = param(
            "base_pre_rate",
            0.2 * torch.ones(n_cell_types, n_genes, device=device),
            constraint=constraints.positive,
        )
        base_pre_disp = param(
            "base_pre_disp",
            torch.ones(n_cell_types, n_genes, device=device),
            constraint=constraints.positive,
        )

        pre_rate = pi_p.unsqueeze(-1) * base_pre_rate
        pre_disp = base_pre_disp
        pre_logits = nb_logits(pre_rate, pre_disp)
        sample(
            "pre_counts",
            NegativeBinomial(total_count=pre_disp, logits=pre_logits).to_event(2),
            obs=batch.pre_counts,
        )

        gene_shift = param(
            "gene_shift",
            0.05 * torch.randn(config.latent_dim, n_cell_types, n_genes, device=device),
        )
        latent_shift = torch.einsum("pl,lcg->pcg", latent_z, gene_shift)
        delta_bias = param(
            "delta_bias",
            torch.zeros(n_cell_types, n_genes, device=device),
        )
        on_rate = torch.nn.functional.softplus(base_pre_rate + latent_shift + delta_bias)
        on_rate = pi_t.unsqueeze(-1) * on_rate
        on_disp = param(
            "on_disp",
            torch.ones(n_cell_types, n_genes, device=device),
            constraint=constraints.positive,
        )
        on_logits = nb_logits(on_rate, on_disp)
        qt_stats = deterministic("qt_mean", (pi_t.unsqueeze(-1) * on_rate).sum(-2))
        sample(
            "on_counts",
            NegativeBinomial(total_count=on_disp, logits=on_logits).to_event(2),
            obs=batch.on_counts,
        )

        response_weights = param(
            "response_weights",
            torch.randn(n_genes, device=device),
        )
        subtype_effect = param(
            "response_subtype",
            torch.zeros(config.n_subtypes, device=device),
        )[subtype]
        latent_effect = torch.sum(latent_u, dim=-1)
        qt_summary = (qt_stats * response_weights).sum(-1)
        response_loc = qt_summary + subtype_effect + latent_effect
        response_scale = param(
            "response_scale",
            torch.tensor(0.5, device=device),
            constraint=constraints.positive,
        )
        sample(
            "responses",
            Normal(response_loc, response_scale),
            obs=batch.responses,
        )
