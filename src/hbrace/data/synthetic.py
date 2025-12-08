"""Simple simulator to unblock local experimentation before real data ingestion."""
from __future__ import annotations

from typing import Optional

import torch

from hbrace.config import ModelConfig
from hbrace.data import PatientBatch


def sample_synthetic_batch(
    config: ModelConfig,
    num_patients: int,
    total_counts: float = 40.0,
    response_noise: float = 0.1,
    device: Optional[torch.device] = None,
) -> PatientBatch:
    """Generate a batch following the qualitative structure of the PDF proposal."""

    device = device or torch.device("cpu")
    n_cell_types = config.n_cell_types
    n_genes = config.n_genes

    subtype_logits = torch.zeros(config.n_subtypes, device=device)
    subtype_ids = torch.distributions.Categorical(logits=subtype_logits).sample(
        (num_patients,)
    )

    dirichlet = torch.distributions.Dirichlet(
        concentration=torch.ones(n_cell_types, device=device)
    )
    pre_mix = dirichlet.sample((num_patients,))
    on_mix = dirichlet.sample((num_patients,))

    # Use Negative Binomial draws to mimic over-dispersed gene counts.
    nb = torch.distributions.NegativeBinomial(
        total_count=total_counts,
        probs=0.5 * torch.ones((), device=device),
    )
    pre_counts = nb.sample((num_patients, n_cell_types, n_genes))
    on_counts = nb.sample((num_patients, n_cell_types, n_genes))

    treatment_effect = (on_counts.sum(-1) - pre_counts.sum(-1)).mean(-1)
    subtype_offset = torch.nn.functional.one_hot(
        subtype_ids, config.n_subtypes
    ).float() @ torch.linspace(-0.5, 0.5, config.n_subtypes, device=device)
    responses = torch.tanh(0.01 * treatment_effect + subtype_offset)
    responses = responses + response_noise * torch.randn_like(responses)

    return PatientBatch(
        subtype_ids=subtype_ids,
        pre_counts=pre_counts,
        on_counts=on_counts,
        responses=responses,
        pre_mix=pre_mix,
        on_mix=on_mix,
    )
