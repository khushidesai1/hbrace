"""Typed containers for the tensors passed into the Pyro model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class PatientBatch:
    """Observed variables grouped per patient.

    Attributes mirror the notation in the proposal:
    - ``subtype_ids`` corresponds tos_i (one-hot or integer labels)
    - ``pre_counts`` corresponds top_ij counts aggregated per cell type
    - ``on_counts`` correspondst_ik
    - ``responses`` corresponds toy_i
    - ``pre_mix`` and ``on_mix`` hold optional empirical simplex weightsπ^p_i, π^t_i
    """

    subtype_ids: torch.Tensor
    pre_counts: torch.Tensor
    on_counts: torch.Tensor
    responses: torch.Tensor
    pre_mix: torch.Tensor | None = None
    on_mix: torch.Tensor | None = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "subtype_ids": self.subtype_ids,
            "pre_counts": self.pre_counts,
            "on_counts": self.on_counts,
            "responses": self.responses,
            "pre_mix": self.pre_mix,
            "on_mix": self.on_mix,
        }

    def to(self, device: torch.device) -> "PatientBatch":
        """Move all tensors to a common device for accelerator training."""

        kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in self.as_dict().items()
        }
        return PatientBatch(**kwargs)
