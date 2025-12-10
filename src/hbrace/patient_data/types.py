from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hbrace.patient_data.utils import compute_cell_type_proportions


@dataclass
class PatientBatch:
    """Patient-level tensors consumed by the Pyro model."""

    pre_counts: torch.Tensor  # shape (N, C, G)
    on_counts: torch.Tensor  # shape (N, C, G)
    cell_type_proportions: torch.Tensor # shape (N, C)
    responses: torch.Tensor  # shape (N,)
    subtype_ids: torch.Tensor  # shape (N,)
    transition_prior: torch.Tensor  # shape (C, C)

    def to(self, device: torch.device | str) -> "PatientBatch":
        """Move all tensors to a specific device.
        
        Args:
            device: Device to move tensors to.

        Returns:
            A PatientBatch object.
        """

        device = torch.device(device)
        return PatientBatch(
            pre_counts=self.pre_counts.to(device),
            on_counts=self.on_counts.to(device),
            cell_type_proportions=self.cell_type_proportions.to(device),
            responses=self.responses.to(device),
            subtype_ids=self.subtype_ids.to(device),
            transition_prior=self.transition_prior.to(device),
        )


@dataclass
class SimConfig:
    """Configuration for the fully synthetic data simulation."""
    n_patients: int = 50
    n_subtypes: int = 3
    n_cell_types: int = 5
    n_genes: int = 100
    d_z: int = 3  # dim of treatment effect z_i
    r_u: int = 2  # dim of confounder u_i
    sigma_D: float = 0.5  # std for Delta_cg
    sigma_W: float = 0.5  # std for W_P
    sigma_eps: float = 0.1  # std for epsilon_i in eta^t
    seed: int = 0


@dataclass
class SimulatedData:
    """Container for synthetic cohort plus utilities to feed the model."""

    config: SimConfig
    pre_counts: np.ndarray  # (N, C, G)
    on_counts: np.ndarray  # (N, C, G)
    responses: np.ndarray  # (N,)
    subtype_ids: np.ndarray  # (N,)
    pi_p: np.ndarray  # (N, C)
    pi_t: np.ndarray  # (N, C)
    transition_prior: np.ndarray  # (C, C)
    pre_cell_types: Optional[List[np.ndarray]] = None
    post_cell_types: Optional[List[np.ndarray]] = None
    min_cells: int = 500
    max_cells: int = 1000
    extra_params: Optional[Dict[str, Any]] = None

    def to_patient_batch(self, device: torch.device | str = "cpu") -> PatientBatch:
        """
        Move tensors to a device and wrap them as a PatientBatch.
        
        Args:
            device: Device to move tensors to.

        Returns:
            A PatientBatch object on the specified device.
        """

        device = torch.device(device)

        cell_type_proportions = compute_cell_type_proportions(
            self.pre_cell_types,
            self.subtype_ids,
            self.pi_p,
        )

        return PatientBatch(
            pre_counts=torch.as_tensor(self.pre_counts, dtype=torch.float32, device=device),
            on_counts=torch.as_tensor(self.on_counts, dtype=torch.float32, device=device),
            cell_type_proportions=torch.as_tensor(cell_type_proportions, dtype=torch.float32, device=device),
            responses=torch.as_tensor(self.responses, dtype=torch.float32, device=device),
            subtype_ids=torch.as_tensor(self.subtype_ids, dtype=torch.long, device=device),
            transition_prior=torch.as_tensor(self.transition_prior, dtype=torch.float32, device=device),
        )
