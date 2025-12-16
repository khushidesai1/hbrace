from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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
        )


@dataclass
class SimConfig:
    """Configuration for the fully synthetic data simulation."""
    n_patients: int = 50
    n_subtypes: int = 3
    n_cell_types: int = 5
    n_genes: int = 100
    min_cells: int = 500
    max_cells: int = 1000
    d_z: int = 3  # dim of treatment effect z_i
    r_u: int = 2  # dim of confounder u_i
    sigma_D: float = 0.5  # std for Delta_cg
    sigma_W: float = 0.5  # std for W_P
    sigma_eps: float = 0.1  # std for epsilon_i in eta^t
    beta_t_active_frac: float = 0.1  # fraction of genes truly predictive
    beta_t_active_scale: float = 2.0  # std for active gene coefficients
    beta_t_inactive_loc: float = 0.0  # mean for inactive gene coefficients
    beta_t_inactive_scale: float = 0.0  # std for inactive gene coefficients
    response_base_rate: float = 0.3  # baseline response rate
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
    pre_cell_types: Optional[List[np.ndarray]] = None
    post_cell_types: Optional[List[np.ndarray]] = None
    extra_params: Optional[Dict[str, Any]] = None

    def to_patient_batch(
        self,
        device: torch.device | str = "cpu",
        indices: Optional[Sequence[int]] = None,
    ) -> PatientBatch:
        """
        Move tensors to a device and wrap them as a PatientBatch.
        
        Args:
            device: Device to move tensors to.
            indices: Optional subset of patient indices to include.

        Returns:
            A PatientBatch object on the specified device.
        """

        device = torch.device(device)

        n_patients = self.pre_counts.shape[0]
        if indices is None:
            idx = np.arange(n_patients)
        else:
            idx = np.asarray(indices)
            if idx.dtype == bool:
                idx = np.nonzero(idx)[0]
            idx = np.atleast_1d(idx)
        if idx.size == 0:
            raise ValueError("indices is empty; cannot build a PatientBatch.")

        pre_counts = self.pre_counts[idx]
        on_counts = self.on_counts[idx]
        responses = self.responses[idx]
        subtype_ids = self.subtype_ids[idx]

        cell_type_lists = None
        if self.pre_cell_types is not None:
            cell_type_lists = [self.pre_cell_types[i] for i in idx.tolist()]
        else:
            raise ValueError("pre_cell_types is required to compute cell type proportions.")

        cell_type_proportions = compute_cell_type_proportions(
            cell_type_lists,
            subtype_ids,
            n_subtypes=self.config.n_subtypes,
            n_cell_types=self.config.n_cell_types,
        )

        return PatientBatch(
            pre_counts=torch.as_tensor(pre_counts, dtype=torch.float32, device=device),
            on_counts=torch.as_tensor(on_counts, dtype=torch.float32, device=device),
            cell_type_proportions=torch.as_tensor(cell_type_proportions, dtype=torch.float32, device=device),
            responses=torch.as_tensor(responses, dtype=torch.float32, device=device),
            subtype_ids=torch.as_tensor(subtype_ids, dtype=torch.long, device=device),
        )
