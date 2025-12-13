import numpy as np
from typing import List, Tuple, Optional

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax function."""
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def clr(pi: np.ndarray) -> np.ndarray:
    """
    Centered log-ratio transform.
    pi: (..., C), each row >0 and sums to 1
    """
    pi = np.clip(pi, 1e-12, 1.0)
    log_pi = np.log(pi)
    mean_log = log_pi.mean(axis=-1, keepdims=True)
    return log_pi - mean_log


def inv_clr(eta: np.ndarray) -> np.ndarray:
    """
    Inverse clr transform.
    eta: (..., C)
    """
    ex = np.exp(eta)
    return ex / ex.sum(axis=-1, keepdims=True)


def nb_from_mu_phi(mu: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (mu, phi) -> (n, p) for numpy's Negative Binomial.
    Convention:
        n = phi  (dispersion / 'size')
        p = phi / (phi + mu)
    Then mean = n*(1-p)/p = mu, var = mu + mu^2/phi.
    """
    phi = np.clip(phi, 1e-6, None)
    mu = np.clip(mu, 1e-6, None)
    n = phi
    p = phi / (phi + mu)
    return n, p


def sample_nb(mu: np.ndarray, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from NB with mean mu and dispersion phi.
    mu, phi: (...,) broadcastable
    Returns integer counts of same shape.
    """
    n, p = nb_from_mu_phi(mu, phi)
    return rng.negative_binomial(n, p)

def collapse_cells(
    cells: List[np.ndarray], cell_types: List[np.ndarray], n_cell_types: int
) -> np.ndarray:
    """Sum cell-level gene counts into patient x cell-type x gene tensors."""

    n_patients = len(cells)
    n_genes = cells[0].shape[1]
    aggregated = np.zeros((n_patients, n_cell_types, n_genes), dtype=np.int64)
    for idx, (counts, types) in enumerate(zip(cells, cell_types)):
        for c in range(n_cell_types):
            mask = types == c
            if not np.any(mask):
                continue
            aggregated[idx, c] = counts[mask].mean(axis=0)
    return aggregated

def compute_cell_type_proportions(
    cell_type_lists: Optional[List[np.ndarray]],
    subtype_ids: np.ndarray,
    n_subtypes: Optional[int] = None,
    n_cell_types: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the average cell type proportions across patients for each subtype.
    
    Args:
        cell_type_lists: List of cell types for each patient.
        subtype_ids: Subtype ids for each patient.
        n_subtypes: Optional explicit number of subtypes to ensure consistent shape.
        n_cell_types: Optional explicit number of cell types to ensure consistent shape.

    Returns:
        Average cell type proportions across patients for each subtype.
    """
    subtype_ids_np = np.asarray(subtype_ids)
    if n_subtypes is None:
        if subtype_ids_np.size == 0:
            raise ValueError("Cannot infer n_subtypes from empty subtype_ids.")
        n_subtypes = int(subtype_ids_np.max()) + 1

    if cell_type_lists is None:
        raise ValueError("cell_type_lists must be provided to compute proportions.")

    # Derive number of cell types from the global max across patients if not provided.
    if n_cell_types is None:
        n_cell_types = max(int(ct.max()) for ct in cell_type_lists) + 1
    n_patients = len(cell_type_lists)

    patient_cell_type_proportions = np.zeros((n_patients, n_cell_types), dtype=np.float32)
    for i, cell_types in enumerate(cell_type_lists):
        counts = np.bincount(cell_types, minlength=n_cell_types)
        patient_cell_type_proportions[i] = counts / counts.sum()

    subtype_cell_type_proportions = np.zeros((n_subtypes, n_cell_types), dtype=np.float32)
    for subtype in range(n_subtypes):
        patient_idxs = subtype_ids_np == subtype
        if not np.any(patient_idxs):
            continue
        subtype_cell_type_proportions[subtype] = patient_cell_type_proportions[patient_idxs].mean(axis=0)
    return subtype_cell_type_proportions
