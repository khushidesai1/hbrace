import numpy as np
from typing import List, Tuple

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
    """Average cell-level gene counts into patient x cell-type x gene tensors."""

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
