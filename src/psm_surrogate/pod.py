"""Proper Orthogonal Decomposition (POD) for model-order reduction.

POD compresses a high-dimensional field snapshot matrix ``X`` (space x time) into a
handful of spatial modes plus their time-varying coefficients, via the SVD
``X = U S V^T``. Keeping the first ``r`` modes yields the reduced coefficients
``q = U_r^T X`` that the LSTM later learns to predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import norm, svd


@dataclass
class PODResult:
    """Outputs of a truncated POD."""

    U: np.ndarray            # spatial modes, (n_locations, n_time)
    S: np.ndarray            # singular values, (n_time,)
    VT: np.ndarray           # right singular vectors, (n_time, n_time)
    modes: np.ndarray        # truncated modes U[:, :r], (n_locations, r)
    coefficients: np.ndarray  # q = modes^T @ X, (r, n_time)
    reconstruction: np.ndarray  # modes @ q, (n_locations, n_time)
    num_modes: int


def compute_pod(variable: np.ndarray, num_modes: int) -> PODResult:
    """Run the (economy) SVD and project onto the leading ``num_modes`` modes."""
    U, S, VT = svd(variable, full_matrices=False)
    modes = U[:, :num_modes]
    coefficients = modes.T @ variable
    reconstruction = modes @ coefficients
    return PODResult(
        U=U,
        S=S,
        VT=VT,
        modes=modes,
        coefficients=coefficients,
        reconstruction=reconstruction,
        num_modes=num_modes,
    )


def cumulative_energy(S: np.ndarray) -> np.ndarray:
    """Fraction of cumulative singular-value 'energy' captured by the first k modes."""
    return np.cumsum(S) / np.sum(S)


def energy_content(S: np.ndarray, num_modes: int) -> float:
    """Percentage of energy retained by keeping ``num_modes`` modes."""
    return float(np.sum(S[:num_modes]) / np.sum(S) * 100.0)


def reconstruction_error(variable: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Per-snapshot relative L2 reconstruction error, in percent.

    ``err[t] = ||x_t - x_hat_t|| / ||x_t|| * 100`` for each time column ``t``.
    """
    n_time = variable.shape[1]
    err = np.zeros(n_time)
    for t in range(n_time):
        err[t] = norm(variable[:, t] - reconstruction[:, t]) / norm(variable[:, t]) * 100.0
    return err
