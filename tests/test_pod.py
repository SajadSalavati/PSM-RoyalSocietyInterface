"""Tests for the POD module."""

import numpy as np

from psm_surrogate.pod import (
    compute_pod,
    cumulative_energy,
    energy_content,
    reconstruction_error,
)


def _low_rank_matrix(n_loc=80, n_time=60, rank=5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_loc, rank))
    B = rng.standard_normal((rank, n_time))
    return A @ B


def test_pod_shapes():
    X = _low_rank_matrix()
    res = compute_pod(X, num_modes=5)
    assert res.modes.shape == (80, 5)
    assert res.coefficients.shape == (5, 60)
    assert res.reconstruction.shape == X.shape


def test_low_rank_is_recovered_exactly():
    # A rank-5 matrix should be (near) perfectly reconstructed with 5 modes.
    X = _low_rank_matrix(rank=5)
    res = compute_pod(X, num_modes=5)
    assert np.allclose(X, res.reconstruction, atol=1e-8)
    assert energy_content(res.S, 5) > 99.999


def test_reconstruction_error_nonnegative():
    X = _low_rank_matrix()
    res = compute_pod(X, num_modes=3)
    err = reconstruction_error(X, res.reconstruction)
    assert err.shape == (X.shape[1],)
    assert np.all(err >= 0)


def test_cumulative_energy_monotonic_and_bounded():
    X = _low_rank_matrix()
    res = compute_pod(X, num_modes=5)
    cum = cumulative_energy(res.S)
    assert np.all(np.diff(cum) >= -1e-12)  # non-decreasing
    assert np.isclose(cum[-1], 1.0)
