"""Tests for scaling, windowing and splitting."""

import numpy as np

from psm_surrogate.preprocessing import (
    MinMaxScaler1D,
    build_io_matrices,
    make_windows,
    sequential_split,
)


def test_minmax_roundtrip():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((50, 4))
    scaler = MinMaxScaler1D.fit(arr)
    scaled = scaler.transform(arr)
    assert scaled.min() >= -1e-9 and scaled.max() <= 1 + 1e-9
    assert np.allclose(scaler.inverse_transform(scaled), arr)


def test_build_io_matrices_shapes():
    n, r = 100, 6
    t = np.linspace(1, n, n)
    v = np.sin(t)
    P = np.cos(t)
    q = np.random.default_rng(0).standard_normal((r, n))
    inp, out = build_io_matrices(t, v, P, q)
    assert inp.shape == (n, r + 3)
    assert out.shape == (n, r)


def test_make_windows_shapes():
    data_in = np.arange(100 * 5).reshape(100, 5).astype(float)
    data_out = np.arange(100 * 2).reshape(100, 2).astype(float)
    tw = 10
    x, y = make_windows(data_in, data_out, tw)
    assert x.shape == (100 - tw - 1, tw, 5)
    assert y.shape == (100 - tw - 1, 2)
    # first target is the output row one step past the first window
    assert np.array_equal(y[0], data_out[tw])


def test_sequential_split_no_leakage():
    x = np.arange(80 * 3 * 2).reshape(80, 3, 2).astype(float)
    y = np.arange(80 * 2).reshape(80, 2).astype(float)
    xtr, ytr, xte, yte = sequential_split(x, y, 0.8)
    assert xtr.shape[0] == 64 and xte.shape[0] == 16
    # test set comes strictly after the train set (chronological order kept)
    assert np.array_equal(xte[0], x[64])
