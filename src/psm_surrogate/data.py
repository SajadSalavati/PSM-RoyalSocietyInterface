"""Loading CFD snapshots and inlet boundary conditions.

Mirrors the data-handling cells of the original notebooks, but with the paths
supplied by :class:`~psm_surrogate.config.ExperimentConfig` instead of hard-coded
Google-Drive locations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_field(npz_path: str, field_key: str) -> np.ndarray:
    """Load one physical field and return it as a ``(n_locations, n_time)`` matrix.

    The archives store fields as ``(time, location)``; POD expects space along the
    rows, so we transpose (this is the ``variable = field.T`` step in the notebooks).
    """
    with np.load(npz_path) as data:
        field = data[field_key]
    return field.T


def load_boundary_conditions(
    csv_path: str, n_snapshots: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the inlet waveform CSV and resample it onto a uniform time grid.

    The CSV holds two independently-sampled waveforms:

    * velocity ``v`` at times ``t1``
    * pressure ``P`` at times ``t2`` (padded with NaNs to share the column height)

    Each is linearly interpolated to ``n_snapshots`` uniformly spaced points, matching
    the number of CFD snapshots.

    Returns
    -------
    (t, v_interp, P_interp) : each of shape ``(n_snapshots,)``
    """
    df = pd.read_csv(csv_path)

    tv_values = df["t1"].tolist()
    v_values = df["v"].tolist()
    interp_v = interp1d(tv_values, v_values)
    uniform_tv = np.linspace(min(tv_values), max(tv_values), n_snapshots)
    v_interp = interp_v(uniform_tv)

    tp_values = [x for x in df["t2"].tolist() if not math.isnan(x)]
    P_values = [x for x in df["P"].tolist() if not math.isnan(x)]
    interp_p = interp1d(tp_values, P_values)
    uniform_tp = np.linspace(min(tp_values), max(tp_values), n_snapshots)
    P_interp = interp_p(uniform_tp)

    t = np.linspace(1, n_snapshots, num=n_snapshots)
    return t, v_interp, P_interp
