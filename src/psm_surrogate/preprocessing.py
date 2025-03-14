"""Turn POD coefficients + boundary conditions into scaled, windowed LSTM tensors.

Pipeline (matching the notebooks):

1. Assemble the feature matrix ``[t, v(t), P(t), q(t)]`` and the target ``q(t)``.
2. Min-max scale both to ``[0, 1]`` (scalers are kept so predictions can be inverted).
3. Slide a fixed-length window over time to form ``(sample, time_window, features)``.
4. Split sequentially (no shuffle) into train / test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MinMaxScaler1D:
    """Column-wise min-max scaler with an explicit inverse transform."""

    min_: np.ndarray
    max_: np.ndarray

    @classmethod
    def fit(cls, arr: np.ndarray) -> MinMaxScaler1D:
        return cls(min_=np.min(arr, axis=0), max_=np.max(arr, axis=0))

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.min_[np.newaxis, :]) / (self.max_ - self.min_)[np.newaxis, :]

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * (self.max_ - self.min_)[np.newaxis, :] + self.min_[np.newaxis, :]


def build_io_matrices(
    t: np.ndarray, v_interp: np.ndarray, P_interp: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate driving signals with POD coefficients into (inputs, outputs).

    inputs  : ``(n_time, 3 + num_modes)`` -> [time, velocity, pressure, q_1..q_r]
    outputs : ``(n_time, num_modes)``     -> q_1..q_r (what the LSTM predicts)
    """
    inputs = np.concatenate(
        (
            t.reshape(-1, 1),
            v_interp.reshape(-1, 1),
            P_interp.reshape(-1, 1),
            coefficients.T,
        ),
        axis=1,
    )
    outputs = np.copy(coefficients).T
    return inputs, outputs


def make_windows(
    input_scaled: np.ndarray, output_scaled: np.ndarray, time_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice into overlapping sequences.

    Sample ``i`` uses inputs ``[i : i+time_window]`` to predict the output at
    ``i+time_window`` (one step ahead of the window).
    """
    x, y = [], []
    for i in range(0, len(input_scaled) - time_window - 1):
        x.append(input_scaled[i : i + time_window, :])
        y.append(output_scaled[i + time_window, :])
    return np.array(x), np.array(y)


def sequential_split(
    total_x: np.ndarray, total_y: np.ndarray, test_split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/test split (first ``test_split`` fraction is training)."""
    n_train = int(test_split * total_x.shape[0])
    x_train = total_x[:n_train]
    y_train = total_y[:n_train]
    x_test = total_x[n_train:]
    y_test = total_y[n_train:]
    return x_train, y_train, x_test, y_test
