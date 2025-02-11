"""Inference: roll the trained LSTM over the test sequences."""

from __future__ import annotations

import numpy as np
import torch

from .lstm import LSTMNet


def predict(model: LSTMNet, x_test: torch.Tensor, device: torch.device) -> np.ndarray:
    """Predict the POD coefficients for each test window.

    Reproduces the notebook's step-by-step evaluation over ``x_test`` and returns a
    ``(n_samples - 1, num_modes)`` array of predicted (scaled) coefficients.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        x = x_test[0:1, :, :].to(device)
        current = model(x)
        for i in range(1, x_test.shape[0]):
            predictions.append(current)
            x = x_test[i : i + 1, :, :].to(device)
            current = model(x)

    return torch.cat(predictions, dim=0).cpu().detach().numpy()
