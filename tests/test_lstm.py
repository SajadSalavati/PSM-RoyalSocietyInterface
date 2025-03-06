"""Tests for the LSTM model."""

import torch

from psm_surrogate.config import load_config
from psm_surrogate.lstm import LSTMNet, build_model


def test_forward_output_shape():
    model = LSTMNet(input_size=15, hidden_size=32, output_size=12, num_layers=2, dropout=0.1)
    x = torch.randn(4, 10, 15)  # (batch, time_window, features)
    out = model(x)
    assert out.shape == (4, 12)


def test_relu_readout_nonnegative():
    model = LSTMNet(input_size=8, hidden_size=16, output_size=5, num_layers=1, dropout=0.0)
    out = model(torch.randn(3, 7, 8))
    assert torch.all(out >= 0)


def test_build_model_from_config():
    cfg = load_config("configs/base_mid_u.yaml")
    model = build_model(cfg, device=torch.device("cpu"))
    x = torch.randn(2, cfg.time_window, cfg.input_size)
    assert model(x).shape == (2, cfg.output_size)
