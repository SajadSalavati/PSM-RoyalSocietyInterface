"""The LSTM that predicts POD temporal coefficients from the driving signals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


class LSTMNet(nn.Module):
    """Stacked LSTM followed by a linear read-out.

    Architecture matches the original notebooks: a multi-layer LSTM whose final
    time-step hidden state is mapped to ``output_size`` POD coefficients through a
    single linear layer with a ReLU. Weights use Xavier-normal initialization.

    The hidden/cell states are created on the same device as the input, so the
    module works transparently on CPU or GPU without a global ``device`` variable.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)
            elif "bias" in name:
                init.constant_(param, 0.0)
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        # use the last time step, then a non-negative linear read-out
        return torch.relu(self.fc(out[:, -1, :]))


def build_model(config, device: torch.device | None = None) -> LSTMNet:
    """Instantiate :class:`LSTMNet` from an :class:`ExperimentConfig`."""
    model = LSTMNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    if device is not None:
        model = model.to(device)
    return model
