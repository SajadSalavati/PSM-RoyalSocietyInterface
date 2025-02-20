"""Training loop with early stopping and best-checkpoint saving."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .config import ExperimentConfig
from .lstm import LSTMNet


def resolve_device(prefer_cpu: bool = False) -> torch.device:
    """Pick CUDA when available (and not explicitly disabled), else CPU."""
    if prefer_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: LSTMNet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
    verbose: bool = True,
) -> LSTMNet:
    """Train ``model`` in place, checkpointing the best test loss.

    Mirrors the notebook loop: full-batch-style manual mini-batching, MSE loss,
    Adam, and early stopping on the test loss after ``early_stop_patience`` epochs
    without improvement. The best weights are written to
    ``<models_dir>/<weight_filename>`` and reloaded before returning.
    """
    model = model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    os.makedirs(config.models_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.models_dir, config.weight_filename)

    best_loss = float("inf")
    early_stop_count = 0

    for epoch in range(config.num_epochs):
        model.train()
        for i in range(0, x_train.shape[0], config.batch_size):
            batch_x = x_train[i : i + config.batch_size]
            batch_y = y_train[i : i + config.batch_size]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train)
            test_loss = criterion(model(x_test), y_test)

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}], "
                f"Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}"
            )

        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stop_count += 1
            if early_stop_count == config.early_stop_patience:
                if verbose:
                    print(f"Early stopping after {epoch + 1} epochs")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def load_checkpoint(model: LSTMNet, config: ExperimentConfig, device: torch.device) -> LSTMNet:
    """Load previously trained weights for inference."""
    checkpoint_path = os.path.join(config.models_dir, config.weight_filename)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)
