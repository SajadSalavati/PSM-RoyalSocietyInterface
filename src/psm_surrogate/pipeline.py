"""High-level orchestration of the POD + LSTM stages for one experiment.

Ties together data loading, POD, preprocessing, training and inference so the CLI
(and notebooks) can call a single function per stage instead of repeating the wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from . import data as data_mod
from . import pod as pod_mod
from . import preprocessing as prep
from .config import ExperimentConfig
from .lstm import build_model
from .train import load_checkpoint, resolve_device, train_model


@dataclass
class PreparedData:
    """Everything downstream stages need, computed once from the raw archive."""

    pod: pod_mod.PODResult
    input_scaler: prep.MinMaxScaler1D
    output_scaler: prep.MinMaxScaler1D
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def run_pod(config: ExperimentConfig, save: bool = True) -> pod_mod.PODResult:
    """Load the field, run truncated POD, and (optionally) save S + error to results/."""
    variable = data_mod.load_field(config.data_path, config.field_key)
    result = pod_mod.compute_pod(variable, config.num_modes)
    error = pod_mod.reconstruction_error(variable, result.reconstruction)
    print(
        f"[{config.name}] {config.num_modes} modes -> "
        f"{pod_mod.energy_content(result.S, config.num_modes):.2f}% energy"
    )
    if save:
        os.makedirs(config.results_dir, exist_ok=True)
        out = os.path.join(config.results_dir, config.pod_result_filename)
        np.savez(out, S=result.S, Reconst_error=error)
        print(f"[{config.name}] wrote {out}")
    return result


def prepare_data(config: ExperimentConfig) -> PreparedData:
    """Full preprocessing chain from raw archive to train/test tensors."""
    variable = data_mod.load_field(config.data_path, config.field_key)
    pod_result = pod_mod.compute_pod(variable, config.num_modes)

    t, v_interp, P_interp = data_mod.load_boundary_conditions(
        config.bc_csv, config.n_snapshots
    )
    inputs, outputs = prep.build_io_matrices(t, v_interp, P_interp, pod_result.coefficients)

    input_scaler = prep.MinMaxScaler1D.fit(inputs)
    output_scaler = prep.MinMaxScaler1D.fit(outputs)
    input_scaled = input_scaler.transform(inputs)
    output_scaled = output_scaler.transform(outputs)

    total_x, total_y = prep.make_windows(input_scaled, output_scaled, config.time_window)
    x_train, y_train, x_test, y_test = prep.sequential_split(
        total_x, total_y, config.test_split
    )

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return PreparedData(
        pod=pod_result,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        x_train=to_tensor(x_train),
        y_train=to_tensor(y_train),
        x_test=to_tensor(x_test),
        y_test=to_tensor(y_test),
    )


def run_train(config: ExperimentConfig, prefer_cpu: bool = False):
    """Prepare data and train the LSTM, saving the best checkpoint."""
    prepared = prepare_data(config)
    device = resolve_device(prefer_cpu=prefer_cpu)
    print(f"[{config.name}] training on {device}")
    model = build_model(config, device)
    model = train_model(
        model,
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
        config,
        device,
    )
    return model, prepared


def run_infer(config: ExperimentConfig, prefer_cpu: bool = True) -> np.ndarray:
    """Load the trained checkpoint and predict test-set coefficients."""
    from .infer import predict

    prepared = prepare_data(config)
    device = resolve_device(prefer_cpu=prefer_cpu)
    model = build_model(config, device)
    model = load_checkpoint(model, config, device)
    return predict(model, prepared.x_test, device)
