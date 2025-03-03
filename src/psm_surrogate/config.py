"""Experiment configuration.

A single :class:`ExperimentConfig` replaces the hand-edited constants that were
duplicated across the twelve notebooks (``model``, ``bc_name``, ``num_modes``,
hyper-parameters, hard-coded Google-Drive paths, ...). Configs are stored as YAML
under ``configs/`` so a run is fully described by data, not by editing code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import yaml


@dataclass
class ExperimentConfig:
    """All knobs for one POD+LSTM experiment.

    Attributes are grouped conceptually:

    * identity: ``name``, ``model``, ``bc_name``, ``variable_name``
    * data:     ``field_key``, ``npz_filename``, ``data_root``, ``bc_csv``, ``n_snapshots``
    * POD:      ``num_modes``
    * sequence: ``time_window``, ``test_split``
    * LSTM:     ``hidden_size`` ... ``early_stop_patience``
    * outputs:  ``models_dir``, ``results_dir``
    """

    # --- identity ---------------------------------------------------------
    name: str
    model: str          # "Base" (healthy) | "Sten" (stenosis)
    bc_name: str        # "Mid" (mid-plane) | "Wall"
    variable_name: str  # human label, e.g. "v_x", "Pressure", "WSS"

    # --- data -------------------------------------------------------------
    field_key: str      # key inside the .npz, e.g. "v_x", "pre", "wss", "osi"
    npz_filename: str
    data_root: str = "data/raw"
    bc_csv: str = "data/raw/BC.csv"
    n_snapshots: int = 1000

    # --- POD --------------------------------------------------------------
    num_modes: int = 12

    # --- sequence construction -------------------------------------------
    time_window: int = 100
    test_split: float = 0.8

    # --- LSTM hyper-parameters -------------------------------------------
    hidden_size: int = 512
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 1.0e-5
    num_epochs: int = 200000
    batch_size: int = 20
    early_stop_patience: int = 100

    # --- outputs ----------------------------------------------------------
    models_dir: str = "models"
    results_dir: str = "results"

    # extra fields tolerated but ignored (forward compatibility)
    extra: dict = field(default_factory=dict)

    @property
    def data_path(self) -> str:
        """Full path to the CFD snapshot archive for this experiment."""
        return os.path.join(self.data_root, self.model, self.bc_name, self.npz_filename)

    @property
    def input_size(self) -> int:
        """LSTM input width = POD modes + 3 driving signals (t, velocity, pressure)."""
        return self.num_modes + 3

    @property
    def output_size(self) -> int:
        return self.num_modes

    @property
    def weight_filename(self) -> str:
        return f"LSTMNet_{self.model}_{self.bc_name}_{self.variable_name}.pt"

    @property
    def pod_result_filename(self) -> str:
        return f"POD_data_{self.model}_{self.bc_name}_{self.variable_name}.npz"

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Unknown keys are collected into ``extra`` instead of raising, so newer configs
    stay loadable by older code.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    known = {f for f in ExperimentConfig.__dataclass_fields__ if f != "extra"}
    kwargs = {k: v for k, v in raw.items() if k in known}
    extra = {k: v for k, v in raw.items() if k not in known}
    if extra:
        kwargs["extra"] = extra
    return ExperimentConfig(**kwargs)
