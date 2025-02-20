"""POD + LSTM surrogate model for patient-specific carotid-artery hemodynamics.

The package turns the twelve near-identical research notebooks into a single,
configuration-driven pipeline:

    CFD snapshots --> POD (SVD reduction) --> LSTM (temporal coefficients) --> reconstruction

Each experiment (healthy/stenosis, mid-plane/wall, velocity/pressure/WSS/OSI) is
described by a small YAML file under ``configs/`` instead of a copy of the code.
"""

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
__version__ = "0.1.0"
