# Predictive Surrogate Model of Carotid-Artery Hemodynamics (POD + LSTM)

[![CI](https://github.com/SajadSalavati/PSM-RoyalSocietyInterface/actions/workflows/ci.yml/badge.svg)](https://github.com/SajadSalavati/PSM-RoyalSocietyInterface/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A reduced-order **surrogate model** that predicts patient-specific blood-flow
hemodynamics in a carotid artery orders of magnitude faster than full CFD.
It combines **Proper Orthogonal Decomposition (POD)** for spatial model-order
reduction with a **Long Short-Term Memory (LSTM)** network that learns the temporal
evolution of the reduced coordinates from the inlet boundary conditions.

> Companion code for the paper *"A Predictive Surrogate Model of Blood Hemodynamics
> for Patient-Specific Carotid Artery Stenosis"* (Journal of the Royal Society Interface).

---

## How it works

```
   CFD snapshots            POD (SVD)               LSTM                reconstruction
  X ∈ ℝ^(space×time)  ──▶  q = Uᵣᵀ X  ──▶  q̂(t) = f(t, v_in, P_in)  ──▶  X̂ = Uᵣ q̂
   (high-dimensional)     r ≪ space          (temporal model)          (full field)
```

1. **POD** compresses each field (velocity components, pressure, WSS, OSI) into a few
   spatial modes that capture ≥99% of the energy — typically 2–45 modes instead of
   thousands of mesh points.
2. **LSTM** takes the driving signals (time, inlet velocity, inlet pressure) over a
   sliding window and predicts the POD coefficients one step ahead.
3. **Reconstruction** maps the predicted coefficients back to the full spatial field.

Two anatomies (**healthy** / **stenosis**) × two regions (**mid-plane** /
**vessel wall**) × the relevant physical fields give **12 experiments**, each defined
by a single YAML file in [`configs/`](configs/).

---

## Repository structure

```
.
├── src/psm_surrogate/     # the installable package
│   ├── config.py          # ExperimentConfig + YAML loader
│   ├── data.py            # load CFD fields and inlet boundary conditions
│   ├── pod.py             # SVD-based POD, energy & reconstruction error
│   ├── preprocessing.py   # scaling, time-windowing, chronological split
│   ├── lstm.py            # LSTM architecture
│   ├── train.py           # training loop with early stopping + checkpointing
│   ├── infer.py           # sequence rollout / prediction
│   ├── plots.py           # publication-style figures
│   ├── pipeline.py        # stage orchestration (data→POD→train→infer)
│   └── cli.py             # command-line entry point
├── configs/               # 12 experiment configs (replace the 12 old notebooks)
├── notebooks/             # original research notebooks (kept as historical artifacts)
├── results/               # precomputed POD singular values/errors + paper figures
├── data/                  # data guide (raw data lives on Zenodo, git-ignored)
├── tests/                 # pytest unit tests (synthetic data, no downloads)
├── pyproject.toml         # packaging + ruff + pytest config
├── requirements.txt
└── .github/workflows/     # CI (lint + tests on Python 3.9 & 3.11)
```

---

## Installation

```bash
git clone https://github.com/SajadSalavati/PSM-RoyalSocietyInterface.git
cd PSM-RoyalSocietyInterface

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # installs the package + dev tools (pytest, ruff)
```

---

## Usage

Every experiment is driven by a config file — no code editing required:

```bash
# POD only: compute singular values + reconstruction error, save to results/
python -m psm_surrogate.cli --config configs/base_mid_u.yaml --stage pod

# Train the LSTM (best checkpoint saved under models/)
python -m psm_surrogate.cli --config configs/sten_wall_wss.yaml --stage train

# Full pipeline: POD → train → inference
python -m psm_surrogate.cli --config configs/base_mid_u.yaml --stage all --cpu
```

Or from Python:

```python
from psm_surrogate.config import load_config
from psm_surrogate.pipeline import run_pod, run_train

cfg = load_config("configs/base_mid_u.yaml")
pod_result = run_pod(cfg)
model, prepared = run_train(cfg, prefer_cpu=True)
```

---

## Data

Raw CFD archives are **not** version-controlled (they are large binaries). They are
published on **Zenodo** — see [`data/README.md`](data/README.md) for the expected
layout, archive keys, and `BC.csv` format. Small precomputed POD arrays used for the
figures are kept in [`results/`](results/).

---

## Selected results

Cumulative POD energy (few modes capture ≥99%) and per-snapshot reconstruction error:

| Mid-plane velocity | Vessel wall (pressure / WSS) |
| ------------------ | ---------------------------- |
| ![Cumulative energy, mid-plane](results/Cumulative%20Energy%20-%20Mid%20Plane.jpg) | ![Cumulative energy, wall](results/Cumulative%20Energy%20-%20Wall.jpg) |
| ![Reconstruction error, mid-plane](results/Reconstruction%20Error%20-%20Mid%20Plane.jpg) | ![Reconstruction error, wall](results/Reconstruction%20Error%20-%20Wall.jpg) |

---

## Development

```bash
ruff check .     # lint + import sorting
pytest           # run the unit-test suite
```

CI runs both on every push and pull request (see `.github/workflows/ci.yml`).

---

## Citation

If you use this work, please cite the paper:

```bibtex
@article{salavati_psm_carotid,
  title   = {A Predictive Surrogate Model of Blood Hemodynamics for
             Patient-Specific Carotid Artery Stenosis},
  journal = {Journal of the Royal Society Interface},
  author  = {Salavatidezfouli, Sajad and others}
}
```

## License

Released under the [MIT License](LICENSE).
