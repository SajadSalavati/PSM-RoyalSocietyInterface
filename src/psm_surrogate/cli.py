"""Command-line entry point for the POD + LSTM surrogate pipeline.

Examples
--------
    # Run POD only and store the singular values / reconstruction error
    python -m psm_surrogate.cli --config configs/base_mid_u.yaml --stage pod

    # Train the LSTM (forcing CPU)
    python -m psm_surrogate.cli --config configs/base_mid_u.yaml --stage train --cpu

    # Full pipeline: POD -> train -> inference
    python -m psm_surrogate.cli --config configs/base_mid_u.yaml --stage all
"""

from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_infer, run_pod, run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psm-surrogate",
        description="POD + LSTM surrogate for patient-specific carotid hemodynamics.",
    )
    parser.add_argument(
        "--config", required=True, help="Path to an experiment YAML (see configs/)."
    )
    parser.add_argument(
        "--stage",
        choices=["pod", "train", "infer", "all"],
        default="all",
        help="Which stage(s) to run.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if a GPU is available."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    print(f"Loaded experiment '{config.name}' from {args.config}")

    if args.stage in ("pod", "all"):
        run_pod(config)
    if args.stage in ("train", "all"):
        run_train(config, prefer_cpu=args.cpu)
    if args.stage in ("infer", "all"):
        preds = run_infer(config, prefer_cpu=True)
        print(f"[{config.name}] inference produced predictions of shape {preds.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
