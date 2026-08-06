#!/usr/bin/env python3
"""与 ``main.py experiments`` 共享同一实现的独立脚本入口。"""

from __future__ import annotations

import argparse

from src.experiments import run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the shared multi-seed experiment suite"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 23, 31, 42])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", default="results/experiment_summary.json")
    args = parser.parse_args()
    result = run_experiment_suite(
        seeds=args.seeds, epochs=args.epochs, output_path=args.output
    )
    for model, metrics in result["aggregate"].items():
        print(model, {name: round(value["mean"], 4) for name, value in metrics.items()})


if __name__ == "__main__":
    main()
