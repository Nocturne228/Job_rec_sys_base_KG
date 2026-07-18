#!/usr/bin/env python3
from src.experiments import run_experiment_suite

if __name__ == "__main__":
    result = run_experiment_suite()
    for model, metrics in result["aggregate"].items():
        print(model, {name: round(value["mean"], 4) for name, value in metrics.items()})
