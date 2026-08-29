"""Train baseline DRL configuration (no GNN, no PSO)."""
from __future__ import annotations

import argparse

from experiments.ablation import run_configs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    run_configs(["DRL"], config_path=a.config)
