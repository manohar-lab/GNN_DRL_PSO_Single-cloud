"""
GNN-DRL-PSO for QoS-Aware Single-Cloud Service Composition — main entry.

Usage
-----
Run the full pipeline end-to-end:

    python main.py --stage all

Individual stages (recommended for debugging):

    python main.py --stage preprocess
    python main.py --stage train_gnn
    python main.py --stage ablation            # runs all 4 configurations x 5 seeds
    python main.py --stage evaluate --name GNN_DRL_PSO --seed 42

Run a subset of configurations / seeds (fast smoke test):

    python main.py --stage ablation --configs DRL DRL_PSO --seeds 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["all", "preprocess", "train_gnn", "ablation",
                             "evaluate", "report"])
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Subset of ablation configurations to run "
                         "(default: all in config.yaml)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--name", default=None, help="Configuration name for evaluate")
    ap.add_argument("--seed", type=int, default=None, help="Seed for evaluate")
    a = ap.parse_args()

    if a.stage in ("preprocess", "all"):
        from experiments.preprocess import run as pp_run
        pp_run(a.config)

    if a.stage in ("train_gnn", "all"):
        from experiments.train_gnn import run as gnn_run
        gnn_run(a.config)

    if a.stage in ("ablation", "all"):
        from experiments.ablation import run_configs
        from experiments.utils import load_config
        cfg = load_config(a.config)
        configs = a.configs or cfg["ablation"]["configurations"]
        run_configs(configs, a.config, a.seeds)

    if a.stage == "evaluate":
        if a.name is None or a.seed is None:
            ap.error("--stage evaluate requires --name and --seed")
        from experiments.evaluate import evaluate
        print(evaluate(a.name, a.seed, a.config))

    if a.stage == "report":
        # Rebuild ablation tables + statistical comparison + figures
        # from existing run_*.json files (no retraining).
        from experiments.regenerate import rebuild_from_logs
        rebuild_from_logs(a.config, a.configs, a.seeds)


if __name__ == "__main__":
    main()
