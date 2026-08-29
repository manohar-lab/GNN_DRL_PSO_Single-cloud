"""
Rebuild aggregate tables + statistical comparison + figures + research
interpretation from existing run_<CONFIG>_seed<SEED>.json files.

Useful when training was interrupted or when you only want to
regenerate reports without retraining.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiments.utils import (
    add_project_root_to_sys_path,
    load_config,
    resolve_path,
    save_json,
    timestamp,
)

add_project_root_to_sys_path()


def _load_summary(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["summary"]
    except Exception:
        return None


def rebuild_from_logs(config_path: str | None = None,
                      config_names: List[str] | None = None,
                      seeds: List[int] | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    configs = config_names or cfg["ablation"]["configurations"]
    seeds = seeds or cfg["experiment"]["seeds"]

    results_root = resolve_path(cfg["experiment"]["results_root"])
    logs_dir = results_root / "logs"
    tables_dir = results_root / "tables"
    fig_dir = results_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg_name in configs:
        for seed in seeds:
            p = logs_dir / f"run_{cfg_name}_seed{seed}.json"
            s = _load_summary(p)
            if s is None:
                print(f"[regenerate] missing/corrupt {p}")
                continue
            rows.append(s)
    if not rows:
        raise RuntimeError("No run_*.json files found — cannot rebuild.")
    results_df = pd.DataFrame(rows)
    results_df.to_csv(tables_dir / "runs_summary.csv", index=False)
    print(f"[regenerate] wrote {tables_dir/'runs_summary.csv'} "
          f"({len(results_df)} rows)")

    ablation_rows = []
    for name in configs:
        sub = results_df[results_df["config"] == name]
        if sub.empty:
            continue
        ablation_rows.append({
            "Configuration":            name,
            "Mean QoS (test)":          float(sub["mean_test_fitness"].mean()),
            "Std":                      float(sub["mean_test_fitness"].std(ddof=0)),
            "Mean DRL Inference (s)":   float(sub["mean_drl_inference_seconds_per_episode"].mean()),
            "Mean PSO (s)":             float(sub["mean_pso_seconds_per_episode"].mean()),
            "Total Seconds":            float(sub["total_seconds"].mean()),
        })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(tables_dir / "ablation_results.csv", index=False)
    print(f"[regenerate] wrote ablation_results.csv:\n"
          f"{ablation_df.to_string(index=False)}")

    if "GNN_DRL_PSO" in configs and "DRL_PSO" in configs:
        from evaluation.statistics import paired_comparison
        prop = (results_df[results_df["config"] == "GNN_DRL_PSO"]
                .sort_values("seed")["mean_test_fitness"].to_numpy())
        base = (results_df[results_df["config"] == "DRL_PSO"]
                .sort_values("seed")["mean_test_fitness"].to_numpy())
        if len(prop) == len(base) and len(prop) >= 2:
            stat = paired_comparison(prop, base,
                                     name_a="GNN_DRL_PSO", name_b="DRL_PSO")
            pd.DataFrame([stat]).to_csv(tables_dir / "statistical_results.csv",
                                        index=False)
            print(f"[regenerate] paired comparison\n{stat}")

    from evaluation.visualization import generate_all_figures
    generate_all_figures(
        run_json_dir=logs_dir,
        tables_dir=tables_dir,
        figures_dir=fig_dir,
        config_names=configs, seeds=seeds,
    )
    print(f"[regenerate] figures -> {fig_dir}")

    from evaluation.metrics import interpret_results
    interpretation = interpret_results(
        results_df=results_df,
        tables_dir=tables_dir,
        config_names=configs,
    )
    save_json(
        {
            "timestamp":     timestamp(),
            "config_names":  configs,
            "seeds":         seeds,
            "interpretation": interpretation,
        },
        logs_dir / "research_interpretation.json",
    )
    print("\n===== RESEARCH INTERPRETATION =====\n" + interpretation["text"])
    return results_df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    rebuild_from_logs(a.config, a.configs, a.seeds)
