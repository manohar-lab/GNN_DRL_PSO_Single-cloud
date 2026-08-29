"""
Phase 13-15 orchestrator.

Runs every configuration in `cfg['ablation']['configurations']` across
every seed in `cfg['experiment']['seeds']`. Persists per-run JSONs and
per-config CSV summaries into results/tables/ + results/logs/ + generates
figures into results/figures/.

Once all runs finish it invokes evaluation.statistics to compute the
paired comparison between GNN_DRL_PSO and DRL_PSO (Phase 14) and
evaluation.visualization to produce the standard figure set (Phase 15).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from experiments.utils import (
    add_project_root_to_sys_path,
    load_config,
    make_run_logger,
    pick_device,
    resolve_path,
    save_json,
    set_seed,
    timestamp,
)

add_project_root_to_sys_path()

from preprocessing.data_loader import load_train_test              # noqa: E402
from preprocessing.normalizer import QoSNormalizer                 # noqa: E402
from experiments.runner import train_and_evaluate                  # noqa: E402


def _serialize_metrics(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compress episode metrics for JSON storage."""
    out = []
    for m in metrics:
        out.append({
            "episode":           m["episode"],
            "seconds":           m.get("seconds", 0.0),
            "initial_fitness":   m["initial_fitness"],
            "final_fitness":     m["final_fitness"],
            "task_sequence":     m["task_sequence"],
            "final_composition": m["final_composition"],
            "valid_steps":       m["valid_steps"],
            "success":           m["success"],
            "normalized_qos":    m["normalized_qos"],
            "raw_qos":           m["raw_qos"],
            "pso_seconds":       m.get("pso_seconds", 0.0),
            "pso_history":       m.get("pso_history", []),
        })
    return out


def run_configs(config_names: List[str], config_path: str | None = None,
                seeds: List[int] | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    if seeds is None:
        seeds = cfg["experiment"]["seeds"]
    device = pick_device(cfg["experiment"]["device"])

    logger = make_run_logger(
        "ablation",
        resolve_path(cfg["experiment"]["results_root"]) / "logs",
    )
    logger.info(f"device={device}")
    logger.info(f"configs={config_names}  seeds={seeds}")
    logger.info(f"episodes/seed={cfg['experiment']['num_episodes']}")

    train_df, test_df = load_train_test(
        resolve_path(cfg["dataset"]["train_path"]),
        resolve_path(cfg["dataset"]["test_path"]),
    )
    normalizer = QoSNormalizer.load(
        resolve_path(cfg["dataset"]["processed_dir"]) / "normalizer.json"
    )

    logs_dir = resolve_path(cfg["experiment"]["results_root"]) / "logs"
    ckpt_dir = resolve_path(cfg["experiment"]["checkpoints_root"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for config_name in config_names:
        for seed in seeds:
            logger.info(f"\n===== {config_name}  seed={seed}  =====")
            set_seed(seed)
            t0 = time.time()
            result = train_and_evaluate(
                config_name=config_name,
                seed=seed,
                cfg=cfg,
                train_df=train_df,
                test_df=test_df,
                normalizer=normalizer,
                device=device,
                logger=logger,
                checkpoint_dir=ckpt_dir,
            )
            dt = time.time() - t0
            summary = result["summary"]
            summary["total_seconds"] = dt
            all_rows.append(summary)
            # Persist per-run JSON
            run_json = {
                "summary":        summary,
                "train_metrics":  _serialize_metrics(result["train_metrics"]),
                "test_metrics":   _serialize_metrics(result["test_metrics"]),
            }
            save_json(run_json, logs_dir / f"run_{config_name}_seed{seed}.json")
            logger.info(f"[done] {config_name}/seed{seed} in {dt:.1f}s "
                        f"train_last10={summary['mean_train_fitness_last_10']:.4f} "
                        f"test_mean={summary['mean_test_fitness']:.4f}")

    results_df = pd.DataFrame(all_rows)
    tables_dir = resolve_path(cfg["experiment"]["results_root"]) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "runs_summary.csv", index=False)
    logger.info(f"wrote runs_summary.csv -> {tables_dir/'runs_summary.csv'}")

    # -----------------------------------------------------------------
    # Ablation aggregate table
    # -----------------------------------------------------------------
    ablation_rows = []
    for name in config_names:
        sub = results_df[results_df["config"] == name]
        ablation_rows.append({
            "Configuration":       name,
            "Mean QoS (test)":     float(sub["mean_test_fitness"].mean()),
            "Std":                 float(sub["mean_test_fitness"].std(ddof=0)),
            "Mean DRL Inference (s)":
                float(sub["mean_drl_inference_seconds_per_episode"].mean()),
            "Mean PSO (s)":        float(sub["mean_pso_seconds_per_episode"].mean()),
            "Total Seconds":       float(sub["total_seconds"].mean()),
        })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(tables_dir / "ablation_results.csv", index=False)
    logger.info(f"wrote ablation_results.csv:\n{ablation_df.to_string(index=False)}")

    # -----------------------------------------------------------------
    # Statistical comparison (only when both proposed and baseline
    # are in the run set)
    # -----------------------------------------------------------------
    if "GNN_DRL_PSO" in config_names and "DRL_PSO" in config_names:
        from evaluation.statistics import paired_comparison
        prop = (results_df[results_df["config"] == "GNN_DRL_PSO"]
                .sort_values("seed")["mean_test_fitness"].to_numpy())
        base = (results_df[results_df["config"] == "DRL_PSO"]
                .sort_values("seed")["mean_test_fitness"].to_numpy())
        stat = paired_comparison(prop, base,
                                 name_a="GNN_DRL_PSO", name_b="DRL_PSO")
        pd.DataFrame([stat]).to_csv(tables_dir / "statistical_results.csv",
                                    index=False)
        logger.info(f"paired comparison\n{stat}")
    else:
        logger.info("Skipping statistical comparison (need both DRL_PSO and GNN_DRL_PSO)")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    from evaluation.visualization import generate_all_figures
    fig_dir = resolve_path(cfg["experiment"]["results_root"]) / "figures"
    generate_all_figures(
        run_json_dir=logs_dir,
        tables_dir=tables_dir,
        figures_dir=fig_dir,
        config_names=config_names,
        seeds=seeds,
    )
    logger.info(f"figures written to {fig_dir}")

    # -----------------------------------------------------------------
    # Research interpretation
    # -----------------------------------------------------------------
    from evaluation.metrics import interpret_results
    interpretation = interpret_results(
        results_df=results_df,
        tables_dir=tables_dir,
        config_names=config_names,
    )
    save_json(
        {
            "timestamp":     timestamp(),
            "config_names":  config_names,
            "seeds":         seeds,
            "interpretation": interpretation,
        },
        resolve_path(cfg["experiment"]["results_root"]) /
        "logs" / "research_interpretation.json",
    )
    logger.info("\n===== RESEARCH INTERPRETATION =====\n" + interpretation["text"])

    return results_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Subset of configurations to run.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    configs = a.configs or cfg["ablation"]["configurations"]
    run_configs(configs, a.config, a.seeds)


if __name__ == "__main__":
    main()
