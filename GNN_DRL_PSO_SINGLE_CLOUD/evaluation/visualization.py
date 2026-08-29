"""
Publication-style plots.

All figures are 100% reproducible from the JSON logs under results/logs/
and the CSV tables under results/tables/. Do NOT edit values by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)


def _load_run(run_dir: Path, cfg_name: str, seed: int) -> Optional[dict]:
    p = run_dir / f"run_{cfg_name}_seed{seed}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------
def qos_comparison_bar(ablation_df: pd.DataFrame, out: Path):
    plt.figure(figsize=(7, 4))
    order = list(ablation_df["Configuration"])
    means = list(ablation_df["Mean QoS (test)"])
    stds  = list(ablation_df["Std"])
    ax = plt.bar(order, means, yerr=stds, capsize=4, color=sns.color_palette("deep", len(order)))
    plt.ylabel("Mean test QoS fitness")
    plt.title("QoS Comparison (mean ± std across seeds)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def convergence_curve(run_dir: Path, config_names: List[str],
                      seeds: List[int], out: Path):
    plt.figure(figsize=(7, 4))
    for cfg in config_names:
        curves = []
        for s in seeds:
            r = _load_run(run_dir, cfg, s)
            if r is None:
                continue
            best_so_far = []
            best = -np.inf
            for m in r["train_metrics"]:
                best = max(best, m["final_fitness"])
                best_so_far.append(best)
            curves.append(best_so_far)
        if not curves:
            continue
        L = min(len(c) for c in curves)
        curves = np.asarray([c[:L] for c in curves])
        m = curves.mean(axis=0)
        s = curves.std(axis=0)
        x = np.arange(L)
        plt.plot(x, m, label=cfg, linewidth=2)
        plt.fill_between(x, m - s, m + s, alpha=0.15)
    plt.xlabel("Training episode")
    plt.ylabel("Best-so-far QoS fitness")
    plt.title("Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def execution_time_bar(ablation_df: pd.DataFrame, out: Path):
    plt.figure(figsize=(7, 4))
    x = np.arange(len(ablation_df))
    w = 0.35
    plt.bar(x - w/2, ablation_df["Mean DRL Inference (s)"],
            w, label="DRL inference (s/ep)")
    plt.bar(x + w/2, ablation_df["Mean PSO (s)"],
            w, label="PSO refinement (s/ep)")
    plt.xticks(x, ablation_df["Configuration"], rotation=15)
    plt.ylabel("Seconds per episode")
    plt.title("Execution-time breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def qos_metrics_bar(run_dir: Path, config_names: List[str],
                    seeds: List[int], out: Path):
    """Per-attribute mean normalized QoS across test episodes."""
    rows = []
    for cfg in config_names:
        for s in seeds:
            r = _load_run(run_dir, cfg, s)
            if r is None:
                continue
            for m in r["test_metrics"]:
                for k, v in m["normalized_qos"].items():
                    rows.append({"config": cfg, "attr": k, "val": float(v)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    agg = df.groupby(["config", "attr"])["val"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=agg, x="attr", y="val", hue="config")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean normalized QoS (test)")
    plt.title("Per-attribute QoS")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def ablation_bar(ablation_df: pd.DataFrame, out: Path):
    plt.figure(figsize=(7, 4))
    order = list(ablation_df["Configuration"])
    means = list(ablation_df["Mean QoS (test)"])
    plt.bar(order, means, color=sns.color_palette("mako", len(order)))
    plt.ylabel("Mean test QoS fitness")
    plt.title("Ablation study")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def overhead_scatter(ablation_df: pd.DataFrame, out: Path):
    plt.figure(figsize=(6, 4))
    for _, row in ablation_df.iterrows():
        plt.scatter(row["Mean DRL Inference (s)"] + row["Mean PSO (s)"],
                    row["Mean QoS (test)"], s=80,
                    label=row["Configuration"])
    plt.xlabel("Seconds per episode (DRL + PSO)")
    plt.ylabel("Mean QoS")
    plt.title("Cost vs quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def fitness_distribution(run_dir: Path, config_names: List[str],
                         seeds: List[int], out: Path):
    rows = []
    for cfg in config_names:
        for s in seeds:
            r = _load_run(run_dir, cfg, s)
            if r is None:
                continue
            for m in r["test_metrics"]:
                rows.append({"config": cfg,
                             "fitness": float(m["final_fitness"])})
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="config", y="fitness")
    sns.stripplot(data=df, x="config", y="fitness", color="black",
                  alpha=0.3, size=2.5)
    plt.xticks(rotation=15)
    plt.ylabel("Test-episode QoS fitness")
    plt.title("Distribution of test-episode fitness")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------
def generate_all_figures(
    run_json_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    config_names: List[str],
    seeds: List[int],
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    ablation_csv = tables_dir / "ablation_results.csv"
    if not ablation_csv.exists():
        print(f"[visualization] {ablation_csv} missing; skipping figures.")
        return
    ablation_df = pd.read_csv(ablation_csv)
    qos_comparison_bar(ablation_df, figures_dir / "qos_comparison.png")
    convergence_curve(run_json_dir, config_names, seeds,
                      figures_dir / "convergence.png")
    execution_time_bar(ablation_df, figures_dir / "execution_time.png")
    qos_metrics_bar(run_json_dir, config_names, seeds,
                    figures_dir / "qos_metrics.png")
    ablation_bar(ablation_df, figures_dir / "ablation.png")
    overhead_scatter(ablation_df, figures_dir / "overhead.png")
    fitness_distribution(run_json_dir, config_names, seeds,
                         figures_dir / "fitness_distribution.png")
