"""
Per-experiment metrics + research interpretation.

`interpret_results` generates a natural-language interpretation from
the empirically observed numbers. It explicitly refuses to force the
conclusion that GNN improves the pipeline: the wording follows the
data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _mean_std(df: pd.DataFrame, config: str, key: str = "mean_test_fitness"):
    sub = df[df["config"] == config][key].to_numpy()
    if sub.size == 0:
        return None, None
    return float(sub.mean()), float(sub.std(ddof=0))


def interpret_results(
    results_df: pd.DataFrame,
    tables_dir: str | Path,
    config_names: List[str],
) -> Dict[str, str]:
    """Produce a paragraph-length interpretation string + tag."""
    tables_dir = Path(tables_dir)
    prop_name = "GNN_DRL_PSO" if "GNN_DRL_PSO" in config_names else None
    base_name = "DRL_PSO"     if "DRL_PSO"     in config_names else None
    if prop_name is None or base_name is None:
        return {
            "tag": "insufficient",
            "text": ("Interpretation requires both DRL_PSO and GNN_DRL_PSO "
                     "runs, but at least one is missing from this batch."),
        }

    prop_mean, prop_std = _mean_std(results_df, prop_name)
    base_mean, base_std = _mean_std(results_df, base_name)
    if prop_mean is None or base_mean is None:
        return {"tag": "no_data", "text": "No matching runs to interpret."}

    diff = prop_mean - base_mean
    rel = (diff / base_mean * 100.0) if base_mean != 0 else float("nan")

    prop_time = float(results_df[results_df["config"] == prop_name]
                      ["mean_drl_inference_seconds_per_episode"].mean())
    base_time = float(results_df[results_df["config"] == base_name]
                      ["mean_drl_inference_seconds_per_episode"].mean())
    overhead = prop_time - base_time

    # Read statistical results if available
    stat_path = tables_dir / "statistical_results.csv"
    stat_txt = ""
    tag = "no_evidence"
    if stat_path.exists():
        stats = pd.read_csv(stat_path).iloc[0].to_dict()
        stat_txt = (
            f"Paired evaluation across {len(results_df[results_df['config']==prop_name])} "
            f"seeds gave a mean difference of {stats['mean_diff']:+.4f} "
            f"({stats['rel_diff_pct']:+.2f}%). "
            f"Wilcoxon p-value = {stats['wilcoxon_p']:.4f}. "
            f"Cohen's d = {stats['cohens_d']:.3f}. "
        )
        p = float(stats["wilcoxon_p"])
        d = float(stats["cohens_d"])
        if p < 0.05 and diff > 0:
            tag = "significant_improvement" if abs(d) >= 0.5 else "significant_small_effect"
        elif p < 0.05 and diff < 0:
            tag = "significant_degradation"
        else:
            tag = "insufficient_evidence"

    convergence_note = ""
    if tag == "insufficient_evidence" and diff > 0:
        convergence_note = (
            "GNN produced a small non-significant improvement — this may "
            "reflect improved convergence or generalization that a larger "
            "training budget or more seeds could confirm. "
        )

    if diff > 0:
        cost_note = (
            f"GNN adds an average of {overhead:+.4f}s of DRL inference "
            f"per episode ({prop_time:.4f}s vs {base_time:.4f}s). "
        )
    else:
        cost_note = (
            f"GNN did not improve QoS in this experiment "
            f"({prop_mean:.4f} vs {base_mean:.4f}) and incurs "
            f"{overhead:+.4f}s per-episode DRL overhead. "
        )

    if tag == "significant_improvement":
        headline = ("GNN-based service dependency representations significantly "
                    "improved QoS-aware composition over the DRL-PSO baseline.")
    elif tag == "significant_small_effect":
        headline = ("GNN-based representations produced a statistically "
                    "significant but small effect over the DRL-PSO baseline.")
    elif tag == "significant_degradation":
        headline = ("GNN-based representations significantly REDUCED QoS "
                    "relative to the DRL-PSO baseline in this experiment.")
    else:
        headline = ("There is insufficient statistical evidence that GNN-based "
                    "representations change QoS relative to the DRL-PSO baseline "
                    "in the tested regime.")

    text = (
        f"{headline}\n"
        f"    GNN_DRL_PSO : mean={prop_mean:.4f} std={prop_std:.4f}\n"
        f"    DRL_PSO     : mean={base_mean:.4f} std={base_std:.4f}\n"
        f"    Δ mean = {diff:+.4f}  ({rel:+.2f}%)\n"
        f"    {stat_txt}"
        f"{convergence_note}"
        f"{cost_note}"
        "Note: p > 0.05 does NOT prove the methods are equivalent — "
        "it means the current experimental budget does not provide "
        "sufficient evidence of a statistically significant difference."
    )
    return {"tag": tag, "text": text}
