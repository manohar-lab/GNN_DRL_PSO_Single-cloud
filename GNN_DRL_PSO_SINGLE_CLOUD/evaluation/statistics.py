"""Paired comparison statistics (Wilcoxon + paired t + effect sizes)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats as sstats


def paired_comparison(a: np.ndarray, b: np.ndarray,
                      name_a: str = "A", name_b: str = "B",
                      alpha: float = 0.05) -> Dict[str, float]:
    """Compare two paired samples (one entry per seed).

    Returns a dict of statistics suitable for CSV export.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.size < 2:
        return {
            "name_a": name_a, "name_b": name_b,
            "n_pairs": int(a.size),
            "mean_a": float(a.mean()) if a.size else float("nan"),
            "mean_b": float(b.mean()) if b.size else float("nan"),
            "mean_diff": float("nan"),
            "rel_diff_pct": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
            "ttest_stat": float("nan"),
            "ttest_p": float("nan"),
            "shapiro_p": float("nan"),
            "cohens_d": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "alpha": alpha,
            "conclusion": "not_enough_pairs",
        }

    diff = a - b
    mean_a = float(a.mean()); mean_b = float(b.mean())
    mean_d = float(diff.mean()); std_d = float(diff.std(ddof=1) if diff.size > 1 else 0.0)
    rel = (mean_d / mean_b * 100.0) if mean_b != 0 else float("nan")

    # Wilcoxon (nonparametric)
    try:
        w_stat, w_p = sstats.wilcoxon(diff, zero_method="wilcox",
                                      alternative="two-sided", nan_policy="propagate")
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    # Paired t
    try:
        t_stat, t_p = sstats.ttest_rel(a, b, nan_policy="propagate")
    except Exception:
        t_stat, t_p = float("nan"), float("nan")
    # Shapiro-Wilk on paired differences
    try:
        _, sh_p = sstats.shapiro(diff)
    except Exception:
        sh_p = float("nan")
    # Cohen's d (paired)
    cohens = mean_d / std_d if std_d > 0 else 0.0
    # 95% CI on mean difference (t-based)
    n = diff.size
    if n > 1 and std_d > 0:
        se = std_d / np.sqrt(n)
        tcrit = float(sstats.t.ppf(1 - alpha / 2.0, df=n - 1))
        ci_low  = mean_d - tcrit * se
        ci_high = mean_d + tcrit * se
    else:
        ci_low = ci_high = float("nan")

    if not np.isfinite(w_p):
        conclusion = "test_undefined"
    elif w_p < alpha and mean_d > 0:
        conclusion = f"{name_a} > {name_b} (significant)"
    elif w_p < alpha and mean_d < 0:
        conclusion = f"{name_a} < {name_b} (significant)"
    else:
        conclusion = "insufficient_evidence_of_difference"

    return {
        "name_a": name_a, "name_b": name_b,
        "n_pairs": int(n),
        "mean_a": mean_a, "mean_b": mean_b,
        "mean_diff": mean_d,
        "rel_diff_pct": float(rel),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p":    float(w_p),
        "ttest_stat":    float(t_stat),
        "ttest_p":       float(t_p),
        "shapiro_p":     float(sh_p),
        "cohens_d":      float(cohens),
        "ci_low":        float(ci_low),
        "ci_high":       float(ci_high),
        "alpha":         float(alpha),
        "conclusion":    conclusion,
    }
