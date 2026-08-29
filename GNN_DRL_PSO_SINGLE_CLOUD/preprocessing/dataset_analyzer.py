"""Dataset inspection and analysis (Phase 1 of the pipeline)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import QOS_COLUMNS, ID_COLUMNS, CLASS_MAP


def analyze_dataset(df: pd.DataFrame, label: str = "dataset") -> dict:
    """Print a compact schema/QoS report and return it as a dict."""
    report = {"label": label, "n_rows": int(len(df))}
    print("=" * 68)
    print(f"[dataset_analyzer] {label}: {len(df)} rows, {df.shape[1]} columns")
    print("-" * 68)
    print("Columns / dtypes:")
    for c in df.columns:
        na = int(df[c].isna().sum())
        print(f"  {c:<26} {str(df[c].dtype):<10} missing={na}")
    print("-" * 68)
    print("QoS attribute statistics:")
    stats = df[QOS_COLUMNS].describe().T[["min", "max", "mean", "std"]]
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))
    report["qos_stats"] = stats.to_dict()

    if "service_classification" in df.columns:
        counts = df["service_classification"].value_counts().sort_index()
        print("-" * 68)
        print("Service Classification distribution (1=Platinum ... 4=Bronze):")
        for k, v in counts.items():
            print(f"  {int(k)} ({CLASS_MAP[int(k)]:<9}): {int(v)}")
        report["classification_distribution"] = {
            int(k): int(v) for k, v in counts.items()
        }
    print("-" * 68)
    print("Sample records:")
    print(df.head(3).to_string(index=False))
    print("=" * 68)
    return report
