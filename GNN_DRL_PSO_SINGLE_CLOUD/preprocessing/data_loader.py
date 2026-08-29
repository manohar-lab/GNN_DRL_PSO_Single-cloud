"""
Load and split the QWS Dataset v2.0.

The raw file `QWS2.txt` is comma-separated with 11 columns per data row:
  1  response_time     (ms,     lower is better)
  2  availability      (%,      higher is better)
  3  throughput        (inv/s,  higher is better)
  4  successability    (%,      higher is better)
  5  reliability       (%,      higher is better)
  6  compliance        (%,      higher is better)
  7  best_practices    (%,      higher is better)
  8  latency           (ms,     lower is better)
  9  documentation     (%,      higher is better)
 10  service_name      (str)
 11  wsdl_address      (str)

Header lines start with '##'. We ignore them and preserve the original
file untouched (as required by the problem statement).

Derived column
--------------
`service_classification` (1..4  ->  Platinum/Gold/Silver/Bronze) is
computed with a transparent, deterministic WsRF-quartile rule because
the QWS v2 release removed this column (it only existed in v1). The
derivation happens ONLY on the training partition and is then applied
identically to test using the same quartile boundaries — this avoids
information leakage from test into train.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


QOS_COLUMNS = [
    "response_time",
    "availability",
    "throughput",
    "successability",
    "reliability",
    "compliance",
    "best_practices",
    "latency",
    "documentation",
]
ID_COLUMNS = ["service_name", "wsdl_address"]
ALL_COLUMNS = QOS_COLUMNS + ID_COLUMNS
CLASS_MAP = {1: "Platinum", 2: "Gold", 3: "Silver", 4: "Bronze"}


def load_qws_dataset(raw_path: str | Path) -> pd.DataFrame:
    """Load the raw QWS2.txt into a DataFrame.

    Parameters
    ----------
    raw_path : path to QWS2.txt

    Returns
    -------
    DataFrame with 11 columns (all QoS numeric + service_name + wsdl_address).
    Rows: 2507 (for QWS v2).
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}\n"
            "Place QWS2.txt at data/raw/QWS2.txt before running preprocessing."
        )

    # A handful of records have commas inside the WSDL URL, which breaks
    # naive CSV parsing. Read the file line-by-line and split with
    # maxsplit=10 so the WSDL URL absorbs any trailing commas.
    rows = []
    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("##") or line.startswith("#"):
                continue
            parts = line.rstrip("\r\n").split(",", maxsplit=10)
            if len(parts) != len(ALL_COLUMNS):
                # skip malformed row transparently
                continue
            rows.append(parts)
    df = pd.DataFrame(rows, columns=ALL_COLUMNS)

    # Force numeric on QoS columns
    for col in QOS_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[QOS_COLUMNS].isna().any().any():
        n_bad = df[QOS_COLUMNS].isna().any(axis=1).sum()
        # Report but do not silently drop — problem statement requires
        # transparent handling of missing values.
        print(f"[data_loader] WARNING: {n_bad} rows contain NaN in QoS columns. Dropping.")
        df = df.dropna(subset=QOS_COLUMNS).reset_index(drop=True)

    df["service_name"] = df["service_name"].astype(str).str.strip()
    df["wsdl_address"] = df["wsdl_address"].astype(str).str.strip()

    return df


def _wsrf(df: pd.DataFrame) -> np.ndarray:
    """Compute the WsRF ranking value for each service (higher = better)."""
    scores = np.zeros(len(df), dtype=np.float64)
    for col in QOS_COLUMNS:
        v = df[col].to_numpy(dtype=np.float64)
        vmin, vmax = float(v.min()), float(v.max())
        rng = vmax - vmin
        if rng == 0.0:
            f = np.zeros_like(v)
        elif col in ("response_time", "latency"):
            f = (vmax - v) / rng                   # lower is better
        else:
            f = (v - vmin) / rng                   # higher is better
        scores += f
    return scores / len(QOS_COLUMNS)               # average -> [0, 1]


def _classify_by_quartile(scores: np.ndarray, boundaries: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Bucket WsRF scores into 4 classes:
       Platinum (top 25%) -> 1,  Gold -> 2,  Silver -> 3,  Bronze -> 4.

    If `boundaries` is None it is derived from `scores` (training).
    Otherwise it is reused (testing).
    """
    if boundaries is None:
        boundaries = np.quantile(scores, [0.25, 0.5, 0.75])
    # 4 = Bronze (lowest), 1 = Platinum (highest)
    labels = np.full_like(scores, fill_value=4, dtype=np.int64)
    labels[scores >= boundaries[0]] = 3      # >= Q25 => Silver or better
    labels[scores >= boundaries[1]] = 2      # >= Q50 => Gold or better
    labels[scores >= boundaries[2]] = 1      # >= Q75 => Platinum
    return labels, boundaries


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
    train_out: str | Path,
    test_out: str | Path,
    processed_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Deterministic 80/20 (default) shuffle + write CSVs.

    Also derives `service_classification` on the training set and
    applies the same quartile boundaries to the test set. Boundaries
    are persisted for reproducibility.
    """
    train_out = Path(train_out); test_out = Path(test_out)
    processed_dir = Path(processed_dir)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(round(len(df) * train_ratio))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train_df = df.iloc[train_idx].reset_index(drop=True).copy()
    test_df = df.iloc[test_idx].reset_index(drop=True).copy()

    # Derive Service Classification on train, freeze boundaries, apply to test.
    train_scores = _wsrf(train_df)
    train_labels, boundaries = _classify_by_quartile(train_scores, None)
    train_df["service_classification"] = train_labels
    train_df["service_class_name"] = [CLASS_MAP[int(l)] for l in train_labels]

    test_scores = _wsrf(test_df)
    test_labels, _ = _classify_by_quartile(test_scores, boundaries)
    test_df["service_classification"] = test_labels
    test_df["service_class_name"] = [CLASS_MAP[int(l)] for l in test_labels]

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    metadata = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_ratio": float(train_ratio),
        "split_seed": int(seed),
        "wsrf_quartile_boundaries": boundaries.tolist(),
        "class_map": CLASS_MAP,
        "qos_columns": QOS_COLUMNS,
    }
    (processed_dir / "split_metadata.json").write_text(
        __import__("json").dumps(metadata, indent=2)
    )
    return train_df, test_df, metadata


def load_train_test(train_path: str | Path, test_path: str | Path
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously written train.csv / test.csv."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df
