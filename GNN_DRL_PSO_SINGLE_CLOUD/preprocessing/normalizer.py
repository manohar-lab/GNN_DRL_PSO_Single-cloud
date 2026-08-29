"""
QoS normalizer.

Fitted ONLY on the training set. The same min/max are reused on the
test set — this avoids information leakage from the test partition
into training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .data_loader import QOS_COLUMNS


class QoSNormalizer:
    """Min-max normalizer for QoS attributes.

    Parameters
    ----------
    directions : mapping column -> 'min' | 'max'
        - 'max' (higher is better): f(q) = (q - qmin) / (qmax - qmin)
        - 'min' (lower is better):  f(q) = (qmax - q) / (qmax - qmin)

    The normalized value is in [0, 1] where 1 always means BEST.
    """

    def __init__(self, directions: Dict[str, str]):
        self.directions = dict(directions)
        self.q_min: Dict[str, float] = {}
        self.q_max: Dict[str, float] = {}
        self.fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, columns: Iterable[str] = QOS_COLUMNS) -> "QoSNormalizer":
        for c in columns:
            v = df[c].to_numpy(dtype=np.float64)
            self.q_min[c] = float(np.min(v))
            self.q_max[c] = float(np.max(v))
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, columns: Iterable[str] = QOS_COLUMNS) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("QoSNormalizer.transform called before .fit()")
        out = df.copy()
        for c in columns:
            v = df[c].to_numpy(dtype=np.float64)
            qmin, qmax = self.q_min[c], self.q_max[c]
            rng = qmax - qmin
            if rng == 0:
                nv = np.zeros_like(v)
            elif self.directions[c] == "max":
                nv = (v - qmin) / rng
            else:                     # 'min' - lower is better
                nv = (qmax - v) / rng
            # Clip to [0, 1] in case test has values outside train range
            out[c] = np.clip(nv, 0.0, 1.0)
        return out

    def fit_transform(self, df, columns=QOS_COLUMNS):
        return self.fit(df, columns).transform(df, columns)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "directions": self.directions,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "fitted": self.fitted,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "QoSNormalizer":
        p = json.loads(Path(path).read_text())
        norm = cls(p["directions"])
        norm.q_min = {k: float(v) for k, v in p["q_min"].items()}
        norm.q_max = {k: float(v) for k, v in p["q_max"].items()}
        norm.fitted = bool(p["fitted"])
        return norm
