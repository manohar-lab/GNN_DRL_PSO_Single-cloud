"""Per-attribute QoS normalization utilities (used by the fitness function)."""
from __future__ import annotations

from typing import Dict

import numpy as np


def normalize_qos(
    raw_values: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    directions: np.ndarray,           # 1 == higher is better, 0 == lower is better
) -> np.ndarray:
    """Vectorized min-max normalization respecting direction.

    Parameters
    ----------
    raw_values : (D,) or (N, D)
    q_min, q_max, directions : (D,)

    Returns
    -------
    (D,) or (N, D) normalized values in [0, 1]  (1 = best).
    """
    raw_values = np.asarray(raw_values, dtype=np.float64)
    q_min = np.asarray(q_min, dtype=np.float64)
    q_max = np.asarray(q_max, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.int64)
    rng = np.where((q_max - q_min) == 0, 1.0, q_max - q_min)

    higher = (raw_values - q_min) / rng
    lower = (q_max - raw_values) / rng
    out = np.where(directions == 1, higher, lower)
    return np.clip(out, 0.0, 1.0)
