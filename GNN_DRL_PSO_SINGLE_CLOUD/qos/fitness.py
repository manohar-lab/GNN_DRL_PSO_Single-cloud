"""
QoS Fitness for a service composition.

Given a list of selected service indices (one per abstract task), the
fitness function:

  1. Aggregates the raw QoS attributes across the composition using
     a per-attribute aggregation rule that reflects sequential
     workflow execution:

        response_time, latency        : SUM      (times add)
        cost                          : SUM      (not present in QWS2)
        throughput                    : MIN      (bottleneck)
        availability, reliability,
        successability                : PRODUCT  (independent probabilities)
        compliance, best_practices,
        documentation                 : MEAN     (structural averages)

     These rules are configurable via `aggregation_rules`.

  2. Normalizes each aggregated attribute to [0, 1] using the min/max
     of the underlying training-set raw QoS (fed via `qos_min` /
     `qos_max`) then flips direction so that 1 = best.

  3. Returns the weighted sum F(x) = sum_j w_j * f_j(q_j(x)) in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


DEFAULT_AGGREGATION: Dict[str, str] = {
    "response_time": "sum",
    "latency": "sum",
    "throughput": "min",
    "availability": "prod",
    "reliability": "prod",
    "successability": "prod",
    "compliance": "mean",
    "best_practices": "mean",
    "documentation": "mean",
}


@dataclass
class FitnessResult:
    fitness: float
    raw_agg: Dict[str, float]              # aggregated raw QoS per attribute
    normalized: Dict[str, float]           # per-attribute [0, 1] score
    weighted_contribution: Dict[str, float]
    valid: bool
    reason: str = ""


class QoSFitness:
    """Config-driven QoS fitness evaluator."""

    def __init__(
        self,
        qos_columns: Sequence[str],
        weights: Dict[str, float],
        directions: Dict[str, str],            # 'max'|'min' per attribute
        qos_min: Dict[str, float],
        qos_max: Dict[str, float],
        aggregation_rules: Optional[Dict[str, str]] = None,
        invalid_penalty: float = 0.0,
    ):
        self.qos_columns = list(qos_columns)
        self.weights = {c: float(weights[c]) for c in self.qos_columns}
        self.directions = {c: str(directions[c]) for c in self.qos_columns}
        self.qos_min = {c: float(qos_min[c]) for c in self.qos_columns}
        self.qos_max = {c: float(qos_max[c]) for c in self.qos_columns}
        self.aggregation_rules = {
            c: (aggregation_rules or DEFAULT_AGGREGATION).get(c, "mean")
            for c in self.qos_columns
        }
        self.invalid_penalty = float(invalid_penalty)

        w_sum = sum(self.weights.values())
        if not np.isclose(w_sum, 1.0, atol=1e-3):
            raise ValueError(
                f"QoS weights must sum to 1.0, got {w_sum:.4f}. "
                f"Weights: {self.weights}"
            )

    # ------------------------------------------------------------------
    def _aggregate(self, raw_matrix: np.ndarray) -> Dict[str, float]:
        """raw_matrix shape: (n_services_in_composition, n_qos)."""
        out: Dict[str, float] = {}
        for i, col in enumerate(self.qos_columns):
            v = raw_matrix[:, i]
            rule = self.aggregation_rules[col]
            if rule == "sum":
                out[col] = float(v.sum())
            elif rule == "min":
                out[col] = float(v.min())
            elif rule == "prod":
                # Convert % to probability if values look like percentages
                if float(v.max()) > 1.0:
                    p = np.clip(v / 100.0, 0.0, 1.0)
                    out[col] = float(np.prod(p) * 100.0)
                else:
                    out[col] = float(np.prod(v))
            elif rule == "mean":
                out[col] = float(v.mean())
            else:
                raise ValueError(f"Unknown aggregation rule '{rule}' for {col}")
        return out

    def _normalize_and_weight(self, agg: Dict[str, float]
                              ) -> tuple[Dict[str, float], Dict[str, float]]:
        normalized: Dict[str, float] = {}
        contribs: Dict[str, float] = {}
        for c in self.qos_columns:
            qmin, qmax = self.qos_min[c], self.qos_max[c]
            rng = qmax - qmin
            v = agg[c]
            if rng == 0:
                n = 0.0
            elif self.directions[c] == "max":
                n = (v - qmin) / rng
            else:
                n = (qmax - v) / rng
            # For attributes whose aggregation may push values outside
            # the per-service min/max range (e.g. summed latency),
            # simply clip. This is standard practice in the QoS-aware
            # service composition literature.
            n = float(np.clip(n, 0.0, 1.0))
            normalized[c] = n
            contribs[c] = self.weights[c] * n
        return normalized, contribs

    # ------------------------------------------------------------------
    def evaluate(
        self,
        service_indices: Sequence[int],
        qos_matrix: np.ndarray,                # (n_total_services, n_qos)
        valid: bool = True,
        reason: str = "",
    ) -> FitnessResult:
        """Return the fitness of one composition.

        `service_indices` are integer positions into `qos_matrix`.
        `qos_matrix` columns must be in the same order as `qos_columns`.
        """
        if not valid:
            return FitnessResult(
                fitness=self.invalid_penalty,
                raw_agg={c: 0.0 for c in self.qos_columns},
                normalized={c: 0.0 for c in self.qos_columns},
                weighted_contribution={c: 0.0 for c in self.qos_columns},
                valid=False,
                reason=reason,
            )
        if len(service_indices) == 0:
            return FitnessResult(
                fitness=self.invalid_penalty,
                raw_agg={c: 0.0 for c in self.qos_columns},
                normalized={c: 0.0 for c in self.qos_columns},
                weighted_contribution={c: 0.0 for c in self.qos_columns},
                valid=False,
                reason="empty composition",
            )
        idx = np.asarray(service_indices, dtype=np.int64)
        raw = qos_matrix[idx, :]
        agg = self._aggregate(raw)
        normalized, contribs = self._normalize_and_weight(agg)
        f = float(sum(contribs.values()))
        return FitnessResult(
            fitness=f,
            raw_agg=agg,
            normalized=normalized,
            weighted_contribution=contribs,
            valid=True,
            reason=reason,
        )
