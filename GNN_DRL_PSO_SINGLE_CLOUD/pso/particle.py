"""Particle abstraction for discrete PSO on service compositions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Particle:
    """A candidate composition + swap-tendency velocity.

    position : list of service INDICES (one per abstract task).
    velocity : list of floats in [0, 1] representing the per-slot
               probability of being resampled in the next iteration.
    """
    position: List[int]
    velocity: List[float]
    fitness: float = -np.inf
    pbest_position: List[int] = field(default_factory=list)
    pbest_fitness: float = -np.inf

    def clone_position(self) -> List[int]:
        return list(self.position)
