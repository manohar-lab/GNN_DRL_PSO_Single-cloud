"""
Discrete Particle Swarm Optimization for service composition.

Encoding
--------
A particle is a list of service INDICES (one per abstract task of the
workflow).  Each slot is drawn from `candidates[task]`, i.e. the list
of services whose classification matches the abstract task at that
position.  This guarantees every particle is a syntactically valid
composition.

Velocity update (adapted for discrete space)
--------------------------------------------
For each slot i we maintain a per-slot "swap probability"

    v_i(t+1) = w * v_i(t)
             + c1 * r1 * I(x_i != pbest_i)
             + c2 * r2 * I(x_i != gbest_i)

Then, with probability sigmoid(v_i(t+1)), slot i is RESAMPLED:
   * If a swap is triggered we pick a service from the same candidate
     pool.  With probability 0.5 we bias toward the gbest slot value
     (if the class matches), otherwise a uniform draw over
     `candidates[task_i]`.

This construction respects:
   * the discrete nature of the problem,
   * the classification constraint of every slot,
   * the standard w / c1 / c2 semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Optional

import numpy as np

from .particle import Particle


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


@dataclass
class PSOResult:
    best_position: List[int]
    best_fitness: float
    history: List[float]                       # per iteration best-so-far
    n_evaluations: int


class DiscretePSO:
    def __init__(
        self,
        candidates_by_task: List[np.ndarray],
        fitness_fn: Callable[[Sequence[int]], float],
        num_particles: int = 20,
        num_iterations: int = 15,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        init_perturb: float = 0.4,
        rng: Optional[np.random.Generator] = None,
    ):
        self.candidates = [np.asarray(c, dtype=np.int64) for c in candidates_by_task]
        self.fitness_fn = fitness_fn
        self.num_particles = int(num_particles)
        self.num_iterations = int(num_iterations)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.init_perturb = float(init_perturb)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_evaluations = 0

    # ------------------------------------------------------------------
    def _sample_slot(self, task_idx: int, avoid: Optional[int] = None) -> int:
        cand = self.candidates[task_idx]
        if cand.size == 1:
            return int(cand[0])
        # Sample uniform, rejecting `avoid` when possible
        for _ in range(6):
            v = int(self.rng.choice(cand))
            if v != avoid:
                return v
        return int(self.rng.choice(cand))

    def _init_population(self, seed_position: Sequence[int]) -> List[Particle]:
        n_tasks = len(seed_position)
        population: List[Particle] = []
        seed_arr = np.asarray(seed_position, dtype=np.int64)
        for p in range(self.num_particles):
            if p == 0:
                pos = list(seed_arr)
            else:
                pos = list(seed_arr)
                n_perturb = max(1, int(round(self.init_perturb * n_tasks)))
                slots = self.rng.choice(n_tasks, size=n_perturb, replace=False)
                for s in slots:
                    pos[int(s)] = self._sample_slot(int(s), avoid=pos[int(s)])
            velocity = list(self.rng.uniform(-0.5, 0.5, size=n_tasks))
            f = float(self.fitness_fn(pos))
            self.n_evaluations += 1
            population.append(
                Particle(
                    position=pos,
                    velocity=velocity,
                    fitness=f,
                    pbest_position=list(pos),
                    pbest_fitness=f,
                )
            )
        return population

    # ------------------------------------------------------------------
    def optimize(self, seed_position: Sequence[int]) -> PSOResult:
        pop = self._init_population(seed_position)
        gbest_idx = int(np.argmax([p.fitness for p in pop]))
        gbest_position = list(pop[gbest_idx].position)
        gbest_fitness = float(pop[gbest_idx].fitness)
        history = [gbest_fitness]

        n_tasks = len(seed_position)
        for _ in range(self.num_iterations):
            for particle in pop:
                new_position = list(particle.position)
                for i in range(n_tasks):
                    r1, r2 = float(self.rng.random()), float(self.rng.random())
                    diff_p = 1.0 if particle.position[i] != particle.pbest_position[i] else 0.0
                    diff_g = 1.0 if particle.position[i] != gbest_position[i] else 0.0
                    particle.velocity[i] = (
                        self.w * particle.velocity[i]
                        + self.c1 * r1 * diff_p
                        + self.c2 * r2 * diff_g
                    )
                    swap_prob = _sigmoid(particle.velocity[i])
                    if self.rng.random() < swap_prob:
                        # Prefer copying from gbest when it belongs to the
                        # right candidate pool (it always does — same task).
                        if self.rng.random() < 0.5:
                            new_position[i] = int(gbest_position[i])
                        else:
                            new_position[i] = self._sample_slot(
                                i, avoid=particle.position[i]
                            )
                particle.position = new_position
                particle.fitness = float(self.fitness_fn(new_position))
                self.n_evaluations += 1
                if particle.fitness > particle.pbest_fitness:
                    particle.pbest_fitness = particle.fitness
                    particle.pbest_position = list(new_position)
                if particle.fitness > gbest_fitness:
                    gbest_fitness = particle.fitness
                    gbest_position = list(new_position)
            history.append(gbest_fitness)

        return PSOResult(
            best_position=list(gbest_position),
            best_fitness=float(gbest_fitness),
            history=history,
            n_evaluations=int(self.n_evaluations),
        )
