"""
Single-cloud service composition environment.

State
-----
At each step the agent observes:
  * a compact numerical state vector containing
      - the current task's one-hot classification tag
      - a summary of already selected services' normalized QoS
      - a progress indicator (step / workflow_length)
  * (optionally) the fixed graph embedding matrix -- exposed via
    `get_graph_embeddings()` so agents that use GNN can concatenate
    it into their policy input.

Action
------
Discrete over the FULL catalog of candidate services (size = n_services).
Services whose classification does NOT match the current abstract task
are masked out.

Reward
------
Sparse: 0.0 at every non-terminal step, and the QoS fitness of the full
composition at the terminal step.  A step that returns an invalid
action gets `invalid_penalty` and terminates the episode.

The environment is deliberately Gym-like but NOT built on top of `gym`
to avoid heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qos.fitness import QoSFitness, FitnessResult


CLASS_TO_IDX = {"Platinum": 0, "Gold": 1, "Silver": 2, "Bronze": 3}


@dataclass
class StepInfo:
    valid: bool
    picked_service: int
    task_tag: str
    fitness_so_far: float = 0.0
    reason: str = ""


class SingleCloudComposeEnv:
    """Discrete-action single-cloud QoS service composition environment."""

    def __init__(
        self,
        services_df: pd.DataFrame,
        qos_columns: Sequence[str],
        fitness: QoSFitness,
        workflow_length: int = 5,
        task_pool: Sequence[str] = ("Platinum", "Gold", "Silver", "Bronze"),
        allow_duplicate_services: bool = False,
        invalid_penalty: float = 0.0,
    ):
        self.services_df = services_df.reset_index(drop=True)
        self.qos_columns = list(qos_columns)
        self.fitness = fitness
        self.workflow_length = int(workflow_length)
        self.task_pool = list(task_pool)
        self.allow_duplicate_services = bool(allow_duplicate_services)
        self.invalid_penalty = float(invalid_penalty)

        self.n_services = len(self.services_df)
        self.qos_matrix = self.services_df[self.qos_columns].to_numpy(dtype=np.float64)

        # Precompute candidate index lists per classification bucket.
        # Fast masking during step().
        self._candidates_by_class: Dict[str, np.ndarray] = {}
        for cls_name in self.task_pool:
            mask = (self.services_df["service_class_name"] == cls_name).to_numpy()
            self._candidates_by_class[cls_name] = np.where(mask)[0].astype(np.int64)
            if self._candidates_by_class[cls_name].size == 0:
                raise ValueError(
                    f"Service class '{cls_name}' has no candidate services in the dataset. "
                    "Cannot construct workflows requiring it."
                )

        # Rolling episode state
        self._task_seq: List[str] = []
        self._selected: List[int] = []
        self._step_idx: int = 0
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return (
            len(CLASS_TO_IDX)                       # current-task one-hot
            + len(self.qos_columns)                 # running mean normalized QoS
            + 1                                     # progress
            + self.workflow_length * len(CLASS_TO_IDX)   # full workflow tags
        )

    @property
    def action_dim(self) -> int:
        return int(self.n_services)

    def get_qos_matrix(self) -> np.ndarray:
        return self.qos_matrix.copy()

    def candidates_for(self, task_tag: str) -> np.ndarray:
        return self._candidates_by_class[task_tag]

    def action_mask(self) -> np.ndarray:
        """Boolean mask (n_services,) with valid actions == True."""
        if self._step_idx >= self.workflow_length:
            return np.zeros(self.n_services, dtype=bool)
        task = self._task_seq[self._step_idx]
        mask = np.zeros(self.n_services, dtype=bool)
        mask[self._candidates_by_class[task]] = True
        if not self.allow_duplicate_services and self._selected:
            mask[np.array(self._selected, dtype=np.int64)] = False
        return mask

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None,
              task_sequence: Optional[Sequence[str]] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if task_sequence is None:
            self._task_seq = list(
                self._rng.choice(self.task_pool, size=self.workflow_length, replace=True)
            )
        else:
            if len(task_sequence) != self.workflow_length:
                raise ValueError(
                    f"task_sequence length {len(task_sequence)} != workflow_length "
                    f"{self.workflow_length}"
                )
            self._task_seq = list(task_sequence)
        self._selected = []
        self._step_idx = 0
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, StepInfo]:
        if self._step_idx >= self.workflow_length:
            raise RuntimeError("step() called after episode terminated.")

        current_task = self._task_seq[self._step_idx]
        mask = self.action_mask()
        if action < 0 or action >= self.n_services or not mask[action]:
            info = StepInfo(
                valid=False,
                picked_service=int(action),
                task_tag=current_task,
                reason="invalid action (class mismatch, out-of-range, or duplicate)",
            )
            state = self._build_state()
            return state, float(self.invalid_penalty), True, info

        self._selected.append(int(action))
        self._step_idx += 1

        done = self._step_idx >= self.workflow_length
        reward = 0.0
        info = StepInfo(valid=True, picked_service=int(action), task_tag=current_task)

        if done:
            fr = self.fitness.evaluate(self._selected, self.qos_matrix, valid=True)
            reward = float(fr.fitness)
            info.fitness_so_far = reward

        return self._build_state(), reward, done, info

    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        # current task one-hot
        cur_oh = np.zeros(len(CLASS_TO_IDX), dtype=np.float32)
        if self._step_idx < self.workflow_length:
            cur_oh[CLASS_TO_IDX[self._task_seq[self._step_idx]]] = 1.0

        # running mean of NORMALIZED QoS for services already selected
        # (uses fitness's per-attribute min/max)
        if self._selected:
            raw = self.qos_matrix[np.array(self._selected, dtype=np.int64), :]
            norm = np.zeros_like(raw)
            for i, c in enumerate(self.qos_columns):
                qmin = self.fitness.qos_min[c]
                qmax = self.fitness.qos_max[c]
                rng = qmax - qmin if (qmax - qmin) != 0 else 1.0
                if self.fitness.directions[c] == "max":
                    norm[:, i] = (raw[:, i] - qmin) / rng
                else:
                    norm[:, i] = (qmax - raw[:, i]) / rng
            running = np.clip(norm.mean(axis=0), 0.0, 1.0).astype(np.float32)
        else:
            running = np.zeros(len(self.qos_columns), dtype=np.float32)

        progress = np.array(
            [self._step_idx / max(1, self.workflow_length)],
            dtype=np.float32,
        )

        # full workflow one-hots (fixed layout so PPO can learn the plan)
        wf = np.zeros(self.workflow_length * len(CLASS_TO_IDX), dtype=np.float32)
        for t, tag in enumerate(self._task_seq):
            wf[t * len(CLASS_TO_IDX) + CLASS_TO_IDX[tag]] = 1.0

        return np.concatenate([cur_oh, running, progress, wf]).astype(np.float32)

    # ------------------------------------------------------------------
    @property
    def task_sequence(self) -> List[str]:
        return list(self._task_seq)

    @property
    def selected(self) -> List[int]:
        return list(self._selected)

    def evaluate_composition(self, service_indices: Sequence[int]) -> FitnessResult:
        return self.fitness.evaluate(service_indices, self.qos_matrix, valid=True)
