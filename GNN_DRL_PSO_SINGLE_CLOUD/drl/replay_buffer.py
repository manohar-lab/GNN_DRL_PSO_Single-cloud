"""On-policy rollout buffer used by PPO."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)

    def add(self, state, action, log_prob, reward, value, done, mask):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        self.masks.append(np.asarray(mask, dtype=bool))

    def clear(self):
        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()
        self.masks.clear()

    def __len__(self):
        return len(self.states)

    def compute_gae(self, gamma: float, gae_lambda: float, last_value: float = 0.0):
        n = len(self.rewards)
        rewards = np.asarray(self.rewards, dtype=np.float64)
        values = np.asarray(self.values + [last_value], dtype=np.float64)
        dones = np.asarray(self.dones, dtype=np.float64)
        advantages = np.zeros(n, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(n)):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages.astype(np.float32), returns.astype(np.float32)

    def to_tensors(self, device: str = "cpu"):
        return {
            "states":     torch.tensor(np.stack(self.states),  dtype=torch.float32,  device=device),
            "actions":    torch.tensor(self.actions,           dtype=torch.long,     device=device),
            "log_probs":  torch.tensor(self.log_probs,         dtype=torch.float32,  device=device),
            "masks":      torch.tensor(np.stack(self.masks),   dtype=torch.bool,     device=device),
        }
