"""PPO agent with clipped objective and generalized advantage estimation."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import ActorCriticPolicy
from .replay_buffer import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        policy: ActorCriticPolicy,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.device = device
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.update_epochs = int(update_epochs)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, mask: np.ndarray,
                      deterministic: bool = False):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        action, log_prob, value = self.policy.act(s, m, deterministic=deterministic)
        return int(action.item()), float(log_prob.item()), float(value.item())

    # ------------------------------------------------------------------
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Run PPO updates on the currently buffered rollout."""
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "n_updates": 0}

        advantages, returns = self.buffer.compute_gae(
            gamma=self.gamma, gae_lambda=self.gae_lambda, last_value=last_value
        )
        # Advantage normalization
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        tensors = self.buffer.to_tensors(self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns,    dtype=torch.float32, device=self.device)

        n = adv_t.size(0)
        idx_all = np.arange(n)

        policy_losses, value_losses, entropies = [], [], []
        for _ in range(self.update_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.batch_size):
                b = idx_all[start:start + self.batch_size]
                b_t = torch.as_tensor(b, dtype=torch.long, device=self.device)
                st = tensors["states"][b_t]
                ac = tensors["actions"][b_t]
                ol = tensors["log_probs"][b_t]
                mk = tensors["masks"][b_t]
                new_lp, ent, val = self.policy.evaluate_actions(st, ac, mk)
                ratio = torch.exp(new_lp - ol)
                surr1 = ratio * adv_t[b_t]
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_range,
                                    1.0 + self.clip_range) * adv_t[b_t]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(val, ret_t[b_t])
                entropy = ent.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optim.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        self.buffer.clear()
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy":    float(np.mean(entropies)),
            "n_updates":  int(len(policy_losses)),
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"policy_state": self.policy.state_dict()}, path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(sd["policy_state"])
