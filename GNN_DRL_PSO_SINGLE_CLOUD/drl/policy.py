"""Actor-critic policy with optional GNN service embeddings.

The policy operates in two mutually consistent modes:

1. `use_gnn = False`  (baseline: DRL / DRL-PSO)
     * Actor produces per-service logits from an MLP over the raw state
       + a learned service embedding table.

2. `use_gnn = True`   (proposed: GNN-DRL / GNN-DRL-PSO)
     * A precomputed GNN embedding matrix `service_embeddings`
       (shape (N, gnn_out)) is passed in from outside.
     * If `gnn_trainable = True` this matrix is a `nn.Parameter`
       (jointly optimized by PPO).
     * If `gnn_trainable = False` it is a fixed buffer (frozen GNN).

For fair comparison across ablations, the ONLY difference between
baseline and proposed policies is where the service embeddings come
from — everything else (state MLP, actor/critic heads) is identical.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticPolicy(nn.Module):

    def __init__(
        self,
        state_dim: int,
        n_services: int,
        service_embed_dim: int = 32,
        hidden_dim: int = 128,
        use_gnn: bool = False,
        gnn_trainable: bool = False,
        service_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_services = int(n_services)
        self.service_embed_dim = int(service_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_gnn = bool(use_gnn)
        self.gnn_trainable = bool(gnn_trainable)

        # Service embeddings ------------------------------------------
        if self.use_gnn:
            if service_embeddings is None:
                raise ValueError(
                    "use_gnn=True requires service_embeddings (shape (N, D))."
                )
            emb = service_embeddings.detach().clone().float()
            if emb.shape != (self.n_services, self.service_embed_dim):
                raise ValueError(
                    f"service_embeddings has shape {tuple(emb.shape)}, expected "
                    f"({self.n_services}, {self.service_embed_dim})."
                )
            if self.gnn_trainable:
                self.service_embeddings = nn.Parameter(emb)
            else:
                self.register_buffer("service_embeddings", emb)
        else:
            self.service_embeddings = nn.Embedding(self.n_services, self.service_embed_dim)
            nn.init.xavier_uniform_(self.service_embeddings.weight)

        # State encoder -----------------------------------------------
        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )

        # Actor head: score = <state_repr, service_embedding>
        self.actor_proj = nn.Linear(self.hidden_dim, self.service_embed_dim)

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    def _service_matrix(self) -> torch.Tensor:
        if self.use_gnn:
            return self.service_embeddings                     # (N, D)
        return self.service_embeddings.weight                  # (N, D)

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        """Return action logits and state value.

        state       : (B, state_dim)
        action_mask : (B, N)   True => valid  (float 0/1 also works)
        """
        h = self.state_mlp(state)                              # (B, H)
        q = self.actor_proj(h)                                 # (B, D)
        svc = self._service_matrix()                           # (N, D)
        logits = q @ svc.t()                                   # (B, N)
        if action_mask is not None:
            mask = action_mask.to(dtype=torch.bool)
            logits = logits.masked_fill(~mask, float("-1e9"))
        value = self.critic(h).squeeze(-1)                     # (B,)
        return logits, value

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, state: torch.Tensor, action_mask: torch.Tensor,
            deterministic: bool = False):
        """Sample an action + return log_prob and value."""
        logits, value = self.forward(state, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, states, actions, action_masks):
        logits, values = self.forward(states, action_masks)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values
