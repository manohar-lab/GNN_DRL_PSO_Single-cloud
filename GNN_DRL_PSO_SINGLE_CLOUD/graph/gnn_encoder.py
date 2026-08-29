"""
GNN encoder + a lightweight unsupervised pretraining routine
(graph autoencoder objective) for the 'frozen' training mode.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy imports of torch_geometric so top-level imports of this package
# succeed even in tests that never build the graph.
try:
    from torch_geometric.nn import GraphSAGE, GCN
    _HAS_TG = True
except Exception:                                    # pragma: no cover
    _HAS_TG = False


class GNNEncoder(nn.Module):
    """Two-layer GNN encoder (GraphSAGE by default)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "graphsage",
    ):
        super().__init__()
        if not _HAS_TG:
            raise ImportError(
                "torch_geometric is required. Install with:\n"
                "    pip install torch-geometric"
            )
        gnn_type = gnn_type.lower()
        if gnn_type == "graphsage":
            self.gnn = GraphSAGE(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif gnn_type == "gcn":
            self.gnn = GCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn(x, edge_index)


class InnerProductDecoder(nn.Module):
    """Reconstructs an edge probability as sigmoid(z_i . z_j)."""

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))


def pretrain_gnn_autoencoder(
    encoder: GNNEncoder,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Unsupervised GAE pretraining: reconstruct edges from embeddings.

    We use positive edges (existing) vs. negative edges (randomly sampled
    non-edges) as a binary classification task.  This provides a valid
    representation-learning objective when no supervised labels exist —
    which is the situation with QWS.
    """
    encoder = encoder.to(device)
    decoder = InnerProductDecoder().to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    losses = []

    n_nodes = int(x.size(0))
    for ep in range(epochs):
        encoder.train()
        optim.zero_grad()
        z = encoder(x, edge_index)
        # Positive edges
        pos_pred = decoder(z, edge_index)
        # Negative edges: random pairs (may occasionally coincide with
        # a positive; acceptable for a proxy objective).
        neg_src = torch.randint(0, n_nodes, (edge_index.size(1),), device=device)
        neg_dst = torch.randint(0, n_nodes, (edge_index.size(1),), device=device)
        neg_pred = decoder(z, torch.stack([neg_src, neg_dst]))
        pos_loss = F.binary_cross_entropy(
            pos_pred, torch.ones_like(pos_pred)
        )
        neg_loss = F.binary_cross_entropy(
            neg_pred, torch.zeros_like(neg_pred)
        )
        loss = pos_loss + neg_loss
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  [GAE pretrain] epoch {ep+1}/{epochs} loss={loss.item():.4f}")
    return {"losses": losses}


@torch.no_grad()
def compute_embeddings(
    encoder: GNNEncoder,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    encoder.eval()
    return encoder(x.to(device), edge_index.to(device)).cpu()
