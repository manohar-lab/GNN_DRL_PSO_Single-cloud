"""Utility for saving/loading a `GraphData` object."""
from __future__ import annotations

from pathlib import Path

import torch

from .graph_builder import GraphData


def save_graph(g: GraphData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": g.x.cpu(),
            "edge_index": g.edge_index.cpu(),
            "class_labels": g.class_labels.cpu(),
            "num_nodes": g.num_nodes,
            "num_features": g.num_features,
            "edge_stats": g.edge_stats,
        },
        path,
    )


def load_graph(path: str | Path) -> GraphData:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return GraphData(
        x=d["x"],
        edge_index=d["edge_index"],
        class_labels=d["class_labels"],
        num_nodes=d["num_nodes"],
        num_features=d["num_features"],
        edge_stats=d["edge_stats"],
    )
