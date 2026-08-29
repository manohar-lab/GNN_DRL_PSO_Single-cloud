"""
Deterministic construction of the service dependency graph.

Because the QWS v2 dataset does not ship explicit service dependencies,
we construct a transparent graph based on service classification and
QoS similarity — this is documented as the "derivation method" the
problem statement asks for.

Graph construction rule
-----------------------
Nodes  : one per service in the training partition.
Edges  : (1) INTRA-class kNN.
             For every service, we connect it to its K nearest neighbors
             within the SAME classification bucket (Platinum/Gold/
             Silver/Bronze), using Euclidean distance on the normalized
             QoS vector.  Intuition: services that behave similarly
             under the same quality tier are compositionally
             interchangeable.
         (2) INTER-class kNN across ADJACENT classifications
             (Platinum<->Gold, Gold<->Silver, Silver<->Bronze).
             Same kNN mechanism.  Intuition: services from adjacent
             quality tiers commonly participate together in real
             compositions.
         (3) The final graph is (optionally) made bidirectional.
Node
features: the normalized QoS vector + a one-hot classification tag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


CLASS_ORDER = ["Platinum", "Gold", "Silver", "Bronze"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


@dataclass
class GraphData:
    x: torch.Tensor                    # (N, F) node features
    edge_index: torch.Tensor           # (2, E)
    class_labels: torch.Tensor         # (N,)
    num_nodes: int
    num_features: int
    edge_stats: Dict[str, int]

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.class_labels = self.class_labels.to(device)
        return self


def build_service_graph(
    services_df: pd.DataFrame,
    normalized_qos: np.ndarray,        # (N, D) with columns aligned to services_df
    qos_columns: Sequence[str],
    intra_class_knn: int = 5,
    inter_class_knn: int = 3,
    bidirectional: bool = True,
) -> GraphData:
    """Build the service dependency graph."""
    if len(services_df) != normalized_qos.shape[0]:
        raise ValueError("services_df and normalized_qos must have the same N.")
    n = int(len(services_df))
    class_names = services_df["service_class_name"].astype(str).to_numpy()
    class_labels = np.array([CLASS_TO_IDX[c] for c in class_names], dtype=np.int64)

    # Indices per class
    idx_per_class: Dict[str, np.ndarray] = {
        c: np.where(class_names == c)[0] for c in CLASS_ORDER
    }

    edges_src: List[int] = []
    edges_dst: List[int] = []
    intra_edges = 0
    inter_edges = 0

    # (1) intra-class kNN
    for c in CLASS_ORDER:
        idx = idx_per_class[c]
        if idx.size <= 1:
            continue
        k = min(intra_class_knn + 1, idx.size)  # +1 because self is included
        knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        knn.fit(normalized_qos[idx])
        _, nbrs = knn.kneighbors(normalized_qos[idx])
        for i_local, row in enumerate(nbrs):
            src_global = int(idx[i_local])
            for j in row[1:]:  # skip self
                dst_global = int(idx[int(j)])
                edges_src.append(src_global)
                edges_dst.append(dst_global)
                intra_edges += 1

    # (2) inter-class kNN between adjacent classes
    for i in range(len(CLASS_ORDER) - 1):
        c1, c2 = CLASS_ORDER[i], CLASS_ORDER[i + 1]
        a, b = idx_per_class[c1], idx_per_class[c2]
        if a.size == 0 or b.size == 0:
            continue
        k = min(inter_class_knn, b.size)
        knn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(normalized_qos[b])
        _, nbrs = knn.kneighbors(normalized_qos[a])
        for i_local, row in enumerate(nbrs):
            src_global = int(a[i_local])
            for j in row:
                dst_global = int(b[int(j)])
                edges_src.append(src_global)
                edges_dst.append(dst_global)
                inter_edges += 1

    if len(edges_src) == 0:
        # Degenerate but keep the graph valid.
        edges_src.append(0); edges_dst.append(0)

    edge_index = torch.tensor(
        np.stack([np.asarray(edges_src, dtype=np.int64),
                  np.asarray(edges_dst, dtype=np.int64)], axis=0),
        dtype=torch.long,
    )
    if bidirectional:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        # Deduplicate
        edge_index = torch.unique(edge_index, dim=1)

    # Node features: normalized QoS + one-hot classification
    D_qos = normalized_qos.shape[1]
    C = len(CLASS_ORDER)
    x_np = np.zeros((n, D_qos + C), dtype=np.float32)
    x_np[:, :D_qos] = normalized_qos.astype(np.float32)
    for i in range(n):
        x_np[i, D_qos + class_labels[i]] = 1.0
    x = torch.from_numpy(x_np)

    return GraphData(
        x=x,
        edge_index=edge_index,
        class_labels=torch.from_numpy(class_labels),
        num_nodes=n,
        num_features=D_qos + C,
        edge_stats={
            "intra_class": int(intra_edges),
            "inter_class": int(inter_edges),
            "total_after_dedup": int(edge_index.shape[1]),
        },
    )
