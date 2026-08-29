"""
Phase 7 wrapper: pretrain the GNN autoencoder on the TRAINING graph.

Also serves as an integration test that graph + GNN + PyG all work.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.utils import (
    add_project_root_to_sys_path,
    load_config,
    make_run_logger,
    pick_device,
    project_root,
    resolve_path,
    save_json,
    set_seed,
)

add_project_root_to_sys_path()

from preprocessing.data_loader import QOS_COLUMNS, load_train_test          # noqa
from preprocessing.normalizer import QoSNormalizer                          # noqa
from graph.graph_builder import build_service_graph                          # noqa
from graph.graph_dataset import save_graph                                   # noqa
from graph.gnn_encoder import (                                              # noqa
    GNNEncoder,
    compute_embeddings,
    pretrain_gnn_autoencoder,
)


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seeds"][0])
    device = pick_device(cfg["experiment"]["device"])

    logger = make_run_logger("train_gnn",
                             resolve_path(cfg["experiment"]["results_root"]) / "logs")
    logger.info(f"device={device}")

    train_path = resolve_path(cfg["dataset"]["train_path"])
    test_path = resolve_path(cfg["dataset"]["test_path"])
    processed_dir = resolve_path(cfg["dataset"]["processed_dir"])
    train_df, test_df = load_train_test(train_path, test_path)
    logger.info(f"train={len(train_df)}, test={len(test_df)}")

    normalizer = QoSNormalizer.load(processed_dir / "normalizer.json")
    train_norm = normalizer.transform(train_df, columns=QOS_COLUMNS)
    normalized_qos = train_norm[QOS_COLUMNS].to_numpy(dtype=np.float32)

    logger.info("building service graph on TRAIN partition ...")
    graph = build_service_graph(
        train_df,
        normalized_qos,
        qos_columns=QOS_COLUMNS,
        intra_class_knn=cfg["graph"]["intra_class_knn"],
        inter_class_knn=cfg["graph"]["inter_class_knn"],
        bidirectional=cfg["graph"]["bidirectional"],
    )
    logger.info(f"graph: {graph.num_nodes} nodes / {graph.edge_index.shape[1]} edges  "
                f"(intra={graph.edge_stats['intra_class']}, "
                f"inter={graph.edge_stats['inter_class']})")

    graph_out = resolve_path("data/processed/train_graph.pt")
    save_graph(graph, graph_out)
    logger.info(f"graph saved  -> {graph_out}")

    encoder = GNNEncoder(
        in_channels=graph.num_features,
        hidden_channels=cfg["gnn"]["hidden_dim"],
        out_channels=cfg["gnn"]["embedding_dim"],
        num_layers=cfg["gnn"]["num_layers"],
        dropout=cfg["gnn"]["dropout"],
        gnn_type=cfg["gnn"]["type"],
    )
    logger.info(f"pretraining GNN encoder ({cfg['gnn']['type']}, "
                f"H={cfg['gnn']['hidden_dim']}, D={cfg['gnn']['embedding_dim']}, "
                f"layers={cfg['gnn']['num_layers']})")

    t0 = time.time()
    hist = pretrain_gnn_autoencoder(
        encoder=encoder,
        x=graph.x,
        edge_index=graph.edge_index,
        epochs=cfg["gnn"]["pretrain_epochs"],
        lr=cfg["gnn"]["learning_rate"],
        device=device,
        verbose=True,
    )
    dt = time.time() - t0
    logger.info(f"pretraining complete in {dt:.2f}s (final loss={hist['losses'][-1]:.4f})")

    model_path = resolve_path(cfg["experiment"]["models_root"]) / "gnn_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "config": {
                "in_channels": graph.num_features,
                "hidden_channels": cfg["gnn"]["hidden_dim"],
                "out_channels":    cfg["gnn"]["embedding_dim"],
                "num_layers":      cfg["gnn"]["num_layers"],
                "dropout":         cfg["gnn"]["dropout"],
                "gnn_type":        cfg["gnn"]["type"],
            },
            "num_train_services": graph.num_nodes,
            "pretrain_history":   hist["losses"],
        },
        model_path,
    )
    logger.info(f"gnn model saved -> {model_path}")

    z = compute_embeddings(encoder, graph.x, graph.edge_index, device=device)
    logger.info(f"embeddings shape = {tuple(z.shape)}  mean={z.mean().item():.4f}  "
                f"std={z.std().item():.4f}")

    save_json(
        {
            "final_loss":      float(hist["losses"][-1]),
            "pretrain_seconds": float(dt),
            "n_nodes":         int(graph.num_nodes),
            "n_edges":         int(graph.edge_index.shape[1]),
            "edge_stats":      graph.edge_stats,
        },
        resolve_path(cfg["experiment"]["results_root"]) / "logs" / "train_gnn_summary.json",
    )
    return {"final_loss": float(hist["losses"][-1]), "seconds": float(dt)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    run(a.config)
