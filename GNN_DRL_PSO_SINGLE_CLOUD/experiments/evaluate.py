"""
Evaluate an already-trained policy on the test partition.

This module reuses `runner.train_and_evaluate` in "evaluation" mode by
loading previously saved checkpoints and skipping the training loop.
For simplicity, and because the training loop is short by design, the
default entry point simply re-runs ablation for the specified configs.
"""
from __future__ import annotations

import argparse
import sys
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

from preprocessing.data_loader import QOS_COLUMNS, load_train_test
from preprocessing.normalizer import QoSNormalizer
from experiments.runner import (
    _build_env,
    _make_pso,
    _run_episode,
    build_service_embeddings,
    make_fitness,
    uses_gnn,
    uses_pso,
)
from drl.policy import ActorCriticPolicy
from drl.ppo_agent import PPOAgent
from graph.gnn_encoder import GNNEncoder


def evaluate(config_name: str, seed: int, config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    set_seed(seed)
    device = pick_device(cfg["experiment"]["device"])

    train_df, test_df = load_train_test(
        resolve_path(cfg["dataset"]["train_path"]),
        resolve_path(cfg["dataset"]["test_path"]),
    )
    normalizer = QoSNormalizer.load(
        resolve_path(cfg["dataset"]["processed_dir"]) / "normalizer.json"
    )
    fitness = make_fitness(cfg, normalizer, train_df)

    ckpt = resolve_path(cfg["experiment"]["checkpoints_root"])
    gnn_encoder = None
    if uses_gnn(config_name):
        p = ckpt / f"{config_name}_seed{seed}_gnn.pt"
        # Fallback to global gnn_model.pt if per-seed checkpoint is absent
        global_gnn = resolve_path(cfg["experiment"]["models_root"]) / "gnn_model.pt"
        gnn_encoder = _load_gnn_encoder(p if p.exists() else global_gnn, cfg)

    test_env = _build_env(test_df, fitness, cfg)
    test_embeddings, _ = build_service_embeddings(
        config_name, test_df, normalizer, cfg,
        seed=seed, gnn_encoder=gnn_encoder, device=device,
    )
    policy = ActorCriticPolicy(
        state_dim=test_env.state_dim,
        n_services=test_env.n_services,
        service_embed_dim=cfg["gnn"]["embedding_dim"],
        hidden_dim=cfg["ppo"]["hidden_dim"],
        use_gnn=True,
        gnn_trainable=False,
        service_embeddings=test_embeddings,
    )
    trained = ckpt / f"{config_name}_seed{seed}_policy.pt"
    if trained.exists():
        sd = torch.load(trained, map_location=device, weights_only=False)
        sd = {k: v for k, v in sd.items() if not k.startswith("service_embeddings")}
        policy.load_state_dict(sd, strict=False)

    agent = PPOAgent(policy=policy, device=device)
    pso = _make_pso(cfg, seed + 1) if uses_pso(config_name) else None

    n_eval = 30
    ep_metrics = []
    for ep in range(n_eval):
        info = _run_episode(test_env, agent, pso,
                            episode_seed=999_000_000 + seed * 10000 + ep,
                            training=False)
        info["episode"] = ep
        ep_metrics.append(info)

    out = {
        "config": config_name,
        "seed":  seed,
        "n_eval": n_eval,
        "mean_test_fitness": float(np.mean([m["final_fitness"] for m in ep_metrics])),
        "std_test_fitness":  float(np.std([m["final_fitness"] for m in ep_metrics])),
    }
    save_json(
        {"summary": out, "metrics": ep_metrics},
        resolve_path(cfg["experiment"]["results_root"]) / "logs" /
        f"evaluate_{config_name}_seed{seed}.json",
    )
    return out


def _load_gnn_encoder(path: Path, cfg) -> GNNEncoder:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        econf = obj["config"]
    else:
        econf = {
            "in_channels": 13,          # 9 QoS + 4 class one-hot
            "hidden_channels": cfg["gnn"]["hidden_dim"],
            "out_channels":    cfg["gnn"]["embedding_dim"],
            "num_layers":      cfg["gnn"]["num_layers"],
            "dropout":         cfg["gnn"]["dropout"],
            "gnn_type":        cfg["gnn"]["type"],
        }
    enc = GNNEncoder(**econf)
    if isinstance(obj, dict) and "state_dict" in obj:
        enc.load_state_dict(obj["state_dict"])
    else:
        enc.load_state_dict(obj)
    return enc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    print(evaluate(a.name, a.seed, a.config))
