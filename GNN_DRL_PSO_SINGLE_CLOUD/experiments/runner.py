"""
Core training + evaluation runner.

`train_and_evaluate(config_name, seed, cfg, ...)` runs a single
(configuration, seed) instance and returns dictionaries of per-episode
metrics for both TRAIN and TEST phases.

Configurations supported (from config.yaml -> ablation.configurations):
    "DRL"            : PPO with a fixed random-projection service encoder,
                       no PSO refinement.
    "GNN_DRL"        : PPO with pretrained-GNN service embeddings,
                       no PSO refinement.
    "DRL_PSO"        : PPO with random-projection service encoder
                       + PSO refinement of the PPO composition.
    "GNN_DRL_PSO"    : PPO with pretrained-GNN service embeddings
                       + PSO refinement of the PPO composition.

Fairness
--------
All four configurations use the SAME
   * dataset partitions and preprocessing artifacts,
   * QoS fitness definition,
   * workflow generator (identical seeds per episode),
   * PPO hyperparameters,
   * PSO hyperparameters (used only when in-config),
   * state MLP architecture and critic head.
The only differences are (a) presence of graph-derived embeddings
and (b) presence of the PSO refinement stage.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.utils import (
    make_run_logger,
    resolve_path,
    set_seed,
    save_json,
)
from preprocessing.data_loader import QOS_COLUMNS, load_train_test
from preprocessing.normalizer import QoSNormalizer
from graph.graph_builder import build_service_graph, GraphData
from graph.gnn_encoder import GNNEncoder, compute_embeddings
from environment.single_cloud_env import SingleCloudComposeEnv
from qos.fitness import QoSFitness
from drl.policy import ActorCriticPolicy
from drl.ppo_agent import PPOAgent
from pso.discrete_pso import DiscretePSO


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def uses_gnn(config_name: str) -> bool:
    return config_name.upper().startswith("GNN")


def uses_pso(config_name: str) -> bool:
    return config_name.upper().endswith("PSO")


def make_fitness(cfg: Dict[str, Any], normalizer: QoSNormalizer,
                 raw_df: pd.DataFrame) -> QoSFitness:
    """QoS fitness uses PER-ATTRIBUTE min/max computed on TRAIN raw values
    (same as the normalizer)."""
    q_min: Dict[str, float] = {}
    q_max: Dict[str, float] = {}
    for c in QOS_COLUMNS:
        v = raw_df[c].to_numpy(dtype=np.float64)
        q_min[c] = float(v.min())
        q_max[c] = float(v.max())
    return QoSFitness(
        qos_columns=QOS_COLUMNS,
        weights=cfg["qos"]["weights"],
        directions=cfg["dataset"]["qos_direction"],
        qos_min=q_min,
        qos_max=q_max,
        invalid_penalty=cfg["qos"]["invalid_penalty"],
    )


def make_random_projection(in_dim: int, out_dim: int, seed: int) -> nn.Linear:
    """Fixed, deterministic random projection used by non-GNN configs.

    This is a legitimate baseline representation: no graph structure,
    but service features are still projected into `service_embed_dim`
    the SAME way at train and test — inductive."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        w = torch.empty(out_dim, in_dim)
        nn.init.orthogonal_(w, generator=g)
        proj.weight.copy_(w)
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


def features_for_services(
    df: pd.DataFrame,
    normalized_qos: np.ndarray,
) -> torch.Tensor:
    """Node features = normalized QoS + one-hot classification (13 dims)."""
    from graph.graph_builder import CLASS_ORDER, CLASS_TO_IDX
    n = len(df); D = normalized_qos.shape[1]; C = len(CLASS_ORDER)
    x = np.zeros((n, D + C), dtype=np.float32)
    x[:, :D] = normalized_qos.astype(np.float32)
    class_labels = np.array(
        [CLASS_TO_IDX[c] for c in df["service_class_name"].astype(str)],
        dtype=np.int64,
    )
    for i in range(n):
        x[i, D + class_labels[i]] = 1.0
    return torch.from_numpy(x)


def _build_env(
    df: pd.DataFrame,
    fitness: QoSFitness,
    cfg: Dict[str, Any],
) -> SingleCloudComposeEnv:
    return SingleCloudComposeEnv(
        services_df=df,
        qos_columns=QOS_COLUMNS,
        fitness=fitness,
        workflow_length=cfg["workflow"]["length"],
        task_pool=cfg["workflow"]["task_pool"],
        allow_duplicate_services=cfg["workflow"]["allow_duplicate_services"],
        invalid_penalty=cfg["qos"]["invalid_penalty"],
    )


# ---------------------------------------------------------------------
# Config-specific embeddings (train-time and test-time)
# ---------------------------------------------------------------------
def build_service_embeddings(
    config_name: str,
    partition_df: pd.DataFrame,
    normalizer: QoSNormalizer,
    cfg: Dict[str, Any],
    seed: int,
    gnn_encoder: Optional[GNNEncoder] = None,
    device: str = "cpu",
):
    """Return (service_embeddings_tensor, extra_dict) for a partition."""
    norm_df = normalizer.transform(partition_df, columns=QOS_COLUMNS)
    normalized = norm_df[QOS_COLUMNS].to_numpy(dtype=np.float32)
    feats = features_for_services(partition_df, normalized)          # (N, F_raw)

    embed_dim = int(cfg["gnn"]["embedding_dim"])
    if uses_gnn(config_name):
        assert gnn_encoder is not None, "GNN configs require gnn_encoder"
        # Build partition graph (structure-aware)
        graph = build_service_graph(
            partition_df,
            normalized,
            qos_columns=QOS_COLUMNS,
            intra_class_knn=cfg["graph"]["intra_class_knn"],
            inter_class_knn=cfg["graph"]["inter_class_knn"],
            bidirectional=cfg["graph"]["bidirectional"],
        )
        z = compute_embeddings(gnn_encoder, graph.x, graph.edge_index, device=device)
        return z.detach().clone().float(), {"graph_edges": int(graph.edge_index.shape[1])}
    else:
        proj = make_random_projection(feats.shape[1], embed_dim, seed=seed)
        with torch.no_grad():
            z = proj(feats)
        return z.detach().clone().float(), {"graph_edges": 0}


# ---------------------------------------------------------------------
# One (config, seed) run
# ---------------------------------------------------------------------
def _run_episode(
    env: SingleCloudComposeEnv,
    agent: PPOAgent,
    pso: Optional[DiscretePSO],
    episode_seed: int,
    training: bool,
) -> Dict[str, Any]:
    """One full episode. Optionally add PSO refinement.

    When `training=True`, transitions are pushed into the PPO buffer and
    (if PSO is used) the terminal reward is overwritten with the PSO
    fitness so that the policy learns from the refined outcome.
    """
    state = env.reset(seed=episode_seed)
    ep_step_infos = []
    trans = []                       # (state, action, log_prob, r, value, done, mask)
    while True:
        mask = env.action_mask()
        action, log_prob, value = agent.select_action(state, mask,
                                                     deterministic=not training)
        next_state, r, done, info = env.step(action)
        trans.append((state.copy(), int(action), float(log_prob),
                      float(r), float(value), bool(done),
                      mask.copy()))
        ep_step_infos.append(info)
        state = next_state
        if done:
            break

    composition = env.selected
    task_seq = env.task_sequence
    initial_fit = env.evaluate_composition(composition).fitness

    pso_history: List[float] = []
    pso_seconds = 0.0
    if pso is not None:
        candidates = [env.candidates_for(t) for t in task_seq]

        def fitfn(pos):
            return env.evaluate_composition(pos).fitness

        t0 = time.time()
        pso.candidates = candidates       # rebind per-task candidates
        pso.fitness_fn = fitfn
        pso.n_evaluations = 0             # reset counter for this episode
        result = pso.optimize(composition)
        pso_seconds = time.time() - t0
        final_composition = result.best_position
        final_fitness = result.best_fitness
        pso_history = result.history
    else:
        final_composition = composition
        final_fitness = initial_fit

    # Push transitions into PPO buffer if training. Override terminal reward
    # with the FINAL fitness (post-PSO if PSO is used) — this is exactly the
    # 'reward-feedback-to-DRL' loop in the paper.
    if training:
        for i, (s, a, lp, r, v, d, m) in enumerate(trans):
            r_use = float(final_fitness) if (i == len(trans) - 1 and d) else r
            agent.buffer.add(s, a, lp, r_use, v, d, m)

    fr = env.evaluate_composition(final_composition)
    per_qos = {c: fr.normalized[c] for c in QOS_COLUMNS}
    raw_qos = {c: fr.raw_agg[c]    for c in QOS_COLUMNS}
    n_valid_steps = sum(1 for i in ep_step_infos if i.valid)

    return {
        "task_sequence":    task_seq,
        "initial_composition": composition,
        "final_composition":   final_composition,
        "initial_fitness":  float(initial_fit),
        "final_fitness":    float(final_fitness),
        "pso_history":      pso_history,
        "pso_seconds":      float(pso_seconds),
        "normalized_qos":   per_qos,
        "raw_qos":          raw_qos,
        "valid_steps":      int(n_valid_steps),
        "success":          bool(n_valid_steps == env.workflow_length),
    }


def _make_pso(cfg: Dict[str, Any], seed: int) -> DiscretePSO:
    return DiscretePSO(
        candidates_by_task=[np.array([0])],   # placeholder, overwritten per episode
        fitness_fn=lambda x: 0.0,              # placeholder
        num_particles=cfg["pso"]["num_particles"],
        num_iterations=cfg["pso"]["num_iterations"],
        w=cfg["pso"]["w"],
        c1=cfg["pso"]["c1"],
        c2=cfg["pso"]["c2"],
        init_perturb=cfg["pso"]["init_perturb"],
        rng=np.random.default_rng(seed),
    )


def train_and_evaluate(
    config_name: str,
    seed: int,
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    normalizer: QoSNormalizer,
    device: str,
    logger=None,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train on TRAIN partition, then evaluate on TEST partition."""
    set_seed(seed)
    if logger is None:
        logger = make_run_logger(
            f"{config_name}_seed{seed}",
            resolve_path(cfg["experiment"]["results_root"]) / "logs",
        )

    # QoS fitness: min/max fitted on the training partition (same as normalizer)
    fitness = make_fitness(cfg, normalizer, train_df)

    # ------------------------------- TRAIN partition setup ---------------
    train_env = _build_env(train_df, fitness, cfg)
    train_norm = normalizer.transform(train_df, columns=QOS_COLUMNS)
    train_normalized_qos = train_norm[QOS_COLUMNS].to_numpy(dtype=np.float32)

    gnn_encoder: Optional[GNNEncoder] = None
    if uses_gnn(config_name):
        # Build the training graph and pretrain a GNN.
        graph = build_service_graph(
            train_df,
            train_normalized_qos,
            qos_columns=QOS_COLUMNS,
            intra_class_knn=cfg["graph"]["intra_class_knn"],
            inter_class_knn=cfg["graph"]["inter_class_knn"],
            bidirectional=cfg["graph"]["bidirectional"],
        )
        gnn_encoder = GNNEncoder(
            in_channels=graph.num_features,
            hidden_channels=cfg["gnn"]["hidden_dim"],
            out_channels=cfg["gnn"]["embedding_dim"],
            num_layers=cfg["gnn"]["num_layers"],
            dropout=cfg["gnn"]["dropout"],
            gnn_type=cfg["gnn"]["type"],
        )
        from graph.gnn_encoder import pretrain_gnn_autoencoder
        logger.info(f"[{config_name}/seed{seed}] pretraining GNN "
                    f"({cfg['gnn']['pretrain_epochs']} epochs)")
        pretrain_gnn_autoencoder(
            gnn_encoder,
            graph.x,
            graph.edge_index,
            epochs=cfg["gnn"]["pretrain_epochs"],
            lr=cfg["gnn"]["learning_rate"],
            device=device,
            verbose=False,
        )
        gnn_encoder.eval()

    # Precompute service embeddings for the TRAIN partition
    train_embeddings, _ = build_service_embeddings(
        config_name, train_df, normalizer, cfg, seed=seed,
        gnn_encoder=gnn_encoder, device=device,
    )

    # ------------------------------- Policy + agent ----------------------
    policy = ActorCriticPolicy(
        state_dim=train_env.state_dim,
        n_services=train_env.n_services,
        service_embed_dim=cfg["gnn"]["embedding_dim"],
        hidden_dim=cfg["ppo"]["hidden_dim"],
        use_gnn=True,                        # unified path (embeddings passed in)
        gnn_trainable=(uses_gnn(config_name) and cfg["gnn"]["training_mode"] == "joint"),
        service_embeddings=train_embeddings,
    )
    agent = PPOAgent(
        policy=policy,
        learning_rate=cfg["ppo"]["learning_rate"],
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_range=cfg["ppo"]["clip_range"],
        value_coef=cfg["ppo"]["value_coef"],
        entropy_coef=cfg["ppo"]["entropy_coef"],
        update_epochs=cfg["ppo"]["update_epochs"],
        batch_size=cfg["ppo"]["batch_size"],
        max_grad_norm=cfg["ppo"]["max_grad_norm"],
        device=device,
    )
    pso = _make_pso(cfg, seed) if uses_pso(config_name) else None

    # ------------------------------- Training loop -----------------------
    num_episodes = int(cfg["experiment"]["num_episodes"])
    rollout_size = int(cfg["ppo"]["rollout_size"])
    train_metrics: List[Dict[str, Any]] = []

    inference_times: List[float] = []
    pso_times: List[float] = []
    ep_start = time.time()
    for ep in range(num_episodes):
        ep_seed = seed * 10000 + ep
        t_ep0 = time.time()
        # -- inference-only timer within episode: policy forward passes
        # (approximate; PSO time is measured separately inside _run_episode)
        m_before = time.time()
        ep_info = _run_episode(train_env, agent, pso, ep_seed, training=True)
        m_after = time.time()
        # Approximate DRL inference time = total episode time - PSO seconds
        drl_time = (m_after - m_before) - ep_info["pso_seconds"]
        inference_times.append(max(0.0, drl_time))
        pso_times.append(ep_info["pso_seconds"])
        ep_info["episode"] = ep
        ep_info["seconds"] = m_after - t_ep0
        train_metrics.append(ep_info)
        # PPO update every `rollout_size` episodes
        if (ep + 1) % max(1, rollout_size) == 0:
            upd = agent.update(last_value=0.0)
            ep_info["ppo_update"] = upd
        if (ep + 1) % max(1, num_episodes // 10) == 0:
            logger.info(
                f"[{config_name}/seed{seed}] ep {ep+1}/{num_episodes} "
                f"init={ep_info['initial_fitness']:.4f} "
                f"final={ep_info['final_fitness']:.4f}"
            )

    train_seconds = time.time() - ep_start
    # Final PPO update flush
    if len(agent.buffer) > 0:
        agent.update(last_value=0.0)

    # ------------------------------- Evaluation on TEST partition --------
    test_env = _build_env(test_df, fitness, cfg)
    test_embeddings, test_info = build_service_embeddings(
        config_name, test_df, normalizer, cfg, seed=seed,
        gnn_encoder=gnn_encoder, device=device,
    )

    # Transfer state_mlp + actor_proj + critic; substitute the service
    # embedding matrix for the test partition (dimensions differ).
    test_policy = ActorCriticPolicy(
        state_dim=test_env.state_dim,
        n_services=test_env.n_services,
        service_embed_dim=cfg["gnn"]["embedding_dim"],
        hidden_dim=cfg["ppo"]["hidden_dim"],
        use_gnn=True,
        gnn_trainable=False,
        service_embeddings=test_embeddings,
    )
    sd = policy.state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith("service_embeddings")}
    test_policy.load_state_dict(sd, strict=False)
    test_agent = PPOAgent(
        policy=test_policy,
        learning_rate=cfg["ppo"]["learning_rate"],
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_range=cfg["ppo"]["clip_range"],
        value_coef=cfg["ppo"]["value_coef"],
        entropy_coef=cfg["ppo"]["entropy_coef"],
        update_epochs=cfg["ppo"]["update_epochs"],
        batch_size=cfg["ppo"]["batch_size"],
        max_grad_norm=cfg["ppo"]["max_grad_norm"],
        device=device,
    )

    test_pso = _make_pso(cfg, seed + 1) if uses_pso(config_name) else None
    test_metrics: List[Dict[str, Any]] = []
    test_seconds_start = time.time()
    n_test_episodes = max(20, num_episodes // 2)
    for ep in range(n_test_episodes):
        ep_seed = 999_000_000 + seed * 10000 + ep
        m0 = time.time()
        info = _run_episode(test_env, test_agent, test_pso, ep_seed, training=False)
        info["episode"] = ep
        info["seconds"] = time.time() - m0
        test_metrics.append(info)
    test_seconds = time.time() - test_seconds_start

    # ------------------------------- Checkpoints -------------------------
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(),
                   checkpoint_dir / f"{config_name}_seed{seed}_policy.pt")
        if gnn_encoder is not None:
            torch.save(gnn_encoder.state_dict(),
                       checkpoint_dir / f"{config_name}_seed{seed}_gnn.pt")

    summary = {
        "config": config_name,
        "seed": seed,
        "n_train_services": int(len(train_df)),
        "n_test_services":  int(len(test_df)),
        "num_train_episodes": num_episodes,
        "num_test_episodes":  n_test_episodes,
        "mean_train_fitness_last_10":
            float(np.mean([m["final_fitness"] for m in train_metrics[-10:]])),
        "mean_test_fitness":
            float(np.mean([m["final_fitness"] for m in test_metrics])),
        "std_test_fitness":
            float(np.std([m["final_fitness"] for m in test_metrics])),
        "mean_test_initial_fitness":
            float(np.mean([m["initial_fitness"] for m in test_metrics])),
        "mean_drl_inference_seconds_per_episode":
            float(np.mean(inference_times)) if inference_times else 0.0,
        "mean_pso_seconds_per_episode":
            float(np.mean(pso_times)) if pso_times else 0.0,
        "train_seconds":         float(train_seconds),
        "test_seconds":          float(test_seconds),
        "test_success_rate":     float(np.mean([m["success"] for m in test_metrics])),
        "uses_gnn":              uses_gnn(config_name),
        "uses_pso":              uses_pso(config_name),
    }
    return {
        "summary": summary,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
