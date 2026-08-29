"""
Unit tests for the GNN-DRL-PSO pipeline.

Run:
    cd GNN_DRL_PSO_SINGLE_CLOUD
    pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import pytest
import torch


# =====================================================================
# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def small_dataset():
    """Tiny synthetic dataset used ONLY by unit tests (never by experiments)."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "response_time":  rng.uniform(50, 500, n),
        "availability":   rng.uniform(50, 100, n),
        "throughput":     rng.uniform(1, 20, n),
        "successability": rng.uniform(50, 100, n),
        "reliability":    rng.uniform(50, 100, n),
        "compliance":     rng.uniform(50, 100, n),
        "best_practices": rng.uniform(50, 100, n),
        "latency":        rng.uniform(1, 200, n),
        "documentation":  rng.uniform(0, 100, n),
        "service_name":   [f"svc_{i}" for i in range(n)],
        "wsdl_address":   [f"http://x/{i}.wsdl" for i in range(n)],
    })
    # Deterministic classification (10 per bucket)
    cls = np.array([1]*10 + [2]*10 + [3]*10 + [4]*10)
    names = {1:"Platinum", 2:"Gold", 3:"Silver", 4:"Bronze"}
    df["service_classification"] = cls
    df["service_class_name"] = [names[c] for c in cls]
    return df.reset_index(drop=True)


# =====================================================================
# Dataset loading
# =====================================================================
def test_data_loader_reads_real_qws2():
    from preprocessing.data_loader import load_qws_dataset, QOS_COLUMNS
    root = Path(__file__).resolve().parent.parent
    p = root / "data" / "raw" / "QWS2.txt"
    if not p.exists():
        pytest.skip("QWS2.txt not present")
    df = load_qws_dataset(p)
    assert df.shape[1] == 11
    assert set(QOS_COLUMNS).issubset(df.columns)
    assert len(df) > 0


# =====================================================================
# Normalizer
# =====================================================================
def test_normalizer_train_only_fit(small_dataset):
    from preprocessing.data_loader import QOS_COLUMNS
    from preprocessing.normalizer import QoSNormalizer
    directions = {c: ("min" if c in ("response_time", "latency") else "max")
                  for c in QOS_COLUMNS}
    norm = QoSNormalizer(directions).fit(small_dataset)
    out = norm.transform(small_dataset, columns=QOS_COLUMNS)
    v = out[QOS_COLUMNS].to_numpy()
    assert v.min() >= 0.0 - 1e-9 and v.max() <= 1.0 + 1e-9


# =====================================================================
# QoS fitness
# =====================================================================
def test_qos_fitness_in_unit_interval(small_dataset):
    from preprocessing.data_loader import QOS_COLUMNS
    from qos.fitness import QoSFitness
    directions = {c: ("min" if c in ("response_time", "latency") else "max")
                  for c in QOS_COLUMNS}
    weights = {c: 1.0 / len(QOS_COLUMNS) for c in QOS_COLUMNS}
    q_min = {c: float(small_dataset[c].min()) for c in QOS_COLUMNS}
    q_max = {c: float(small_dataset[c].max()) for c in QOS_COLUMNS}
    fitness = QoSFitness(QOS_COLUMNS, weights, directions, q_min, q_max)
    mat = small_dataset[QOS_COLUMNS].to_numpy(dtype=np.float64)
    fr = fitness.evaluate([0, 10, 20, 30], mat)
    assert 0.0 <= fr.fitness <= 1.0
    assert fr.valid


# =====================================================================
# Graph construction
# =====================================================================
def test_graph_has_expected_edges(small_dataset):
    from preprocessing.data_loader import QOS_COLUMNS
    from preprocessing.normalizer import QoSNormalizer
    from graph.graph_builder import build_service_graph
    directions = {c: ("min" if c in ("response_time", "latency") else "max")
                  for c in QOS_COLUMNS}
    norm_df = QoSNormalizer(directions).fit_transform(small_dataset)
    normalized = norm_df[QOS_COLUMNS].to_numpy(dtype=np.float32)
    g = build_service_graph(small_dataset, normalized, QOS_COLUMNS,
                            intra_class_knn=3, inter_class_knn=2)
    assert g.num_nodes == len(small_dataset)
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] > 0
    assert g.num_features == len(QOS_COLUMNS) + 4     # QoS + 4-class one-hot


# =====================================================================
# GNN forward pass
# =====================================================================
def test_gnn_forward_dims(small_dataset):
    from preprocessing.data_loader import QOS_COLUMNS
    from preprocessing.normalizer import QoSNormalizer
    from graph.graph_builder import build_service_graph
    from graph.gnn_encoder import GNNEncoder
    directions = {c: ("min" if c in ("response_time", "latency") else "max")
                  for c in QOS_COLUMNS}
    norm_df = QoSNormalizer(directions).fit_transform(small_dataset)
    normalized = norm_df[QOS_COLUMNS].to_numpy(dtype=np.float32)
    g = build_service_graph(small_dataset, normalized, QOS_COLUMNS,
                            intra_class_knn=3, inter_class_knn=2)
    enc = GNNEncoder(in_channels=g.num_features, hidden_channels=16,
                     out_channels=8, num_layers=2, dropout=0.0)
    z = enc(g.x, g.edge_index)
    assert z.shape == (len(small_dataset), 8)


# =====================================================================
# Environment
# =====================================================================
def test_env_reset_step_success(small_dataset):
    from preprocessing.data_loader import QOS_COLUMNS
    from qos.fitness import QoSFitness
    from environment.single_cloud_env import SingleCloudComposeEnv
    directions = {c: ("min" if c in ("response_time", "latency") else "max")
                  for c in QOS_COLUMNS}
    weights = {c: 1.0/len(QOS_COLUMNS) for c in QOS_COLUMNS}
    q_min = {c: float(small_dataset[c].min()) for c in QOS_COLUMNS}
    q_max = {c: float(small_dataset[c].max()) for c in QOS_COLUMNS}
    fit = QoSFitness(QOS_COLUMNS, weights, directions, q_min, q_max)
    env = SingleCloudComposeEnv(small_dataset, QOS_COLUMNS, fit,
                                workflow_length=4,
                                task_pool=["Platinum","Gold","Silver","Bronze"])
    state = env.reset(seed=1)
    assert state.shape == (env.state_dim,)
    for _ in range(4):
        mask = env.action_mask()
        assert mask.any()
        action = int(np.where(mask)[0][0])
        state, r, done, info = env.step(action)
        assert info.valid
    assert done


def test_env_invalid_action(small_dataset):
    from preprocessing.data_loader import QOS_COLUMNS
    from qos.fitness import QoSFitness
    from environment.single_cloud_env import SingleCloudComposeEnv
    directions = {c: ("min" if c in ("response_time", "latency") else "max")
                  for c in QOS_COLUMNS}
    weights = {c: 1.0/len(QOS_COLUMNS) for c in QOS_COLUMNS}
    q_min = {c: float(small_dataset[c].min()) for c in QOS_COLUMNS}
    q_max = {c: float(small_dataset[c].max()) for c in QOS_COLUMNS}
    fit = QoSFitness(QOS_COLUMNS, weights, directions, q_min, q_max)
    env = SingleCloudComposeEnv(small_dataset, QOS_COLUMNS, fit,
                                workflow_length=4,
                                task_pool=["Platinum","Gold","Silver","Bronze"])
    env.reset(seed=1)
    state, r, done, info = env.step(-1)
    assert done and not info.valid


# =====================================================================
# PSO
# =====================================================================
def test_pso_produces_valid_particles():
    from pso.discrete_pso import DiscretePSO
    cand = [np.array([0,1,2,3]), np.array([4,5,6,7]), np.array([8,9,10,11])]
    fn = lambda pos: -sum(pos) / 100.0
    pso = DiscretePSO(cand, fn, num_particles=8, num_iterations=6,
                      rng=np.random.default_rng(42))
    result = pso.optimize(seed_position=[3, 4, 8])
    for i, v in enumerate(result.best_position):
        assert v in list(cand[i])
    assert isinstance(result.best_fitness, float)
    assert len(result.history) == 7


# =====================================================================
# PPO update (short)
# =====================================================================
def test_ppo_update_short(small_dataset):
    from preprocessing.data_loader import QOS_COLUMNS
    from qos.fitness import QoSFitness
    from environment.single_cloud_env import SingleCloudComposeEnv
    from drl.policy import ActorCriticPolicy
    from drl.ppo_agent import PPOAgent
    directions = {c: ("min" if c in ("response_time", "latency") else "max")
                  for c in QOS_COLUMNS}
    weights = {c: 1.0/len(QOS_COLUMNS) for c in QOS_COLUMNS}
    q_min = {c: float(small_dataset[c].min()) for c in QOS_COLUMNS}
    q_max = {c: float(small_dataset[c].max()) for c in QOS_COLUMNS}
    fit = QoSFitness(QOS_COLUMNS, weights, directions, q_min, q_max)
    env = SingleCloudComposeEnv(small_dataset, QOS_COLUMNS, fit,
                                workflow_length=3,
                                task_pool=["Platinum","Gold","Silver"])
    torch.manual_seed(0)
    # Random-projection service embeddings so the policy works without a GNN
    feats = torch.randn(env.n_services, 13)
    emb   = torch.nn.Linear(13, 8, bias=False)(feats).detach()
    policy = ActorCriticPolicy(state_dim=env.state_dim, n_services=env.n_services,
                               service_embed_dim=8, hidden_dim=16,
                               use_gnn=True, gnn_trainable=False,
                               service_embeddings=emb)
    agent = PPOAgent(policy, batch_size=4, update_epochs=2)
    for _ in range(3):
        s = env.reset(seed=1)
        while True:
            mask = env.action_mask()
            a, lp, v = agent.select_action(s, mask)
            ns, r, done, info = env.step(a)
            agent.buffer.add(s, a, lp, r, v, done, mask)
            s = ns
            if done: break
    out = agent.update()
    assert out["n_updates"] > 0


# =====================================================================
# Statistics helper
# =====================================================================
def test_paired_comparison_basic():
    from evaluation.statistics import paired_comparison
    a = np.array([0.60, 0.62, 0.61, 0.63, 0.62])
    b = np.array([0.55, 0.58, 0.56, 0.59, 0.57])
    out = paired_comparison(a, b, "prop", "base")
    assert out["mean_diff"] > 0
    assert 0.0 <= out["wilcoxon_p"] <= 1.0
