# PRD — GNN-DRL-PSO for QoS-Aware Single-Cloud Service Composition

## Original problem statement
Implement (from scratch) the research pipeline
**"GNN-DRL-PSO for QoS-Aware Single-Cloud Service Composition"** — an
extension of the base paper "A Hybrid Deep Reinforcement and Swarm
Optimization Strategy for Intelligent Cloud Service Composition."
Must be executable on Windows via VS Code, reproducible, modular,
and scientifically valid (fair baseline, no test-set leakage,
data-driven conclusions).

## User personas
- **Research author** — needs a clean, publishable pipeline that
  produces figures/tables/statistics without manual editing.
- **Reviewer** — needs to reproduce results from a single command.
- **Follow-up student** — needs each module to be understandable in
  isolation and each ablation to be a config switch.

## Architecture (implemented)
```
QWS2.txt → preprocess → train/test CSVs + normalizer + WsRF-quartile
       → build_service_graph → pretrain GNN (autoencoder)
       → PPO agent (actor-critic) → episode composition
       → discrete PSO refinement → QoS fitness → PPO reward
       → ablation across {DRL, GNN_DRL, DRL_PSO, GNN_DRL_PSO} × 5 seeds
       → tables + figures + paired statistical comparison + interpretation
```

## Core requirements (satisfied)
- [x] 80/20 deterministic split of QWS2 by service, seed fixed
- [x] Normalizer fit on TRAIN only, applied to test — no leakage
- [x] Service Classification derived transparently via WsRF quartile
- [x] Configurable weighted QoS fitness with per-attribute aggregation
- [x] Single-cloud environment; no cross-cloud actions
- [x] Service dependency graph documented and deterministic
- [x] 2-layer GraphSAGE GNN with graph-autoencoder pretraining
- [x] PPO with clipped objective, GAE(λ), entropy regularization
- [x] Discrete PSO with proper particle/velocity/pbest/gbest updates
- [x] Reward feedback: PSO fitness → terminal PPO reward
- [x] Four-way ablation study (DRL, GNN_DRL, DRL_PSO, GNN_DRL_PSO)
- [x] Wilcoxon + paired-t + Shapiro + Cohen's d + 95 % CI
- [x] Convergence plot with same criterion for all methods
- [x] Full figure set + tables + auto-generated interpretation
- [x] Unit tests (10 tests, all passing)
- [x] Windows-compatible (no OS-specific paths)

## Implemented modules
```
preprocessing/  data_loader.py, dataset_analyzer.py, normalizer.py
qos/            fitness.py, normalization.py
environment/    single_cloud_env.py
graph/          graph_builder.py, graph_dataset.py, gnn_encoder.py
drl/            policy.py, ppo_agent.py, replay_buffer.py
pso/            discrete_pso.py, particle.py
experiments/    preprocess.py, train_gnn.py, train_drl.py,
                train_gnn_drl_pso.py, evaluate.py, ablation.py,
                regenerate.py, runner.py, utils.py
evaluation/     metrics.py, statistics.py, visualization.py
tests/          test_all.py (10 tests)
main.py, config/config.yaml, requirements.txt, README.md
```

## What's been implemented (2026-02)
- Full Phase 1 – Phase 15 of the research plan
- All 20 (4 configs × 5 seeds) training runs completed
- All 7 required figures generated (qos_comparison, convergence,
  execution_time, qos_metrics, ablation, overhead, fitness_distribution)
- Statistical comparison, ablation aggregate, and per-run JSONs
- Research summary at `results/RESEARCH_SUMMARY.txt`

## Actual result (data-driven, NOT forced)
GNN_DRL_PSO vs DRL_PSO: Δ mean = +0.0015 (+0.26 %), Wilcoxon p = 1.0,
Cohen's d = 0.216.  **Insufficient evidence of a statistically
significant difference** in the tested regime — matches outcome (B)
enumerated in the research plan.

## Backlog / P1
- **Joint GNN training**: set `gnn.training_mode: joint`; needs a
  slightly different PPO loop that keeps the GNN in-graph.
- **Larger budget**: 20 seeds × 200 episodes to tighten the CI.
- **Learned linear encoder for non-GNN baselines** to remove the
  known limitation of the random-orthogonal baseline.
- **Contrastive graph learning** as an alternative pretraining
  objective (may better align with composition reward).
- **Workflow-level train/test split** rather than service split
  (needs explicit workflow annotations, currently synthesized).

## How to reproduce
```
cd GNN_DRL_PSO_SINGLE_CLOUD
python main.py --stage preprocess
python main.py --stage train_gnn
python main.py --stage ablation
python main.py --stage report
```
