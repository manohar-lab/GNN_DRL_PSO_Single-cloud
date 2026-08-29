"""
Phase 1-3: dataset inspection + train/test split + normalization.

Running this once produces
  data/train/train.csv
  data/test/test.csv
  data/processed/split_metadata.json
  data/processed/normalizer.json
  data/processed/dataset_summary.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.utils import (
    add_project_root_to_sys_path,
    load_config,
    project_root,
    resolve_path,
    save_json,
    set_seed,
)

add_project_root_to_sys_path()

from preprocessing.data_loader import (        # noqa: E402
    QOS_COLUMNS,
    load_qws_dataset,
    split_train_test,
)
from preprocessing.dataset_analyzer import analyze_dataset   # noqa: E402
from preprocessing.normalizer import QoSNormalizer            # noqa: E402


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["dataset"]["split_seed"])

    raw_path = resolve_path(cfg["dataset"]["raw_path"])
    train_path = resolve_path(cfg["dataset"]["train_path"])
    test_path = resolve_path(cfg["dataset"]["test_path"])
    processed_dir = resolve_path(cfg["dataset"]["processed_dir"])

    print(f"\n[preprocess] loading raw dataset from {raw_path}")
    df = load_qws_dataset(raw_path)
    print(f"[preprocess] loaded {len(df)} rows x {df.shape[1]} cols")

    summary_raw = analyze_dataset(df, label="Full raw QWS2")

    print("\n[preprocess] splitting 80/20 (deterministic)")
    train_df, test_df, split_meta = split_train_test(
        df,
        train_ratio=cfg["dataset"]["train_test_split"],
        seed=cfg["dataset"]["split_seed"],
        train_out=train_path,
        test_out=test_path,
        processed_dir=processed_dir,
    )
    print(f"[preprocess] wrote train.csv  ->  {train_path}")
    print(f"[preprocess] wrote test.csv   ->  {test_path}")

    summary_train = analyze_dataset(train_df, label="Train (80%)")
    summary_test = analyze_dataset(test_df, label="Test  (20%)")

    print("\n[preprocess] fitting QoS normalizer on TRAIN only")
    directions = cfg["dataset"]["qos_direction"]
    normalizer = QoSNormalizer(directions=directions)
    normalizer.fit(train_df, columns=QOS_COLUMNS)
    normalizer.save(processed_dir / "normalizer.json")
    print(f"[preprocess] wrote normalizer -> {processed_dir/'normalizer.json'}")

    summary = {
        "raw_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test":  int(len(test_df)),
        "qos_columns": QOS_COLUMNS,
        "split": split_meta,
        "raw_stats":    summary_raw.get("qos_stats", {}),
        "train_stats":  summary_train.get("qos_stats", {}),
        "test_stats":   summary_test.get("qos_stats", {}),
        "class_distribution": {
            "train": summary_train.get("classification_distribution", {}),
            "test":  summary_test.get("classification_distribution", {}),
        },
        "normalizer_q_min": normalizer.q_min,
        "normalizer_q_max": normalizer.q_max,
    }
    save_json(summary, processed_dir / "dataset_summary.json")
    print(f"[preprocess] wrote summary    -> {processed_dir/'dataset_summary.json'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    run(a.config)
