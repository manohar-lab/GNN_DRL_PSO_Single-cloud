"""
Utilities: config loading, seed control, path management, logging.
Every experiment module goes through this helper.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def project_root() -> Path:
    """Return the absolute path of the GNN_DRL_PSO_SINGLE_CLOUD directory."""
    return Path(__file__).resolve().parent.parent


def add_project_root_to_sys_path() -> None:
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = project_root() / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg_path: str) -> Path:
    """Interpret a config path relative to the project root."""
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    return (project_root() / p).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        torch.use_deterministic_algorithms(False)  # allow non-deterministic ops on CPU
    except Exception:
        pass


def pick_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_run_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            "%H:%M:%S")
    fh = logging.FileHandler(log_dir / f"{name}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
