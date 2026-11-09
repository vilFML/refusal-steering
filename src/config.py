"""Config loading, artifact paths and seeding."""
import os
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRS = ("data", "vectors", "outputs", "results")


def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths():
    """Artifact folders live under $STEERING_DATA_ROOT (e.g. a Drive folder in Colab),
    or under the repo root when the variable is not set."""
    root = Path(os.environ.get("STEERING_DATA_ROOT", REPO_ROOT))
    paths = {name: root / name for name in ARTIFACT_DIRS}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
