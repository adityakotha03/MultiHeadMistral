from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype_from_name(name: str) -> torch.dtype:
    lower_name = name.lower()
    if lower_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lower_name in {"fp16", "float16", "half"}:
        return torch.float16
    if lower_name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
