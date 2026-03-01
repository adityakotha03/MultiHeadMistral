from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class MultiTokenMistralConfig:
    base_model_name: str = "mistralai/Ministral-3-3B-Instruct-2512"
    dataset_name: str = "mbpp"
    train_split: str = "train"
    eval_ratio: float = 0.05

    num_future_heads: int = 4
    head_loss_weights: Optional[List[float]] = None
    max_seq_len: int = 1024

    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_steps: int = 1200
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    auto_resume_from_latest: bool = False

    seed: int = 42
    output_dir: str = "outputs/mt_3b_demo"

    report_to_wandb: bool = True
    wandb_project: str = "ministral-multitoken"
    wandb_run_name: str = "mt-heads-demo"

    benchmark_num_prompts: int = 30
    benchmark_max_new_tokens: int = 128

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MultiTokenMistralConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> "MultiTokenMistralConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MultiTokenMistralConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {k: v for k, v in raw.items() if k in valid_keys}
        if "lora_target_modules" in filtered and isinstance(filtered["lora_target_modules"], list):
            filtered["lora_target_modules"] = tuple(filtered["lora_target_modules"])
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
