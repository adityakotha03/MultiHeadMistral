from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from transformers import Trainer, TrainerCallback, TrainingArguments

from .config import MultiTokenMistralConfig
from .data_mbpp import CausalLMCollator, build_mbpp_datasets
from .model_multitoken import build_training_multitoken_model, load_tokenizer
from .utils import ensure_dir, load_env_file, set_seed


class MultiTokenTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_head_log_step = -1

    def compute_loss(  # type: ignore[override]
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        del num_items_in_batch
        outputs = model(**inputs)
        loss = outputs["loss"]

        if model.training:
            current_step = int(self.state.global_step)
            if current_step != self._last_head_log_step:
                self._last_head_log_step = current_step
                head_losses = outputs.get("head_losses")
                if head_losses is not None:
                    logs = {
                        f"train/loss_head_{head_index}": float(head_losses[head_index].detach().float().cpu().item())
                        for head_index in range(head_losses.numel())
                    }
                    self.log(logs)

        if return_outputs:
            return loss, outputs
        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):  # type: ignore[override]
        # Force PyTorch serialization for checkpoints to avoid safetensors tied-weight errors
        # with wrapper models that expose shared lm_head/embed_tokens tensors.
        checkpoint_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, "pytorch_model.bin"))

        processor = getattr(self, "processing_class", None)
        if processor is None:
            processor = getattr(self, "tokenizer", None)
        if processor is not None:
            processor.save_pretrained(checkpoint_dir)

        torch.save(self.args, os.path.join(checkpoint_dir, "training_args.bin"))


class PerfAndLossCallback(TrainerCallback):
    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self._step_start: Optional[float] = None
        self._last_step_time: Optional[float] = None

    def on_step_begin(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        if self._step_start is None:
            return
        self._last_step_time = max(time.perf_counter() - self._step_start, 1e-6)

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs):
        del control, kwargs
        if logs is None:
            return

        if "loss" in logs:
            logs["train/loss_total"] = float(logs["loss"])
        if "eval_loss" in logs:
            logs["eval/loss_total"] = float(logs["eval_loss"])

        if self._last_step_time is not None and state.global_step > 0:
            tokens_per_step = args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps) * self.max_seq_len
            logs["perf/step_time_sec"] = float(self._last_step_time)
            logs["perf/tokens_per_sec"] = float(tokens_per_step / self._last_step_time)

        if torch.cuda.is_available():
            logs["gpu/max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ministal 3B with multi-token heads (QLoRA + W&B).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output_dir override.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional num_train_steps override.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Checkpoint path to resume from, or "latest" to pick latest checkpoint-* in output_dir.',
    )
    return parser.parse_args()


def _find_latest_checkpoint(output_dir: str | Path) -> Optional[str]:
    path = Path(output_dir)
    if not path.exists():
        return None

    candidates = []
    for child in path.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        suffix = child.name.replace("checkpoint-", "", 1)
        if suffix.isdigit():
            candidates.append((int(suffix), child))

    for _, checkpoint_dir in sorted(candidates, key=lambda x: x[0], reverse=True):
        if _is_resume_checkpoint_valid(checkpoint_dir):
            return str(checkpoint_dir)
    return None


def _can_read_torch_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        _ = torch.load(path, map_location="cpu")
        return True
    except Exception as exc:
        print(f"[warn] Skipping unreadable checkpoint file: {path} ({exc})")
        return False


def _is_resume_checkpoint_valid(checkpoint_dir: Path) -> bool:
    model_file = checkpoint_dir / "pytorch_model.bin"
    trainer_state_file = checkpoint_dir / "trainer_state.json"

    if not trainer_state_file.exists():
        return False
    if not _can_read_torch_file(model_file):
        return False

    optional_state_files = [
        checkpoint_dir / "optimizer.pt",
        checkpoint_dir / "scheduler.pt",
        checkpoint_dir / "scaler.pt",
        checkpoint_dir / "rng_state.pth",
    ]
    for state_file in optional_state_files:
        if state_file.exists() and not _can_read_torch_file(state_file):
            return False
    return True


def _resolve_resume_checkpoint(
    cfg: MultiTokenMistralConfig,
    cli_resume_from_checkpoint: Optional[str],
) -> Optional[str]:
    if cli_resume_from_checkpoint:
        if cli_resume_from_checkpoint.lower() == "latest":
            latest = _find_latest_checkpoint(cfg.output_dir)
            if latest is None:
                raise RuntimeError(
                    f'No checkpoint-* directories found in "{cfg.output_dir}" to resume from.'
                )
            return latest
        if not Path(cli_resume_from_checkpoint).exists():
            raise RuntimeError(f'Checkpoint path does not exist: "{cli_resume_from_checkpoint}"')
        return cli_resume_from_checkpoint

    if cfg.auto_resume_from_latest:
        return _find_latest_checkpoint(cfg.output_dir)
    return None


def main() -> None:
    args = parse_args()
    load_env_file(".env")

    cfg = MultiTokenMistralConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_steps is not None:
        cfg.num_train_steps = args.max_steps

    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required in environment or .env for loading the base model.")

    if cfg.report_to_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
        os.environ.setdefault("WANDB_NAME", cfg.wandb_run_name)

    tokenizer = load_tokenizer(cfg.base_model_name, hf_token=hf_token)
    train_dataset, eval_dataset = build_mbpp_datasets(tokenizer=tokenizer, cfg=cfg)
    model = build_training_multitoken_model(cfg=cfg, hf_token=hf_token)
    loaded_in_4bit = bool(getattr(model.base_model, "is_loaded_in_4bit", False))

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    ta_kwargs = {
        "output_dir": cfg.output_dir,
        "max_steps": cfg.num_train_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "logging_steps": cfg.logging_steps,
        "eval_steps": cfg.eval_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "report_to": ["wandb"] if cfg.report_to_wandb else [],
        "run_name": cfg.wandb_run_name,
        "remove_unused_columns": False,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "optim": "paged_adamw_8bit" if loaded_in_4bit else "adamw_torch",
        "gradient_checkpointing": True,
        "dataloader_pin_memory": True,
        "logging_first_step": True,
    }

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "save_safetensors" in ta_params:
        ta_kwargs["save_safetensors"] = False
    else:
        print("[warn] TrainingArguments has no save_safetensors arg; using torch-only checkpoint save override.")

    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError(
            "This transformers version has neither eval_strategy nor evaluation_strategy in TrainingArguments."
        )

    training_args = TrainingArguments(**ta_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": CausalLMCollator(tokenizer),
        "callbacks": [PerfAndLossCallback(max_seq_len=cfg.max_seq_len)],
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = MultiTokenTrainer(**trainer_kwargs)

    resume_checkpoint = _resolve_resume_checkpoint(cfg, args.resume_from_checkpoint)
    if resume_checkpoint:
        print(f"[info] Resuming from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    eval_metrics = trainer.evaluate()

    model.save_multitoken(cfg.output_dir, cfg=cfg)
    tokenizer.save_pretrained(cfg.output_dir)
    trainer.save_state()
    Path(cfg.output_dir, "final_eval_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(eval_metrics, indent=2))
    print(f"Saved artifacts to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
