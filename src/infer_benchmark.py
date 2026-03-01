from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

from .config import MultiTokenMistralConfig
from .model_multitoken import (
    load_inference_multitoken_model,
    load_multitoken_from_checkpoint,
    load_tokenizer,
)
from .utils import cuda_sync, ensure_dir, load_env_file, set_seed


def _extract_prompt(example: Dict[str, object]) -> str:
    for key in ("text", "prompt", "problem"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_mbpp_prompts(num_prompts: int, seed: int) -> List[str]:
    all_splits = load_dataset("mbpp")
    split_name = "test" if "test" in all_splits else "train"
    split = all_splits[split_name]

    prompts: List[str] = []
    for row in split:
        prompt = _extract_prompt(row)
        if prompt:
            prompts.append(
                "### Instruction:\n"
                "Write a correct Python function for the following task.\n"
                f"{prompt}\n\n"
                "### Response:\n"
            )

    if not prompts:
        raise RuntimeError("Could not extract prompts from MBPP.")

    rng = random.Random(seed)
    if len(prompts) > num_prompts:
        prompts = rng.sample(prompts, num_prompts)
    return prompts


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def run_baseline_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[float, int, str]:
    device = _model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    cuda_sync()
    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    cuda_sync()
    elapsed = time.perf_counter() - start

    new_tokens = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return elapsed, new_tokens, output_text


@torch.no_grad()
def draft_verify_decode(
    wrapper_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, object]:
    device = _model_device(wrapper_model.base_model)
    eos_token_id = tokenizer.eos_token_id

    encoded = tokenizer(prompt, return_tensors="pt")
    current_ids = encoded["input_ids"].to(device)

    generated_tokens: List[int] = []
    drafted_total = 0
    drafted_accepted = 0
    forward_calls = 0

    cuda_sync()
    start = time.perf_counter()

    while len(generated_tokens) < max_new_tokens:
        remaining = max_new_tokens - len(generated_tokens)
        model_outputs = wrapper_model(input_ids=current_ids, use_cache=False)
        forward_calls += 1

        next_logits = model_outputs["logits_head0"][:, -1, :]
        last_hidden = model_outputs["last_hidden_state"][:, -1, :]

        drafted_tokens = [int(torch.argmax(next_logits, dim=-1).item())]
        for head in wrapper_model.aux_heads:
            head_logits = head(last_hidden)
            drafted_tokens.append(int(torch.argmax(head_logits, dim=-1).item()))
        drafted_tokens = drafted_tokens[:remaining]
        drafted_total += len(drafted_tokens)

        draft_tensor = torch.tensor([drafted_tokens], dtype=current_ids.dtype, device=device)
        verify_input = torch.cat([current_ids, draft_tensor], dim=1)

        verify_outputs = wrapper_model.base_model(input_ids=verify_input, return_dict=True)
        forward_calls += 1
        verify_logits = verify_outputs.logits

        context_len = current_ids.shape[1]
        verifier_tokens: List[int] = []
        for token_index in range(len(drafted_tokens)):
            position = context_len - 1 + token_index
            token_id = int(torch.argmax(verify_logits[:, position, :], dim=-1).item())
            verifier_tokens.append(token_id)

        accepted: List[int] = []
        matched = 0
        for draft_token, verifier_token in zip(drafted_tokens, verifier_tokens):
            if draft_token == verifier_token:
                accepted.append(draft_token)
                matched += 1
            else:
                accepted.append(verifier_token)
                break
        drafted_accepted += matched

        if not accepted:
            accepted = [verifier_tokens[0]]

        stop = False
        accepted_for_append: List[int] = []
        for token in accepted:
            accepted_for_append.append(token)
            generated_tokens.append(token)
            if eos_token_id is not None and token == eos_token_id:
                stop = True
                break
            if len(generated_tokens) >= max_new_tokens:
                stop = True
                break

        append_tensor = torch.tensor([accepted_for_append], dtype=current_ids.dtype, device=device)
        current_ids = torch.cat([current_ids, append_tensor], dim=1)

        if stop:
            break

    cuda_sync()
    elapsed = time.perf_counter() - start
    decoded_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    return {
        "elapsed_sec": elapsed,
        "new_tokens": len(generated_tokens),
        "text": decoded_text,
        "drafted_total": drafted_total,
        "drafted_accepted": drafted_accepted,
        "forward_calls": forward_calls,
    }


def summarize_mode(total_time: float, total_tokens: int, num_prompts: int) -> Dict[str, float]:
    avg_latency = float(total_time / num_prompts) if num_prompts > 0 else 0.0
    tokens_per_sec = float(total_tokens / total_time) if total_time > 0 else 0.0
    return {"total_time_sec": float(total_time), "tokens_generated": int(total_tokens), "tokens_per_sec": tokens_per_sec, "avg_latency_sec": float(avg_latency)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs multi-token head decoding.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Final exported run directory (contains adapter/, multi_token_heads.pt, multitoken_config.json).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Trainer checkpoint directory (checkpoint-*) for interrupted runs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path (YAML/JSON) used during training; needed when loading from checkpoint_dir.",
    )
    parser.add_argument("--num_prompts", type=int, default=None, help="Number of prompts to benchmark.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens per prompt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default=None, help="Optional explicit JSON output path.")
    return parser.parse_args()


def _load_cfg_from_any(path: str | Path) -> MultiTokenMistralConfig:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return MultiTokenMistralConfig.from_json(path)
    return MultiTokenMistralConfig.from_yaml(path)


def _resolve_checkpoint_cfg(args: argparse.Namespace, run_dir_hint: Optional[Path]) -> MultiTokenMistralConfig:
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise RuntimeError(f"Config path does not exist: {cfg_path}")
        return _load_cfg_from_any(cfg_path)

    if run_dir_hint is not None:
        exported_cfg = run_dir_hint / "multitoken_config.json"
        if exported_cfg.exists():
            return MultiTokenMistralConfig.from_json(exported_cfg)

    default_cfg = Path("configs/default.yaml")
    if default_cfg.exists():
        print("[warn] Using configs/default.yaml for checkpoint inference config.")
        return MultiTokenMistralConfig.from_yaml(default_cfg)

    raise RuntimeError(
        "Could not resolve config for checkpoint inference. "
        "Pass --config path/to/config.yaml used during training."
    )


def main() -> None:
    args = parse_args()
    load_env_file(".env")
    set_seed(args.seed)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required in environment or .env for loading the base model.")

    if args.model_dir is None and args.checkpoint_dir is None:
        raise RuntimeError("Pass either --model_dir (final export) or --checkpoint_dir (interrupted checkpoint).")

    model_dir: Optional[Path] = Path(args.model_dir) if args.model_dir else None
    checkpoint_dir: Optional[Path] = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    if checkpoint_dir is not None:
        run_dir_hint = model_dir if model_dir is not None else checkpoint_dir.parent
        cfg = _resolve_checkpoint_cfg(args, run_dir_hint=run_dir_hint)
        artifacts_root = checkpoint_dir
    else:
        assert model_dir is not None
        cfg_path = model_dir / "multitoken_config.json"
        cfg = MultiTokenMistralConfig.from_json(cfg_path)
        artifacts_root = model_dir

    num_prompts = args.num_prompts or cfg.benchmark_num_prompts
    max_new_tokens = args.max_new_tokens or cfg.benchmark_max_new_tokens
    prompts = load_mbpp_prompts(num_prompts=num_prompts, seed=args.seed)

    tokenizer_dir = None
    if model_dir is not None and (model_dir / "tokenizer_config.json").exists():
        tokenizer_dir = model_dir
    elif checkpoint_dir is not None and (checkpoint_dir.parent / "tokenizer_config.json").exists():
        tokenizer_dir = checkpoint_dir.parent
    tokenizer_source = str(tokenizer_dir) if tokenizer_dir is not None else cfg.base_model_name
    tokenizer = load_tokenizer(tokenizer_source, hf_token=hf_token)

    if checkpoint_dir is not None:
        wrapper_model = load_multitoken_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            cfg=cfg,
            hf_token=hf_token,
        )
    else:
        assert model_dir is not None
        wrapper_model, _ = load_inference_multitoken_model(model_dir=model_dir, cfg=cfg, hf_token=hf_token)
    base_model = wrapper_model.base_model

    baseline_total_time = 0.0
    baseline_total_tokens = 0
    multi_total_time = 0.0
    multi_total_tokens = 0
    drafted_total = 0
    drafted_accepted = 0
    forward_calls_total = 0
    examples = []

    for prompt in tqdm(prompts, desc="Running benchmark"):
        baseline_elapsed, baseline_tokens, baseline_text = run_baseline_generate(
            base_model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        baseline_total_time += baseline_elapsed
        baseline_total_tokens += baseline_tokens

        multi_result = draft_verify_decode(
            wrapper_model=wrapper_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        multi_total_time += float(multi_result["elapsed_sec"])
        multi_total_tokens += int(multi_result["new_tokens"])
        drafted_total += int(multi_result["drafted_total"])
        drafted_accepted += int(multi_result["drafted_accepted"])
        forward_calls_total += int(multi_result["forward_calls"])

        if len(examples) < 3:
            examples.append(
                {
                    "prompt": prompt,
                    "baseline_text": baseline_text,
                    "multi_token_text": str(multi_result["text"]),
                }
            )

    baseline_summary = summarize_mode(baseline_total_time, baseline_total_tokens, len(prompts))
    multi_summary = summarize_mode(multi_total_time, multi_total_tokens, len(prompts))
    acceptance_rate = float(drafted_accepted / drafted_total) if drafted_total > 0 else 0.0

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir.resolve()) if model_dir is not None else None,
        "checkpoint_dir": str(checkpoint_dir.resolve()) if checkpoint_dir is not None else None,
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "baseline": baseline_summary,
        "multi_token": {
            **multi_summary,
            "draft_acceptance_rate": acceptance_rate,
            "drafted_total_tokens": drafted_total,
            "drafted_accepted_tokens": drafted_accepted,
            "forward_calls": forward_calls_total,
        },
        "speedup_by_latency": (
            baseline_summary["total_time_sec"] / multi_summary["total_time_sec"]
            if multi_summary["total_time_sec"] > 0
            else 0.0
        ),
        "speedup_by_throughput": (
            multi_summary["tokens_per_sec"] / baseline_summary["tokens_per_sec"]
            if baseline_summary["tokens_per_sec"] > 0
            else 0.0
        ),
        "examples": examples,
    }

    output_path = Path(args.save_path) if args.save_path else artifacts_root / "benchmark_results.json"
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Saved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
