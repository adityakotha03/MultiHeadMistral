from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from .config import MultiTokenMistralConfig
from .model_multitoken import (
    load_inference_multitoken_model,
    load_multitoken_from_checkpoint,
    load_raw_base_model_for_inference,
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
    fixed_length: bool,
) -> Dict[str, object]:
    device = _model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_kwargs: Dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if fixed_length:
        generate_kwargs["min_new_tokens"] = max_new_tokens

    cuda_sync()
    start = time.perf_counter()
    try:
        output_ids = model.generate(**inputs, **generate_kwargs)
    except TypeError:
        # Fallback for transformers builds where min_new_tokens is unavailable.
        if "min_new_tokens" in generate_kwargs:
            del generate_kwargs["min_new_tokens"]
            generate_kwargs["min_length"] = int(inputs["input_ids"].shape[1] + max_new_tokens)
        output_ids = model.generate(**inputs, **generate_kwargs)
    cuda_sync()
    elapsed = time.perf_counter() - start

    new_tokens = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {
        "elapsed_sec": elapsed,
        "new_tokens": new_tokens,
        "text": output_text,
    }


@torch.no_grad()
def draft_verify_decode(
    wrapper_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    fixed_length: bool,
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
            if not fixed_length and eos_token_id is not None and token == eos_token_id:
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
    return {
        "total_time_sec": float(total_time),
        "tokens_generated": int(total_tokens),
        "tokens_per_sec": tokens_per_sec,
        "avg_latency_sec": float(avg_latency),
    }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _metric_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"median": 0.0, "mean": 0.0, "p90": 0.0}
    return {
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "p90": _percentile(values, 0.90),
    }


def _run_prompt_set(
    prompts: List[str],
    wrapper_model,
    adapted_base_model,
    raw_base_model,
    tokenizer,
    max_new_tokens: int,
    fixed_length: bool,
    progress_desc: str,
    include_examples: bool,
    include_per_prompt: bool,
) -> Dict[str, object]:
    adapted_total_time = 0.0
    adapted_total_tokens = 0
    raw_total_time = 0.0
    raw_total_tokens = 0
    multi_total_time = 0.0
    multi_total_tokens = 0
    drafted_total = 0
    drafted_accepted = 0
    forward_calls_total = 0
    per_prompt_rows: List[Dict[str, object]] = []
    examples: List[Dict[str, str]] = []

    iterator = tqdm(prompts, desc=progress_desc, leave=False) if prompts else []
    for prompt_index, prompt in enumerate(iterator, start=1):
        adapted_result = run_baseline_generate(
            adapted_base_model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            fixed_length=fixed_length,
        )
        raw_result = None
        if raw_base_model is not None:
            raw_result = run_baseline_generate(
                raw_base_model,
                tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                fixed_length=fixed_length,
            )
        multi_result = draft_verify_decode(
            wrapper_model=wrapper_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            fixed_length=fixed_length,
        )

        adapted_elapsed = float(adapted_result["elapsed_sec"])
        multi_elapsed = float(multi_result["elapsed_sec"])
        adapted_tokens = int(adapted_result["new_tokens"])
        multi_tokens = int(multi_result["new_tokens"])
        raw_elapsed = float(raw_result["elapsed_sec"]) if raw_result is not None else 0.0
        raw_tokens = int(raw_result["new_tokens"]) if raw_result is not None else 0

        adapted_total_time += adapted_elapsed
        adapted_total_tokens += adapted_tokens
        raw_total_time += raw_elapsed
        raw_total_tokens += raw_tokens
        multi_total_time += multi_elapsed
        multi_total_tokens += multi_tokens
        drafted_total += int(multi_result["drafted_total"])
        drafted_accepted += int(multi_result["drafted_accepted"])
        forward_calls_total += int(multi_result["forward_calls"])

        if include_per_prompt:
            adapted_tps = float(adapted_tokens / adapted_elapsed) if adapted_elapsed > 0 else 0.0
            multi_tps = float(multi_tokens / multi_elapsed) if multi_elapsed > 0 else 0.0
            row: Dict[str, object] = {
                "prompt_index": prompt_index,
                "adapted_baseline_elapsed_sec": adapted_elapsed,
                "multi_elapsed_sec": multi_elapsed,
                "adapted_baseline_tokens": adapted_tokens,
                "multi_tokens": multi_tokens,
                "adapted_baseline_tokens_per_sec": adapted_tps,
                "multi_tokens_per_sec": multi_tps,
                "speedup_multi_vs_adapted_by_latency": float(adapted_elapsed / multi_elapsed)
                if multi_elapsed > 0
                else 0.0,
                "speedup_multi_vs_adapted_by_throughput": float(multi_tps / adapted_tps)
                if adapted_tps > 0
                else 0.0,
            }
            if raw_result is not None:
                raw_tps = float(raw_tokens / raw_elapsed) if raw_elapsed > 0 else 0.0
                row.update(
                    {
                        "raw_base_elapsed_sec": raw_elapsed,
                        "raw_base_tokens": raw_tokens,
                        "raw_base_tokens_per_sec": raw_tps,
                        "speedup_multi_vs_raw_by_latency": float(raw_elapsed / multi_elapsed)
                        if multi_elapsed > 0
                        else 0.0,
                        "speedup_multi_vs_raw_by_throughput": float(multi_tps / raw_tps) if raw_tps > 0 else 0.0,
                        "speedup_adapted_vs_raw_by_latency": float(raw_elapsed / adapted_elapsed)
                        if adapted_elapsed > 0
                        else 0.0,
                        "speedup_adapted_vs_raw_by_throughput": float(adapted_tps / raw_tps) if raw_tps > 0 else 0.0,
                    }
                )
            per_prompt_rows.append(row)

        if include_examples and len(examples) < 3:
            example_row: Dict[str, str] = {
                "prompt": prompt,
                "adapted_baseline_text": str(adapted_result["text"]),
                "multi_token_text": str(multi_result["text"]),
            }
            if raw_result is not None:
                example_row["raw_base_text"] = str(raw_result["text"])
            examples.append(example_row)

    adapted_summary = summarize_mode(adapted_total_time, adapted_total_tokens, len(prompts))
    raw_summary = summarize_mode(raw_total_time, raw_total_tokens, len(prompts))
    multi_summary = summarize_mode(multi_total_time, multi_total_tokens, len(prompts))
    adapted_acceptance_rate = 1.0 if adapted_total_tokens > 0 else 0.0
    raw_acceptance_rate = 1.0 if raw_total_tokens > 0 else 0.0
    draft_acceptance_rate = float(drafted_accepted / drafted_total) if drafted_total > 0 else 0.0
    adapted_summary["token_acceptance_rate"] = adapted_acceptance_rate
    if raw_base_model is not None:
        raw_summary["token_acceptance_rate"] = raw_acceptance_rate
    multi_summary.update(
        {
            "token_acceptance_rate": draft_acceptance_rate,
            "draft_acceptance_rate": draft_acceptance_rate,
            "drafted_total_tokens": drafted_total,
            "drafted_accepted_tokens": drafted_accepted,
            "forward_calls": forward_calls_total,
        }
    )

    result: Dict[str, object] = {
        "adapted_baseline": adapted_summary,
        "multi_token": multi_summary,
        "speedup_multi_vs_adapted_by_latency": (
            adapted_summary["total_time_sec"] / multi_summary["total_time_sec"]
            if multi_summary["total_time_sec"] > 0
            else 0.0
        ),
        "speedup_multi_vs_adapted_by_throughput": (
            multi_summary["tokens_per_sec"] / adapted_summary["tokens_per_sec"]
            if adapted_summary["tokens_per_sec"] > 0
            else 0.0
        ),
    }
    if raw_base_model is not None:
        result["raw_base"] = raw_summary
        result["speedup_multi_vs_raw_by_latency"] = (
            raw_summary["total_time_sec"] / multi_summary["total_time_sec"]
            if multi_summary["total_time_sec"] > 0
            else 0.0
        )
        result["speedup_multi_vs_raw_by_throughput"] = (
            multi_summary["tokens_per_sec"] / raw_summary["tokens_per_sec"]
            if raw_summary["tokens_per_sec"] > 0
            else 0.0
        )
        result["speedup_adapted_vs_raw_by_latency"] = (
            raw_summary["total_time_sec"] / adapted_summary["total_time_sec"]
            if adapted_summary["total_time_sec"] > 0
            else 0.0
        )
        result["speedup_adapted_vs_raw_by_throughput"] = (
            adapted_summary["tokens_per_sec"] / raw_summary["tokens_per_sec"]
            if raw_summary["tokens_per_sec"] > 0
            else 0.0
        )
    if include_examples:
        result["examples"] = examples
    if include_per_prompt:
        result["per_prompt"] = per_prompt_rows
    return result


def _aggregate_repeats(repeat_results: List[Dict[str, object]]) -> Dict[str, object]:
    if not repeat_results:
        raise RuntimeError("No repeat results to aggregate.")

    adapted_total_time: List[float] = []
    adapted_tokens_per_sec: List[float] = []
    adapted_latency: List[float] = []
    adapted_tokens_generated: List[float] = []
    adapted_acceptance: List[float] = []

    raw_total_time: List[float] = []
    raw_tokens_per_sec: List[float] = []
    raw_latency: List[float] = []
    raw_tokens_generated: List[float] = []
    raw_acceptance: List[float] = []
    has_raw = "raw_base" in repeat_results[0]

    multi_total_time: List[float] = []
    multi_tokens_per_sec: List[float] = []
    multi_latency: List[float] = []
    multi_tokens_generated: List[float] = []
    multi_acceptance: List[float] = []
    multi_forward_calls: List[float] = []
    multi_drafted_total: List[float] = []
    multi_drafted_accepted: List[float] = []

    speedup_multi_vs_adapted_latency: List[float] = []
    speedup_multi_vs_adapted_throughput: List[float] = []
    speedup_multi_vs_raw_latency: List[float] = []
    speedup_multi_vs_raw_throughput: List[float] = []
    speedup_adapted_vs_raw_latency: List[float] = []
    speedup_adapted_vs_raw_throughput: List[float] = []

    for repeat_result in repeat_results:
        adapted = repeat_result["adapted_baseline"]  # type: ignore[index]
        multi = repeat_result["multi_token"]  # type: ignore[index]

        adapted_total_time.append(float(adapted["total_time_sec"]))  # type: ignore[index]
        adapted_tokens_per_sec.append(float(adapted["tokens_per_sec"]))  # type: ignore[index]
        adapted_latency.append(float(adapted["avg_latency_sec"]))  # type: ignore[index]
        adapted_tokens_generated.append(float(adapted["tokens_generated"]))  # type: ignore[index]
        adapted_acceptance.append(float(adapted["token_acceptance_rate"]))  # type: ignore[index]

        multi_total_time.append(float(multi["total_time_sec"]))  # type: ignore[index]
        multi_tokens_per_sec.append(float(multi["tokens_per_sec"]))  # type: ignore[index]
        multi_latency.append(float(multi["avg_latency_sec"]))  # type: ignore[index]
        multi_tokens_generated.append(float(multi["tokens_generated"]))  # type: ignore[index]
        multi_acceptance.append(float(multi["token_acceptance_rate"]))  # type: ignore[index]
        multi_forward_calls.append(float(multi["forward_calls"]))  # type: ignore[index]
        multi_drafted_total.append(float(multi["drafted_total_tokens"]))  # type: ignore[index]
        multi_drafted_accepted.append(float(multi["drafted_accepted_tokens"]))  # type: ignore[index]

        speedup_multi_vs_adapted_latency.append(float(repeat_result["speedup_multi_vs_adapted_by_latency"]))
        speedup_multi_vs_adapted_throughput.append(float(repeat_result["speedup_multi_vs_adapted_by_throughput"]))

        if has_raw:
            raw = repeat_result["raw_base"]  # type: ignore[index]
            raw_total_time.append(float(raw["total_time_sec"]))  # type: ignore[index]
            raw_tokens_per_sec.append(float(raw["tokens_per_sec"]))  # type: ignore[index]
            raw_latency.append(float(raw["avg_latency_sec"]))  # type: ignore[index]
            raw_tokens_generated.append(float(raw["tokens_generated"]))  # type: ignore[index]
            raw_acceptance.append(float(raw["token_acceptance_rate"]))  # type: ignore[index]

            speedup_multi_vs_raw_latency.append(float(repeat_result["speedup_multi_vs_raw_by_latency"]))
            speedup_multi_vs_raw_throughput.append(float(repeat_result["speedup_multi_vs_raw_by_throughput"]))
            speedup_adapted_vs_raw_latency.append(float(repeat_result["speedup_adapted_vs_raw_by_latency"]))
            speedup_adapted_vs_raw_throughput.append(float(repeat_result["speedup_adapted_vs_raw_by_throughput"]))

    aggregate: Dict[str, object] = {
        "adapted_baseline": {
            "total_time_sec": _metric_stats(adapted_total_time),
            "tokens_per_sec": _metric_stats(adapted_tokens_per_sec),
            "avg_latency_sec": _metric_stats(adapted_latency),
            "tokens_generated": _metric_stats(adapted_tokens_generated),
            "token_acceptance_rate": _metric_stats(adapted_acceptance),
        },
        "multi_token": {
            "total_time_sec": _metric_stats(multi_total_time),
            "tokens_per_sec": _metric_stats(multi_tokens_per_sec),
            "avg_latency_sec": _metric_stats(multi_latency),
            "tokens_generated": _metric_stats(multi_tokens_generated),
            "token_acceptance_rate": _metric_stats(multi_acceptance),
            "draft_acceptance_rate": _metric_stats(multi_acceptance),
            "forward_calls": _metric_stats(multi_forward_calls),
            "drafted_total_tokens": _metric_stats(multi_drafted_total),
            "drafted_accepted_tokens": _metric_stats(multi_drafted_accepted),
        },
        "speedups": {
            "multi_vs_adapted_latency": _metric_stats(speedup_multi_vs_adapted_latency),
            "multi_vs_adapted_throughput": _metric_stats(speedup_multi_vs_adapted_throughput),
        },
    }
    if has_raw:
        aggregate["raw_base"] = {
            "total_time_sec": _metric_stats(raw_total_time),
            "tokens_per_sec": _metric_stats(raw_tokens_per_sec),
            "avg_latency_sec": _metric_stats(raw_latency),
            "tokens_generated": _metric_stats(raw_tokens_generated),
            "token_acceptance_rate": _metric_stats(raw_acceptance),
        }
        speedups = aggregate["speedups"]  # type: ignore[index]
        speedups["multi_vs_raw_latency"] = _metric_stats(speedup_multi_vs_raw_latency)  # type: ignore[index]
        speedups["multi_vs_raw_throughput"] = _metric_stats(speedup_multi_vs_raw_throughput)  # type: ignore[index]
        speedups["adapted_vs_raw_latency"] = _metric_stats(speedup_adapted_vs_raw_latency)  # type: ignore[index]
        speedups["adapted_vs_raw_throughput"] = _metric_stats(speedup_adapted_vs_raw_throughput)  # type: ignore[index]

    return aggregate


def _median_mode_summary(mode_aggregate: Dict[str, object], include_multi_fields: bool = False) -> Dict[str, float]:
    summary = {
        "total_time_sec": float(mode_aggregate["total_time_sec"]["median"]),  # type: ignore[index]
        "tokens_generated": float(mode_aggregate["tokens_generated"]["median"]),  # type: ignore[index]
        "tokens_per_sec": float(mode_aggregate["tokens_per_sec"]["median"]),  # type: ignore[index]
        "avg_latency_sec": float(mode_aggregate["avg_latency_sec"]["median"]),  # type: ignore[index]
        "token_acceptance_rate": float(mode_aggregate["token_acceptance_rate"]["median"]),  # type: ignore[index]
    }
    if include_multi_fields:
        summary.update(
            {
                "draft_acceptance_rate": float(mode_aggregate["draft_acceptance_rate"]["median"]),  # type: ignore[index]
                "forward_calls": float(mode_aggregate["forward_calls"]["median"]),  # type: ignore[index]
                "drafted_total_tokens": float(mode_aggregate["drafted_total_tokens"]["median"]),  # type: ignore[index]
                "drafted_accepted_tokens": float(mode_aggregate["drafted_accepted_tokens"]["median"]),  # type: ignore[index]
            }
        )
    return summary


def _print_benchmark_table(repeat_results: List[Dict[str, object]], aggregate: Dict[str, object]) -> None:
    has_raw = bool(repeat_results and "raw_base" in repeat_results[0])
    print("\nPer-repeat summary")
    if has_raw:
        print(
            "repeat | raw_tps | adap_tps | multi_tps | multi/raw_x | multi/adap_x | raw_lat(s) | adap_lat(s) | multi_lat(s) | accept"
        )
    else:
        print("repeat | adap_tps | multi_tps | multi/adap_x | adap_lat(s) | multi_lat(s) | accept")
    for repeat in repeat_results:
        adapted = repeat["adapted_baseline"]  # type: ignore[index]
        multi = repeat["multi_token"]  # type: ignore[index]
        repeat_index = int(repeat["repeat_index"])
        if has_raw:
            raw = repeat["raw_base"]  # type: ignore[index]
            print(
                f"{repeat_index:>6} | "
                f"{float(raw['tokens_per_sec']):>7.2f} | "
                f"{float(adapted['tokens_per_sec']):>8.2f} | "
                f"{float(multi['tokens_per_sec']):>9.2f} | "
                f"{float(repeat['speedup_multi_vs_raw_by_throughput']):>11.3f} | "
                f"{float(repeat['speedup_multi_vs_adapted_by_throughput']):>12.3f} | "
                f"{float(raw['avg_latency_sec']):>10.4f} | "
                f"{float(adapted['avg_latency_sec']):>11.4f} | "
                f"{float(multi['avg_latency_sec']):>12.4f} | "
                f"{float(multi['token_acceptance_rate']):>6.3f}"
            )
        else:
            print(
                f"{repeat_index:>6} | "
                f"{float(adapted['tokens_per_sec']):>8.2f} | "
                f"{float(multi['tokens_per_sec']):>9.2f} | "
                f"{float(repeat['speedup_multi_vs_adapted_by_throughput']):>12.3f} | "
                f"{float(adapted['avg_latency_sec']):>11.4f} | "
                f"{float(multi['avg_latency_sec']):>12.4f} | "
                f"{float(multi['token_acceptance_rate']):>6.3f}"
            )

    speedups = aggregate["speedups"]  # type: ignore[index]
    print("\nAggregate medians")
    print(
        "multi/adapted throughput speedup (median): "
        f"{float(speedups['multi_vs_adapted_throughput']['median']):.3f}"
    )
    print(
        "multi/adapted latency speedup (median): "
        f"{float(speedups['multi_vs_adapted_latency']['median']):.3f}"
    )
    if has_raw:
        print(
            "multi/raw throughput speedup (median): "
            f"{float(speedups['multi_vs_raw_throughput']['median']):.3f}"
        )
        print(
            "multi/raw latency speedup (median): "
            f"{float(speedups['multi_vs_raw_latency']['median']):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark raw base AR vs adapted AR vs adapted multi-token decoding."
    )
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
    parser.add_argument("--num_prompts", type=int, default=None, help="Number of measured prompts.")
    parser.add_argument("--warmup_prompts", type=int, default=None, help="Warmup prompts excluded from metrics.")
    parser.add_argument("--repeats", type=int, default=None, help="Number of measured repeats.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens per prompt.")

    fixed_group = parser.add_mutually_exclusive_group()
    fixed_group.add_argument(
        "--fixed_length",
        dest="fixed_length",
        action="store_true",
        help="Force exact max_new_tokens for timing fairness.",
    )
    fixed_group.add_argument(
        "--eos_aware",
        dest="fixed_length",
        action="store_false",
        help="Stop when EOS is generated (more realistic, less stable timing).",
    )
    parser.set_defaults(fixed_length=None)

    per_prompt_group = parser.add_mutually_exclusive_group()
    per_prompt_group.add_argument(
        "--include_per_prompt",
        dest="include_per_prompt",
        action="store_true",
        help="Include per-prompt timing rows in output JSON.",
    )
    per_prompt_group.add_argument(
        "--no_include_per_prompt",
        dest="include_per_prompt",
        action="store_false",
        help="Do not include per-prompt rows in output JSON.",
    )
    parser.set_defaults(include_per_prompt=None)

    raw_base_group = parser.add_mutually_exclusive_group()
    raw_base_group.add_argument(
        "--include_raw_base",
        dest="include_raw_base",
        action="store_true",
        help="Include original raw base-model AR benchmark (A/B/C comparison).",
    )
    raw_base_group.add_argument(
        "--no_include_raw_base",
        dest="include_raw_base",
        action="store_false",
        help="Skip raw base-model run and only compare adapted AR vs multi-token.",
    )
    parser.set_defaults(include_raw_base=None)

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
        cfg = MultiTokenMistralConfig.from_json(model_dir / "multitoken_config.json")
        artifacts_root = model_dir

    num_prompts = args.num_prompts or cfg.benchmark_num_prompts
    max_new_tokens = args.max_new_tokens or cfg.benchmark_max_new_tokens
    warmup_prompts = cfg.benchmark_warmup_prompts if args.warmup_prompts is None else args.warmup_prompts
    repeats = cfg.benchmark_repeats if args.repeats is None else args.repeats
    fixed_length = cfg.benchmark_fixed_length if args.fixed_length is None else bool(args.fixed_length)
    include_per_prompt = (
        cfg.benchmark_include_per_prompt if args.include_per_prompt is None else bool(args.include_per_prompt)
    )
    include_raw_base = (
        cfg.benchmark_include_raw_base if args.include_raw_base is None else bool(args.include_raw_base)
    )

    if num_prompts <= 0:
        raise RuntimeError(f"num_prompts must be > 0, got {num_prompts}")
    if max_new_tokens <= 0:
        raise RuntimeError(f"max_new_tokens must be > 0, got {max_new_tokens}")
    if warmup_prompts < 0:
        raise RuntimeError(f"warmup_prompts must be >= 0, got {warmup_prompts}")
    if repeats <= 0:
        raise RuntimeError(f"repeats must be > 0, got {repeats}")

    total_prompts_needed = num_prompts + warmup_prompts
    sampled_prompts = load_mbpp_prompts(num_prompts=total_prompts_needed, seed=args.seed)
    if len(sampled_prompts) <= warmup_prompts:
        raise RuntimeError(
            "Not enough prompts after warmup split. "
            f"Requested warmup={warmup_prompts}, got total={len(sampled_prompts)}"
        )

    warmup_set = sampled_prompts[:warmup_prompts]
    measured_prompts = sampled_prompts[warmup_prompts:]
    if len(measured_prompts) < num_prompts:
        print(
            "[warn] Could not sample requested measured prompt count. "
            f"Requested {num_prompts}, using {len(measured_prompts)}."
        )

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
    adapted_base_model = wrapper_model.base_model
    raw_base_model = None
    if include_raw_base:
        raw_base_model = load_raw_base_model_for_inference(cfg=cfg, hf_token=hf_token)

    warmup_summary: Dict[str, object] = {"num_prompts": len(warmup_set)}
    if warmup_set:
        print(f"Running warmup on {len(warmup_set)} prompts (excluded from metrics).")
        warmup_result = _run_prompt_set(
            prompts=warmup_set,
            wrapper_model=wrapper_model,
            adapted_base_model=adapted_base_model,
            raw_base_model=raw_base_model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            fixed_length=fixed_length,
            progress_desc="Warmup",
            include_examples=False,
            include_per_prompt=False,
        )
        warmup_summary["adapted_baseline"] = warmup_result["adapted_baseline"]
        if "raw_base" in warmup_result:
            warmup_summary["raw_base"] = warmup_result["raw_base"]
        warmup_summary["multi_token"] = warmup_result["multi_token"]

    repeat_results: List[Dict[str, object]] = []
    examples: List[Dict[str, str]] = []
    for repeat_index in range(repeats):
        repeat_result = _run_prompt_set(
            prompts=measured_prompts,
            wrapper_model=wrapper_model,
            adapted_base_model=adapted_base_model,
            raw_base_model=raw_base_model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            fixed_length=fixed_length,
            progress_desc=f"Repeat {repeat_index + 1}/{repeats}",
            include_examples=(repeat_index == 0),
            include_per_prompt=include_per_prompt,
        )
        repeat_result["repeat_index"] = repeat_index + 1
        if repeat_index == 0 and "examples" in repeat_result:
            examples = list(repeat_result["examples"])  # type: ignore[arg-type]
            del repeat_result["examples"]
        repeat_results.append(repeat_result)

    aggregate = _aggregate_repeats(repeat_results)
    adapted_median_summary = _median_mode_summary(aggregate["adapted_baseline"])  # type: ignore[arg-type]
    multi_median_summary = _median_mode_summary(aggregate["multi_token"], include_multi_fields=True)  # type: ignore[arg-type]
    raw_median_summary = (
        _median_mode_summary(aggregate["raw_base"])  # type: ignore[arg-type]
        if "raw_base" in aggregate
        else None
    )

    speedup_by_latency = float(aggregate["speedups"]["multi_vs_adapted_latency"]["median"])  # type: ignore[index]
    speedup_by_throughput = float(aggregate["speedups"]["multi_vs_adapted_throughput"]["median"])  # type: ignore[index]
    speedup_multi_vs_raw_by_latency = (
        float(aggregate["speedups"]["multi_vs_raw_latency"]["median"])  # type: ignore[index]
        if "multi_vs_raw_latency" in aggregate["speedups"]  # type: ignore[operator]
        else None
    )
    speedup_multi_vs_raw_by_throughput = (
        float(aggregate["speedups"]["multi_vs_raw_throughput"]["median"])  # type: ignore[index]
        if "multi_vs_raw_throughput" in aggregate["speedups"]  # type: ignore[operator]
        else None
    )
    speedup_adapted_vs_raw_by_latency = (
        float(aggregate["speedups"]["adapted_vs_raw_latency"]["median"])  # type: ignore[index]
        if "adapted_vs_raw_latency" in aggregate["speedups"]  # type: ignore[operator]
        else None
    )
    speedup_adapted_vs_raw_by_throughput = (
        float(aggregate["speedups"]["adapted_vs_raw_throughput"]["median"])  # type: ignore[index]
        if "adapted_vs_raw_throughput" in aggregate["speedups"]  # type: ignore[operator]
        else None
    )

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir.resolve()) if model_dir is not None else None,
        "checkpoint_dir": str(checkpoint_dir.resolve()) if checkpoint_dir is not None else None,
        "protocol": {
            "num_prompts": len(measured_prompts),
            "warmup_prompts": len(warmup_set),
            "repeats": repeats,
            "max_new_tokens": max_new_tokens,
            "fixed_length": fixed_length,
            "seed": args.seed,
            "include_per_prompt": include_per_prompt,
            "include_raw_base": include_raw_base,
        },
        "warmup": warmup_summary,
        "repeats": repeat_results,
        "aggregate": aggregate,
        # Backward-compatible summary fields.
        "num_prompts": len(measured_prompts),
        "max_new_tokens": max_new_tokens,
        "baseline": adapted_median_summary,
        "adapted_baseline": adapted_median_summary,
        "raw_base": raw_median_summary,
        "multi_token": multi_median_summary,
        "speedup_by_latency": speedup_by_latency,
        "speedup_by_throughput": speedup_by_throughput,
        "speedup_multi_vs_adapted_by_latency": speedup_by_latency,
        "speedup_multi_vs_adapted_by_throughput": speedup_by_throughput,
        "speedup_multi_vs_raw_by_latency": speedup_multi_vs_raw_by_latency,
        "speedup_multi_vs_raw_by_throughput": speedup_multi_vs_raw_by_throughput,
        "speedup_adapted_vs_raw_by_latency": speedup_adapted_vs_raw_by_latency,
        "speedup_adapted_vs_raw_by_throughput": speedup_adapted_vs_raw_by_throughput,
        "examples": examples,
    }

    output_path = Path(args.save_path) if args.save_path else artifacts_root / "benchmark_results.json"
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    _print_benchmark_table(repeat_results, aggregate)
    print(json.dumps(results, indent=2))
    print(f"Saved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
