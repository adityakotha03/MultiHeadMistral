from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoConfig

from .config import MultiTokenMistralConfig
from .utils import ensure_dir, load_env_file, set_seed


DEFAULT_TASKS = ("arc_easy", "hellaswag", "piqa")
METRIC_PRIORITY = (
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,none",
    "em,none",
    "f1,none",
    "pass@1,create_test",
    "pass@1,none",
    "pass_at_1,extract_code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-eval harness automatically for raw base, adapted LoRA, and multi-token variants."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Final exported run directory containing adapter/ and multitoken_config.json.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path (YAML/JSON). Defaults to model_dir/multitoken_config.json.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated lm-eval tasks (default: arc_easy,hellaswag,piqa).",
    )
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot count for all tasks.")
    parser.add_argument(
        "--limit",
        type=float,
        default=100.0,
        help="Per-task example limit for quick runs. Use <=0 to disable limit.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help='lm-eval batch size, e.g. "auto", "auto:4", or "8".',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device for lm-eval, e.g. "cuda:0" or "cpu". Defaults to cuda:0 if available else cpu.',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where lm-eval artifacts are saved (default: model_dir/lm_eval).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed passed to lm-eval and local sampling logic.",
    )
    parser.add_argument(
        "--include_raw_base",
        action="store_true",
        default=True,
        help="Include original base model evaluation (default: true).",
    )
    parser.add_argument(
        "--no_include_raw_base",
        dest="include_raw_base",
        action="store_false",
        help="Skip raw base model; run adapted-only.",
    )
    parser.add_argument(
        "--include_adapted",
        action="store_true",
        default=True,
        help="Include adapted LoRA model evaluation (default: true).",
    )
    parser.add_argument(
        "--no_include_adapted",
        dest="include_adapted",
        action="store_false",
        help="Skip adapted model; run raw-only.",
    )
    parser.add_argument(
        "--include_multi_token",
        action="store_true",
        default=True,
        help="Include multi-token variant in report (default: true).",
    )
    parser.add_argument(
        "--no_include_multi_token",
        dest="include_multi_token",
        action="store_false",
        help="Skip multi-token variant.",
    )
    parser.add_argument(
        "--force_multi_token_run",
        action="store_true",
        help=(
            "Run a separate lm-eval pass for multi-token variant. "
            "By default, multi-token quality is reused from adapted_lora (same AR/head-0 behavior in lm-eval)."
        ),
    )
    parser.add_argument(
        "--confirm_run_unsafe_code",
        action="store_true",
        help="Pass through for code-execution tasks like humaneval/mbpp.",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template in lm-eval run.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Save per-sample outputs from lm-eval.",
    )
    return parser.parse_args()


def _parse_tasks(value: str) -> List[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    if not tasks:
        raise RuntimeError("No tasks parsed from --tasks.")
    return tasks


def _load_cfg(args: argparse.Namespace, model_dir: Path) -> MultiTokenMistralConfig:
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise RuntimeError(f"Config path does not exist: {cfg_path}")
        if cfg_path.suffix.lower() == ".json":
            return MultiTokenMistralConfig.from_json(cfg_path)
        return MultiTokenMistralConfig.from_yaml(cfg_path)

    default_cfg = model_dir / "multitoken_config.json"
    if default_cfg.exists():
        return MultiTokenMistralConfig.from_json(default_cfg)
    raise RuntimeError(
        f"Could not find config. Expected {default_cfg} or pass --config path/to/config.yaml."
    )


def _to_model_args_string(items: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in items.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "True" if value else "False"
        parts.append(f"{key}={value}")
    return ",".join(parts)


def _quantization_model_args(
    cfg: MultiTokenMistralConfig,
    hf_token: Optional[str],
) -> Dict[str, Any]:
    if not cfg.use_4bit:
        return {}

    config = _load_auto_config(cfg.base_model_name, hf_token=hf_token)
    if _is_native_fp8_config(config):
        print("[info] Native FP8 metadata detected; not forcing bitsandbytes 4-bit in lm-eval.")
        return {}

    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": cfg.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": cfg.bnb_4bit_use_double_quant,
        "bnb_4bit_compute_dtype": cfg.bnb_4bit_compute_dtype,
    }


def _load_auto_config(model_name_or_path: str, hf_token: Optional[str]) -> Optional[Any]:
    try:
        return AutoConfig.from_pretrained(
            model_name_or_path,
            token=hf_token,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"[warn] Could not inspect model config metadata: {exc}")
        return None


def _is_native_fp8_config(config: Optional[Any]) -> bool:
    if config is None:
        return False

    qcfg = getattr(config, "quantization_config", None)
    if qcfg is None:
        maybe_dict = config.to_dict() if hasattr(config, "to_dict") else {}
        qcfg = maybe_dict.get("quantization_config")
    if qcfg is None:
        return False

    if hasattr(qcfg, "to_dict"):
        qcfg = qcfg.to_dict()
    if not isinstance(qcfg, dict):
        return "fp8" in str(qcfg).lower()

    method = qcfg.get("quant_method") or qcfg.get("quantization_method")
    if isinstance(method, str) and "fp8" in method.lower():
        return True
    return "fp8" in json.dumps(qcfg, sort_keys=True).lower()


def _build_lm_eval_command(
    *,
    model_backend: str,
    tasks: List[str],
    model_args: str,
    output_path: Path,
    num_fewshot: int,
    batch_size: str,
    seed: int,
    trust_remote_code: bool,
    limit: Optional[float],
    device: Optional[str],
    confirm_run_unsafe_code: bool,
    apply_chat_template: bool,
    log_samples: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        model_backend,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
        "--seed",
        str(seed),
    ]
    if trust_remote_code:
        cmd.append("--trust_remote_code")
    if limit is not None and limit > 0:
        if float(limit).is_integer():
            cmd.extend(["--limit", str(int(limit))])
        else:
            cmd.extend(["--limit", str(limit)])
    if device:
        cmd.extend(["--device", device])
    if confirm_run_unsafe_code:
        cmd.append("--confirm_run_unsafe_code")
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if log_samples:
        cmd.append("--log_samples")
    return cmd


def _run_command_with_legacy_fallback(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return proc

    stderr = proc.stderr or ""
    stdout = proc.stdout or ""
    combined = f"{stdout}\n{stderr}".lower()
    if "no module named lm_eval" in combined:
        raise RuntimeError(
            "lm-eval is not installed. Install with: pip install \"lm_eval[hf]\""
        )

    # Newer lm-eval uses subcommands, but old versions use single-command CLI.
    if "unrecognized arguments: run" in combined:
        legacy_cmd = [arg for arg in cmd if arg != "run"]
        legacy_proc = subprocess.run(legacy_cmd, capture_output=True, text=True)
        if legacy_proc.returncode == 0:
            return legacy_proc
        raise RuntimeError(
            "lm-eval failed with both modern and legacy CLI formats.\n"
            f"Legacy stderr:\n{legacy_proc.stderr}\nLegacy stdout:\n{legacy_proc.stdout}"
        )

    raise RuntimeError(
        "lm-eval command failed.\n"
        f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )


def _looks_like_missing_backend_error(message: str) -> bool:
    lower = message.lower()
    return (
        "unrecognized model" in lower
        or "unknown model" in lower
        or "not in the model registry" in lower
        or "invalid choice" in lower
    )


def _run_with_model_class_fallback(
    *,
    base_cmd: List[str],
    output_path: Path,
    model_args_payload: Dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    # Try variants because lm-eval versions differ in how they resolve custom classes.
    payload_variants: List[Dict[str, Any]] = []
    payload_variants.append(dict(model_args_payload))

    dotted_variant = dict(model_args_payload)
    dotted_variant["model_class"] = "transformers.AutoModelForImageTextToText"
    payload_variants.append(dotted_variant)

    no_override_variant = dict(model_args_payload)
    no_override_variant.pop("model_class", None)
    payload_variants.append(no_override_variant)

    errors: List[str] = []
    for idx, payload in enumerate(payload_variants, start=1):
        run_cmd = list(base_cmd)
        try:
            arg_index = run_cmd.index("--model_args") + 1
        except ValueError:
            raise RuntimeError("Internal error: --model_args not found in lm-eval command.")
        run_cmd[arg_index] = _to_model_args_string(payload)

        if idx > 1:
            print(f"[warn] Retrying mistral3 lm-eval load with model_args variant #{idx}.")

        try:
            return _run_command_with_legacy_fallback(run_cmd)
        except RuntimeError as exc:
            errors.append(str(exc))

    merged_errors = "\n\n---\n\n".join(errors)
    raise RuntimeError(
        "All mistral3 lm-eval model-class loading attempts failed.\n"
        f"Tried variants in {output_path}.\n\n{merged_errors}"
    )


def _run_mistral3_with_adapter_fallbacks(
    *,
    tasks: List[str],
    output_path: Path,
    num_fewshot: int,
    batch_size: str,
    seed: int,
    trust_remote_code: bool,
    limit: Optional[float],
    device: Optional[str],
    confirm_run_unsafe_code: bool,
    apply_chat_template: bool,
    log_samples: bool,
    base_payload: Dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    # Preferred path from lm-eval PR #3487: dedicated adapter backend.
    adapter_cmd = _build_lm_eval_command(
        model_backend="hf-mistral3",
        tasks=tasks,
        model_args=_to_model_args_string(base_payload),
        output_path=output_path,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        limit=limit,
        device=device,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        apply_chat_template=apply_chat_template,
        log_samples=log_samples,
    )
    try:
        return _run_command_with_legacy_fallback(adapter_cmd)
    except RuntimeError as exc:
        msg = str(exc)
        if not _looks_like_missing_backend_error(msg):
            raise
        print(
            "[warn] lm-eval backend 'hf-mistral3' not available in this installation; "
            "falling back to 'hf' model_class strategy."
        )

    # Compatibility fallback for older lm-eval versions.
    hf_payload = dict(base_payload)
    hf_payload["model_class"] = "AutoModelForImageTextToText"
    hf_cmd = _build_lm_eval_command(
        model_backend="hf",
        tasks=tasks,
        model_args=_to_model_args_string(hf_payload),
        output_path=output_path,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        limit=limit,
        device=device,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        apply_chat_template=apply_chat_template,
        log_samples=log_samples,
    )
    return _run_with_model_class_fallback(
        base_cmd=hf_cmd,
        output_path=output_path,
        model_args_payload=hf_payload,
    )


def _find_latest_results_json(output_root: Path) -> Path:
    candidates = sorted(
        output_root.rglob("results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    fallback = sorted(
        output_root.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if fallback:
        return fallback[0]
    raise RuntimeError(f"No lm-eval JSON output found under: {output_root}")


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))


def _metric_candidates(task_metrics: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key, value in task_metrics.items():
        lower_key = str(key).lower()
        if "stderr" in lower_key:
            continue
        if _is_numeric(value):
            keys.append(str(key))
    return keys


def _pick_metric(task_metrics: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    keys = _metric_candidates(task_metrics)
    if not keys:
        return None, None

    for preferred in METRIC_PRIORITY:
        if preferred in task_metrics and _is_numeric(task_metrics[preferred]):
            return preferred, float(task_metrics[preferred])

    first_key = sorted(keys)[0]
    return first_key, float(task_metrics[first_key])


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    collected = list(values)
    if not collected:
        return None
    return float(statistics.fmean(collected))


def _build_comparison_rows(
    tasks: List[str],
    raw_results: Optional[Dict[str, Dict[str, Any]]],
    adapted_results: Optional[Dict[str, Dict[str, Any]]],
    multi_results: Optional[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        raw_task = (raw_results or {}).get(task, {})
        adapted_task = (adapted_results or {}).get(task, {})
        multi_task = (multi_results or {}).get(task, {})

        raw_metric_key, raw_metric_val = _pick_metric(raw_task) if raw_task else (None, None)
        adapted_metric_key, adapted_metric_val = _pick_metric(adapted_task) if adapted_task else (None, None)
        multi_metric_key, multi_metric_val = _pick_metric(multi_task) if multi_task else (None, None)

        aligned_metric = multi_metric_key or adapted_metric_key or raw_metric_key
        metric_names = {
            "raw": raw_metric_key,
            "adapted": adapted_metric_key,
            "multi": multi_metric_key,
        }
        non_null_metric_names = {v for v in metric_names.values() if v is not None}
        if len(non_null_metric_names) > 1:
            aligned_metric = (
                f"raw={metric_names['raw']} | adapted={metric_names['adapted']} | multi={metric_names['multi']}"
            )

        row = {
            "task": task,
            "metric": aligned_metric,
            "raw_base": raw_metric_val,
            "adapted_lora": adapted_metric_val,
            "multi_token": multi_metric_val,
            "delta_adapted_minus_raw": None,
            "delta_multi_minus_raw": None,
            "delta_multi_minus_adapted": None,
        }
        if raw_metric_val is not None and adapted_metric_val is not None:
            row["delta_adapted_minus_raw"] = float(adapted_metric_val - raw_metric_val)
        if raw_metric_val is not None and multi_metric_val is not None:
            row["delta_multi_minus_raw"] = float(multi_metric_val - raw_metric_val)
        if adapted_metric_val is not None and multi_metric_val is not None:
            row["delta_multi_minus_adapted"] = float(multi_metric_val - adapted_metric_val)
        rows.append(row)
    return rows


def _to_markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| task | metric | raw_base | adapted_lora | multi_token | "
        "delta(adapted-raw) | delta(multi-raw) | delta(multi-adapted) |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        raw_val = "-" if row["raw_base"] is None else f"{float(row['raw_base']):.4f}"
        adapted_val = "-" if row["adapted_lora"] is None else f"{float(row['adapted_lora']):.4f}"
        multi_val = "-" if row["multi_token"] is None else f"{float(row['multi_token']):.4f}"
        delta_adapted = (
            "-" if row["delta_adapted_minus_raw"] is None else f"{float(row['delta_adapted_minus_raw']):+.4f}"
        )
        delta_multi_raw = (
            "-" if row["delta_multi_minus_raw"] is None else f"{float(row['delta_multi_minus_raw']):+.4f}"
        )
        delta_multi_adapted = (
            "-"
            if row["delta_multi_minus_adapted"] is None
            else f"{float(row['delta_multi_minus_adapted']):+.4f}"
        )
        lines.append(
            f"| {row['task']} | {row['metric'] or '-'} | {raw_val} | {adapted_val} | {multi_val} | "
            f"{delta_adapted} | {delta_multi_raw} | {delta_multi_adapted} |"
        )
    return "\n".join(lines)


def _print_rows(rows: List[Dict[str, Any]]) -> None:
    print("\nTask comparison (lm-eval)")
    print(
        "task | metric | raw_base | adapted_lora | multi_token | "
        "delta(adapted-raw) | delta(multi-raw) | delta(multi-adapted)"
    )
    for row in rows:
        raw_val = "-" if row["raw_base"] is None else f"{float(row['raw_base']):.4f}"
        adapted_val = "-" if row["adapted_lora"] is None else f"{float(row['adapted_lora']):.4f}"
        multi_val = "-" if row["multi_token"] is None else f"{float(row['multi_token']):.4f}"
        delta_adapted = (
            "-" if row["delta_adapted_minus_raw"] is None else f"{float(row['delta_adapted_minus_raw']):+.4f}"
        )
        delta_multi_raw = (
            "-" if row["delta_multi_minus_raw"] is None else f"{float(row['delta_multi_minus_raw']):+.4f}"
        )
        delta_multi_adapted = (
            "-"
            if row["delta_multi_minus_adapted"] is None
            else f"{float(row['delta_multi_minus_adapted']):+.4f}"
        )
        print(
            f"{row['task']} | {row['metric'] or '-'} | {raw_val} | {adapted_val} | {multi_val} | "
            f"{delta_adapted} | {delta_multi_raw} | {delta_multi_adapted}"
        )


def main() -> None:
    args = parse_args()
    load_env_file(".env")
    set_seed(args.seed)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise RuntimeError(f"model_dir does not exist: {model_dir}")
    adapter_dir = model_dir / "adapter"
    if not adapter_dir.exists():
        raise RuntimeError(f"adapter dir not found: {adapter_dir}")

    cfg = _load_cfg(args, model_dir=model_dir)
    tasks = _parse_tasks(args.tasks)

    if not args.include_raw_base and not args.include_adapted and not args.include_multi_token:
        raise RuntimeError("Nothing to evaluate. Enable at least one of raw_base, adapted, or multi_token.")

    output_dir = Path(args.output_dir) if args.output_dir else (model_dir / "lm_eval")
    ensure_dir(output_dir)

    hf_token = os.getenv("HF_TOKEN")
    quant_args = _quantization_model_args(cfg=cfg, hf_token=hf_token)

    default_device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() or _cuda_available() else "cpu"
    device = args.device or default_device

    base_config = _load_auto_config(cfg.base_model_name, hf_token=hf_token)
    model_type = str(getattr(base_config, "model_type", "")).lower() if base_config is not None else ""

    common_base_args: Dict[str, Any] = {
        "pretrained": cfg.base_model_name,
        "dtype": "auto",
    }
    common_base_args.update(quant_args)

    run_artifacts: Dict[str, Any] = {}
    raw_results: Optional[Dict[str, Dict[str, Any]]] = None
    adapted_results: Optional[Dict[str, Dict[str, Any]]] = None
    multi_results: Optional[Dict[str, Dict[str, Any]]] = None

    if args.include_raw_base:
        raw_out = output_dir / "raw_base"
        ensure_dir(raw_out)
        raw_model_args = _to_model_args_string(common_base_args)
        print(f"[info] Running raw_base lm-eval on tasks: {','.join(tasks)}")
        if model_type == "mistral3":
            raw_proc = _run_mistral3_with_adapter_fallbacks(
                tasks=tasks,
                output_path=raw_out,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                seed=args.seed,
                trust_remote_code=True,
                limit=args.limit if args.limit > 0 else None,
                device=device,
                confirm_run_unsafe_code=args.confirm_run_unsafe_code,
                apply_chat_template=args.apply_chat_template,
                log_samples=args.log_samples,
                base_payload=dict(common_base_args),
            )
        else:
            raw_cmd = _build_lm_eval_command(
                model_backend="hf",
                tasks=tasks,
                model_args=raw_model_args,
                output_path=raw_out,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                seed=args.seed,
                trust_remote_code=True,
                limit=args.limit if args.limit > 0 else None,
                device=device,
                confirm_run_unsafe_code=args.confirm_run_unsafe_code,
                apply_chat_template=args.apply_chat_template,
                log_samples=args.log_samples,
            )
            raw_proc = _run_command_with_legacy_fallback(raw_cmd)
        raw_json = _find_latest_results_json(raw_out)
        raw_payload = json.loads(raw_json.read_text(encoding="utf-8"))
        raw_results = raw_payload.get("results", {})
        run_artifacts["raw_base"] = {
            "command": "mistral3_auto_fallback" if model_type == "mistral3" else raw_cmd,
            "output_json": str(raw_json),
            "stdout_tail": "\n".join((raw_proc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((raw_proc.stderr or "").splitlines()[-20:]),
            "task_results": raw_results,
        }

    if args.include_adapted or args.include_multi_token:
        adapted_out = output_dir / "adapted_lora"
        ensure_dir(adapted_out)
        adapted_model_args_payload = dict(common_base_args)
        adapted_model_args_payload["peft"] = str(adapter_dir.resolve())
        adapted_model_args = _to_model_args_string(adapted_model_args_payload)
        print(f"[info] Running adapted_lora lm-eval on tasks: {','.join(tasks)}")
        if model_type == "mistral3":
            adapted_proc = _run_mistral3_with_adapter_fallbacks(
                tasks=tasks,
                output_path=adapted_out,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                seed=args.seed,
                trust_remote_code=True,
                limit=args.limit if args.limit > 0 else None,
                device=device,
                confirm_run_unsafe_code=args.confirm_run_unsafe_code,
                apply_chat_template=args.apply_chat_template,
                log_samples=args.log_samples,
                base_payload=dict(adapted_model_args_payload),
            )
        else:
            adapted_cmd = _build_lm_eval_command(
                model_backend="hf",
                tasks=tasks,
                model_args=adapted_model_args,
                output_path=adapted_out,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                seed=args.seed,
                trust_remote_code=True,
                limit=args.limit if args.limit > 0 else None,
                device=device,
                confirm_run_unsafe_code=args.confirm_run_unsafe_code,
                apply_chat_template=args.apply_chat_template,
                log_samples=args.log_samples,
            )
            adapted_proc = _run_command_with_legacy_fallback(adapted_cmd)
        adapted_json = _find_latest_results_json(adapted_out)
        adapted_payload = json.loads(adapted_json.read_text(encoding="utf-8"))
        adapted_run_results = adapted_payload.get("results", {})

        if args.include_adapted:
            adapted_results = adapted_run_results
            run_artifacts["adapted_lora"] = {
                "command": "mistral3_auto_fallback" if model_type == "mistral3" else adapted_cmd,
                "output_json": str(adapted_json),
                "stdout_tail": "\n".join((adapted_proc.stdout or "").splitlines()[-20:]),
                "stderr_tail": "\n".join((adapted_proc.stderr or "").splitlines()[-20:]),
                "task_results": adapted_results,
            }

        if args.include_multi_token:
            if args.force_multi_token_run:
                multi_out = output_dir / "multi_token"
                ensure_dir(multi_out)
                print(f"[info] Running multi_token lm-eval on tasks: {','.join(tasks)}")
                if model_type == "mistral3":
                    multi_proc = _run_mistral3_with_adapter_fallbacks(
                        tasks=tasks,
                        output_path=multi_out,
                        num_fewshot=args.num_fewshot,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        trust_remote_code=True,
                        limit=args.limit if args.limit > 0 else None,
                        device=device,
                        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
                        apply_chat_template=args.apply_chat_template,
                        log_samples=args.log_samples,
                        base_payload=dict(adapted_model_args_payload),
                    )
                    multi_command: Any = "mistral3_auto_fallback"
                else:
                    multi_cmd = _build_lm_eval_command(
                        model_backend="hf",
                        tasks=tasks,
                        model_args=adapted_model_args,
                        output_path=multi_out,
                        num_fewshot=args.num_fewshot,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        trust_remote_code=True,
                        limit=args.limit if args.limit > 0 else None,
                        device=device,
                        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
                        apply_chat_template=args.apply_chat_template,
                        log_samples=args.log_samples,
                    )
                    multi_proc = _run_command_with_legacy_fallback(multi_cmd)
                    multi_command = multi_cmd

                multi_json = _find_latest_results_json(multi_out)
                multi_payload = json.loads(multi_json.read_text(encoding="utf-8"))
                multi_results = multi_payload.get("results", {})
                run_artifacts["multi_token"] = {
                    "command": multi_command,
                    "output_json": str(multi_json),
                    "stdout_tail": "\n".join((multi_proc.stdout or "").splitlines()[-20:]),
                    "stderr_tail": "\n".join((multi_proc.stderr or "").splitlines()[-20:]),
                    "task_results": multi_results,
                    "quality_mode": "explicit_multi_token_run_head0_ar",
                }
            else:
                multi_results = adapted_run_results
                run_artifacts["multi_token"] = {
                    "command": "reused_from_adapted_lora",
                    "output_json": str(adapted_json),
                    "task_results": multi_results,
                    "quality_mode": (
                        "shared_with_adapted_lora_head0_ar; "
                        "lm-eval does not execute draft-verify multi-token decoding"
                    ),
                }

    comparison_rows = _build_comparison_rows(
        tasks=tasks,
        raw_results=raw_results,
        adapted_results=adapted_results,
        multi_results=multi_results,
    )
    _print_rows(comparison_rows)

    raw_vals = [float(row["raw_base"]) for row in comparison_rows if row["raw_base"] is not None]
    adapted_vals = [float(row["adapted_lora"]) for row in comparison_rows if row["adapted_lora"] is not None]
    multi_vals = [float(row["multi_token"]) for row in comparison_rows if row["multi_token"] is not None]
    delta_vals = [
        float(row["delta_adapted_minus_raw"])
        for row in comparison_rows
        if row["delta_adapted_minus_raw"] is not None
    ]
    delta_multi_raw_vals = [
        float(row["delta_multi_minus_raw"])
        for row in comparison_rows
        if row["delta_multi_minus_raw"] is not None
    ]
    delta_multi_adapted_vals = [
        float(row["delta_multi_minus_adapted"])
        for row in comparison_rows
        if row["delta_multi_minus_adapted"] is not None
    ]

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir.resolve()),
        "config_source": args.config or str((model_dir / "multitoken_config.json").resolve()),
        "tasks": tasks,
        "protocol": {
            "num_fewshot": args.num_fewshot,
            "limit": args.limit if args.limit > 0 else None,
            "batch_size": args.batch_size,
            "device": device,
            "include_raw_base": bool(args.include_raw_base),
            "include_adapted": bool(args.include_adapted),
            "include_multi_token": bool(args.include_multi_token),
            "force_multi_token_run": bool(args.force_multi_token_run),
            "confirm_run_unsafe_code": args.confirm_run_unsafe_code,
            "apply_chat_template": args.apply_chat_template,
            "log_samples": args.log_samples,
        },
        "variants": run_artifacts,
        "comparison": comparison_rows,
        "aggregate": {
            "raw_base_mean_metric": _mean_or_none(raw_vals),
            "adapted_lora_mean_metric": _mean_or_none(adapted_vals),
            "multi_token_mean_metric": _mean_or_none(multi_vals),
            "mean_delta_adapted_minus_raw": _mean_or_none(delta_vals),
            "mean_delta_multi_minus_raw": _mean_or_none(delta_multi_raw_vals),
            "mean_delta_multi_minus_adapted": _mean_or_none(delta_multi_adapted_vals),
        },
        "notes": [
            "This script compares raw base, adapted LoRA, and multi-token quality in lm-eval.",
            "lm-eval quality uses AR/head-0 behavior; draft-verify multi-token speedups are benchmarked in src.infer_benchmark.",
        ],
    }

    summary_json_path = output_dir / "lm_eval_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table_md_path = output_dir / "lm_eval_comparison.md"
    table_md_path.write_text(_to_markdown_table(comparison_rows) + "\n", encoding="utf-8")

    print("\n[done] Saved:")
    print(f"- {summary_json_path}")
    print(f"- {table_md_path}")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
