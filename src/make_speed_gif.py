from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch
from datasets import load_dataset

from .config import MultiTokenMistralConfig
from .utils import cuda_sync, ensure_dir, load_env_file, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a side-by-side speed comparison GIF for one prompt.")
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
        help="Optional checkpoint-* path for interrupted runs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path (YAML/JSON) required when loading from checkpoint_dir.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Optional explicit full prompt text.")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help=(
            "Optional plain task/question text. "
            "It will be wrapped into a neutral instruction-response prompt template."
        ),
    )
    parser.add_argument(
        "--code_template",
        action="store_true",
        help="Wrap --question with the code-generation instruction template (old behavior).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
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
    parser.set_defaults(fixed_length=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--playback_speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. >1 makes animation faster.",
    )
    parser.add_argument("--width", type=int, default=1560)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--font_size", type=int, default=20)
    parser.add_argument("--font_path", type=str, default=None, help="Optional .ttf font path.")
    parser.add_argument(
        "--output_gif",
        type=str,
        default=None,
        help="Output GIF path (default: model_dir/speed_side_by_side.gif).",
    )
    return parser.parse_args()


def _load_cfg(args: argparse.Namespace, model_dir: Optional[Path], checkpoint_dir: Optional[Path]) -> MultiTokenMistralConfig:
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise RuntimeError(f"Config path does not exist: {cfg_path}")
        if cfg_path.suffix.lower() == ".json":
            return MultiTokenMistralConfig.from_json(cfg_path)
        return MultiTokenMistralConfig.from_yaml(cfg_path)

    if model_dir is not None:
        cfg_json = model_dir / "multitoken_config.json"
        if cfg_json.exists():
            return MultiTokenMistralConfig.from_json(cfg_json)

    if checkpoint_dir is not None:
        run_dir = checkpoint_dir.parent
        cfg_json = run_dir / "multitoken_config.json"
        if cfg_json.exists():
            return MultiTokenMistralConfig.from_json(cfg_json)

    default_yaml = Path("configs/default.yaml")
    if default_yaml.exists():
        return MultiTokenMistralConfig.from_yaml(default_yaml)
    raise RuntimeError("Could not resolve config. Pass --config path/to/config.yaml.")


def _choose_prompt(args: argparse.Namespace) -> str:
    if args.question and args.question.strip():
        question = args.question.strip()
        if args.code_template:
            return (
                "### Instruction:\n"
                "Write a correct Python function for the following task.\n"
                f"{question}\n\n"
                "### Response:\n"
            )
        return (
            "### Instruction:\n"
            f"{question}\n\n"
            "### Response:\n"
        )
    if args.prompt and args.prompt.strip():
        return args.prompt.strip()
    prompts = load_mbpp_prompts(num_prompts=1, seed=args.seed)
    return prompts[0]


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
    if len(prompts) <= num_prompts:
        return prompts[:num_prompts]

    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(len(prompts), generator=rng).tolist()
    return [prompts[i] for i in perm[:num_prompts]]


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
    import time

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
        if "min_new_tokens" in generate_kwargs:
            del generate_kwargs["min_new_tokens"]
            generate_kwargs["min_length"] = int(inputs["input_ids"].shape[1] + max_new_tokens)
        output_ids = model.generate(**inputs, **generate_kwargs)
    cuda_sync()
    elapsed = time.perf_counter() - start

    new_tokens = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
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
    import time

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
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {
        "elapsed_sec": elapsed,
        "new_tokens": len(generated_tokens),
        "text": decoded_text,
        "drafted_total": drafted_total,
        "drafted_accepted": drafted_accepted,
        "forward_calls": forward_calls,
    }


def _load_font(font_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as exc:
            print(f"[warn] Could not load font at {font_path}: {exc}. Falling back to default font.")
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if not text:
        return [""]

    paragraphs = text.replace("\r", "").replace("\t", "    ").split("\n")
    lines: List[str] = []
    for para in paragraphs:
        words = para.split(" ")
        current = ""
        for word in words:
            test = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), test, font=font)
            width = int(bbox[2] - bbox[0])
            if width <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                # Hard-wrap single long token.
                if int(draw.textbbox((0, 0), word, font=font)[2]) > max_width:
                    chunk = ""
                    for ch in word:
                        candidate = f"{chunk}{ch}"
                        if int(draw.textbbox((0, 0), candidate, font=font)[2]) <= max_width:
                            chunk = candidate
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = ch
                    current = chunk
                else:
                    current = word
        if current:
            lines.append(current)
        if para != paragraphs[-1]:
            lines.append("")
    return lines


def _clip_text(text: str, max_chars: int = 1500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _progress_color(mode_key: str) -> Tuple[int, int, int]:
    if mode_key == "raw_base":
        return (52, 152, 219)
    if mode_key == "adapted_lora":
        return (231, 76, 60)
    return (39, 174, 96)


def _create_frames(
    results: Dict[str, Dict[str, object]],
    width: int,
    height: int,
    fps: int,
    playback_speed: float,
    font: ImageFont.ImageFont,
) -> List[Image.Image]:
    title_font = font
    panel_font = font
    small_font = font

    modes = ["raw_base", "adapted_lora", "multi_token"]
    labels = {
        "raw_base": "Raw Base (AR)",
        "adapted_lora": "Adapted LoRA (AR)",
        "multi_token": "Hydra Mistral (Multi-Token)",
    }

    elapsed_values = [float(results[m]["elapsed_sec"]) for m in modes]
    max_elapsed = max(max(elapsed_values), 1e-6)
    effective_max_elapsed = max_elapsed / max(playback_speed, 1e-6)
    total_frames = max(24, int(effective_max_elapsed * fps) + 1)
    frame_ms = int(1000 / max(fps, 1))

    margin = 24
    header_h = 86
    footer_h = 22
    panel_gap = 18
    panel_w = (width - 2 * margin - 2 * panel_gap) // 3
    panel_h = height - margin - header_h - margin

    frames: List[Image.Image] = []
    for frame_idx in range(total_frames):
        t = (frame_idx / max(fps, 1)) * max(playback_speed, 1e-6)
        img = Image.new("RGB", (width, height), (14, 18, 24))
        draw = ImageDraw.Draw(img)

        title = "Hydra Mistral Speed Demo: Same Prompt, 3 Variants"
        draw.text((margin, 24), title, font=title_font, fill=(245, 247, 250))

        for i, mode in enumerate(modes):
            x = margin + i * (panel_w + panel_gap)
            y = header_h
            draw.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=12, fill=(25, 31, 42), outline=(60, 70, 88))

            elapsed = max(float(results[mode]["elapsed_sec"]), 1e-6)
            new_tokens = int(results[mode]["new_tokens"])
            tps = float(new_tokens / elapsed) if elapsed > 0 else 0.0
            progress = max(0.0, min(1.0, t / elapsed))

            full_text = _clip_text(str(results[mode]["text"]))
            reveal_chars = int(len(full_text) * progress)
            shown_text = full_text[:reveal_chars]

            label = labels[mode]
            draw.text((x + 16, y + 14), label, font=panel_font, fill=(238, 241, 245))
            stats = f"{elapsed:.2f}s | {tps:.2f} tok/s | {new_tokens} tokens"
            draw.text((x + 16, y + 44), stats, font=small_font, fill=(169, 178, 193))

            bar_x0 = x + 16
            bar_y0 = y + 76
            bar_x1 = x + panel_w - 16
            bar_y1 = bar_y0 + 10
            draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=5, fill=(40, 46, 58))
            fill_x = bar_x0 + int((bar_x1 - bar_x0) * progress)
            draw.rounded_rectangle((bar_x0, bar_y0, fill_x, bar_y1), radius=5, fill=_progress_color(mode))

            text_x = x + 16
            text_y = y + 96
            text_w = panel_w - 32
            wrapped_lines = _wrap_text(draw, shown_text, font=panel_font, max_width=text_w)

            line_bbox = draw.textbbox((0, 0), "Ag", font=panel_font)
            line_h = int((line_bbox[3] - line_bbox[1]) + 8)
            max_lines = max(1, (panel_h - 116 - footer_h) // max(line_h, 1))
            visible_lines = wrapped_lines[-max_lines:]
            for line_idx, line in enumerate(visible_lines):
                draw.text((text_x, text_y + line_idx * line_h), line, font=panel_font, fill=(220, 226, 235))

            draw.text((x + 16, y + panel_h - 20), f"Progress: {progress * 100:.0f}%", font=small_font, fill=(145, 156, 174))

        frames.append(img)

    # Hold last frame for ~1.5s for readability.
    hold_count = max(1, int(1.5 * fps))
    frames.extend([frames[-1].copy() for _ in range(hold_count)])

    # Store frame duration metadata in first frame info for save().
    frames[0].info["duration"] = frame_ms
    return frames


def main() -> None:
    args = parse_args()
    from .model_multitoken import (
        load_inference_multitoken_model,
        load_multitoken_from_checkpoint,
        load_raw_base_model_for_inference,
        load_tokenizer,
    )

    load_env_file(".env")
    set_seed(args.seed)

    if args.model_dir is None and args.checkpoint_dir is None:
        raise RuntimeError("Pass either --model_dir or --checkpoint_dir.")

    model_dir = Path(args.model_dir) if args.model_dir else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    cfg = _load_cfg(args=args, model_dir=model_dir, checkpoint_dir=checkpoint_dir)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required in .env or environment.")

    tokenizer_source = cfg.base_model_name
    if model_dir is not None and (model_dir / "tokenizer_config.json").exists():
        tokenizer_source = str(model_dir)
    tokenizer = load_tokenizer(tokenizer_source, hf_token=hf_token)

    if checkpoint_dir is not None:
        wrapper_model = load_multitoken_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            cfg=cfg,
            hf_token=hf_token,
        )
    else:
        assert model_dir is not None
        wrapper_model, _ = load_inference_multitoken_model(
            model_dir=model_dir,
            cfg=cfg,
            hf_token=hf_token,
        )

    adapted_model = wrapper_model.base_model
    raw_model = load_raw_base_model_for_inference(cfg=cfg, hf_token=hf_token)

    prompt = _choose_prompt(args)
    print("[info] Running one-sample generation for all three variants...")

    raw_result = run_baseline_generate(
        raw_model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        fixed_length=args.fixed_length,
    )
    adapted_result = run_baseline_generate(
        adapted_model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        fixed_length=args.fixed_length,
    )
    multi_result = draft_verify_decode(
        wrapper_model=wrapper_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        fixed_length=args.fixed_length,
    )

    results = {
        "raw_base": raw_result,
        "adapted_lora": adapted_result,
        "multi_token": multi_result,
    }

    font = _load_font(args.font_path, args.font_size)
    frames = _create_frames(
        results=results,
        width=args.width,
        height=args.height,
        fps=max(args.fps, 1),
        playback_speed=max(args.playback_speed, 1e-6),
        font=font,
    )

    output_path = Path(args.output_gif) if args.output_gif else (
        (model_dir / "speed_side_by_side.gif") if model_dir is not None else (checkpoint_dir / "speed_side_by_side.gif")
    )
    ensure_dir(output_path.parent)

    frame_duration_ms = int(1000 / max(args.fps, 1))
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=frame_duration_ms,
        optimize=False,
    )

    print(f"[done] GIF saved: {output_path}")
    print(
        "[summary] "
        f"raw={float(raw_result['elapsed_sec']):.2f}s, "
        f"adapted={float(adapted_result['elapsed_sec']):.2f}s, "
        f"multi={float(multi_result['elapsed_sec']):.2f}s"
    )


if __name__ == "__main__":
    main()
