from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import MultiTokenMistralConfig
from .utils import torch_dtype_from_name

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None  # type: ignore[assignment]

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None  # type: ignore[assignment]


def build_quantization_config(cfg: MultiTokenMistralConfig) -> Optional[BitsAndBytesConfig]:
    if not cfg.use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch_dtype_from_name(cfg.bnb_4bit_compute_dtype),
    )


def load_tokenizer(model_name_or_path: str, hf_token: Optional[str] = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _single_gpu_or_cpu_device_map() -> Optional[Dict[str, int]]:
    if torch.cuda.is_available():
        return {"": 0}
    return None


def _load_auto_config(model_name_or_path: str, hf_token: Optional[str]):
    try:
        return AutoConfig.from_pretrained(
            model_name_or_path,
            token=hf_token,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"[warn] Could not inspect model config metadata: {exc}")
        return None


def _is_native_fp8_config(config) -> bool:
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


def _resolve_quantization_config(
    cfg: MultiTokenMistralConfig,
    model_name_or_path: str,
    hf_token: Optional[str],
) -> Tuple[Optional[BitsAndBytesConfig], bool, str]:
    config = _load_auto_config(model_name_or_path, hf_token)
    model_type = str(getattr(config, "model_type", "")).lower() if config is not None else ""

    if not cfg.use_4bit:
        return None, False, model_type

    if _is_native_fp8_config(config):
        print(
            "[info] Detected native FP8 quantization metadata on the base model; "
            "disabling bitsandbytes 4-bit loading for compatibility."
        )
        return None, True, model_type
    return build_quantization_config(cfg), False, model_type


def _load_model_from_family(
    model_name_or_path: str,
    hf_token: Optional[str],
    quantization_config: Optional[BitsAndBytesConfig],
    model_type: str,
):
    kwargs = {
        "token": hf_token,
        "trust_remote_code": True,
        "device_map": _single_gpu_or_cpu_device_map(),
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    if model_type == "mistral3":
        if AutoModelForImageTextToText is None:
            raise RuntimeError(
                "Model config is mistral3, but AutoModelForImageTextToText is unavailable. "
                "Upgrade transformers to a version that supports Mistral3."
            )
        return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **kwargs)

    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def _load_model_with_auto_fallback(
    model_name_or_path: str,
    hf_token: Optional[str],
    quantization_config: Optional[BitsAndBytesConfig],
    model_type: str,
):
    try:
        return _load_model_from_family(
            model_name_or_path=model_name_or_path,
            hf_token=hf_token,
            quantization_config=quantization_config,
            model_type=model_type,
        )
    except ValueError as exc:
        message = str(exc)
        # Mistral3 is currently exposed through image-text-to-text auto models.
        retry_on_image_text = (
            "Unrecognized configuration class" in message
            and "Mistral3Config" in message
            and AutoModelForImageTextToText is not None
        )
        if not retry_on_image_text:
            raise
        return AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            token=hf_token,
            trust_remote_code=True,
            device_map=_single_gpu_or_cpu_device_map(),
            **({"quantization_config": quantization_config} if quantization_config is not None else {}),
        )


def _iter_model_candidates(model):
    yield model
    candidate_attrs = (
        "model",
        "base_model",
        "language_model",
        "text_model",
        "llm",
        "decoder",
    )
    seen = {id(model)}
    for attr in candidate_attrs:
        child = getattr(model, attr, None)
        if child is not None and id(child) not in seen:
            seen.add(id(child))
            yield child


def _safe_enable_gradient_checkpointing(model) -> bool:
    for candidate in _iter_model_candidates(model):
        fn = getattr(candidate, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                fn()
                return True
            except TypeError:
                fn({})
                return True
            except Exception:
                continue
    return False


def _safe_disable_gradient_checkpointing(model) -> bool:
    for candidate in _iter_model_candidates(model):
        fn = getattr(candidate, "gradient_checkpointing_disable", None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception:
                continue
    return False


def _safe_enable_input_require_grads(model) -> bool:
    for candidate in _iter_model_candidates(model):
        fn = getattr(candidate, "enable_input_require_grads", None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception:
                continue
    return False


def _set_use_cache_flag(model, use_cache: bool) -> None:
    for candidate in _iter_model_candidates(model):
        cfg = getattr(candidate, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                cfg.use_cache = use_cache
            except Exception:
                pass


def _prepare_model_for_training_kbit(base_model):
    try:
        return prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    except TypeError:
        try:
            return prepare_model_for_kbit_training(base_model, False)
        except TypeError:
            print("[warn] Could not call prepare_model_for_kbit_training with checkpointing disabled; skipping it.")
            return base_model


def load_base_model_for_training(
    cfg: MultiTokenMistralConfig,
    hf_token: Optional[str] = None,
):
    quantization_config, fp8_detected, model_type = _resolve_quantization_config(
        cfg=cfg,
        model_name_or_path=cfg.base_model_name,
        hf_token=hf_token,
    )
    base_model = _load_model_with_auto_fallback(
        model_name_or_path=cfg.base_model_name,
        hf_token=hf_token,
        quantization_config=quantization_config,
        model_type=model_type,
    )
    _set_use_cache_flag(base_model, use_cache=False)
    if quantization_config is not None:
        base_model = _prepare_model_for_training_kbit(base_model)
    elif fp8_detected:
        print("[info] Skipping prepare_model_for_kbit_training because base model is FP8-native.")

    if not _safe_enable_gradient_checkpointing(base_model):
        print("[warn] Could not enable gradient checkpointing on base model or nested text model.")
    _safe_enable_input_require_grads(base_model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora_target_modules),
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_base_model_for_inference(
    cfg: MultiTokenMistralConfig,
    adapter_dir: str | Path,
    hf_token: Optional[str] = None,
):
    quantization_config, _, model_type = _resolve_quantization_config(
        cfg=cfg,
        model_name_or_path=cfg.base_model_name,
        hf_token=hf_token,
    )
    base_model = _load_model_with_auto_fallback(
        model_name_or_path=cfg.base_model_name,
        hf_token=hf_token,
        quantization_config=quantization_config,
        model_type=model_type,
    )
    adapted_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    _set_use_cache_flag(adapted_model, use_cache=True)
    adapted_model.eval()
    return adapted_model


class MultiTokenMistralModel(nn.Module):
    def __init__(
        self,
        base_model,
        num_future_heads: int,
        head_loss_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if num_future_heads < 1:
            raise ValueError("num_future_heads must be >= 1")

        self.base_model = base_model
        self.num_future_heads = num_future_heads

        output_embedding = self.base_model.get_output_embeddings()
        if output_embedding is None or output_embedding.weight is None:
            raise RuntimeError("Base model has no output embedding; cannot build multi-token heads.")
        vocab_size = int(output_embedding.weight.shape[0])
        hidden_size = int(output_embedding.weight.shape[1])
        self.aux_heads = nn.ModuleList(
            [nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(num_future_heads - 1)]
        )

        with torch.no_grad():
            for head in self.aux_heads:
                head.weight.copy_(output_embedding.weight.detach().to(head.weight.device, head.weight.dtype))

        weights = self._build_head_weights(head_loss_weights)
        self.register_buffer("head_loss_weights", weights, persistent=True)
        self._align_aux_heads_to_base()

    def _align_aux_heads_to_base(self) -> None:
        if not self.aux_heads:
            return
        output_embedding = self.base_model.get_output_embeddings()
        if output_embedding is None:
            return
        target_device = output_embedding.weight.device
        target_dtype = output_embedding.weight.dtype
        self.aux_heads.to(device=target_device, dtype=target_dtype)

    def _build_head_weights(self, weights: Optional[List[float]]) -> torch.Tensor:
        if weights is None:
            return torch.full((self.num_future_heads,), 1.0 / float(self.num_future_heads), dtype=torch.float32)
        if len(weights) != self.num_future_heads:
            raise ValueError(
                f"head_loss_weights length ({len(weights)}) must match num_future_heads ({self.num_future_heads})"
            )
        tensor_weights = torch.tensor(weights, dtype=torch.float32)
        weight_sum = float(tensor_weights.sum().item())
        if weight_sum <= 0:
            raise ValueError("head_loss_weights must sum to a positive value")
        return tensor_weights / weight_sum

    def _loss_for_offset(self, logits: torch.Tensor, labels: torch.Tensor, offset: int) -> torch.Tensor:
        if logits.size(1) <= offset:
            return logits.new_zeros(())
        shifted_logits = logits[:, :-offset, :].contiguous()
        shifted_labels = labels[:, offset:].contiguous()
        if shifted_labels.numel() == 0:
            return logits.new_zeros(())
        return F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        del gradient_checkpointing_kwargs
        if not _safe_enable_gradient_checkpointing(self.base_model):
            raise AttributeError("Base model does not support gradient_checkpointing_enable")

    def gradient_checkpointing_disable(self) -> None:
        if not _safe_disable_gradient_checkpointing(self.base_model):
            raise AttributeError("Base model does not support gradient_checkpointing_disable")

    def enable_input_require_grads(self) -> None:
        _safe_enable_input_require_grads(self.base_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=use_cache,
        )
        hidden_states = base_outputs.hidden_states[-1]
        logits_all_heads: List[torch.Tensor] = [base_outputs.logits]
        for head in self.aux_heads:
            logits_all_heads.append(head(hidden_states))

        result: Dict[str, torch.Tensor] = {
            "logits_head0": logits_all_heads[0],
            "logits_all_heads": logits_all_heads,  # type: ignore[assignment]
            "last_hidden_state": hidden_states,
        }
        if use_cache and hasattr(base_outputs, "past_key_values"):
            result["past_key_values"] = base_outputs.past_key_values  # type: ignore[assignment]

        if labels is not None:
            losses = [
                self._loss_for_offset(logits_all_heads[head_index], labels, offset=head_index + 1)
                for head_index in range(self.num_future_heads)
            ]
            head_losses = torch.stack(losses)
            weights = self.head_loss_weights.to(device=head_losses.device, dtype=head_losses.dtype)
            result["head_losses"] = head_losses
            result["loss"] = torch.sum(head_losses * weights)

        return result

    def save_multitoken(self, output_dir: str | Path, cfg: Optional[MultiTokenMistralConfig] = None) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        adapter_path = output_path / "adapter"
        self.base_model.save_pretrained(str(adapter_path))
        torch.save(
            {
                "num_future_heads": self.num_future_heads,
                "head_loss_weights": self.head_loss_weights.detach().cpu(),
                "aux_heads": self.aux_heads.state_dict(),
            },
            output_path / "multi_token_heads.pt",
        )
        if cfg is not None:
            cfg.save_json(output_path / "multitoken_config.json")

    def load_aux_heads(self, state_path: str | Path) -> None:
        payload = torch.load(state_path, map_location="cpu")
        if int(payload["num_future_heads"]) != self.num_future_heads:
            raise ValueError(
                f"Saved head count {payload['num_future_heads']} does not match current {self.num_future_heads}"
            )
        self.aux_heads.load_state_dict(payload["aux_heads"], strict=True)
        if "head_loss_weights" in payload:
            with torch.no_grad():
                saved = payload["head_loss_weights"].to(dtype=self.head_loss_weights.dtype)
                self.head_loss_weights.copy_(saved)
        self._align_aux_heads_to_base()


def build_training_multitoken_model(
    cfg: MultiTokenMistralConfig,
    hf_token: Optional[str] = None,
) -> MultiTokenMistralModel:
    base = load_base_model_for_training(cfg, hf_token=hf_token)
    return MultiTokenMistralModel(
        base_model=base,
        num_future_heads=cfg.num_future_heads,
        head_loss_weights=cfg.head_loss_weights,
    )


def load_inference_multitoken_model(
    model_dir: str | Path,
    cfg: Optional[MultiTokenMistralConfig] = None,
    hf_token: Optional[str] = None,
) -> Tuple[MultiTokenMistralModel, MultiTokenMistralConfig]:
    model_path = Path(model_dir)
    if cfg is None:
        cfg = MultiTokenMistralConfig.from_json(model_path / "multitoken_config.json")

    base = load_base_model_for_inference(
        cfg=cfg,
        adapter_dir=model_path / "adapter",
        hf_token=hf_token,
    )
    model = MultiTokenMistralModel(
        base_model=base,
        num_future_heads=cfg.num_future_heads,
        head_loss_weights=cfg.head_loss_weights,
    )
    model.load_aux_heads(model_path / "multi_token_heads.pt")
    _set_use_cache_flag(model.base_model, use_cache=True)
    model.eval()
    return model, cfg


def _load_checkpoint_state_dict(checkpoint_dir: str | Path) -> Dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_dir)
    pytorch_bin = checkpoint_path / "pytorch_model.bin"
    if pytorch_bin.exists():
        try:
            payload = torch.load(pytorch_bin, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint weights from {pytorch_bin}: {exc}. "
                "This checkpoint is likely incomplete/corrupted (often from interrupted save). "
                "Use an earlier checkpoint-* directory."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected a state dict at {pytorch_bin}, got {type(payload)}")
        return payload

    safetensors_file = checkpoint_path / "model.safetensors"
    if safetensors_file.exists():
        if load_safetensors_file is None:
            raise RuntimeError(
                "Found model.safetensors checkpoint but safetensors is not installed."
            )
        return load_safetensors_file(str(safetensors_file))

    raise RuntimeError(
        f"No checkpoint weights found in {checkpoint_path}. "
        "Expected pytorch_model.bin or model.safetensors."
    )


def load_multitoken_from_checkpoint(
    checkpoint_dir: str | Path,
    cfg: MultiTokenMistralConfig,
    hf_token: Optional[str] = None,
) -> MultiTokenMistralModel:
    model = build_training_multitoken_model(cfg=cfg, hf_token=hf_token)
    state_dict = _load_checkpoint_state_dict(checkpoint_dir)
    load_result = model.load_state_dict(state_dict, strict=False)

    if load_result.missing_keys:
        print(f"[warn] Missing keys while loading checkpoint: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"[warn] Unexpected keys while loading checkpoint: {len(load_result.unexpected_keys)}")

    _set_use_cache_flag(model.base_model, use_cache=True)
    model.eval()
    return model
