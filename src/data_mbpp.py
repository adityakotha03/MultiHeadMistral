from __future__ import annotations

from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import MultiTokenMistralConfig


def _pick_first_non_empty(example: Dict[str, object], keys: List[str]) -> str:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_problem_and_solution(example: Dict[str, object]) -> Tuple[str, str]:
    problem = _pick_first_non_empty(example, ["text", "prompt", "problem"])
    solution = _pick_first_non_empty(example, ["code", "canonical_solution", "solution"])
    return problem, solution


def _format_training_text(problem: str, solution: str) -> str:
    return (
        "### Instruction:\n"
        "Write a correct Python function for the following task.\n"
        f"{problem}\n\n"
        "### Response:\n"
        f"{solution}"
    )


def build_mbpp_datasets(
    tokenizer: PreTrainedTokenizerBase, cfg: MultiTokenMistralConfig
) -> Tuple[Dataset, Dataset]:
    raw_train = load_dataset(cfg.dataset_name, split=cfg.train_split)

    def is_usable(example: Dict[str, object]) -> bool:
        problem, solution = _extract_problem_and_solution(example)
        return bool(problem) and bool(solution)

    filtered = raw_train.filter(is_usable, desc="Filtering empty MBPP records")

    if cfg.eval_ratio > 0:
        split = filtered.train_test_split(test_size=cfg.eval_ratio, seed=cfg.seed)
        raw_train_split = split["train"]
        raw_eval_split = split["test"]
    else:
        raw_train_split = filtered
        eval_size = min(128, len(filtered))
        raw_eval_split = filtered.select(range(eval_size))

    def tokenize_record(example: Dict[str, object]) -> Dict[str, List[int]]:
        problem, solution = _extract_problem_and_solution(example)
        text = _format_training_text(problem, solution)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_len,
            padding=False,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            if len(input_ids) >= cfg.max_seq_len:
                input_ids[-1] = eos_token_id
                attention_mask[-1] = 1
            elif input_ids[-1] != eos_token_id:
                input_ids.append(eos_token_id)
                attention_mask.append(1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    train_dataset = raw_train_split.map(
        tokenize_record,
        remove_columns=raw_train_split.column_names,
        desc="Tokenizing train split",
    )
    eval_dataset = raw_eval_split.map(
        tokenize_record,
        remove_columns=raw_eval_split.column_names,
        desc="Tokenizing eval split",
    )
    return train_dataset, eval_dataset


class CausalLMCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, object]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
