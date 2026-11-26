# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GRPO trainer setup for Studio.

Calls ``PatchFastRL("GRPO")`` *before* importing ``GRPOTrainer`` / ``GRPOConfig`` from TRL so
Unsloth's memory-efficient GRPO patches (including Gemma 4 gradient checkpointing & KL stability)
are active. Supports both conversational prompts (``[{"role": "user", "content": "..."}]``) and
plain-text prompts paired with verifiable target columns such as ``answer``.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Sequence

from loggers import get_logger

from core.training.grpo_rewards import (
    DEFAULT_GRPO_REWARD_FUNCS,
    resolve_grpo_reward_funcs,
)

logger = get_logger(__name__)

GRPO_METHODS = ("GRPO",)
GRPO_LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo")

_SFT_ONLY_CONFIG_KEYS = frozenset(
    {
        "dataset_text_field",
        "packing",
        "eval_packing",
        "max_seq_length",
        "completion_only_loss",
        "assistant_only_loss",
        "dataset_num_proc",
    }
)

_CHAT_TEMPLATE_PROBE = "unsloth-grpo-template-probe"


def normalize_grpo_method(method: Any) -> Optional[str]:
    """Return 'GRPO' when the requested training method is GRPO, else None."""
    normalized = str(method or "").strip().upper()
    return normalized if normalized in GRPO_METHODS else None


def is_grpo_method(method: Any) -> bool:
    return normalize_grpo_method(method) is not None


def normalize_grpo_loss_type(loss_type: Any) -> str:
    normalized = str(loss_type or "").strip().lower() or "grpo"
    if normalized not in GRPO_LOSS_TYPES:
        raise ValueError(
            f"Unknown GRPO loss type {loss_type!r}; expected one of {', '.join(GRPO_LOSS_TYPES)}."
        )
    return normalized


def grpo_dataset_mismatch(method: Any, dataset: Any) -> Optional[str]:
    """Reject a GRPO run paired with a dataset that has no valid prompt column."""
    if not is_grpo_method(method):
        return None
    if isinstance(dataset, dict) and "dataset" in dataset:
        dataset = dataset["dataset"]

    from utils.datasets.grpo import is_grpo_dataset

    if is_grpo_dataset(dataset):
        return None

    return (
        "Dataset mismatch: GRPO training needs a dataset with a `prompt` (or `question`/`problem`) "
        "column, optionally paired with a verifiable `answer` column. Choose a GRPO dataset or "
        "switch the training objective to SFT."
    )


def _tokenizer_of(processing_class: Any) -> Any:
    return getattr(processing_class, "tokenizer", processing_class)


def _chat_template(tokenizer: Any) -> Optional[str]:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = next(iter(template.values()), None)
    if isinstance(template, str) and template.strip():
        return template
    return None


def _looks_templated(prompt: str, tokenizer: Any) -> bool:
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": _CHAT_TEMPLATE_PROBE}],
            tokenize = False,
            add_generation_prompt = True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info(f"Could not render the chat template for the GRPO dataset: {exc}\n")
        return True
    if not isinstance(rendered, str):
        return True
    prefix = rendered.split(_CHAT_TEMPLATE_PROBE)[0]
    return bool(prefix.strip()) and prefix in prompt


def _prompt_string_to_messages(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        return dict(row)
    messages = []
    system = row.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    updated = dict(row)
    updated["prompt"] = messages
    return updated


def prepare_grpo_dataset(
    dataset: Any,
    processing_class: Any,
    num_proc: Optional[int] = None,
) -> Any:
    """Keep conversational prompts untouched; wrap plain string prompts as chat messages when templated."""
    tokenizer = _tokenizer_of(processing_class)
    if tokenizer is None or _chat_template(tokenizer) is None or not hasattr(dataset, "map"):
        return dataset

    try:
        first = dataset[0]
    except Exception as exc:  # noqa: BLE001
        logger.info(f"GRPO dataset left unconverted: {exc}\n")
        return dataset
    if not isinstance(first, dict):
        return dataset
    if isinstance(first.get("prompt"), (list, tuple)):
        return dataset
    prompt = first.get("prompt")
    if not isinstance(prompt, str) or _looks_templated(prompt, tokenizer):
        return dataset

    try:
        return dataset.map(
            _prompt_string_to_messages,
            num_proc = num_proc,
            desc = "Applying the chat template to GRPO prompts",
        )
    except TypeError:
        return dataset.map(_prompt_string_to_messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not convert GRPO string prompts to messages: {exc}\n")
        return dataset


def _ensure_pad_token_and_left_padding(processing_class: Any) -> Any:
    tokenizer = _tokenizer_of(processing_class)
    if tokenizer is None:
        return processing_class
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            try:
                tokenizer.pad_token = eos
                logger.info("GRPO training: pad_token unset, reusing eos_token\n")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"GRPO training: could not set pad_token: {exc}\n")
    try:
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
    except Exception:  # noqa: BLE001
        pass
    return processing_class


def _config_kwargs(config_cls: Any, config_args: Dict[str, Any]) -> Dict[str, Any]:
    fields = getattr(config_cls, "__dataclass_fields__", None)
    if fields:
        return {
            key: value
            for key, value in config_args.items()
            if key in fields and getattr(fields[key], "init", True)
        }
    return {key: value for key, value in config_args.items() if key not in _SFT_ONLY_CONFIG_KEYS}


def _trainer_tokenizer_kwarg(trainer_cls: Any) -> str:
    try:
        parameters = inspect.signature(trainer_cls.__init__).parameters
    except (TypeError, ValueError):
        return "processing_class"
    if "processing_class" in parameters or "tokenizer" not in parameters:
        return "processing_class"
    return "tokenizer"


def _apply_unsloth_patch() -> None:
    try:
        from unsloth import PatchFastRL
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Unsloth PatchFastRL for GRPO unavailable: {exc}\n")
        return
    try:
        PatchFastRL("GRPO")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Unsloth PatchFastRL for GRPO failed: {exc}\n")


def _import_grpo_classes() -> Any:
    from trl import GRPOConfig, GRPOTrainer

    return GRPOTrainer, GRPOConfig


def _clamp_grpo_lengths(
    model: Any,
    max_seq_length: Optional[int],
    max_prompt_length: Optional[int],
    max_completion_length: Optional[int],
) -> tuple[int, int]:
    context = getattr(model, "max_seq_length", None)
    budget = int(max_seq_length or 2048)
    if isinstance(context, int) and context > 0:
        budget = min(budget, context)
    budget = max(2, budget)

    prompt_len = int(max_prompt_length or min(512, budget // 2))
    prompt_len = max(1, min(prompt_len, max(1, budget - 1)))

    remaining = max(1, budget - prompt_len)
    completion_len = int(max_completion_length or min(512, remaining))
    completion_len = max(1, min(completion_len, remaining))
    return prompt_len, completion_len


def setup_grpo_trainer(
    model: Any,
    processing_class: Any,
    train_dataset: Any,
    eval_dataset: Any = None,
    config_args: Optional[Dict[str, Any]] = None,
    num_generations: int = 4,
    max_prompt_length: Optional[int] = 512,
    max_completion_length: Optional[int] = 512,
    max_seq_length: Optional[int] = 2048,
    beta: float = 0.04,
    temperature: float = 0.9,
    epsilon: float = 0.2,
    num_iterations: int = 1,
    loss_type: Any = "grpo",
    reward_funcs: Optional[Sequence[str]] = None,
    reward_weights: Optional[Sequence[float]] = None,
    use_vllm: bool = False,
) -> Any:
    """Build a TRL GRPOTrainer for a Studio run."""
    mismatch = grpo_dataset_mismatch("GRPO", train_dataset)
    if mismatch is not None:
        raise ValueError(mismatch)
    if eval_dataset is not None and grpo_dataset_mismatch("GRPO", eval_dataset):
        raise ValueError(
            "Eval dataset mismatch: GRPO training needs an eval dataset with a `prompt` "
            "(or `question`/`problem`) column."
        )

    normalized_loss = normalize_grpo_loss_type(loss_type)
    resolved_rewards, resolved_weights = resolve_grpo_reward_funcs(
        reward_funcs or DEFAULT_GRPO_REWARD_FUNCS,
        reward_weights = reward_weights,
    )

    _apply_unsloth_patch()
    trainer_cls, config_cls = _import_grpo_classes()

    processing_class = _ensure_pad_token_and_left_padding(processing_class)
    num_proc = (config_args or {}).get("dataset_num_proc")
    train_dataset = prepare_grpo_dataset(train_dataset, processing_class, num_proc)
    if eval_dataset is not None:
        eval_dataset = prepare_grpo_dataset(eval_dataset, processing_class, num_proc)

    prompt_len, completion_len = _clamp_grpo_lengths(
        model,
        max_seq_length = max_seq_length,
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
    )

    grpo_fields = getattr(config_cls, "__dataclass_fields__", None) or {}
    config_kwargs = _config_kwargs(config_cls, config_args or {})
    grpo_overrides: Dict[str, Any] = {
        "num_generations": max(2, int(num_generations)),
        "max_prompt_length": prompt_len,
        "max_completion_length": completion_len,
        "beta": float(beta),
        "temperature": float(temperature),
        "epsilon": float(epsilon),
        "num_iterations": max(1, int(num_iterations)),
        "loss_type": normalized_loss,
        "use_vllm": bool(use_vllm),
        "remove_unused_columns": False,
    }
    if resolved_weights is not None:
        grpo_overrides["reward_weights"] = resolved_weights

    for key, value in grpo_overrides.items():
        if not grpo_fields or key in grpo_fields:
            config_kwargs[key] = value

    num_gen = int(config_kwargs.get("num_generations", max(2, int(num_generations))))
    per_device_bs = int(config_kwargs.get("per_device_train_batch_size", 1) or 1)
    if (
        not grpo_fields or "generation_batch_size" in grpo_fields
    ) and "generation_batch_size" not in config_kwargs:
        if per_device_bs % num_gen != 0:
            config_kwargs["generation_batch_size"] = num_gen

    config = config_cls(**config_kwargs)

    logger.info(
        f"Initializing GRPOTrainer with num_generations={num_gen}, "
        f"max_prompt_length={prompt_len}, max_completion_length={completion_len}, "
        f"beta={float(beta)}, temperature={float(temperature)}, loss_type={normalized_loss}, "
        f"rewards={[fn.__name__ for fn in resolved_rewards]}\n"
    )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "reward_funcs": resolved_rewards,
        "args": config,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer_kwargs[_trainer_tokenizer_kwarg(trainer_cls)] = processing_class
    return trainer_cls(**trainer_kwargs)