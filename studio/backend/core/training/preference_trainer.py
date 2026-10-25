# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DPO / ORPO / CPO trainers for Studio.

TRL >= 0.19 moved ``beta``, ``max_length`` and ``max_prompt_length`` off the trainer kwargs onto
the config object and renamed ``tokenizer=`` to ``processing_class=``. This module therefore builds
the preference config explicitly instead of forwarding the SFT config, which also carries keys no
preference config accepts.

``PatchFastRL`` has to run *before* the trainer class is imported, otherwise TRL resolves its own
unpatched class. The metrics (rewards/*, nll_loss, log_odds_ratio) that Studio already streams come
from TRL's own logging, so nothing else needs wiring here.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from loggers import get_logger

logger = get_logger(__name__)

PREFERENCE_METHODS = ("DPO", "ORPO", "CPO")

# SFTConfig takes these; the preference configs do not. Anything else is filtered against the
# target config's own dataclass fields, because TRL keeps moving knobs between releases.
_SFT_ONLY_CONFIG_KEYS = frozenset(
    {
        "dataset_text_field",
        "packing",
        "eval_packing",
        "max_seq_length",
        "completion_only_loss",
        "assistant_only_loss",
    }
)

# Rendered through the chat template to tell an already-formatted prompt from a raw one.
_CHAT_TEMPLATE_PROBE = "unsloth-preference-template-probe"


def normalize_preference_method(method: Any) -> Optional[str]:
    """Return the canonical method name, or None when this is not a preference method."""
    normalized = str(method or "").strip().upper()
    return normalized if normalized in PREFERENCE_METHODS else None


def preference_dataset_mismatch(method: Any, dataset: Any) -> Optional[str]:
    """Reject a preference method paired with a dataset that has no chosen/rejected columns.

    DPO/ORPO/CPO read chosen/rejected pairs. An SFT dataset would fail late inside TRL, so refuse
    it up front and let the UI report a dataset mismatch instead of a trainer traceback.
    """
    normalized = normalize_preference_method(method)
    if normalized is None:
        return None

    # The text paths hand the trainer a {"dataset": ..., "detected_format": ...} wrapper.
    if isinstance(dataset, dict) and "dataset" in dataset:
        dataset = dataset["dataset"]

    from utils.datasets.preference import is_preference_dataset

    if is_preference_dataset(dataset):
        return None

    return (
        f"Dataset mismatch: {normalized} training needs a preference dataset with chosen and "
        "rejected columns (for example prompt/chosen/rejected). Choose a preference dataset "
        "or switch Preference mode to SFT."
    )


def _tokenizer_of(processing_class: Any) -> Any:
    """VLM runs carry a Processor; the chat template lives on its tokenizer."""
    return getattr(processing_class, "tokenizer", processing_class)


def _chat_template(tokenizer: Any) -> Optional[str]:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = next(iter(template.values()), None)
    if isinstance(template, str) and template.strip():
        return template
    return None


def _looks_templated(prompt: str, tokenizer: Any) -> bool:
    """True when the prompt already carries the model's rendered chat template.

    Preference datasets on the Hub are often pre-formatted; applying the template twice inserts
    duplicate role markers and silently degrades the run. When the template cannot be rendered we
    treat the row as already formatted and leave the dataset untouched.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": _CHAT_TEMPLATE_PROBE}],
            tokenize = False,
            add_generation_prompt = True,
        )
    except Exception as exc:  # noqa: BLE001 - an unusable template must not fail the run
        logger.info(f"Could not render the chat template for the preference dataset: {exc}\n")
        return True
    if not isinstance(rendered, str):
        return True
    prefix = rendered.split(_CHAT_TEMPLATE_PROBE)[0]
    return bool(prefix.strip()) and prefix in prompt


def _to_messages(row: Dict[str, Any]) -> Dict[str, Any]:
    """Render one string preference row as chat messages so TRL applies the chat template."""
    prompt, chosen, rejected = row.get("prompt"), row.get("chosen"), row.get("rejected")
    if not (
        isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)
    ):
        return dict(row)

    prompt_text = prompt
    messages = []
    system = row.get("system")
    if isinstance(system, str) and system.strip():
        system_text = system.strip()
        messages.append({"role": "system", "content": system_text})
        prefix = f"{system_text}\n\n"
        if prompt_text.startswith(prefix):
            prompt_text = prompt_text[len(prefix) :]
    messages.append({"role": "user", "content": prompt_text.strip()})

    updated = dict(row)
    updated["prompt"] = messages
    updated["chosen"] = [{"role": "assistant", "content": chosen.strip()}]
    updated["rejected"] = [{"role": "assistant", "content": rejected.strip()}]
    return updated


def prepare_preference_dataset(
    dataset: Any,
    processing_class: Any,
    num_proc: Optional[int] = None,
) -> Any:
    """Convert plain prompt/chosen/rejected rows into messages for TRL's chat-template pass.

    TRL applies a chat template only to conversational rows (`is_conversational`), so a string
    dataset trains on raw text the model never sees at inference. Rows that are already
    conversational, already templated, or datasets whose tokenizer has no chat template (base
    models, raw text) are left exactly as they are.
    """
    tokenizer = _tokenizer_of(processing_class)
    if tokenizer is None or _chat_template(tokenizer) is None or not hasattr(dataset, "map"):
        return dataset

    try:
        first = dataset[0]
    except Exception as exc:  # noqa: BLE001 - streaming/empty datasets keep their raw form
        logger.info(f"Preference dataset left unconverted: {exc}\n")
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
            _to_messages,
            num_proc = num_proc,
            desc = "Applying the chat template to preference data",
        )
    except TypeError:
        return dataset.map(_to_messages)
    except Exception as exc:  # noqa: BLE001 - never lose a run over the conversion
        logger.warning(f"Could not convert the preference dataset to messages: {exc}\n")
        return dataset


def _ensure_pad_token(processing_class: Any) -> Any:
    """TRL raises at init when the tokenizer has no pad token; eos is the usual stand-in."""
    tokenizer = _tokenizer_of(processing_class)
    if tokenizer is None or getattr(tokenizer, "pad_token_id", None) is not None:
        return processing_class
    eos = getattr(tokenizer, "eos_token", None)
    if eos is None:
        return processing_class
    try:
        tokenizer.pad_token = eos
        logger.info("Preference training: pad_token unset, reusing eos_token\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Preference training: could not set pad_token: {exc}\n")
    return processing_class


def _config_kwargs(config_cls: Any, config_args: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keys the preference config declares."""
    fields = getattr(config_cls, "__dataclass_fields__", None)
    if fields:
        return {
            key: value
            for key, value in config_args.items()
            if key in fields and getattr(fields[key], "init", True)
        }
    return {
        key: value for key, value in config_args.items() if key not in _SFT_ONLY_CONFIG_KEYS
    }


def _trainer_tokenizer_kwarg(trainer_cls: Any) -> str:
    """Preference trainers take `processing_class`; TRL < 0.19 only knows `tokenizer`."""
    try:
        parameters = inspect.signature(trainer_cls.__init__).parameters
    except (TypeError, ValueError):
        return "processing_class"
    if "processing_class" in parameters or "tokenizer" not in parameters:
        return "processing_class"
    return "tokenizer"


def _apply_unsloth_patch(method: str) -> None:
    """Patch TRL before its trainer class is imported, or the unpatched class wins."""
    try:
        from unsloth import PatchFastRL
    except Exception as exc:  # noqa: BLE001 - plain TRL still trains, just slower
        logger.warning(f"Unsloth PatchFastRL for {method} unavailable: {exc}\n")
        return
    try:
        PatchFastRL(method)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Unsloth PatchFastRL for {method} failed: {exc}\n")


def _import_trainer_classes(method: str) -> Any:
    """Import the TRL trainer/config pair, following ORPO/CPO's move to trl.experimental."""
    if method == "DPO":
        from trl import DPOConfig, DPOTrainer

        return DPOTrainer, DPOConfig
    if method == "ORPO":
        try:
            from trl import ORPOConfig, ORPOTrainer
        except ImportError:
            from trl.experimental.orpo import ORPOConfig, ORPOTrainer
        return ORPOTrainer, ORPOConfig
    try:
        from trl import CPOConfig, CPOTrainer
    except ImportError:
        from trl.experimental.cpo import CPOConfig, CPOTrainer
    return CPOTrainer, CPOConfig


def _is_peft_model(model: Any) -> bool:
    """True for a LoRA/QLoRA run, where TRL can disable the adapter instead of copying the model."""
    try:
        from peft import PeftModel
    except Exception:  # noqa: BLE001 - no peft means no adapter to disable
        return False
    return isinstance(model, PeftModel)


def _clamp_max_length(model: Any, max_length: Optional[int]) -> int:
    """A max_length above the model's context crashes TRL's logprob gather."""
    requested = int(max_length or 2048)
    context = getattr(model, "max_seq_length", None)
    if isinstance(context, int) and context > 0:
        return min(requested, context)
    return requested


def setup_preference_trainer(
    method: Any,
    model: Any,
    processing_class: Any,
    train_dataset: Any,
    eval_dataset: Any = None,
    config_args: Optional[Dict[str, Any]] = None,
    beta: float = 0.1,
    cpo_alpha: float = 1.0,
    max_prompt_length: Optional[int] = 512,
    max_length: Optional[int] = 2048,
) -> Any:
    """Build the DPO/ORPO/CPO trainer for a Studio run.

    `config_args` is Studio's shared SFT config dict; only the keys the preference config knows are
    forwarded, so SFT-only knobs never reach TRL.
    """
    normalized = normalize_preference_method(method)
    if normalized is None:
        raise ValueError(
            f"Unknown preference training method {method!r}; expected one of "
            f"{', '.join(PREFERENCE_METHODS)}."
        )

    mismatch = preference_dataset_mismatch(normalized, train_dataset)
    if mismatch is not None:
        raise ValueError(mismatch)
    if eval_dataset is not None and preference_dataset_mismatch(normalized, eval_dataset):
        raise ValueError(
            f"Eval dataset mismatch: {normalized} training needs an eval dataset with chosen "
            "and rejected columns (for example prompt/chosen/rejected)."
        )

    _apply_unsloth_patch(normalized)
    trainer_cls, config_cls = _import_trainer_classes(normalized)

    processing_class = _ensure_pad_token(processing_class)
    num_proc = (config_args or {}).get("dataset_num_proc")
    train_dataset = prepare_preference_dataset(train_dataset, processing_class, num_proc)
    if eval_dataset is not None:
        eval_dataset = prepare_preference_dataset(eval_dataset, processing_class, num_proc)

    max_length = _clamp_max_length(model, max_length)
    prompt_length = int(max_prompt_length or max_length // 2)
    prompt_length = max(1, min(prompt_length, max(1, max_length // 2)))

    config_kwargs = _config_kwargs(config_cls, config_args or {})
    config_kwargs.update(
        {
            "beta": float(beta),
            "max_length": max_length,
            "max_prompt_length": prompt_length,
        }
    )
    if normalized == "CPO":
        config_kwargs["cpo_alpha"] = float(cpo_alpha)
    if normalized in ("ORPO", "CPO"):
        # Their default DPODataCollatorWithPadding reads the prompt/chosen/rejected columns, so
        # TRL sets this itself and warns on every run. DPO keeps Trainer's signature columns.
        config_kwargs["remove_unused_columns"] = False
    # HF's token counter reads one `main_input_name` tensor, but a preference batch carries the
    # prompt and both completions separately, so it warned on every step and never counted.
    config_kwargs["include_num_input_tokens_seen"] = False
    dpo_peft = normalized != "DPO" or _is_peft_model(model)
    if normalized == "DPO" and not dpo_peft:
        # A full finetune has no adapter to disable, and TRL would silently deep-copy the whole
        # model as the reference (2x VRAM). Precomputing the reference logprobs keeps one model.
        if not hasattr(train_dataset, "add_column"):
            raise ValueError(
                "DPO full finetuning needs a map-style dataset to precompute the reference "
                "logprobs. Disable dataset streaming, or train with LoRA/QLoRA."
            )
        config_kwargs["precompute_ref_log_probs"] = True
        logger.info(
            "DPO full finetune: precomputing reference logprobs instead of copying the model\n"
        )
    config = config_cls(**config_kwargs)

    logger.info(
        f"Initializing {normalized}Trainer with beta={float(beta)}, "
        f"max_length={max_length}, max_prompt_length={prompt_length}"
        + (f", cpo_alpha={float(cpo_alpha)}" if normalized == "CPO" else "")
        + "\n"
    )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": config,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if normalized == "DPO":
        # TRL scores the reference by disabling the adapter, or from precomputed logprobs.
        trainer_kwargs["ref_model"] = None
    trainer_kwargs[_trainer_tokenizer_kwarg(trainer_cls)] = processing_class
    return trainer_cls(**trainer_kwargs)
