# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Preference Trainer integration (DPO, ORPO and CPO) for Unsloth Studio.
Wraps Unsloth's PatchDPOTrainer and PatchFastRL with TRL's DPOTrainer / ORPOTrainer / CPOTrainer.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional
from loggers import get_logger

logger = get_logger(__name__)


def preference_dataset_mismatch(training_method: Any, dataset: Any) -> Optional[str]:
    """Reject a preference method paired with a dataset that has no chosen/rejected columns.

    DPO/ORPO/CPO read `chosen`/`rejected` pairs; an SFT dataset would either fail late inside
    TRL or, on the format fallback path, train the SFT objective instead. Refuse it up front
    so the UI can report a dataset mismatch instead of a confusing trainer error.
    """
    method = str(training_method or "").strip().upper()
    if method not in ("DPO", "ORPO", "CPO"):
        return None

    # The text paths hand the trainer a {"dataset": ..., "detected_format": ...} wrapper.
    if isinstance(dataset, dict) and "dataset" in dataset:
        dataset = dataset["dataset"]

    from utils.datasets.preference import is_preference_dataset

    if is_preference_dataset(dataset):
        return None

    return (
        f"Dataset mismatch: {method} training needs a preference dataset with chosen and "
        "rejected columns (for example prompt/chosen/rejected). Choose a preference dataset "
        "or switch Preference mode to SFT."
    )


def _import_orpo_trainer() -> Any:
    """TRL 0.29+ moved ORPO to trl.experimental.orpo; older releases keep it at the top level."""
    try:
        from trl import ORPOConfig, ORPOTrainer
    except ImportError:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer
    return ORPOTrainer, ORPOConfig


def _set_tokenizer_argument(
    trainer_kwargs: Dict[str, Any], trainer_cls: Any, tokenizer: Any
) -> None:
    """TRL renamed the tokenizer argument to `processing_class` in 0.19; pass whichever exists."""
    parameters = inspect.signature(trainer_cls.__init__).parameters
    key = "tokenizer" if "tokenizer" in parameters else "processing_class"
    trainer_kwargs[key] = tokenizer


def setup_dpo_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any],
    config_args: Dict[str, Any],
    dpo_beta: float = 0.1,
    max_prompt_length: Optional[int] = 512,
    max_length: Optional[int] = 2048,
) -> Any:
    """
    Sets up and initializes a DPOTrainer with Unsloth's memory-optimized PatchDPOTrainer.
    """
    try:
        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()
    except Exception as exc:
        logger.warning(f"Unsloth PatchDPOTrainer call encountered: {exc}. Proceeding with standard TRL.")

    from trl import DPOTrainer, DPOConfig

    # Filter out SFT-only or conflicting kwargs if present
    dpo_kwargs = dict(config_args)
    dpo_kwargs.pop("dataset_text_field", None)
    dpo_kwargs.pop("packing", None)
    dpo_kwargs.pop("max_seq_length", None)
    dpo_kwargs.pop("dataset_num_proc", None)

    dpo_config = DPOConfig(
        beta=dpo_beta,
        max_length=max_length or 2048,
        max_prompt_length=max_prompt_length or 512,
        **dpo_kwargs,
    )

    logger.info(f"Initializing DPOTrainer with beta={dpo_beta}, max_prompt_length={max_prompt_length}...")
    trainer_kwargs = {
        "model": model,
        "ref_model": None,  # Unsloth handles ref model internally to save VRAM
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "args": dpo_config,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    return DPOTrainer(**trainer_kwargs)


def setup_cpo_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any],
    config_args: Dict[str, Any],
    dpo_beta: float = 0.1,
    cpo_alpha: float = 1.0,
    max_prompt_length: Optional[int] = 512,
    max_length: Optional[int] = 2048,
) -> Any:
    """
    Sets up and initializes a CPOTrainer with Unsloth's PatchFastRL.
    """
    try:
        from unsloth import PatchFastRL
        PatchFastRL("CPO")
    except Exception as exc:
        logger.warning(f"Unsloth PatchFastRL call encountered: {exc}. Proceeding with standard TRL.")

    from trl import CPOTrainer, CPOConfig

    # Filter out SFT-only or conflicting kwargs if present
    cpo_kwargs = dict(config_args)
    cpo_kwargs.pop("dataset_text_field", None)
    cpo_kwargs.pop("packing", None)
    cpo_kwargs.pop("max_seq_length", None)
    cpo_kwargs.pop("dataset_num_proc", None)

    cpo_config = CPOConfig(
        beta=dpo_beta,
        cpo_alpha=cpo_alpha,
        loss_type="sigmoid",
        max_length=max_length or 2048,
        max_prompt_length=max_prompt_length or 512,
        **cpo_kwargs,
    )

    logger.info(f"Initializing CPOTrainer with beta={dpo_beta}, cpo_alpha={cpo_alpha}...")
    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "args": cpo_config,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    return CPOTrainer(**trainer_kwargs)


def setup_orpo_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any],
    config_args: Dict[str, Any],
    dpo_beta: float = 0.1,
    max_prompt_length: Optional[int] = 512,
    max_length: Optional[int] = 2048,
) -> Any:
    """
    Sets up and initializes an ORPOTrainer with Unsloth's PatchFastRL.

    ORPO is monolithic: it drops the reference model, folding the SFT objective and the
    odds-ratio preference term into one loss, so no ref_model is passed.
    """
    try:
        from unsloth import PatchFastRL
        PatchFastRL("ORPO")
    except Exception as exc:
        logger.warning(f"Unsloth PatchFastRL call encountered: {exc}. Proceeding with standard TRL.")

    ORPOTrainer, ORPOConfig = _import_orpo_trainer()

    # Filter out SFT-only or conflicting kwargs if present
    orpo_kwargs = dict(config_args)
    orpo_kwargs.pop("dataset_text_field", None)
    orpo_kwargs.pop("packing", None)
    orpo_kwargs.pop("max_seq_length", None)
    orpo_kwargs.pop("dataset_num_proc", None)

    orpo_config = ORPOConfig(
        beta=dpo_beta,
        max_length=max_length or 2048,
        max_prompt_length=max_prompt_length or 512,
        **orpo_kwargs,
    )

    logger.info(
        f"Initializing ORPOTrainer with beta={dpo_beta}, "
        f"max_prompt_length={max_prompt_length}..."
    )
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": orpo_config,
    }
    _set_tokenizer_argument(trainer_kwargs, ORPOTrainer, tokenizer)
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    return ORPOTrainer(**trainer_kwargs)
