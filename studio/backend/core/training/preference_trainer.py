# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Preference Trainer integration (DPO and CPO) for Unsloth Studio.
Wraps Unsloth's PatchDPOTrainer and PatchFastRL with TRL's DPOTrainer / CPOTrainer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from loggers import get_logger

logger = get_logger(__name__)


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
