# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Standalone GradScaler support for FORCE_FLOAT32 models training without fp16/bf16 autocast.

On pre-Ampere GPUs (e.g. T4, V100) lacking bfloat16 support, FORCE_FLOAT32 model families
(Gemma 3/3n/4, Qwen 3.5 / Qwen 3 MoE, gpt-oss, GLM-4 MoE) downcast frozen linear weights
to float16 while keeping autocast disabled (``fp16=False``, ``bf16=False``,
``mixed_precision='no'``). Because Accelerate sets ``accelerator.scaler = None`` when
``mixed_precision == 'no'``, backward propagation through float16 weights and intermediate
norms loses small adapter gradients to rounding noise unless a dynamic ``GradScaler`` is
installed on the trainer's ``accelerator``.

Keeping ``args.fp16 = False`` and ``args.bf16 = False`` preserves float32 attention/norm
semantics and avoids mutating process-wide ``ACCELERATE_MIXED_PRECISION``, while attaching a
per-trainer ``GradScaler`` restores loss scaling across ``accelerator.backward()``,
``clip_grad_norm_()``, ``AcceleratedOptimizer.step()``, and ``unsloth_fused_ce_loss``.
"""

from __future__ import annotations

import torch

__all__ = [
    "create_forced_float32_loss_scaler",
    "install_forced_float32_loss_scaler",
    "maybe_install_forced_float32_scaler",
]


def _model_accelerator_device(model):
    """Return a CUDA/XPU parameter device on ``model`` if present, else the first device."""
    if model is None or not hasattr(model, "parameters"):
        return None
    fallback = None
    for param in model.parameters():
        device = getattr(param, "device", None)
        if device is None:
            continue
        if device.type in ("cuda", "xpu"):
            return device
        if fallback is None:
            fallback = device
    return fallback


def _unwrap_optimizer(optimizer):
    """Unwrap Accelerate's wrapper down to the underlying PyTorch optimizer expected by GradScaler."""
    seen = set()
    while optimizer is not None and id(optimizer) not in seen:
        seen.add(id(optimizer))
        inner = getattr(optimizer, "optimizer", None)
        if isinstance(inner, torch.optim.Optimizer) or (
            inner is not None and not isinstance(optimizer, torch.optim.Optimizer)
        ):
            optimizer = inner
        else:
            break
    return optimizer


def create_forced_float32_loss_scaler(
    model,
    args = None,
    accelerator = None,
    *,
    trainer = None,
    is_forced_float32 = None,
):
    """Create a GradScaler for a forced-float32 CUDA/XPU trainer, or return None when inappropriate."""
    from .loader_utils import deepspeed_zero_stage, fsdp_version_requested

    if is_forced_float32 is None:
        is_forced_float32 = bool(getattr(model, "_unsloth_forced_float32", False))
    if not is_forced_float32:
        return None

    if args is not None and (getattr(args, "fp16", False) or getattr(args, "bf16", False)):
        return None

    if accelerator is not None:
        if getattr(accelerator, "scaler", None) is not None:
            return None
        if getattr(accelerator, "_unsloth_pp_stage", None) is not None:
            return None
        dist_type = str(getattr(accelerator, "distributed_type", "") or "").upper()
        if "DEEPSPEED" in dist_type or "FSDP" in dist_type:
            return None

    if trainer is not None and (
        getattr(trainer, "is_deepspeed_enabled", False)
        or getattr(trainer, "is_fsdp_enabled", False)
        or getattr(trainer, "_unsloth_tensor_parallel", False)
        or getattr(trainer, "_unsloth_pipeline_parallel", None) is not None
    ):
        return None

    if (
        getattr(args, "deepspeed", None)
        or getattr(args, "fsdp", None)
        or deepspeed_zero_stage() is not None
        or fsdp_version_requested()
    ):
        return None

    device = _model_accelerator_device(model)
    if device is None or device.type not in ("cuda", "xpu"):
        return None

    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type)
    # torch < 2.3 has only the CUDA scaler.
    return torch.cuda.amp.GradScaler() if device.type == "cuda" else None


def install_forced_float32_loss_scaler(accelerator, scaler, get_optimizer) -> None:
    """Attach ``scaler`` to ``accelerator`` and unscale once before gradient clipping."""
    accelerator.scaler = scaler
    # Accelerate unscales before clipping only when it runs native fp16 itself (a prior fp16
    # Trainer in this process can leave that state); unscaling twice raises in GradScaler.
    if getattr(accelerator, "native_amp", False) and getattr(accelerator, "mixed_precision", None) == "fp16":
        return
    if getattr(accelerator.clip_grad_norm_, "_unsloth_forced_float32_scaler_wrapped", False):
        return
    clip = accelerator.clip_grad_norm_

    def _clip_grad_norm(
        parameters,
        max_norm,
        norm_type = 2,
    ):
        optimizer = _unwrap_optimizer(get_optimizer())
        if optimizer is not None:
            scaler.unscale_(optimizer)
        return clip(parameters, max_norm, norm_type)

    _clip_grad_norm._unsloth_forced_float32_scaler_wrapped = True
    accelerator.clip_grad_norm_ = _clip_grad_norm


def maybe_install_forced_float32_scaler(trainer, model = None, *, force_float32 = None) -> bool:
    """Create and install a standalone GradScaler on ``trainer.accelerator`` when needed."""
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None:
        return False
    target_model = getattr(trainer, "model", None)
    if target_model is None:
        target_model = model
    args = getattr(trainer, "args", None)
    scaler = create_forced_float32_loss_scaler(
        target_model,
        args,
        accelerator,
        trainer = trainer,
        is_forced_float32 = force_float32,
    )
    if scaler is None:
        return False
    install_forced_float32_loss_scaler(
        accelerator,
        scaler,
        lambda: getattr(trainer, "optimizer", None),
    )
    opt = getattr(trainer, "optimizer", None)
    if opt is not None and hasattr(opt, "scaler") and opt.scaler is None:
        opt.scaler = scaler
    return True
