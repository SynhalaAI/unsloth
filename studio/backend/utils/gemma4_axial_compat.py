# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Accept Gemma 4 vision configs written by transformers >= 5.20.

transformers 5.20 made ``axial`` the default (and only) RoPE type of the Gemma 4
vision tower, and serializes it into ``config.json``. The 5.x sidecars Studio
ships (5.5.0 and 5.10.2) only accept ``default`` and resolve every other type
through ``ROPE_INIT_FUNCTIONS``; ``axial`` is not registered there, so loading
such a checkpoint dies with ``KeyError: 'axial'`` while the vision tower is
built.

On those versions ``Gemma4VisionRotaryEmbedding.compute_default_rope_parameters``
*is* the axial 2D RoPE (same frequencies and per-axis rotation), so registering
it under ``"axial"`` restores compatibility without editing the checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_axial_rope(rope_init_functions: dict[str, Any], vision_rope_cls: type) -> bool:
    """Add an ``"axial"`` entry to *rope_init_functions* when the class needs one.

    Returns True only when an entry was added. An existing entry, or a class with
    native axial support (transformers >= 5.20), is left untouched.
    """
    if "axial" in rope_init_functions:
        return False
    if hasattr(vision_rope_cls, "compute_axial_rope_parameters"):
        return False
    compute_default = getattr(vision_rope_cls, "compute_default_rope_parameters", None)
    if compute_default is None:
        return False
    rope_init_functions["axial"] = compute_default
    return True


def install_gemma4_axial_compat() -> bool:
    """Patch the active transformers install; safe to call repeatedly.

    Older transformers resolve the vision rope type before the model is built,
    so the entry has to exist by the time ``from_pretrained`` runs. Workers call
    this right after importing transformers, before any model load.
    """
    try:
        import transformers
        from transformers.models.gemma4 import modeling_gemma4
    except Exception:
        # No Gemma 4 in this transformers version (or no transformers at all).
        return False
    try:
        added = register_axial_rope(
            modeling_gemma4.ROPE_INIT_FUNCTIONS,
            modeling_gemma4.Gemma4VisionRotaryEmbedding,
        )
    except Exception as exc:  # pragma: no cover - defensive, never block a load
        logger.debug("Could not install Gemma 4 axial RoPE compatibility: %s", exc)
        return False
    if added:
        logger.info(
            "Registered Gemma 4 'axial' vision RoPE for transformers %s",
            transformers.__version__,
        )
    return added
