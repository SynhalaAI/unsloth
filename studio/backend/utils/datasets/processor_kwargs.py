# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tokenizer init kwargs that transformers 5 leaks into audio processor kwargs.

A tokenizer using the ``"tokenizers"`` backend copies its padding/truncation state into
``tokenizer.init_kwargs`` while loading (``max_length``, ``pad_to_multiple_of``, ...).
``ProcessorMixin._merge_kwargs`` forwards every init kwarg that is also a valid audio
kwarg into ``audio_kwargs``. Gemma 4's audio feature extractor defaults to
``truncation=True``, so a leaked ``max_length=None`` overrides its 480_000-sample default
and building the first batch raises "When setting ``truncation=True``, make sure that
``max_length`` is defined."

Models saved after such a load carry the keys in their ``tokenizer_config.json``, so the
audio VLM collator has to drop them before the processor is called with audio.
"""

from __future__ import annotations

from typing import Any, Iterable

# Copied out of tokenizer.init_kwargs by _merge_kwargs: both are valid audio kwargs.
LEAKED_AUDIO_KWARG_KEYS = ("max_length", "pad_to_multiple_of")


def strip_leaked_audio_kwargs(
    processor: Any,
    keys: Iterable[str] = LEAKED_AUDIO_KWARG_KEYS,
) -> list[str]:
    """Drop *keys* from a processor's tokenizer init kwargs and return what was removed.

    Tolerates a processor without a tokenizer, and a tokenizer without an init kwargs
    mapping, so partial mocks and plain tokenizers are both safe to pass.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = processor
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, dict):
        return []

    removed = []
    for key in keys:
        if key in init_kwargs:
            init_kwargs.pop(key)
            removed.append(key)
    return removed
