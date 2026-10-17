# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The transformers-5 tokenizer init kwargs that leak into audio processor kwargs.

A model whose tokenizer_config.json carries ``max_length: null`` (written by a save after
transformers 5 stamped the tokenizer's padding state) kills the audio VLM collator, because
_merge_kwargs forwards it into audio_kwargs and Gemma 4's feature extractor refuses
truncation without a max_length. The audio VLM branch drops it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from utils.datasets.processor_kwargs import (
    LEAKED_AUDIO_KWARG_KEYS,
    strip_leaked_audio_kwargs,
)


def _processor(**init_kwargs):
    return SimpleNamespace(tokenizer = SimpleNamespace(init_kwargs = dict(init_kwargs)))


def test_strips_the_keys_transformers_leaks_into_audio_kwargs():
    processor = _processor(max_length = None, pad_to_multiple_of = None)

    assert strip_leaked_audio_kwargs(processor) == ["max_length", "pad_to_multiple_of"]
    assert processor.tokenizer.init_kwargs == {}


def test_keeps_the_legitimate_tokenizer_keys():
    processor = _processor(
        model_max_length = 131072,
        pad_token_type_id = 0,
        padding_side = "left",
        max_length = None,
    )

    strip_leaked_audio_kwargs(processor)

    assert processor.tokenizer.init_kwargs == {
        "model_max_length": 131072,
        "pad_token_type_id": 0,
        "padding_side": "left",
    }


def test_reports_only_the_keys_it_actually_removed():
    processor = _processor(pad_to_multiple_of = 8, model_max_length = 131072)

    assert strip_leaked_audio_kwargs(processor) == ["pad_to_multiple_of"]


def test_accepts_a_bare_tokenizer():
    tokenizer = SimpleNamespace(init_kwargs = {"max_length": None})

    assert strip_leaked_audio_kwargs(tokenizer) == ["max_length"]
    assert tokenizer.init_kwargs == {}


def test_is_a_noop_without_init_kwargs():
    assert strip_leaked_audio_kwargs(SimpleNamespace()) == []
    assert strip_leaked_audio_kwargs(SimpleNamespace(tokenizer = None)) == []
    assert strip_leaked_audio_kwargs(SimpleNamespace(tokenizer = object())) == []


def test_leaked_keys_are_the_ones_merge_kwargs_copies():
    # Both are valid ProcessorMixin audio kwargs, which is why the leak lands in audio_kwargs.
    assert set(LEAKED_AUDIO_KWARG_KEYS) == {"max_length", "pad_to_multiple_of"}


def test_audio_vlm_branch_strips_the_leak_before_building_the_collator():
    source = (
        Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py"
    ).read_text(encoding = "utf-8")
    branch = source.index("elif self.is_audio_vlm and not raw_text_mode:")
    next_branch = source.index("elif self.is_vlm and not raw_text_mode:", branch)

    assert "strip_leaked_audio_kwargs(processor)" in source[branch:next_branch]
