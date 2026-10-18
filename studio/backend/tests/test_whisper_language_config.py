# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the configurable Whisper decoder language.

The trainer used to pin ``whisper_language = "English"`` and
``generation_config.language = "<|en|>"`` for every audio dataset, so a Sinhala
run trained on labels prefixed ``<|en|>`` and hallucinated at inference. This
locks the new plumbing (config field -> route payload -> worker -> trainer) and
the token resolution helper.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unusable imports
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

import pytest  # noqa: E402

from utils.datasets.whisper_language import (  # noqa: E402
    DEFAULT_WHISPER_LANGUAGE,
    DEFAULT_WHISPER_TASK,
    apply_whisper_decoder_language,
    normalize_whisper_task,
    resolve_generation_language,
)

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


# ---------------------------------------------------------------- unit: task + language resolution


def test_default_language_is_still_english():
    assert DEFAULT_WHISPER_LANGUAGE == "English"
    assert DEFAULT_WHISPER_TASK == "transcribe"


def test_normalize_whisper_task_accepts_and_rejects():
    assert normalize_whisper_task("TRANSLATE") == "translate"
    assert normalize_whisper_task("transcribe") == "transcribe"
    assert normalize_whisper_task(None) == "transcribe"
    assert normalize_whisper_task("garbage") == "transcribe"
    assert normalize_whisper_task("") == "transcribe"


def test_resolve_generation_language_passes_through_tokens():
    assert resolve_generation_language("<|si|>", None) == "<|si|>"
    assert resolve_generation_language("<|en|>", None) == "<|en|>"


def test_resolve_generation_language_reads_the_processor_prefix():
    proc = SimpleNamespace(
        tokenizer = SimpleNamespace(
            prefix_tokens = [1, 2, 3],
            convert_ids_to_tokens = lambda x: "<|si|>",
        )
    )
    assert resolve_generation_language("Sinhala", proc) == "<|si|>"


def test_resolve_generation_language_falls_back_for_two_letter_codes():
    assert resolve_generation_language("si", None) == "<|si|>"
    assert resolve_generation_language("EN", None) == "<|en|>"


def test_resolve_generation_language_none_or_blank_is_none():
    assert resolve_generation_language(None) is None
    assert resolve_generation_language("") is None
    assert resolve_generation_language("   ") is None


def test_resolve_generation_language_unknown_name_passes_through():
    assert resolve_generation_language("Sinhala", None) == "Sinhala"


def test_apply_whisper_decoder_language_sets_language_and_task():
    model = SimpleNamespace(generation_config = SimpleNamespace(language = None, task = None))
    proc = SimpleNamespace(
        tokenizer = SimpleNamespace(
            prefix_tokens = [1, 2, 3],
            convert_ids_to_tokens = lambda x: "<|si|>",
        )
    )
    token = apply_whisper_decoder_language(model, proc, "Sinhala", "translate")
    assert token == "<|si|>"
    assert model.generation_config.language == "<|si|>"
    assert model.generation_config.task == "translate"


def test_apply_whisper_decoder_language_is_no_op_without_generation_config():
    model = SimpleNamespace()
    proc = None
    assert apply_whisper_decoder_language(model, proc, "Sinhala", "transcribe") is None


# ---------------------------------------------------------------- contract: the plumbing


def test_training_model_declares_whisper_language_fields():
    src = (Path(__file__).resolve().parents[1] / "models" / "training.py").read_text(encoding = "utf-8")
    assert 'whisper_language: Optional[str] = Field(' in src
    assert 'whisper_task: Optional[str] = Field(' in src

    # When the backend's heavy deps (PyJWT, ...) are available, check the real
    # pydantic defaults too.
    try:
        import models.training as mt
    except Exception:  # noqa: BLE001 - optional in minimal environments
        return
    fields = mt.TrainingStartRequest.model_fields
    assert "whisper_language" in fields
    assert "whisper_task" in fields
    assert fields["whisper_language"].default is None
    assert fields["whisper_task"].default is None


def test_route_forwards_whisper_language_to_the_worker_payload():
    src = (Path(__file__).resolve().parents[1] / "routes" / "training.py").read_text(encoding = "utf-8")
    assert '"whisper_language": request.whisper_language' in src
    assert '"whisper_task": request.whisper_task' in src


def test_worker_forwards_whisper_language_to_load_model():
    src = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(encoding = "utf-8")
    # Two load_model call sites (initial + cached-files fallback) plus the pre-detect
    # label-tokenizer call, which builds the training labels.
    assert src.count('whisper_language = config.get("whisper_language")') == 3
    assert src.count('whisper_task = config.get("whisper_task")') == 3


def test_pre_detect_builds_whisper_labels_with_the_run_language():
    src = (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py").read_text(encoding = "utf-8")
    # The dataset is tokenized by the pre-detect processor, so the label prefix must be
    # chosen there; a generation-config-only fix leaves the labels English.
    assert "def pre_detect_and_load_tokenizer(" in src
    assert "whisper_language: Optional[str] = None," in src
    start = src.index("def pre_detect_and_load_tokenizer(")
    branch = src[start : src.index("def resolve_spark_tts_repo(")]
    assert "language = whisper_language or DEFAULT_WHISPER_LANGUAGE" in branch
    assert "task = normalize_whisper_task(whisper_task)" in branch


def test_trainer_uses_the_helper_instead_of_hardcoding_english():
    src = (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py").read_text(encoding = "utf-8")
    assert "apply_whisper_decoder_language" in src
    assert "DEFAULT_WHISPER_LANGUAGE" in src
    # The hard-coded English decode prompt must be gone from the Whisper branch.
    assert 'self.model.generation_config.language = "<|en|>"' not in src
    assert 'whisper_language = "English"' not in src
    assert 'whisper_task = "transcribe"' not in src
    # The helper is forwarded on the OSError retry path too.
    assert "whisper_language = whisper_language" in src
    assert "whisper_task = whisper_task" in src
