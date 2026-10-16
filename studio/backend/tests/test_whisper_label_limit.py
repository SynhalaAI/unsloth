# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for Whisper rows past config.max_target_positions.

transformers' WhisperForConditionalGeneration.forward raises
``Labels' sequence length N cannot exceed ... 448 tokens.`` for them, and the fix must
happen while the dataset is prepared: the dataset itself (a Hub upload the user does not
rewrite) is never modified, and fitting rows are never truncated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
import json  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402


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

from core.training import trainer as tmod  # noqa: E402
from utils.datasets.whisper_labels import (  # noqa: E402
    DEFAULT_MAX_TARGET_POSITIONS,
    labels_fit_decoder,
    resolve_max_target_positions,
)

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


@pytest.fixture
def audio_trainer(monkeypatch):
    # Avoid MLX substitution on Apple silicon.
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    monkeypatch.setattr(tmod, "ensure_audio_decoding", lambda: True)
    t = tmod.UnslothTrainer()
    t.model_name = "unsloth/whisper-small"
    return t


class _FakeWhisperTokenizer:
    """One id per character, so a row's label length is its text length."""

    class _Features:
        def __init__(self, arrays):
            self.input_features = arrays

    def feature_extractor(self, array, sampling_rate = None):
        return self._Features([list(array)])

    def tokenizer(self, text):
        return types.SimpleNamespace(input_ids = list(range(len(text))))


class _Rows:
    """Dataset stand-in, like _FakeAudioDataset: real Audio() casting needs torchcodec."""

    def __init__(self, texts, with_audio = True):
        self._texts = list(texts)
        self._with_audio = with_audio

    @property
    def column_names(self):
        return (["audio"] if self._with_audio else []) + ["text"]

    def cast_column(self, column, feature):
        if column not in self.column_names:
            return _Rows(self._texts, with_audio = False)
        return self

    def train_test_split(self, test_size = None, seed = None):
        n_eval = max(1, round(len(self._texts) * test_size))
        return {
            "train": _Rows(self._texts[:-n_eval]),
            "test": _Rows(self._texts[-n_eval:]),
        }

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        row = {"text": self._texts[idx]}
        if self._with_audio:
            row["audio"] = {"array": [0.0] * 160, "sampling_rate": 16000}
        return row


# ---------------------------------------------------------------- unit: the limit

def test_nothing_configured_falls_back_to_whispers_own_default(monkeypatch):
    """WhisperConfig.max_target_positions is 448; an uncached repo must read as that, offline."""
    import utils.datasets.whisper_labels as wl

    monkeypatch.setattr(wl, "_config_json", lambda model_name: None)
    assert resolve_max_target_positions() == DEFAULT_MAX_TARGET_POSITIONS == 448
    assert resolve_max_target_positions(model_name = "unsloth/__never_cached_whisper__") == 448


def test_a_local_config_json_raises_the_budget(tmp_path):
    """A checkpoint that ships a larger table trains with it - the cap is read, not hardcoded."""
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "whisper", "max_target_positions": 1024}),
        encoding = "utf-8",
    )
    assert resolve_max_target_positions(model_name = str(tmp_path)) == 1024


def test_a_loaded_model_config_wins_over_its_config_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"max_target_positions": 448}), encoding = "utf-8")
    model = types.SimpleNamespace(config = types.SimpleNamespace(max_target_positions = 2048))
    assert resolve_max_target_positions(model = model, model_name = str(tmp_path)) == 2048


def test_a_malformed_or_absent_config_is_not_a_verdict(tmp_path):
    (tmp_path / "config.json").write_text("not json", encoding = "utf-8")
    assert resolve_max_target_positions(model_name = str(tmp_path)) == 448
    assert resolve_max_target_positions(model_name = str(tmp_path / "missing")) == 448


# ---------------------------------------------------------------- unit: the BOS strip

def test_the_collator_dropped_bos_is_not_counted():
    """DataCollatorSpeechSeq2SeqWithPadding strips one leading BOS, so a 449-id row of
    [BOS, ...448...] fits; the model validates the post-strip batch."""
    BOS = 50257
    assert labels_fit_decoder([BOS] + [7] * 448, 448, BOS)
    assert not labels_fit_decoder([BOS] + [7] * 449, 448, BOS)


def test_without_a_bos_the_raw_length_is_the_counted_length():
    assert labels_fit_decoder([7] * 448, 448, None)
    assert not labels_fit_decoder([7] * 449, 448, None)
    # bos_token_id known but the row does not start with it: nothing to strip.
    assert labels_fit_decoder([7] * 448, 448, 50257)
    assert not labels_fit_decoder([7] * 449, 448, 50257)


# ---------------------------------------------------------------- integration: the skip

def test_rows_past_the_limit_are_skipped_whole_and_the_source_is_untouched(audio_trainer, tmp_path):
    """The model loads after preprocessing, so the budget comes from the checkpoint's
    config.json - and a fitting row must come through untruncated."""
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "whisper", "max_target_positions": 4}),
        encoding = "utf-8",
    )
    audio_trainer.model_name = str(tmp_path)
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _Rows(["okay", "far too long for the decoder"]),
        eval_split = None,
        eval_dataset = None,
    )

    assert train_data == [{"input_features": [0.0] * 160, "labels": [0, 1, 2, 3]}], (
        "a fitting row must be kept whole and an over-long row dropped, not truncated"
    )
    assert eval_data is None
    assert any("max_target_positions" in w for w in audio_trainer.training_progress.warnings)


def test_an_over_limit_row_also_warns_when_it_is_in_the_eval_split(audio_trainer, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"max_target_positions": 4}), encoding = "utf-8")
    audio_trainer.model_name = str(tmp_path)
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _Rows(["okay", "okay", "okay", "okay", "okay", "okay"]),
        eval_split = "validation",
        eval_dataset = _Rows(["ev-1 is way over the budget"]),
    )

    assert len(train_data) == 6, "the limit must not touch rows that fit"
    assert not eval_data, "every eval row was over the limit"
    warnings = audio_trainer.training_progress.warnings
    assert any("skipped 1 row" in w for w in warnings)
    assert any("The dataset was not changed" in w for w in warnings)


def test_a_split_wholly_past_the_limit_says_why(audio_trainer, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"max_target_positions": 3}), encoding = "utf-8")
    audio_trainer.model_name = str(tmp_path)
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    with pytest.raises(ValueError, match = "exceed the 3-token label limit"):
        audio_trainer._preprocess_whisper_dataset(
            _Rows(["xxxx"]), eval_split = None, eval_dataset = None
        )


def test_the_default_budget_keeps_a_realistic_row(audio_trainer):
    """No config readable and a short transcript: nothing is dropped and nothing warns."""
    audio_trainer.model = None
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _Rows(["short transcript"]), eval_split = None, eval_dataset = None
    )

    assert len(train_data) == 1
    assert eval_data is None
    assert not audio_trainer.training_progress.warnings