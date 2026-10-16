# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Replay dataset feature: schema validation and mixing behaviour."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from models.training import TrainingStartRequest  # noqa: E402

BASE = {
    "model_name": "unsloth/gemma-4-E2B-it",
    "training_type": "LoRA/QLoRA",
    "format_type": "alpaca",
    "hf_dataset": "org/data",
}


def _request(**overrides):
    return TrainingStartRequest.model_validate({**BASE, **overrides})


def test_replay_defaults_off():
    assert _request().replay_datasets is None


def test_replay_accepts_a_valid_spec():
    request = _request(replay_datasets=[{"dataset": "org/replay", "probability": 0.2}])
    assert len(request.replay_datasets) == 1
    assert request.replay_datasets[0].split == "train"


def test_replay_probability_bounds_refused():
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValidationError):
            _request(replay_datasets=[{"dataset": "org/r", "probability": bad}])


def test_replay_probabilities_must_leave_the_main_dataset_a_share():
    with pytest.raises(ValidationError):
        _request(replay_datasets=[
            {"dataset": "org/a", "probability": 0.7},
            {"dataset": "org/b", "probability": 0.4},
        ])


def test_replay_refused_with_streaming():
    with pytest.raises(ValidationError):
        _request(dataset_streaming=True, replay_datasets=[{"dataset": "org/r", "probability": 0.2}])


def test_replay_refused_for_raw_cpt_and_preference_runs():
    for overrides in (
        {"format_type": "raw"},
        {"training_type": "Continued Pretraining"},
        {"training_method": "DPO"},
        {"training_method": "CPO"},
        {"is_dataset_audio": True},
    ):
        with pytest.raises(ValidationError):
            _request(**overrides, replay_datasets=[{"dataset": "org/r", "probability": 0.2}])


def test_replay_dataset_id_rejects_path_traversal():
    with pytest.raises(ValidationError):
        _request(replay_datasets=[{"dataset": "../etc/passwd", "probability": 0.2}])


def test_mix_replay_datasets_stops_at_the_main_datasets_end():
    datasets = pytest.importorskip("datasets")
    from core.training.replay_mixing import mix_replay_datasets

    main = datasets.Dataset.from_dict({"text": [f"m{i}" for i in range(10)]})
    replay = datasets.Dataset.from_dict({"text": [f"r{i}" for i in range(100)]})
    mixed = mix_replay_datasets(main, [replay], [0.2], seed=3407)
    assert len(mixed) == len(main)
    assert any(text.startswith("r") for text in mixed["text"])


def test_mix_replay_datasets_rejects_probabilities_covering_everything():
    pytest.importorskip("datasets")
    from core.training.replay_mixing import mix_replay_datasets

    with pytest.raises(ValueError):
        mix_replay_datasets("main", ["replay"], [1.0])


def test_apply_replay_mixing_is_a_noop_without_specs():
    from core.training.replay_mixing import apply_replay_mixing

    info = {"dataset": "sentinel", "success": True}
    result = apply_replay_mixing(info, None, model_name="m", tokenizer=None, format_type="alpaca")
    assert result is info



def test_replay_local_source_accepts_a_path():
    request = _request(
        replay_datasets=[{"source": "local", "dataset": "E:/data/replay.jsonl", "probability": 0.2}]
    )
    assert request.replay_datasets[0].source == "local"


def test_replay_hf_source_rejects_a_filesystem_path():
    with pytest.raises(ValidationError):
        _request(replay_datasets=[{"dataset": "E:/data/replay.jsonl", "probability": 0.2}])


def test_load_local_replay_reads_a_jsonl_file(tmp_path):
    pytest.importorskip("datasets")
    from core.training.replay_mixing import _load_local_replay

    path = tmp_path / "replay.jsonl"
    path.write_text("{\"text\": \"a\"}\n{\"text\": \"b\"}\n", encoding="utf-8")
    loaded = _load_local_replay({"source": "local", "dataset": str(path)})
    assert len(loaded) == 2
    assert list(loaded["text"]) == ["a", "b"]
