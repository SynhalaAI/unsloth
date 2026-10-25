# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
import types

import pytest

try:
    from datasets import Dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

from utils.datasets.preference import (
    detect_preference_columns,
    is_preference_dataset,
    standardize_preference_dataset,
)
from utils.datasets.format_detection import detect_dataset_format
from hub.utils.dataset_format import check_dataset_format
from models.training import TrainingStartRequest
from core.training.trainer import resolve_training_format_type


class MockDataset:
    def __init__(self, data: dict):
        self._data = data
        self.column_names = list(data.keys())

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {k: v[idx] for k, v in self._data.items()}
        return self._data[idx]

    def __iter__(self):
        for i in range(len(next(iter(self._data.values())))):
            yield self[i]

    def __len__(self):
        return len(next(iter(self._data.values())))

    def rename_columns(self, rename_map: dict):
        new_data = {}
        for k, v in self._data.items():
            new_data[rename_map.get(k, k)] = v
        return MockDataset(new_data)

    def map(self, fn, **kwargs):
        new_rows = [fn(self[i]) for i in range(len(self))]
        cols = list(new_rows[0].keys())
        new_data = {c: [r[c] for r in new_rows] for c in cols}
        return MockDataset(new_data)


def test_detect_preference_columns_standard():
    cols = ["prompt", "chosen", "rejected"]
    mapping = detect_preference_columns(cols)
    assert mapping is not None
    assert mapping["chosen"] == "chosen"
    assert mapping["rejected"] == "rejected"
    assert mapping["prompt"] == "prompt"


def test_detect_preference_columns_alternative_names():
    cols = ["instruction", "positive", "negative"]
    mapping = detect_preference_columns(cols)
    assert mapping is not None
    assert mapping["chosen"] == "positive"
    assert mapping["rejected"] == "negative"
    assert mapping["prompt"] == "instruction"


def test_detect_preference_columns_missing():
    cols = ["instruction", "output"]
    assert detect_preference_columns(cols) is None


def _make_dataset(data: dict):
    if _HAS_DATASETS:
        return Dataset.from_dict(data)
    return MockDataset(data)


def test_is_preference_dataset():
    data = {
        "prompt": ["What is 2+2?"],
        "chosen": ["4"],
        "rejected": ["5"],
    }
    ds = _make_dataset(data)
    assert is_preference_dataset(ds) is True

    sft_data = {
        "instruction": ["What is 2+2?"],
        "output": ["4"],
    }
    sft_ds = _make_dataset(sft_data)
    assert is_preference_dataset(sft_ds) is False


def test_standardize_preference_dataset():
    data = {
        "instruction": ["Calculate 10 * 5"],
        "positive": ["50"],
        "negative": ["15"],
    }
    ds = _make_dataset(data)
    standardized = standardize_preference_dataset(ds)

    assert "chosen" in standardized.column_names
    assert "rejected" in standardized.column_names
    assert "prompt" in standardized.column_names
    assert standardized[0]["prompt"] == "Calculate 10 * 5"
    assert standardized[0]["chosen"] == "50"
    assert standardized[0]["rejected"] == "15"


def test_detect_dataset_format_preference():
    data = {
        "prompt": ["Hello"],
        "chosen": ["Hi there!"],
        "rejected": ["Go away"],
    }
    ds = _make_dataset(data)
    fmt = detect_dataset_format(ds)
    assert fmt["format"] == "preference_dpo"
    assert fmt["preference_columns"]["chosen"] == "chosen"
    assert fmt["preference_columns"]["rejected"] == "rejected"


def test_vision_model_check_preserves_text_preference_format():
    dataset = _make_dataset(
        {
            "prompt": ["Hello"],
            "chosen": ["Hi there!"],
            "rejected": ["Go away"],
            "persona": ["helpful"],
        }
    )

    result = check_dataset_format(dataset, is_vlm=True)

    assert result["detected_format"] == "preference_dpo"
    assert result["requires_manual_mapping"] is False
    assert result["is_image"] is False


def test_standardize_preference_dataset_with_system_prompt():
    data = {
        "system": ["You are an expert mathematician."],
        "instruction": ["Calculate 10 * 5"],
        "positive": ["50"],
        "negative": ["15"],
    }
    ds = _make_dataset(data)
    standardized = standardize_preference_dataset(ds)

    assert standardized[0]["prompt"] == "You are an expert mathematician.\n\nCalculate 10 * 5"
    assert standardized[0]["chosen"] == "50"
    assert standardized[0]["rejected"] == "15"


def test_training_start_request_dpo_validation():
    req_dict = {
        "model_name": "unsloth/llama-3-8b-bnb-4bit",
        "training_type": "LoRA/QLoRA",
        "training_method": "DPO",
        "dpo_beta": 0.15,
        "max_prompt_length": 256,
        "format_type": "preference_dpo",
    }
    req = TrainingStartRequest(**req_dict)
    assert req.training_method == "DPO"
    assert req.dpo_beta == 0.15
    assert req.max_prompt_length == 256


def test_training_start_request_cpo_validation():
    req_dict = {
        "model_name": "unsloth/llama-3-8b-bnb-4bit",
        "training_type": "LoRA/QLoRA",
        "training_method": "CPO",
        "dpo_beta": 0.1,
        "cpo_alpha": 1.5,
        "format_type": "preference_cpo",
    }
    req = TrainingStartRequest(**req_dict)
    assert req.training_method == "CPO"
    assert req.cpo_alpha == 1.5
    assert req.dpo_beta == 0.1


def test_training_start_request_orpo_validation():
    req_dict = {
        "model_name": "unsloth/llama-3-8b-bnb-4bit",
        "training_type": "LoRA/QLoRA",
        "training_method": "ORPO",
        "dpo_beta": 0.2,
        "max_prompt_length": 384,
        "format_type": "preference_dpo",
    }
    req = TrainingStartRequest(**req_dict)
    assert req.training_method == "ORPO"
    assert req.dpo_beta == 0.2
    assert req.max_prompt_length == 384


def test_preference_dataset_mismatch_requires_chosen_and_rejected():
    from core.training.preference_trainer import preference_dataset_mismatch

    class Ds:
        def __init__(self, columns):
            self.column_names = columns

    sft_rows = Ds(["instruction", "output"])
    preference_rows = Ds(["prompt", "chosen", "rejected"])

    for method in ("DPO", "ORPO", "CPO", "orpo"):
        message = preference_dataset_mismatch(method, sft_rows)
        assert message is not None
        assert message.startswith("Dataset mismatch:")
        assert preference_dataset_mismatch(method, preference_rows) is None

    # SFT (and an unset method) never consults the dataset shape.
    assert preference_dataset_mismatch("SFT", sft_rows) is None
    assert preference_dataset_mismatch(None, sft_rows) is None

    # The trainer hands over the {"dataset": ...} wrapper, so it must be unwrapped.
    assert preference_dataset_mismatch("ORPO", {"dataset": preference_rows}) is None
    assert preference_dataset_mismatch("ORPO", {"dataset": sft_rows}) is not None


def _fake_orpo_modules(monkeypatch, trainer_cls, config_cls):
    trl = types.ModuleType("trl")
    trl.ORPOTrainer = trainer_cls
    trl.ORPOConfig = config_cls
    unsloth = types.ModuleType("unsloth")
    calls = {}

    def _patch_fast_rl(algorithm):
        calls["algorithm"] = algorithm

    unsloth.PatchFastRL = _patch_fast_rl
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    return calls


def test_setup_orpo_trainer_passes_processing_class_and_strips_sft_kwargs(monkeypatch):
    from core.training import preference_trainer

    recorded = {}

    class FakeORPOConfig:
        def __init__(self, **kwargs):
            recorded["config"] = kwargs

    class FakeORPOTrainer:
        def __init__(self, **kwargs):
            recorded["trainer"] = kwargs

    calls = _fake_orpo_modules(monkeypatch, FakeORPOTrainer, FakeORPOConfig)

    trainer = preference_trainer.setup_orpo_trainer(
        model = object(),
        tokenizer = "tok",
        train_dataset = ["row"],
        eval_dataset = ["eval"],
        config_args = {
            "output_dir": "out",
            "max_seq_length": 2048,
            "dataset_num_proc": 1,
            "packing": True,
            "dataset_text_field": "text",
        },
        dpo_beta = 0.2,
        max_prompt_length = 384,
        max_length = 1024,
    )

    assert isinstance(trainer, FakeORPOTrainer)
    assert calls["algorithm"] == "ORPO"
    assert recorded["config"]["beta"] == 0.2
    assert recorded["config"]["max_length"] == 1024
    assert recorded["config"]["max_prompt_length"] == 384
    assert recorded["config"]["output_dir"] == "out"
    assert "max_seq_length" not in recorded["config"]
    assert "dataset_num_proc" not in recorded["config"]
    assert "packing" not in recorded["config"]
    assert "dataset_text_field" not in recorded["config"]
    assert recorded["trainer"]["processing_class"] == "tok"
    assert "tokenizer" not in recorded["trainer"]
    assert recorded["trainer"]["eval_dataset"] == ["eval"]


def test_setup_orpo_trainer_falls_back_to_trl_experimental(monkeypatch):
    from core.training import preference_trainer

    recorded = {}

    class FakeORPOConfig:
        def __init__(self, **kwargs):
            recorded["config"] = kwargs

    class FakeORPOTrainer:
        def __init__(self, **kwargs):
            recorded["trainer"] = kwargs

    # TRL 0.29+: nothing exported at trl top level, only trl.experimental.orpo.
    trl = types.ModuleType("trl")
    trl.__path__ = []
    experimental = types.ModuleType("trl.experimental")
    experimental.__path__ = []
    orpo = types.ModuleType("trl.experimental.orpo")
    orpo.ORPOTrainer = FakeORPOTrainer
    orpo.ORPOConfig = FakeORPOConfig
    trl.experimental = experimental
    experimental.orpo = orpo
    unsloth = types.ModuleType("unsloth")
    unsloth.PatchFastRL = lambda algorithm: recorded.setdefault("algorithm", algorithm)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.experimental", experimental)
    monkeypatch.setitem(sys.modules, "trl.experimental.orpo", orpo)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)

    trainer = preference_trainer.setup_orpo_trainer(
        model = object(),
        tokenizer = "tok",
        train_dataset = ["row"],
        eval_dataset = None,
        config_args = {"output_dir": "out"},
    )

    assert isinstance(trainer, FakeORPOTrainer)
    assert recorded["algorithm"] == "ORPO"
    assert recorded["trainer"]["processing_class"] == "tok"
    assert "eval_dataset" not in recorded["trainer"]


def test_resolve_training_format_type_uses_detected_preference_format():
    assert resolve_training_format_type({"format_type": "auto"}, {"final_format": "preference_dpo"}, "") == "preference_dpo"
    assert resolve_training_format_type({"format_type": "chatml"}, None, "") == "chatml"
