# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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


def test_resolve_training_format_type_uses_detected_preference_format():
    assert resolve_training_format_type({"format_type": "auto"}, {"final_format": "preference_dpo"}, "") == "preference_dpo"
    assert resolve_training_format_type({"format_type": "chatml"}, None, "") == "chatml"
