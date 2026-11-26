# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import pytest

from core.training import grpo_rewards as gr
from core.training import grpo_trainer as gt
from utils.datasets.format_detection import detect_dataset_format
from utils.datasets.grpo import (
    detect_grpo_columns,
    is_grpo_dataset,
    standardize_grpo_dataset,
)


class _FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = [dict(r) for r in rows]
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]

    def __iter__(self):
        return iter(self._rows)

    def rename_columns(self, mapping: dict[str, str]) -> "_FakeDataset":
        return _FakeDataset(
            [{mapping.get(k, k): v for k, v in row.items()} for row in self._rows]
        )

    def map(self, fn, num_proc=None, desc=None) -> "_FakeDataset":
        del num_proc, desc
        return _FakeDataset([fn(row) for row in self._rows])


SINHALA_MCQ_ROW = {
    "prompt": [
        {
            "role": "user",
            "content": (
                "Samanya kathana Sinhalayedi 'handamin' (crying) kiyana likhitha padaya "
                "wenuwata swabhawikawa kiyanne kohomada?\n"
                "A) Walapemin\nB) Anda anda\nC) Shoka wemin\nD) Handa nagamin\n"
                "Niweradi wikalpa akura pamanak <answer>...</answer> athule liyanna."
            ),
        }
    ],
    "answer": "B",
}


def test_detect_conversational_prompt_answer_grpo_format():
    ds = _FakeDataset([SINHALA_MCQ_ROW])
    fmt = detect_dataset_format(ds)
    assert fmt["format"] == "grpo"
    assert fmt["grpo_columns"] == {"prompt": "prompt", "answer": "answer", "system": None}
    assert is_grpo_dataset(ds) is True

    standardized = standardize_grpo_dataset(ds)
    assert standardized[0]["prompt"] == SINHALA_MCQ_ROW["prompt"]
    assert standardized[0]["answer"] == "B"


def test_detect_plaintext_question_answer_and_standardize():
    ds = _FakeDataset([{"question": "What is 2 + 2?", "solution": "4"}])
    fmt = detect_dataset_format(ds)
    assert fmt["format"] == "grpo"
    assert fmt["grpo_columns"] == {"prompt": "question", "answer": "solution", "system": None}

    standardized = standardize_grpo_dataset(ds)
    assert standardized[0]["prompt"] == "What is 2 + 2?"
    assert standardized[0]["answer"] == "4"


def test_grpo_detection_does_not_steal_sft_or_preference_datasets():
    pref = _FakeDataset([{"prompt": "hi", "chosen": "hello", "rejected": "bye"}])
    assert detect_dataset_format(pref)["format"] == "preference_dpo"
    assert is_grpo_dataset(pref) is False

    alpaca = _FakeDataset([{"instruction": "Say hi", "input": "", "output": "Hi!"}])
    assert detect_dataset_format(alpaca)["format"] == "alpaca"
    assert is_grpo_dataset(alpaca) is False

    chatml = _FakeDataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        ]
    )
    assert detect_dataset_format(chatml)["format"] == "chatml"
    assert is_grpo_dataset(chatml) is False

    # A prompt column containing an assistant turn must not be treated as GRPO.
    prompt_with_assistant = _FakeDataset(
        [
            {
                "prompt": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
                "answer": "B",
            }
        ]
    )
    assert detect_grpo_columns(["prompt", "answer"], [prompt_with_assistant[0]]) is None


def test_answer_match_and_format_rewards_for_conversational_and_string_completions():
    completions = [
        [{"role": "assistant", "content": "<answer>B</answer>"}],
        [{"role": "assistant", "content": "The answer is <answer> b </answer>"}],
        [{"role": "assistant", "content": "<answer>C</answer>"}],
        "<answer>B)</answer>",
        "No tags at all",
    ]
    answers = ["B", "B", "B", "B", "B"]

    match_scores = gr.answer_match_reward(completions = completions, answer = answers)
    assert match_scores == [1.0, 1.0, 0.0, 1.0, 0.0]

    format_scores = gr.answer_format_reward(completions = completions)
    assert format_scores == [0.5, 0.5, 0.5, 0.5, 0.0]

    reasoning_scores = gr.reasoning_format_reward(
        completions = [
            "<think>Let me think.</think><answer>B</answer>",
            "<answer>B</answer>",
            "plain",
        ]
    )
    assert reasoning_scores == [0.5, 0.2, 0.0]


def test_setup_grpo_trainer_patches_and_configures_trl(monkeypatch):
    events: list[str] = []

    @dataclasses.dataclass
    class _FakeGRPOConfig:
        output_dir: str = "out"
        per_device_train_batch_size: int = 1
        num_generations: int = 4
        generation_batch_size: Optional[int] = None
        max_prompt_length: int = 512
        max_completion_length: int = 512
        beta: float = 0.04
        temperature: float = 0.9
        epsilon: float = 0.2
        num_iterations: int = 1
        loss_type: str = "grpo"
        use_vllm: bool = False
        remove_unused_columns: bool = True

    class _FakeGRPOTrainer:
        def __init__(self, model, reward_funcs, args, train_dataset, processing_class, eval_dataset=None):
            self.model = model
            self.reward_funcs = reward_funcs
            self.args = args
            self.train_dataset = train_dataset
            self.processing_class = processing_class
            self.eval_dataset = eval_dataset

    def _fake_patch():
        events.append("patch")

    def _fake_import():
        events.append("import")
        return _FakeGRPOTrainer, _FakeGRPOConfig

    monkeypatch.setattr(gt, "_apply_unsloth_patch", _fake_patch)
    monkeypatch.setattr(gt, "_import_grpo_classes", _fake_import)

    class _FakeTokenizer:
        pad_token_id = None
        eos_token = "<eos>"
        pad_token = None
        padding_side = "right"
        chat_template = "{{ messages }}"

    class _FakeModel:
        max_seq_length = 1024

    ds = _FakeDataset([SINHALA_MCQ_ROW])
    tok = _FakeTokenizer()
    trainer = gt.setup_grpo_trainer(
        model = _FakeModel(),
        processing_class = tok,
        train_dataset = ds,
        config_args = {
            "output_dir": "runs/grpo",
            "per_device_train_batch_size": 1,
            "packing": True,
            "dataset_text_field": "text",
        },
        num_generations = 4,
        max_prompt_length = 768,
        max_completion_length = 512,
        max_seq_length = 2048,
        beta = 0.02,
        temperature = 0.8,
        loss_type = "dr_grpo",
        reward_funcs = ["answer_match", "answer_format"],
    )

    assert events == ["patch", "import"]
    assert tok.pad_token == "<eos>"
    assert tok.padding_side == "left"
    assert trainer.args.max_prompt_length == 768
    # Clamped to model.max_seq_length (1024) - prompt (768) = 256
    assert trainer.args.max_completion_length == 256
    assert trainer.args.generation_batch_size == 4
    assert trainer.args.remove_unused_columns is False
    assert trainer.args.loss_type == "dr_grpo"
    assert len(trainer.reward_funcs) == 2


def test_grpo_kl_early_stopping_patience_and_reset():
    class _State:
        global_step = 0

    class _Control:
        should_training_stop = False

    reasons: list[str] = []
    cb = gt.GrpoKlEarlyStoppingCallback(
        threshold = 0.5,
        patience = 3,
        on_trigger = reasons.append,
    )
    state, control = _State(), _Control()

    for step, kl in [(1, 0.6), (2, 0.7), (3, 0.2), (4, 0.8), (5, 0.9)]:
        state.global_step = step
        cb.on_log(None, state, control, logs = {"kl": kl})
        assert control.should_training_stop is False

    # 3rd consecutive step above 0.5 triggers stop.
    state.global_step = 6
    cb.on_log(None, state, control, logs = {"kl": 0.95})
    assert control.should_training_stop is True
    assert cb.triggered is True
    assert len(reasons) == 1
    assert "step 6" in reasons[0]


def test_grpo_kl_early_stopping_immediate_on_nan():
    class _State:
        global_step = 12

    class _Control:
        should_training_stop = False

    cb = gt.GrpoKlEarlyStoppingCallback(threshold = 0.5, patience = 5)
    state, control = _State(), _Control()
    cb.on_log(None, state, control, logs = {"kl": float("nan")})
    assert control.should_training_stop is True
    assert cb.triggered is True
