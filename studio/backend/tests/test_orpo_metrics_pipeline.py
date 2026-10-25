# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ORPO/preference metrics plumbing: SSE payload -> progress histories -> SQLite history."""

from __future__ import annotations

import os
import sys

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.training.training as tr
import storage.studio_db as sdb
from core.training.training import TrainingBackend, TrainingProgress
from core.training.worker import _create_trainer_progress_callback


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(tr, "account_is_retired", lambda: False)
    monkeypatch.setattr(sdb, "finish_run", lambda **kw: None)
    monkeypatch.setattr(sdb, "insert_metrics_batch", lambda *a, **k: None)
    monkeypatch.setattr(sdb, "update_run_progress", lambda **kw: None)
    b = TrainingBackend()
    monkeypatch.setattr(b, "_start_stop_watchdog", lambda *a, **k: None)
    b.current_job_id = "job-orpo"
    b._db_run_created = True
    b._db_total_steps_set = True
    b._db_config = {"model_name": "m", "hf_dataset": "d"}
    return b


class _EventQueue:
    """Minimal event_queue stand-in: the worker callback only calls .put()."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def put(self, event: dict) -> None:
        self.events.append(event)


def _orpo_event(step: int, **overrides) -> dict:
    event = {
        "type": "progress",
        "step": step,
        "epoch": 0.5,
        "loss": 1.1,
        "learning_rate": 1e-4,
        "total_steps": 10,
        "elapsed_seconds": 5.0,
        "eta_seconds": 1,
        "session_start_step": 0,
        "grad_norm": 0.4,
        "num_tokens": 100,
        "eval_loss": None,
        "cer": None,
        "wer": None,
        "rewards_chosen": 0.8,
        "rewards_rejected": -0.2,
        "rewards_accuracies": 0.75,
        "rewards_margins": 1.0,
        "eval_rewards_accuracies": None,
        "eval_rewards_margins": None,
        "nll_loss": 0.9,
        "log_odds_ratio": 0.3,
        "status_message": None,
    }
    event.update(overrides)
    return event


def test_progress_events_keep_preference_and_orpo_histories(backend):
    b = backend
    b._handle_event(_orpo_event(1))
    b._handle_event(
        _orpo_event(2, rewards_accuracies=0.9, nll_loss=0.7, log_odds_ratio=0.5)
    )

    assert b.reward_accuracy_history == [0.75, 0.9]
    assert b.reward_accuracy_step_history == [1, 2]
    assert b.reward_margin_history == [1.0, 1.0]
    assert b.nll_loss_history == [0.9, 0.7]
    assert b.log_odds_ratio_history == [0.3, 0.5]
    assert b._progress.rewards_accuracies == 0.9
    assert b._progress.nll_loss == 0.7
    assert b._progress.log_odds_ratio == 0.5

    buffered = b._metric_buffer[-1]
    assert buffered["rewards_accuracy"] == 0.9
    assert buffered["rewards_margin"] == 1.0
    assert buffered["nll_loss"] == 0.7
    assert buffered["log_odds_ratio"] == 0.5


def test_eval_line_records_eval_preference_without_duplicating_train_step(backend):
    b = backend
    b._handle_event(_orpo_event(1))
    # The worker repeats the last train values on an eval log line; only eval_* is new.
    b._handle_event(
        {
            "type": "progress",
            "step": 1,
            "loss": 1.1,
            "rewards_accuracies": 0.75,
            "rewards_margins": 1.0,
            "nll_loss": 0.9,
            "log_odds_ratio": 0.3,
            "eval_loss": 1.0,
            "eval_rewards_accuracies": 0.6,
            "eval_rewards_margins": 0.4,
        }
    )

    assert b.reward_accuracy_history == [0.75]
    assert b.nll_loss_history == [0.9]
    assert b.eval_reward_accuracy_history == [0.6]
    assert b.eval_reward_margin_history == [0.4]


def test_worker_progress_event_carries_preference_and_orpo_metrics():
    queue = _EventQueue()
    callback = _create_trainer_progress_callback(queue)
    events = queue.events

    progress = TrainingProgress(step = 1, total_steps = 4, loss = 1.2)
    progress.rewards_accuracies = 0.8
    progress.rewards_margins = 0.9
    progress.nll_loss = 0.7
    progress.log_odds_ratio = 0.25
    callback(progress)

    payload = next(event for event in events if event.get("type") == "progress")
    assert payload["rewards_accuracies"] == 0.8
    assert payload["rewards_margins"] == 0.9
    assert payload["nll_loss"] == 0.7
    assert payload["log_odds_ratio"] == 0.25


def test_worker_progress_event_is_emitted_when_only_preference_metrics_change():
    queue = _EventQueue()
    callback = _create_trainer_progress_callback(queue)
    events = queue.events

    progress = TrainingProgress(step = 2, total_steps = 4, loss = 1.2)
    progress.rewards_accuracies = 0.5
    callback(progress)
    progress.rewards_accuracies = 0.9
    callback(progress)

    progress_events = [event for event in events if event.get("type") == "progress"]
    assert [event["rewards_accuracies"] for event in progress_events] == [0.5, 0.9]


def test_orpo_metrics_survive_the_history_roundtrip(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())

    studio_db.create_run(
        id = "run-orpo",
        model_name = "unsloth/test-model",
        dataset_name = "test-preference",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 3,
    )
    studio_db.insert_metrics_batch(
        "run-orpo",
        [
            {
                "step": 1,
                "loss": 1.0,
                "rewards_accuracy": 0.75,
                "rewards_margin": 1.0,
                "nll_loss": 0.9,
                "log_odds_ratio": 0.3,
            },
            {
                "step": 2,
                "loss": 0.9,
                "rewards_accuracy": 0.9,
                "rewards_margin": 1.1,
                "eval_rewards_accuracy": 0.6,
                "eval_rewards_margin": 0.4,
                "nll_loss": 0.7,
                "log_odds_ratio": 0.5,
            },
        ],
    )

    metrics = studio_db.get_run_metrics("run-orpo")
    # The paired-loop variable must not shadow the plain step list.
    assert metrics["step_history"] == [1, 2]
    assert metrics["reward_accuracy_history"] == [0.75, 0.9]
    assert metrics["reward_accuracy_step_history"] == [1, 2]
    assert metrics["reward_margin_history"] == [1.0, 1.1]
    assert metrics["eval_reward_accuracy_history"] == [0.6]
    assert metrics["eval_reward_accuracy_step_history"] == [2]
    assert metrics["eval_reward_margin_history"] == [0.4]
    assert metrics["nll_loss_history"] == [0.9, 0.7]
    assert metrics["nll_loss_step_history"] == [1, 2]
    assert metrics["log_odds_ratio_history"] == [0.3, 0.5]
    assert metrics["log_odds_ratio_step_history"] == [1, 2]
