# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CER/WER (OCR metrics) pipeline: helper math, the pump's history/buffer
append, and the /status + /metrics surfaces that feed the frontend charts.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.training.ocr_metrics import compute_cer_wer, make_ocr_metrics_fn
from core.training.training import TrainingBackend, TrainingProgress
import routes.training as rt


async def _inline_to_thread(function, /, *args, **kwargs):
    return function(*args, **kwargs)


@pytest.fixture(autouse = True)
def _run_route_helpers_inline(monkeypatch):
    monkeypatch.setattr(rt.asyncio, "to_thread", _inline_to_thread)


def _running_ocr_backend(monkeypatch, job_id = "job_ocr"):
    b = TrainingBackend()
    b.current_job_id = job_id
    b._proc = type("_P", (), {"is_alive": lambda self: True, "pid": 999})()
    b._progress = TrainingProgress(is_training = True, status_message = "Training in progress...")
    b._finalize_run_in_db = lambda **kw: None
    b._ensure_db_run_created = lambda: None
    b._start_stop_watchdog = lambda **kw: None
    monkeypatch.setattr(rt, "get_training_backend", lambda: b)
    return b


# ── ocr_metrics helper math ─────────────────────────────────────────────


def test_compute_cer_wer_perfect_and_partial():
    result = compute_cer_wer(
        references = ["hello world", "kitten"],
        hypotheses = ["hello world", "sitting"],
    )
    # [0.0, ~0.857] and [0.0, ~1.0]; only sanity ranges are pinned so a jiwer
    # version bump cannot flake the test.
    assert 0.0 <= result["cer"] <= 1.0
    assert 0.0 <= result["wer"] <= 1.0
    assert result["cer"] < result["wer"]


def test_compute_cer_wer_handles_empty_references():
    result = compute_cer_wer(
        references = ["", "not blank"],
        hypotheses = ["", "not blank"],
    )
    # Equal empty pair scores perfect, so the mean is 0.0.
    assert result["cer"] == 0.0
    assert result["wer"] == 0.0

    result = compute_cer_wer(references = [""], hypotheses = ["unexpected text"])
    assert result["cer"] == 1.0
    assert result["wer"] == 1.0


def test_make_ocr_metrics_fn_decodes_and_scores(monkeypatch):
    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def batch_decode(self, ids, skip_special_tokens = True):
            # "A B" -> [2, 3] for the reference; ["A X"] -> [2, 4] for the hypothesis.
            mapping = {2: "A", 3: "B", 4: "X"}
            return [" ".join(mapping[i] for i in row if i in mapping) for row in ids]

    fn = make_ocr_metrics_fn(_Tok())

    class _Pred:
        pass

    pred = _Pred()
    pred.predictions = [[[2, 3]]]
    pred.label_ids = [[2, 3]]
    assert fn(pred)["cer"] == 0.0
    assert fn(pred)["wer"] == 0.0

    pred.predictions = [[[2, 3]]]
    pred.label_ids = [[3, 2]]
    assert fn(pred)["cer"] > 0.0


# ── pump → histories → routes ───────────────────────────────────────────


def test_pump_appends_ocr_metrics_to_histories(monkeypatch):
    b = _running_ocr_backend(monkeypatch)
    b._handle_event(
        {
            "type": "progress",
            "step": 2,
            "loss": 1.0,
            "learning_rate": 0.0001,
            "total_steps": 10,
            "cer": 0.35,
            "wer": 0.2,
        }
    )
    b._handle_event(
        {
            "type": "progress",
            "step": 4,
            "loss": 0.9,
            "learning_rate": 0.0001,
            "total_steps": 10,
            "cer": 0.12,
            "wer": 0.05,
        }
    )
    assert b.cer_history == [0.35, 0.12]
    assert b.cer_step_history == [2, 4]
    assert b.wer_history == [0.2, 0.05]
    assert b.wer_step_history == [2, 4]
    # The metric buffer carries the same values for DB persistence.
    assert b._metric_buffer[-1]["cer"] == 0.12
    assert b._metric_buffer[-1]["wer"] == 0.05


def test_pump_ignores_invalid_ocr_metrics(monkeypatch):
    b = _running_ocr_backend(monkeypatch)
    b._handle_event(
        {
            "type": "progress",
            "step": 2,
            "loss": 1.0,
            "learning_rate": 0.0001,
            "total_steps": 10,
            "cer": "not-a-number",
            "wer": float("inf"),
        }
    )
    assert b.cer_history == []
    assert b.wer_history == []
    assert b._metric_buffer[-1]["cer"] is None
    assert b._metric_buffer[-1]["wer"] is None


def test_status_surface_reports_ocr_metrics(monkeypatch):
    b = _running_ocr_backend(monkeypatch)
    b.step_history[:] = [2]
    b.loss_history[:] = [1.0]
    b.lr_history[:] = [0.0001]
    b.cer_history[:] = [0.35]
    b.cer_step_history[:] = [2]
    b.wer_history[:] = [0.2]
    b.wer_step_history[:] = [2]

    status = asyncio.run(rt.get_training_status(current_subject = "tester"))
    assert status.metric_history["cer"] == [0.35]
    assert status.metric_history["cer_steps"] == [2]
    assert status.metric_history["wer"] == [0.2]
    assert status.metric_history["wer_steps"] == [2]


def test_metrics_route_reports_ocr_metrics(monkeypatch):
    b = _running_ocr_backend(monkeypatch)
    b.step_history[:] = [2]
    b.loss_history[:] = [1.0]
    b.lr_history[:] = [0.0001]
    b.cer_history[:] = [0.35, 0.12]
    b.cer_step_history[:] = [2, 4]
    b.wer_history[:] = [0.2, 0.05]
    b.wer_step_history[:] = [2, 4]

    metrics = asyncio.run(rt.get_training_metrics(expected_job_id = "job_ocr", current_subject = "tester"))
    assert metrics.cer_history == [0.35, 0.12]
    assert metrics.cer_step_history == [2, 4]
    assert metrics.wer_history == [0.2, 0.05]
    assert metrics.wer_step_history == [2, 4]