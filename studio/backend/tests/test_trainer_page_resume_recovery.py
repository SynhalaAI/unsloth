# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for Trainer page checkpoint resume:
- Rejects resume when DB history status is 'completed'
- Repairs DB record when run is 'error' or 'stopped'
- Creates a recovered DB record with available data when run record is missing from DB
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi import HTTPException

from storage import studio_db
from models.training import TrainingStartRequest
from routes import training as training_route


def _write_checkpoint_state(out: Path, step: int) -> Path:
    checkpoint = out / f"checkpoint-{step}"
    checkpoint.mkdir(parents = True, exist_ok = True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding = "utf-8"
    )
    torch.save({"weight": torch.ones(1)}, checkpoint / "adapter_model.bin")
    torch.save({"state": {0: torch.ones(1)}}, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": step}, checkpoint / "scheduler.pt")
    return checkpoint


def test_get_latest_run_by_output_dir_exact_and_completed(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())

    out_dir = str(tmp_path / "outputs" / "test-run-1")
    studio_db.create_run(
        id = "run-c1",
        model_name = "test-model",
        dataset_name = "test-dataset",
        config_json = json.dumps({"output_dir": out_dir, "model_name": "test-model"}),
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 100,
        output_dir = out_dir,
    )
    studio_db.finish_run(
        id = "run-c1",
        status = "completed",
        ended_at = "2026-01-01T01:00:00Z",
        final_step = 100,
        final_loss = 0.5,
        duration_seconds = 3600,
        output_dir = None,
        config_json = json.dumps({"output_dir": out_dir, "model_name": "test-model"}),
    )

    found = studio_db.get_latest_run_by_output_dir(out_dir)
    assert found is not None
    assert found["id"] == "run-c1"
    assert found["status"] == "completed"


def test_repair_run_for_resume_unblocks_and_updates(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())

    out_dir = str(tmp_path / "outputs" / "test-run-err")
    studio_db.create_run(
        id = "run-err",
        model_name = "test-model",
        dataset_name = "test-dataset",
        config_json = json.dumps({"model_name": "test-model"}),
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 50,
        output_dir = out_dir,
    )
    studio_db.finish_run(
        id = "run-err",
        status = "error",
        ended_at = "2026-01-01T00:30:00Z",
        final_step = 20,
        final_loss = 1.2,
        duration_seconds = 1800,
        output_dir = out_dir,
        resume_blocked = True,
        error_message = "crashed",
    )

    before = studio_db.get_run("run-err")
    assert before["resume_blocked"] == 1

    repaired = studio_db.repair_run_for_resume(
        "run-err",
        out_dir,
        config_json = json.dumps({"model_name": "test-model", "output_dir": out_dir}),
        total_steps = 100,
        final_step = 20,
    )
    assert repaired is not None
    assert repaired["resume_blocked"] == 0
    assert repaired["total_steps"] == 100
    assert repaired["final_step"] == 20
    assert repaired["output_dir"] == out_dir


def test_create_recovered_run(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())

    out_dir = str(tmp_path / "outputs" / "recovered-run")
    cfg = json.dumps({"model_name": "unsloth/Llama-3", "output_dir": out_dir})

    run = studio_db.create_recovered_run(
        output_dir = out_dir,
        final_step = 40,
        model_name = "unsloth/Llama-3",
        dataset_name = "test-data",
        config_json = cfg,
        total_steps = 100,
    )

    assert run is not None
    assert run["status"] == "stopped"
    assert run["resume_blocked"] == 0
    assert run["final_step"] == 40
    assert run["total_steps"] == 100
    assert run["output_dir"] == out_dir
    assert run["model_name"] == "unsloth/Llama-3"


@pytest.mark.asyncio
async def test_resume_rejects_when_db_run_completed(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())

    out = tmp_path / "outputs" / "completed-run"
    checkpoint = _write_checkpoint_state(out, 50)

    # Seed DB with completed run
    studio_db.create_run(
        id = "comp-1",
        model_name = "unsloth/Qwen",
        dataset_name = "d",
        config_json = json.dumps({"output_dir": str(out), "model_name": "unsloth/Qwen"}),
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 50,
        output_dir = str(out),
    )
    studio_db.finish_run(
        id = "comp-1",
        status = "completed",
        ended_at = "2026-01-01T01:00:00Z",
        final_step = 50,
        final_loss = 0.2,
        duration_seconds = 3600,
        output_dir = None,
        config_json = json.dumps({"output_dir": str(out), "model_name": "unsloth/Qwen"}),
    )

    req = TrainingStartRequest(
        model_name = "unsloth/Qwen",
        training_type = "Full Finetuning",
        format_type = "alpaca",
        resume_from_checkpoint = str(checkpoint),
        hf_token = "",
    )

    with patch("routes.training.normalize_resume_output_dir", return_value = str(checkpoint)):
        with pytest.raises(HTTPException) as exc_info:
            await training_route.start_training(req)
        assert exc_info.value.status_code == 400
        assert "Completed training runs cannot be resumed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_resume_creates_recovered_record_when_missing_from_db(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())

    out = tmp_path / "outputs" / "orphaned-run"
    checkpoint = _write_checkpoint_state(out, 30)

    # No record in DB for this run!
    assert studio_db.get_latest_run_by_output_dir(str(out)) is None

    req = TrainingStartRequest(
        model_name = "unsloth/Qwen",
        training_type = "Full Finetuning",
        format_type = "alpaca",
        resume_from_checkpoint = str(checkpoint),
        hf_dataset = "sample/data",
        max_steps = 100,
        hf_token = "",
    )

    captured = {}
    class FakeBackend:
        current_job_id = "new-job"
        def is_training_active(self):
            return False
        def start_training(self, **kwargs):
            captured.update(kwargs)
            return True
        def resolve_start_request(self, *args, **kwargs):
            pass

    fake_backend = FakeBackend()

    with (
        patch("routes.training.get_training_backend", return_value = fake_backend),
        patch("routes.training.normalize_resume_output_dir", return_value = str(checkpoint)),
        patch("routes.training.load_model_defaults", return_value = {}),
        patch("routes.training._reject_untrainable_model_request") as mock_preflight,
        patch("routes.training._preflight_hf_dataset_request"),
        patch("utils.hardware.ensure_hardware_detected"),
        patch("routes.training._validate_training_platform"),
    ):
        mock_preflight.return_value = type("Preflight", (), {
            "model_name": "unsloth/Qwen",
            "model_local_path": None,
            "cached_model_pin": None,
        })()
        res = await training_route.start_training(req)
        assert res.status == "queued"
        # Verify a recovered run was created in DB
        latest = studio_db.get_latest_run_by_output_dir(str(out))
        assert latest is not None
        assert latest["model_name"] == "unsloth/Qwen"
        assert latest["final_step"] == 30
        assert captured.get("resume_source_run_id") == latest["id"]


@pytest.mark.asyncio
async def test_resume_repairs_error_record_in_db(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())

    out = tmp_path / "outputs" / "error-run"
    checkpoint = _write_checkpoint_state(out, 25)

    # Insert an errored run with resume_blocked = 1
    studio_db.create_run(
        id = "err-1",
        model_name = "unsloth/Qwen",
        dataset_name = "sample/data",
        config_json = json.dumps({"output_dir": str(out), "model_name": "unsloth/Qwen", "hf_dataset": "sample/data"}),
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 50,
        output_dir = str(out),
    )
    studio_db.finish_run(
        id = "err-1",
        status = "error",
        ended_at = "2026-01-01T00:20:00Z",
        final_step = 25,
        final_loss = 1.0,
        duration_seconds = 1200,
        output_dir = str(out),
        resume_blocked = True,
        error_message = "OOM",
    )

    req = TrainingStartRequest(
        model_name = "unsloth/Qwen",
        training_type = "Full Finetuning",
        format_type = "alpaca",
        resume_from_checkpoint = str(checkpoint),
        hf_dataset = "sample/data",
        max_steps = 80,
        hf_token = "",
    )

    captured = {}
    class FakeBackend:
        current_job_id = "new-job-2"
        def is_training_active(self):
            return False
        def start_training(self, **kwargs):
            captured.update(kwargs)
            return True
        def resolve_start_request(self, *args, **kwargs):
            pass

    fake_backend = FakeBackend()

    with (
        patch("routes.training.get_training_backend", return_value = fake_backend),
        patch("routes.training.normalize_resume_output_dir", return_value = str(checkpoint)),
        patch("routes.training.load_model_defaults", return_value = {}),
        patch("routes.training._reject_untrainable_model_request") as mock_preflight,
        patch("routes.training._preflight_hf_dataset_request"),
        patch("utils.hardware.ensure_hardware_detected"),
        patch("routes.training._validate_training_platform"),
    ):
        mock_preflight.return_value = type("Preflight", (), {
            "model_name": "unsloth/Qwen",
            "model_local_path": None,
            "cached_model_pin": None,
        })()
        res = await training_route.start_training(req)
        assert res.status == "queued"
        # Verify the run was repaired: resume_blocked cleared to 0
        repaired = studio_db.get_run("err-1")
        assert repaired["resume_blocked"] == 0
        assert captured.get("resume_source_run_id") == "err-1"
