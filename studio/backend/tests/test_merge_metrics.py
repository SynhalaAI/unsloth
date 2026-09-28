# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the weight-space merge metrics behind the Export page's Test button.

Both modules are loaded by file path, as the sibling merge tests do, so the
suite runs without importing the ``unsloth`` package (whose ``__init__`` wants
an accelerator). The metrics module therefore has to reach for the core merger
lazily, which this suite also pins: a module-scope import would break the API
process on a host without a GPU.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: both modules use postponed annotations, and
    # dataclasses resolves a string annotation through sys.modules[cls.__module__],
    # which is None for a module that was never registered.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def _fake_ensure_export_supported() -> None:
    """The export-capability gate; the analysis is CPU-only, so tests skip it."""


@pytest.fixture(scope = "module")
def core():
    return _load("multi_adapter_merge_under_test", _REPO_ROOT / "unsloth" / "multi_adapter_merge.py")


@pytest.fixture(scope = "module")
def metrics():
    return _load("merge_metrics_under_test", _BACKEND_ROOT / "core" / "export" / "merge_metrics.py")


@pytest.fixture
def analyze(metrics, core, monkeypatch):
    """``analyze_adapters`` bound to the real core merger, loaded by path."""
    monkeypatch.setattr(metrics, "_core", lambda: core)
    return metrics.analyze_adapters


@pytest.fixture(scope = "module")
def export_routes():
    """The export router, loaded by path rather than through ``routes``.

    ``routes/__init__`` imports every router, so ``from routes import export``
    would pull in the training router and its multipart upload dependencies to
    test one endpoint.
    """
    return _load("export_routes_under_test", _BACKEND_ROOT / "routes" / "export.py")


@pytest.fixture
def route_module(core, monkeypatch):
    """The package module the route imports, with the same path-loaded core.

    The route does ``from core.export.merge_metrics import analyze_adapters``,
    so it resolves a different module object than the by-path one above; binding
    both to the same core keeps the two entry points on one implementation.
    """
    import core.export.merge_metrics as packaged

    monkeypatch.setattr(packaged, "_core", lambda: core)
    return packaged


def _factors(out_features: int, in_features: int, rank: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(rank, in_features, generator = generator)
    B = torch.randn(out_features, rank, generator = generator)
    return A, B


def _delta(A, B, scaling: float = 1.0):
    return scaling * (B.to(torch.float32) @ A.to(torch.float32))


def _write_adapter(directory: Path, factors: dict, *, r: int = 4, alpha: int = 4, base: str = "unsloth/test-base"):
    """A minimal PEFT adapter directory the core loader can read."""
    from safetensors.torch import save_file

    directory.mkdir(parents = True, exist_ok = True)
    state = {}
    for key, (A, B) in factors.items():
        state[f"base_model.model.{key}.lora_A.weight"] = A.contiguous()
        state[f"base_model.model.{key}.lora_B.weight"] = B.contiguous()
    save_file(state, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": r,
                "lora_alpha": alpha,
                "base_model_name_or_path": base,
                "target_modules": sorted({key.rsplit(".", 1)[0] for key in factors}),
            }
        ),
        encoding = "utf-8",
    )
    return directory

class TestLowRankIdentities:
    """The cosine path never expands a delta; it has to agree with the real one."""

    def test_norm_matches_the_materialised_delta(self, metrics):
        A, B = _factors(6, 5, 2, seed = 1)
        expected = float(_delta(A, B).pow(2).sum().item())
        assert metrics._factor_norm_sq(A, B) == pytest.approx(expected, rel = 1e-4)

    def test_inner_product_matches_the_materialised_delta(self, metrics):
        A1, B1 = _factors(6, 5, 2, seed = 1)
        A2, B2 = _factors(6, 5, 3, seed = 2)
        expected = float((_delta(A1, B1) * _delta(A2, B2)).sum().item())
        assert metrics._factor_inner(A1, B1, A2, B2) == pytest.approx(expected, rel = 1e-3)

    def test_ranks_may_differ_between_adapters(self, metrics):
        A1, B1 = _factors(6, 5, 2, seed = 1)
        A2, B2 = _factors(6, 5, 3, seed = 2)
        expected = float((_delta(A1, B1) * _delta(A2, B2)).sum().item())
        assert metrics._factor_inner(A1, B1, A2, B2) == pytest.approx(expected, rel = 1e-3)

    def test_sign_scan_counts_only_overlapping_nonzeros(self, metrics):
        # A is (r, in) and B is (out, r). An identity B makes the delta equal A,
        # so the first adapter is [+, +, +, -, -, -] and the second is all +.
        A1 = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        A2 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        B1 = torch.eye(2)
        B2 = torch.eye(2)
        found, comparable = metrics._sign_scan(A1, B1, A2, B2, 1.0, 1.0)
        assert comparable == 6
        assert found == 3  # the negative row disagrees on all three columns

    def test_sign_scan_ignores_positions_one_adapter_leaves_at_zero(self, metrics):
        A1 = torch.tensor([[1.0, 0.0]])
        A2 = torch.tensor([[-1.0, 1.0]])
        B1 = torch.ones(1, 1)
        B2 = torch.ones(1, 1)
        found, comparable = metrics._sign_scan(A1, B1, A2, B2, 1.0, 1.0)
        assert (found, comparable) == (1, 1)


class TestRecommendation:
    """The thresholds that map metrics onto a method."""

    def test_disjoint_adapters_stay_linear(self, metrics):
        assert metrics._recommend(0.0, 0.0, overlap = False)["method"] == "linear"

    def test_agreeing_adapters_stay_linear(self, metrics):
        recommendation = metrics._recommend(cosine = 0.9, conflict = 0.05, overlap = True)
        assert recommendation["method"] == "linear"
        assert recommendation["density"] == 1.0

    def test_opposed_deltas_ask_for_ties(self, metrics):
        assert metrics._recommend(cosine = -0.4, conflict = 0.2, overlap = True)["method"] == "ties"

    def test_dense_disagreement_asks_for_dare_ties(self, metrics):
        recommendation = metrics._recommend(cosine = 0.3, conflict = 0.6, overlap = True)
        assert recommendation["method"] == "dare_ties"

    def test_weak_overlap_asks_for_ties(self, metrics):
        assert metrics._recommend(cosine = 0.2, conflict = 0.2, overlap = True)["method"] == "ties"

    def test_severity_buckets_follow_the_score(self, metrics):
        assert metrics._classify(0.0) == "low"
        assert metrics._classify(0.25) == "moderate"
        assert metrics._classify(0.9) == "high"

class TestAnalyzeAdapters:
    """End to end over adapter directories on disk, through the real core loader."""

    def test_matching_adapters_report_no_interference(self, analyze, tmp_path):
        factors = {
            "model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 3),
            "model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 4),
        }
        first = _write_adapter(tmp_path / "first", factors)
        second = _write_adapter(tmp_path / "second", factors)

        report = analyze([str(first), str(second)])

        assert report["interference"] == "low"
        assert report["score"] == 0.0
        assert report["mean_cosine"] == pytest.approx(1.0, abs = 1e-3)
        assert report["max_sign_conflict_rate"] == 0.0
        assert report["recommendation"]["method"] == "linear"
        (pair,) = report["pairs"]
        assert pair["shared_modules"] == 2
        assert pair["module_overlap"] == 1.0
        assert pair["worst_modules"] == []
        assert [adapter["modules"] for adapter in report["adapters"]] == [2, 2]

    def test_opposed_adapters_are_flagged_and_recommended_for_ties(self, analyze, tmp_path):
        forward = {
            "model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 5),
            "model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 6),
        }
        reversed_ = {key: (A, -B) for key, (A, B) in forward.items()}
        first = _write_adapter(tmp_path / "first", forward)
        second = _write_adapter(tmp_path / "second", reversed_)

        report = analyze([str(first), str(second)])

        assert report["mean_cosine"] == pytest.approx(-1.0, abs = 1e-3)
        assert report["max_sign_conflict_rate"] == pytest.approx(1.0)
        assert report["interference"] == "high"
        assert report["recommendation"]["method"] == "ties"

    def test_disjoint_module_sets_do_not_interfere(self, analyze, tmp_path):
        first = _write_adapter(
            tmp_path / "first",
            {"vision_tower.blocks.0.attn.q_proj": _factors(4, 4, 2, seed = 7)},
        )
        second = _write_adapter(
            tmp_path / "second",
            {"audio_tower.blocks.0.attn.q_proj": _factors(4, 4, 2, seed = 8)},
        )

        report = analyze([str(first), str(second)])

        (pair,) = report["pairs"]
        assert pair["shared_modules"] == 0
        assert pair["module_overlap"] == 0.0
        assert report["interference"] == "low"
        assert report["recommendation"]["method"] == "linear"

    def test_weights_describe_the_merge_that_would_run(self, analyze, tmp_path):
        factors = {"model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 9)}
        first = _write_adapter(tmp_path / "first", factors)
        second = _write_adapter(tmp_path / "second", factors)

        report = analyze([str(first), str(second)], weights = [3.0, 1.0])

        assert report["adapters"][0]["effective_weight"] == pytest.approx(0.75)
        assert report["adapters"][1]["effective_weight"] == pytest.approx(0.25)
        # A pair of identical adapters stays aligned whatever the weights, but
        # the norm ratio follows the weights rather than the raw factors.
        (pair,) = report["pairs"]
        assert pair["cosine"] == pytest.approx(1.0, abs = 1e-3)
        assert pair["norm_ratio"] == pytest.approx(3.0, rel = 1e-3)

    def test_worst_modules_name_the_layer_that_conflicts(self, analyze, tmp_path):
        shared_agreeing = {"model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 10)}
        forward = {
            "model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 10),
            "model.layers.1.self_attn.q_proj": _factors(4, 4, 2, seed = 11),
        }
        opposing_A, opposing_B = _factors(4, 4, 2, seed = 11)
        reversed_ = {
            "model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 10),
            # Negate B only: the delta is B @ A, so flipping B flips its sign
            # (flipping both would leave the product unchanged).
            "model.layers.1.self_attn.q_proj": (opposing_A, -opposing_B),
        }
        first = _write_adapter(tmp_path / "first", forward)
        second = _write_adapter(tmp_path / "second", reversed_)
        assert shared_agreeing  # the first module is identical in both adapters

        report = analyze([str(first), str(second)])

        (pair,) = report["pairs"]
        assert [entry["module"] for entry in pair["worst_modules"]] == [
            "model.layers.1.self_attn.q_proj"
        ]
        assert pair["worst_modules"][0]["sign_conflict_rate"] == pytest.approx(1.0)

    def test_sign_budget_is_reported_when_it_runs_out(self, analyze, tmp_path):
        factors = {"model.layers.0.self_attn.q_proj": _factors(8, 8, 2, seed = 12)}
        first = _write_adapter(tmp_path / "first", factors)
        second = _write_adapter(tmp_path / "second", factors)

        report = analyze([str(first), str(second)], max_sign_elements = 4)

        assert report["sign_scan_truncated"] is True
        assert report["max_sign_conflict_rate"] == 0.0  # identical adapters, sampled

    def test_a_single_adapter_is_rejected(self, analyze, tmp_path):
        only = _write_adapter(tmp_path / "only", {"model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 13)})
        with pytest.raises(ValueError):
            analyze([str(only)])

    def test_weights_must_match_the_adapter_count(self, analyze, tmp_path):
        first = _write_adapter(tmp_path / "first", {"model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 14)})
        second = _write_adapter(tmp_path / "second", {"model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 15)})
        with pytest.raises(ValueError):
            analyze([str(first), str(second)], weights = [1.0])

    def test_adapters_from_different_base_models_are_rejected(self, analyze, tmp_path):
        factors = {"model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 16)}
        first = _write_adapter(tmp_path / "first", factors, base = "unsloth/base-a")
        second = _write_adapter(tmp_path / "second", factors, base = "unsloth/base-b")
        with pytest.raises(ValueError):
            analyze([str(first), str(second)])

class TestAnalyzeRoute:
    """The endpoint contract: a report, or a 400 the user can act on."""

    @staticmethod
    def _client(monkeypatch, tmp_path, export_routes):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from auth.authentication import allow_ambient_hf_token, get_current_subject

        app = FastAPI()
        app.include_router(export_routes.router, prefix = "/api/export")
        app.dependency_overrides[get_current_subject] = lambda: "alice"
        app.dependency_overrides[allow_ambient_hf_token] = lambda: True
        monkeypatch.setattr(
            export_routes, "_ensure_export_supported", _fake_ensure_export_supported
        )
        return export_routes, TestClient(app)

    def test_the_route_returns_a_report(self, monkeypatch, tmp_path, route_module, export_routes):
        factors = {"model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 21)}
        first = _write_adapter(tmp_path / "first", factors)
        second = _write_adapter(tmp_path / "second", factors)
        _routes, client = self._client(monkeypatch, tmp_path, export_routes)

        response = client.post(
            "/api/export/merge/analyze",
            json = {"adapter_paths": [str(first), str(second)]},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True
        assert body["interference"] == "low"
        assert body["recommendation"]["method"] == "linear"
        assert len(body["pairs"]) == 1
        assert len(body["adapters"]) == 2

    def test_the_route_rejects_a_single_adapter(self, monkeypatch, tmp_path, route_module, export_routes):
        only = _write_adapter(
            tmp_path / "only", {"model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 22)}
        )
        _routes, client = self._client(monkeypatch, tmp_path, export_routes)

        response = client.post(
            "/api/export/merge/analyze",
            json = {"adapter_paths": [str(only)]},
        )

        assert response.status_code == 422

    def test_the_route_reports_an_unreadable_adapter(self, monkeypatch, tmp_path, route_module, export_routes):
        good = _write_adapter(
            tmp_path / "good", {"model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 23)}
        )
        _routes, client = self._client(monkeypatch, tmp_path, export_routes)

        response = client.post(
            "/api/export/merge/analyze",
            json = {"adapter_paths": [str(good), str(tmp_path / "missing")]},
        )

        # The core resolves an unknown path as a repo id, so the detail names
        # the offending string rather than saying "missing adapter"; the
        # contract under test is that it is a 400 the user can act on.
        assert response.status_code == 400
        assert "missing" in response.json()["detail"]

    def test_the_route_surfaces_mismatched_base_models(self, monkeypatch, tmp_path, route_module, export_routes):
        factors = {"model.layers.0.mlp.up_proj": _factors(4, 4, 2, seed = 24)}
        first = _write_adapter(tmp_path / "first", factors, base = "unsloth/base-a")
        second = _write_adapter(tmp_path / "second", factors, base = "unsloth/base-b")
        _routes, client = self._client(monkeypatch, tmp_path, export_routes)

        response = client.post(
            "/api/export/merge/analyze",
            json = {"adapter_paths": [str(first), str(second)]},
        )

        assert response.status_code == 400
        assert "base model" in response.json()["detail"].lower()