# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the weight-space merge metrics behind the Export page's Test button.

Both modules are loaded by file path, as the sibling merge tests do, so the
suite runs without importing the ``unsloth`` package (whose ``__init__`` wants
an accelerator). The metrics module therefore has to reach for the core merger
lazily, which this suite also pins: a module-scope import would break the API
process on a host without a GPU.
"""

import builtins
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

    def test_sign_scan_drops_positions_below_the_delta_scale(self, metrics):
        # A reconstructed delta is dense, so `!= 0` counted float noise at
        # positions neither adapter really moved. Those signs are random and held
        # the rate at the 0.5 chance level whatever the adapters did. Here only
        # the first position is a real disagreement; the second is 1e-7 noise,
        # which used to be counted too (comparable 2 instead of 1).
        A1 = torch.tensor([[1.0, 1e-7]])
        A2 = torch.tensor([[-1.0, 1e-7]])
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

    def test_chance_level_disagreement_does_not_ask_for_dare_ties(self, metrics):
        # Two independent adapters disagree on half their positions by
        # construction, so the threshold has to sit above 0.5 or DARE fires on
        # every orthogonal pair - and at density 0.3 DARE sheds 70% of each
        # delta, which is exactly the signal a non-conflicting pair should keep.
        assert metrics._recommend(cosine = 0.02, conflict = 0.5, overlap = True)["method"] != "dare_ties"
        assert metrics._recommend(cosine = 0.02, conflict = 0.55, overlap = True)["method"] != "dare_ties"
        # Genuinely dense disagreement still reaches DARE.
        assert metrics._recommend(cosine = 0.3, conflict = 0.65, overlap = True)["method"] == "dare_ties"

    def test_orthogonal_adapters_do_not_read_low_and_dare_at_once(self, metrics):
        # The Export page showed "Low interference" beside a dare_ties
        # recommendation for this exact shape (cosine ~0.02, conflict ~0.5).
        # Severity and the recommendation come off the same numbers, so they
        # have to agree.
        conflict = 0.5
        score = 0.5 * max(0.0, -0.02) + 0.5 * max(0.0, 2.0 * conflict - 1.0)
        assert metrics._classify(score) == "low"
        assert metrics._recommend(cosine = 0.02, conflict = conflict, overlap = True)["method"] != "dare_ties"

    def test_high_severity_is_reachable_at_a_realistic_opposition(self, metrics):
        # "high" used to need a full -1.0 cosine or a 1.0 conflict rate, so a
        # pair opposed on most weights could never reach it. Both of these are
        # ordinary oppositions and both must now read as high.
        opposed = 0.5 * max(0.0, -0.7) + 0.5 * max(0.0, 2.0 * 0.6 - 1.0)
        assert metrics._classify(opposed) == "high"
        dense = 0.5 * max(0.0, 0.0) + 0.5 * max(0.0, 2.0 * 0.85 - 1.0)
        assert metrics._classify(dense) == "high"
        # A modest opposition lands in the middle rather than at the floor.
        modest = 0.5 * max(0.0, -0.3) + 0.5 * max(0.0, 0.0)
        assert metrics._classify(modest) == "moderate"

    def test_severity_buckets_follow_the_score(self, metrics):
        assert metrics._classify(0.0) == "low"
        assert metrics._classify(0.25) == "moderate"
        assert metrics._classify(0.9) == "high"


class TestDominance:
    """The second axis: agreeing adapters that still leave one carrying the merge.

    Cosine, conflict and score 0 all describe two adapters that face the same
    way; none of them notice that one delta is 27x the other. Before this axis a
    set like that read "low interference".
    """

    def test_even_adapters_have_no_dominance(self, metrics):
        assert metrics._dominance_score(1.0) == 0.0
        # Below 1 is the same gap read the other way round.
        assert metrics._dominance_score(0.5) == 0.0

    def test_the_axis_is_logarithmic(self, metrics):
        # A multiplicative failure: 2x matters to 10x as 10x matters to 50x, so
        # equal ratios must sit equal distances apart.
        two_to_ten = metrics._dominance_score(10.0) - metrics._dominance_score(2.0)
        ten_to_fifty = metrics._dominance_score(50.0) - metrics._dominance_score(10.0)
        assert two_to_ten == pytest.approx(ten_to_fifty, rel = 1e-6)

    def test_a_dominant_adapter_raises_severity_with_no_disagreement(self, metrics):
        # The merge from the bug report: adapters that do not conflict at all
        # (cosine ~0, conflict at chance) but one is far larger.
        assert metrics._classify(metrics._score(0.02, 0.5, 1.0)) == "low"
        assert metrics._classify(metrics._score(0.02, 0.5, 2.0)) == "moderate"
        assert metrics._classify(metrics._score(0.02, 0.5, 5.5)) == "moderate"
        assert metrics._classify(metrics._score(0.02, 0.5, 27.5)) == "high"

    def test_dominance_is_the_worse_axis_not_an_average(self, metrics):
        # Averaging would let a balanced, opposed pair wash out a dominant one
        # and vice versa. Either axis being bad has to be enough on its own.
        opposed_and_even = metrics._score(-0.7, 0.6, 1.0)
        agreeing_and_dominant = metrics._score(0.02, 0.5, 27.5)
        assert metrics._classify(opposed_and_even) == "high"
        assert metrics._classify(agreeing_and_dominant) == "high"
        assert metrics._score(0.0, 0.0, 1.0) == 0.0

    def test_dominance_reads_direction_off_the_pair(self, metrics):
        # norm_ratio is ||first|| / ||second||, so a ratio below 1 means the
        # second adapter is the larger one and must be named as such.
        pairs = [
            {"a": "Small", "b": "Big", "shared_modules": 393, "norm_ratio": 0.2},
            {"a": "Big", "b": "Other", "shared_modules": 245, "norm_ratio": 1.5},
        ]
        ratio, larger, smaller = metrics._dominance(pairs)
        assert ratio == pytest.approx(5.0)
        assert (larger, smaller) == ("Big", "Small")

    def test_pairs_without_shared_modules_are_not_even(self, metrics):
        # A disjoint pair has no ratio at all; it must not be read as 1.0 and
        # must not outrank a real gap.
        pairs = [
            {"a": "x", "b": "y", "shared_modules": 0, "norm_ratio": 1.0},
            {"a": "Small", "b": "Big", "shared_modules": 10, "norm_ratio": 0.25},
        ]
        ratio, larger, smaller = metrics._dominance(pairs)
        assert ratio == pytest.approx(4.0)
        assert (larger, smaller) == ("Big", "Small")

class TestDominance:
    """The widest norm gap between two adapters, surfaced on its own axis.

    Interference (opposition and sign disagreement) used to be the only thing
    that could move severity, so an adapter set in perfect agreement still read
    "low" with one member carrying the whole merge.
    """

    def test_dominance_axis_reads_logarithmically(self, metrics):
        # 1.0 is even by construction; 2x is the first visibly uneven merge and
        # 10x is past the point where one adapter carries the result.
        assert metrics._dominance_score(1.0) == 0.0
        assert metrics._dominance_score(0.5) == 0.0
        assert metrics._classify(metrics._dominance_score(1.6)) == "low"
        assert metrics._classify(metrics._dominance_score(2.0)) == "moderate"
        assert metrics._classify(metrics._dominance_score(5.5)) == "moderate"
        assert metrics._classify(metrics._dominance_score(10.0)) == "high"
        assert metrics._dominance_score(10_000.0) == pytest.approx(1.0)

    def test_severity_takes_the_worse_axis(self, metrics):
        # No dominance: exactly the old number.
        assert metrics._score(0.02, 0.5, 1.0) == pytest.approx(0.0)
        # A 27x gap on an otherwise agreeing set - dominance alone is enough.
        dominated = metrics._score(0.02, 0.5, 27.5)
        assert metrics._classify(dominated) == "high"
        # And a bad disagreement is still visible when the set is balanced.
        assert metrics._score(-0.7, 0.6, 1.0) == pytest.approx(0.45)

    def test_dominance_names_the_outranking_side(self, metrics):
        pairs = [
            {"a": "big", "b": "small", "shared_modules": 10, "norm_ratio": 5.5},
            {"a": "even", "b": "small", "shared_modules": 10, "norm_ratio": 0.4},
        ]
        # 0.4 is the other way round: small outranks even by 2.5x, but big still
        # wins the maximum at 5.5x.
        ratio, larger, smaller = metrics._dominance(pairs)
        assert ratio == pytest.approx(5.5)
        assert (larger, smaller) == ("big", "small")
        # Pairs sharing no modules carry no ratio and must not read as even.
        assert metrics._dominance(
            [{"a": "x", "b": "y", "shared_modules": 0, "norm_ratio": 1.0}]
        ) == (1.0, "", "")

    def test_an_audio_style_gap_reads_high_while_agreeing(self, metrics):
        # One adapter 5.5x the others with a near-zero cosine: nothing disagrees,
        # yet the small one can barely contribute; shrinking it further to a 27x
        # gap with a 0.2 weight only makes that worse.
        assert metrics._classify(metrics._score(0.02, 0.5, 5.5)) == "moderate"
        assert metrics._classify(metrics._score(0.02, 0.5, 27.5)) == "high"


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

    def test_an_uneven_pair_reports_dominance_even_when_it_agrees(self, analyze, tmp_path):
        # Identical factors mirrored by a weight gap: same direction, no
        # conflict, yet one adapter carries the merge. The report has to say so
        # instead of stopping at "low interference".
        factors = {"model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 20)}
        first = _write_adapter(tmp_path / "first", factors)
        second = _write_adapter(tmp_path / "second", factors)

        report = analyze([str(first), str(second)], weights = [50.0, 1.0])

        assert report["dominance"]["ratio"] == pytest.approx(50.0, rel = 1e-3)
        assert report["dominance"]["adapter"] == report["adapters"][0]["name"]
        assert report["dominance"]["against"] == report["adapters"][1]["name"]
        assert report["dominance"]["score"] > 0.0
        # Severity follows the worse axis, so dominance alone lifts the report
        # out of "low" even with a perfect cosine and zero conflict.
        assert report["interference"] == "high"
        (pair,) = report["pairs"]
        assert pair["cosine"] == pytest.approx(1.0, abs = 1e-3)
        assert pair["sign_conflict_rate"] == 0.0

    def test_an_even_pair_reports_no_dominance(self, analyze, tmp_path):
        factors = {"model.layers.0.self_attn.q_proj": _factors(4, 4, 2, seed = 21)}
        first = _write_adapter(tmp_path / "first", factors)
        second = _write_adapter(tmp_path / "second", factors)

        report = analyze([str(first), str(second)])

        assert report["dominance"]["ratio"] == pytest.approx(1.0)
        assert report["dominance"]["score"] == 0.0
        assert report["dominance"]["adapter"] == ""
        assert report["interference"] == "low"

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

class TestTheCoreImport:
    """The core must resolve without the ``unsloth`` package.

    ``import unsloth`` runs ``_gpu_init`` and refuses to import without an
    accelerator, and the analysis is a CPU-only preflight that never loads a
    model - so routing it through the package would take the feature away on
    exactly the hosts that want to check a merge before touching a GPU, and
    would additionally require the package to be installed at all.
    """

    _HELPERS = (
        "_resolve_adapter_path",
        "_load_adapter_config",
        "_validate_adapters",
        "_adapter_display_name",
        "_load_adapter_state_dict",
        "_group_lora_factors",
        "_adapter_scaling",
    )

    def test_it_loads_the_core_even_when_the_package_cannot_be_imported(self, metrics, monkeypatch):
        monkeypatch.setattr(metrics, "_core_module", None)
        real_import = builtins.__import__

        def refuse_unsloth(name, *args, **kwargs):
            if name == "unsloth" or name.startswith("unsloth."):
                raise ModuleNotFoundError("No module named \'unsloth\'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_unsloth)

        core = metrics._core()

        assert Path(core.__file__).parent == metrics._REPO_ROOT / "unsloth"
        for helper in self._HELPERS:
            assert hasattr(core, helper), helper

    def test_it_loads_the_core_once(self, metrics, monkeypatch):
        monkeypatch.setattr(metrics, "_core_module", None)
        assert metrics._core() is metrics._core()

    def test_the_repository_root_points_at_the_checkout(self, metrics):
        assert (metrics._REPO_ROOT / "unsloth" / "multi_adapter_merge.py").is_file()


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

    def test_the_route_names_the_real_cause_when_the_runtime_is_missing(
        self, monkeypatch, tmp_path, route_module, export_routes
    ):
        # A missing `unsloth` package and a missing PyTorch wheel both raise
        # ImportError. Blaming PyTorch sends the user to the wrong fix for the
        # first, so the detail has to name what actually failed.
        def refuse(*args, **kwargs):
            raise ModuleNotFoundError("No module named \'unsloth\'")

        monkeypatch.setattr(route_module, "analyze_adapters", refuse)
        _routes, client = self._client(monkeypatch, tmp_path, export_routes)

        response = client.post(
            "/api/export/merge/analyze",
            json = {"adapter_paths": ["a", "b"]},
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "No module named" in detail
        assert "PyTorch is not installed" not in detail

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