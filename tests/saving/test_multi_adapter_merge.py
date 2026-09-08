# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for unsloth.multi_adapter_merge.

These are CPU-only unit tests that exercise the merge logic using synthetic
adapter state dicts — no GPU or real model weights required.
"""

import json
import os
import tempfile

import pytest
import torch

from unsloth.multi_adapter_merge import (
    MultiAdapterMergeConfig,
    _linear_merge,
    _load_adapter_config,
    _load_adapter_state_dict,
    _reconstruct_deltas,
    _ties_merge,
    _validate_adapters,
    merge_adapters_into_model,
)


# ---------------------------------------------------------------------------
# Helpers: create synthetic adapter directories on disk
# ---------------------------------------------------------------------------

def _make_adapter_dir(
    tmp_dir,
    name,
    base_model="test-org/test-base",
    r=16,
    lora_alpha=16,
    target_modules=("q_proj", "v_proj"),
    out_features=64,
    in_features=64,
    seed=0,
):
    """Create a minimal PEFT adapter directory with random LoRA weights."""
    adapter_path = os.path.join(tmp_dir, name)
    os.makedirs(adapter_path, exist_ok=True)

    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),
    }
    with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
        json.dump(config, f)

    gen = torch.Generator().manual_seed(seed)
    state_dict = {}
    for mod in target_modules:
        key_a = f"base_model.model.model.layers.0.self_attn.{mod}.lora_A.weight"
        key_b = f"base_model.model.model.layers.0.self_attn.{mod}.lora_B.weight"
        state_dict[key_a] = torch.randn(r, in_features, generator=gen)
        state_dict[key_b] = torch.randn(out_features, r, generator=gen)

    from safetensors.torch import save_file
    save_file(state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))
    return adapter_path


# ---------------------------------------------------------------------------
# MultiAdapterMergeConfig tests
# ---------------------------------------------------------------------------

class TestMultiAdapterMergeConfig:
    def test_valid_config(self):
        cfg = MultiAdapterMergeConfig(
            adapter_paths=["a", "b"],
            weights=[0.6, 0.4],
        )
        assert cfg.method == "linear"
        assert abs(sum(cfg.weights) - 1.0) < 1e-6

    def test_weight_normalization(self):
        cfg = MultiAdapterMergeConfig(
            adapter_paths=["a", "b"],
            weights=[3.0, 7.0],
            normalize_weights=True,
        )
        assert abs(cfg.weights[0] - 0.3) < 1e-6
        assert abs(cfg.weights[1] - 0.7) < 1e-6

    def test_no_normalization(self):
        cfg = MultiAdapterMergeConfig(
            adapter_paths=["a"],
            weights=[2.0],
            normalize_weights=False,
        )
        assert cfg.weights == [2.0]

    def test_empty_adapters_raises(self):
        with pytest.raises(ValueError, match="At least one adapter"):
            MultiAdapterMergeConfig(adapter_paths=[], weights=[])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            MultiAdapterMergeConfig(adapter_paths=["a", "b"], weights=[1.0])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown merge method"):
            MultiAdapterMergeConfig(
                adapter_paths=["a"], weights=[1.0], method="unknown"
            )

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="must not sum to zero"):
            MultiAdapterMergeConfig(
                adapter_paths=["a", "b"], weights=[0.0, 0.0]
            )

    def test_bad_density_raises(self):
        with pytest.raises(ValueError, match="density"):
            MultiAdapterMergeConfig(
                adapter_paths=["a"], weights=[1.0], density=0.0
            )


# ---------------------------------------------------------------------------
# Adapter I/O tests
# ---------------------------------------------------------------------------

class TestAdapterIO:
    def test_load_config(self, tmp_path):
        path = _make_adapter_dir(str(tmp_path), "adapter1")
        cfg = _load_adapter_config(path)
        assert cfg["peft_type"] == "LORA"
        assert cfg["r"] == 16

    def test_load_config_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            _load_adapter_config(str(tmp_path / "nonexistent"))

    def test_load_state_dict(self, tmp_path):
        path = _make_adapter_dir(str(tmp_path), "adapter1")
        sd = _load_adapter_state_dict(path)
        assert len(sd) > 0
        for v in sd.values():
            assert v.device == torch.device("cpu")

    def test_load_state_dict_missing_raises(self, tmp_path):
        os.makedirs(tmp_path / "empty_adapter")
        with pytest.raises(FileNotFoundError, match="No adapter weights"):
            _load_adapter_state_dict(str(tmp_path / "empty_adapter"))


# ---------------------------------------------------------------------------
# Delta reconstruction tests
# ---------------------------------------------------------------------------

class TestReconstructDeltas:
    def test_basic_reconstruction(self, tmp_path):
        r, out_f, in_f = 8, 32, 32
        path = _make_adapter_dir(
            str(tmp_path), "a1", r=r, out_features=out_f, in_features=in_f
        )
        cfg = _load_adapter_config(path)
        sd = _load_adapter_state_dict(path)
        deltas = _reconstruct_deltas(sd, cfg)

        # Should produce one delta per target module.
        assert len(deltas) == 2  # q_proj and v_proj

        for key, delta in deltas.items():
            assert delta.shape == (out_f, in_f)
            assert delta.dtype == torch.float32

    def test_scaling_applied(self, tmp_path):
        """Verify that α/r scaling is applied correctly."""
        r, alpha = 4, 8  # scaling = 2.0
        path = _make_adapter_dir(
            str(tmp_path), "a1", r=r, lora_alpha=alpha,
            target_modules=("q_proj",), out_features=16, in_features=16,
        )
        cfg = _load_adapter_config(path)
        sd = _load_adapter_state_dict(path)

        # Manually compute expected delta.
        a_key = [k for k in sd if "lora_A" in k][0]
        b_key = [k for k in sd if "lora_B" in k][0]
        expected = (alpha / r) * sd[b_key].float().mm(sd[a_key].float())

        deltas = _reconstruct_deltas(sd, cfg)
        delta = list(deltas.values())[0]
        assert torch.allclose(delta, expected, atol=1e-5)

    def test_heterogeneous_ranks(self, tmp_path):
        """Adapters with different ranks produce same-shaped deltas."""
        out_f, in_f = 32, 32
        p1 = _make_adapter_dir(
            str(tmp_path), "a1", r=8, out_features=out_f, in_features=in_f, seed=1
        )
        p2 = _make_adapter_dir(
            str(tmp_path), "a2", r=32, out_features=out_f, in_features=in_f, seed=2
        )
        d1 = _reconstruct_deltas(_load_adapter_state_dict(p1), _load_adapter_config(p1))
        d2 = _reconstruct_deltas(_load_adapter_state_dict(p2), _load_adapter_config(p2))

        for key in d1:
            assert d1[key].shape == d2[key].shape


# ---------------------------------------------------------------------------
# Linear merge tests
# ---------------------------------------------------------------------------

class TestLinearMerge:
    def test_equal_weights(self, tmp_path):
        out_f, in_f = 16, 16
        p1 = _make_adapter_dir(str(tmp_path), "a1", out_features=out_f, in_features=in_f, seed=1)
        p2 = _make_adapter_dir(str(tmp_path), "a2", out_features=out_f, in_features=in_f, seed=2)

        d1 = _reconstruct_deltas(_load_adapter_state_dict(p1), _load_adapter_config(p1))
        d2 = _reconstruct_deltas(_load_adapter_state_dict(p2), _load_adapter_config(p2))

        merged = _linear_merge([d1, d2], [0.5, 0.5])
        for key in merged:
            expected = 0.5 * d1[key] + 0.5 * d2[key]
            assert torch.allclose(merged[key], expected, atol=1e-5)

    def test_single_adapter_passthrough(self, tmp_path):
        """Single adapter with weight=1.0 should equal the original delta."""
        p = _make_adapter_dir(str(tmp_path), "a1", out_features=16, in_features=16)
        d = _reconstruct_deltas(_load_adapter_state_dict(p), _load_adapter_config(p))
        merged = _linear_merge([d], [1.0])
        for key in merged:
            assert torch.allclose(merged[key], d[key], atol=1e-6)

    def test_missing_module_treated_as_zero(self, tmp_path):
        """If adapter2 doesn't touch q_proj, its delta is zero for that module."""
        p1 = _make_adapter_dir(
            str(tmp_path), "a1",
            target_modules=("q_proj", "v_proj"), out_features=16, in_features=16, seed=1,
        )
        p2 = _make_adapter_dir(
            str(tmp_path), "a2",
            target_modules=("v_proj",), out_features=16, in_features=16, seed=2,
        )
        d1 = _reconstruct_deltas(_load_adapter_state_dict(p1), _load_adapter_config(p1))
        d2 = _reconstruct_deltas(_load_adapter_state_dict(p2), _load_adapter_config(p2))

        merged = _linear_merge([d1, d2], [0.5, 0.5])
        # q_proj: only from d1 (weight 0.5)
        q_key = [k for k in merged if "q_proj" in k][0]
        assert torch.allclose(merged[q_key], 0.5 * d1[q_key], atol=1e-6)

    def test_custom_weights(self, tmp_path):
        p1 = _make_adapter_dir(str(tmp_path), "a1", out_features=16, in_features=16, seed=1)
        p2 = _make_adapter_dir(str(tmp_path), "a2", out_features=16, in_features=16, seed=2)

        d1 = _reconstruct_deltas(_load_adapter_state_dict(p1), _load_adapter_config(p1))
        d2 = _reconstruct_deltas(_load_adapter_state_dict(p2), _load_adapter_config(p2))

        merged = _linear_merge([d1, d2], [0.8, 0.2])
        for key in merged:
            expected = 0.8 * d1[key] + 0.2 * d2[key]
            assert torch.allclose(merged[key], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# TIES merge tests
# ---------------------------------------------------------------------------

class TestTIESMerge:
    def test_ties_produces_output(self, tmp_path):
        """Smoke test: TIES merge produces non-empty, finite output."""
        p1 = _make_adapter_dir(str(tmp_path), "a1", out_features=16, in_features=16, seed=1)
        p2 = _make_adapter_dir(str(tmp_path), "a2", out_features=16, in_features=16, seed=2)

        d1 = _reconstruct_deltas(_load_adapter_state_dict(p1), _load_adapter_config(p1))
        d2 = _reconstruct_deltas(_load_adapter_state_dict(p2), _load_adapter_config(p2))

        merged = _ties_merge([d1, d2], [0.5, 0.5], density=0.5)
        assert len(merged) > 0
        for v in merged.values():
            assert torch.isfinite(v).all()

    def test_ties_density_1_retains_all(self, tmp_path):
        """With density=1.0, no parameters are trimmed."""
        p1 = _make_adapter_dir(str(tmp_path), "a1", out_features=8, in_features=8, seed=10)
        p2 = _make_adapter_dir(str(tmp_path), "a2", out_features=8, in_features=8, seed=20)

        d1 = _reconstruct_deltas(_load_adapter_state_dict(p1), _load_adapter_config(p1))
        d2 = _reconstruct_deltas(_load_adapter_state_dict(p2), _load_adapter_config(p2))

        merged_full = _ties_merge([d1, d2], [0.5, 0.5], density=1.0)
        merged_half = _ties_merge([d1, d2], [0.5, 0.5], density=0.5)

        # Full density should have at least as many non-zero elements.
        for key in merged_full:
            full_nnz = (merged_full[key] != 0).sum()
            half_nnz = (merged_half[key] != 0).sum()
            assert full_nnz >= half_nnz

    def test_ties_single_adapter_degenerates(self, tmp_path):
        """Single adapter should just scale — no TIES overhead."""
        p = _make_adapter_dir(str(tmp_path), "a1", out_features=16, in_features=16, seed=1)
        d = _reconstruct_deltas(_load_adapter_state_dict(p), _load_adapter_config(p))

        merged = _ties_merge([d], [1.0], density=0.5)
        for key in merged:
            expected = d[key] * 1.0
            assert torch.allclose(merged[key], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_same_base_model_ok(self):
        configs = [
            {"base_model_name_or_path": "org/base", "peft_type": "LORA"},
            {"base_model_name_or_path": "org/base", "peft_type": "LORA"},
        ]
        result = _validate_adapters(configs, ["a", "b"])
        assert result == "org/base"

    def test_different_base_models_raises(self):
        configs = [
            {"base_model_name_or_path": "org/base-a", "peft_type": "LORA"},
            {"base_model_name_or_path": "org/base-b", "peft_type": "LORA"},
        ]
        with pytest.raises(ValueError, match="different base models"):
            _validate_adapters(configs, ["a", "b"])

    def test_non_lora_raises(self):
        configs = [
            {"base_model_name_or_path": "org/base", "peft_type": "LORA"},
            {"base_model_name_or_path": "org/base", "peft_type": "IA3"},
        ]
        with pytest.raises(ValueError, match="only LoRA"):
            _validate_adapters(configs, ["a", "b"])


# ---------------------------------------------------------------------------
# End-to-end merge_adapters_into_model test (CPU, synthetic model)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _make_simple_model(self, in_f=16, out_f=16):
        """Build a trivial nn.Module whose state dict keys match the adapter keys."""
        import torch.nn as nn

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module()])
                layer0 = self.model.layers[0]
                layer0.self_attn = nn.Module()
                layer0.self_attn.q_proj = nn.Linear(in_f, out_f, bias=False)
                layer0.self_attn.v_proj = nn.Linear(in_f, out_f, bias=False)

        model = ToyModel()
        # Zero out weights so we can verify the delta was applied.
        for p in model.parameters():
            p.data.zero_()
        return model

    def test_linear_merge_applied(self, tmp_path):
        in_f = out_f = 16
        model = self._make_simple_model(in_f, out_f)

        p1 = _make_adapter_dir(
            str(tmp_path), "a1", out_features=out_f, in_features=in_f, seed=1
        )
        p2 = _make_adapter_dir(
            str(tmp_path), "a2", out_features=out_f, in_features=in_f, seed=2
        )

        result = merge_adapters_into_model(
            model,
            adapter_paths=[p1, p2],
            weights=[0.5, 0.5],
            method="linear",
        )

        # After merge, weights should no longer be all zeros.
        for name, param in result.named_parameters():
            if "q_proj" in name or "v_proj" in name:
                assert not torch.allclose(param, torch.zeros_like(param))

    def test_ties_merge_applied(self, tmp_path):
        in_f = out_f = 16
        model = self._make_simple_model(in_f, out_f)

        p1 = _make_adapter_dir(
            str(tmp_path), "a1", out_features=out_f, in_features=in_f, seed=3
        )
        p2 = _make_adapter_dir(
            str(tmp_path), "a2", out_features=out_f, in_features=in_f, seed=4
        )

        result = merge_adapters_into_model(
            model,
            adapter_paths=[p1, p2],
            weights=[0.7, 0.3],
            method="ties",
            density=0.5,
        )

        for name, param in result.named_parameters():
            if "q_proj" in name or "v_proj" in name:
                assert torch.isfinite(param).all()
