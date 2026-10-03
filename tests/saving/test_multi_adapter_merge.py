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

The merge arithmetic is PEFT's (``add_weighted_adapter`` these CPU-only tests
cover the Unsloth side of the delegation: config validation, adapter I/O, delta
reconstruction, compatibility checks and an end-to-end merge on a tiny model.
"""

import json
import os
import tempfile

import pytest
import torch

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "unsloth.multi_adapter_merge",
    os.path.join(os.path.dirname(__file__), "..", "..", "unsloth", "multi_adapter_merge.py"),
)
_mod = importlib.util.module_from_spec(_spec)
import sys
sys.modules["unsloth.multi_adapter_merge"] = _mod
_spec.loader.exec_module(_mod)

SUPPORTED_METHODS = _mod.SUPPORTED_METHODS
MultiAdapterMergeConfig = _mod.MultiAdapterMergeConfig
_resolve_adapter_path = _mod._resolve_adapter_path
_load_adapter_config = _mod._load_adapter_config
_load_adapter_state_dict = _mod._load_adapter_state_dict
_reconstruct_deltas = _mod._reconstruct_deltas
_validate_adapters = _mod._validate_adapters
_adapter_display_name = _mod._adapter_display_name
merge_adapters_into_model = _mod.merge_adapters_into_model


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
    def test_supported_methods_are_peft_combination_types(self):
        assert SUPPORTED_METHODS == (
            "linear", "svd", "cat", "ties", "dare_ties", "dare_linear",
            "magnitude_prune",
        )

    def test_valid_config(self):
        cfg = MultiAdapterMergeConfig(adapter_paths=["a", "b"], weights=[1.0, 1.0])
        assert cfg.method == "linear"
        assert cfg.weights == [0.5, 0.5]

    def test_weight_normalization(self):
        cfg = MultiAdapterMergeConfig(
            adapter_paths=["a", "b"], weights=[3.0, 1.0], normalize_weights=True
        )
        assert cfg.weights == [0.75, 0.25]

    def test_no_normalization(self):
        cfg = MultiAdapterMergeConfig(
            adapter_paths=["a", "b"], weights=[3.0, 1.0], normalize_weights=False
        )
        assert cfg.weights == [3.0, 1.0]

    def test_empty_adapters_raises(self):
        with pytest.raises(ValueError):
            MultiAdapterMergeConfig(adapter_paths=[], weights=[])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            MultiAdapterMergeConfig(adapter_paths=["a", "b"], weights=[1.0])

    def test_unknown_method_raises(self):
        for method in ("sce", "della", "breadcrumbs", "multislerp", "model_stock", "bogus"):
            with pytest.raises(ValueError, match="Unknown merge method"):
                MultiAdapterMergeConfig(
                    adapter_paths=["a", "b"], weights=[1.0, 1.0], method=method
                )

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="sum to zero"):
            MultiAdapterMergeConfig(adapter_paths=["a", "b"], weights=[1.0, -1.0])

    def test_bad_density_raises(self):
        for density in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="density"):
                MultiAdapterMergeConfig(
                    adapter_paths=["a", "b"], weights=[1.0, 1.0], density=density
                )

    def test_dare_alias(self):
        cfg = MultiAdapterMergeConfig(
            adapter_paths=["a", "b"], weights=[1.0, 1.0], method="dare"
        )
        assert cfg.method == "dare_ties"

    def test_ctm_alias_maps_to_svd(self):
        cfg = MultiAdapterMergeConfig(
            adapter_paths=["a", "b"], weights=[1.0, 1.0], method="ctm"
        )
        assert cfg.method == "svd"

    def test_bad_target_rank_raises(self):
        for target_rank in (0, -4):
            with pytest.raises(ValueError, match="target_rank"):
                MultiAdapterMergeConfig(
                    adapter_paths=["a", "b"], weights=[1.0, 1.0], target_rank=target_rank
                )

# ---------------------------------------------------------------------------
# Adapter I/O tests
# ---------------------------------------------------------------------------

class TestAdapterIO:
    def test_local_adapter_path_is_preserved(self, tmp_path):
        path = _make_adapter_dir(str(tmp_path), "local_adapter")
        assert _resolve_adapter_path(path) == path

    def test_hf_adapter_repo_is_downloaded(self, monkeypatch):
        calls = {}

        def fake_snapshot_download(repo_id, token = None, allow_patterns = None, max_workers = None):
            calls["repo_id"] = repo_id
            calls["token"] = token
            calls["allow_patterns"] = allow_patterns
            return "/tmp/snapshot"

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

        result = _resolve_adapter_path("org/my-lora", hf_token = "hf_xyz")
        assert result == "/tmp/snapshot"
        assert calls["repo_id"] == "org/my-lora"
        assert calls["token"] == "hf_xyz"
        assert any("adapter_config.json" in pattern for pattern in calls["allow_patterns"])

    def test_load_config(self, tmp_path):
        path = _make_adapter_dir(str(tmp_path), "cfg")
        cfg = _load_adapter_config(path)
        assert cfg["r"] == 16
        assert cfg["peft_type"] == "LORA"

    def test_load_config_missing_raises(self, tmp_path):
        empty = tmp_path / "empty_config"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            _load_adapter_config(str(empty))

    def test_load_state_dict(self, tmp_path):
        path = _make_adapter_dir(str(tmp_path), "sd")
        state = _load_adapter_state_dict(path)
        assert any("lora_A" in key for key in state)

    def test_load_state_dict_missing_raises(self, tmp_path):
        empty = tmp_path / "empty_weights"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            _load_adapter_state_dict(str(empty))

    def test_adapter_display_name_formatting(self):
        assert (
            _adapter_display_name("/tmp/runs/finance_lora/checkpoint-500")
            == "finance_lora (checkpoint-500)"
        )
        assert (
            _adapter_display_name({"repo_id": "org/my-lora", "subfolder": "checkpoint-500"})
            == "org/my-lora (checkpoint-500)"
        )
        assert _adapter_display_name({"repo_id": "org/my-lora"}) == "org/my-lora"
        assert _adapter_display_name("/tmp/runs/finance_lora") == "finance_lora"


# ---------------------------------------------------------------------------
# Delta reconstruction tests  (helpers the metrics preflight reuses)
# ---------------------------------------------------------------------------

class TestReconstructDeltas:
    def test_basic_reconstruction(self, tmp_path):
        path = _make_adapter_dir(
            str(tmp_path), "recon", r = 4, lora_alpha = 4,
            out_features = 8, in_features = 8, seed = 3,
        )
        deltas = _reconstruct_deltas(
            _load_adapter_state_dict(path), _load_adapter_config(path)
        )
        assert set(deltas) == {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        }
        for delta in deltas.values():
            assert delta.shape == (8, 8)

    def test_scaling_applied(self, tmp_path):
        path = _make_adapter_dir(
            str(tmp_path), "scale", r = 4, lora_alpha = 8,
            out_features = 8, in_features = 8, seed = 4,
        )
        state = _load_adapter_state_dict(path)
        deltas = _reconstruct_deltas(state, _load_adapter_config(path))
        key = "model.layers.0.self_attn.q_proj.weight"
        A = state["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].float()
        B = state["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"].float()
        assert torch.allclose(deltas[key], (8 / 4) * (B @ A), atol = 1e-5)

    def test_heterogeneous_ranks(self, tmp_path):
        p1 = _make_adapter_dir(str(tmp_path), "r4", r = 4, out_features = 8, in_features = 8, seed = 5)
        p2 = _make_adapter_dir(str(tmp_path), "r8", r = 8, out_features = 8, in_features = 8, seed = 6)
        d1 = _reconstruct_deltas(_load_adapter_state_dict(p1), _load_adapter_config(p1))
        d2 = _reconstruct_deltas(_load_adapter_state_dict(p2), _load_adapter_config(p2))
        assert all(delta.shape == (8, 8) for delta in d1.values())
        assert all(delta.shape == (8, 8) for delta in d2.values())


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_same_base_model_ok(self):
        configs = [
            {"base_model_name_or_path": "base"},
            {"base_model_name_or_path": "base"},
        ]
        assert _validate_adapters(configs, ["a", "b"]) == "base"

    def test_different_base_models_raises(self):
        configs = [
            {"base_model_name_or_path": "base-a"},
            {"base_model_name_or_path": "base-b"},
        ]
        with pytest.raises(ValueError, match = "different base models"):
            _validate_adapters(configs, ["a", "b"])

    def test_non_lora_raises(self):
        configs = [{"base_model_name_or_path": "base", "peft_type": "IA3"}]
        with pytest.raises(ValueError, match = "only LoRA adapters"):
            _validate_adapters(configs, ["a"])

# ---------------------------------------------------------------------------
# End-to-end PEFT delegation (tiny transformer model; skipped without peft)
# ---------------------------------------------------------------------------

def _tiny_model(seed = 0):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 64,
    )
    torch.manual_seed(seed)
    return LlamaForCausalLM(config)


def _save_tiny_adapter_at(path, seed, rank = 8):
    """Save a real PEFT adapter (non-zero lora_B) at *path*."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        _tiny_model(),
        LoraConfig(
            r = rank,
            lora_alpha = rank * 2,
            target_modules = ["q_proj", "v_proj"],
            task_type = "CAUSAL_LM",
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator = generator) * 0.05)
    path.mkdir(parents = True, exist_ok = True)
    model.save_pretrained(str(path))
    return str(path)


def _save_tiny_adapters(tmp_path, ranks = (8, 8)):
    return [
        _save_tiny_adapter_at(tmp_path / f"adapter{idx}", (1, 2)[idx], rank = rank)
        for idx, rank in enumerate(ranks)
    ]


def _base_state():
    return {key: value.detach().clone() for key, value in _tiny_model().state_dict().items()}


def _touched_keys(state):
    return [key for key in state if "q_proj" in key or "v_proj" in key]


def test_single_adapter_linear_is_base_plus_the_weighted_delta(tmp_path):
    """PEFT folds a single adapter exactly: delta == weight * delta."""
    paths = _save_tiny_adapters(tmp_path)
    base_state = _base_state()

    merged = merge_adapters_into_model(
        _tiny_model(),
        adapter_paths = [paths[0]],
        weights = [0.75],
        method = "linear",
        normalize_weights = False,
    )
    deltas = _reconstruct_deltas(
        _load_adapter_state_dict(paths[0]), _load_adapter_config(paths[0])
    )
    merged_state = merged.state_dict()
    key = next(key for key in base_state if key.endswith("q_proj.weight"))
    expected = base_state[key].float() + 0.75 * deltas[key]
    assert torch.allclose(merged_state[key].float(), expected, atol = 1e-4)


def test_cat_is_the_exact_weighted_delta_sum(tmp_path):
    """`cat` concatenates the factors, so the deltas add exactly."""
    paths = _save_tiny_adapters(tmp_path)
    base_state = _base_state()

    merged = merge_adapters_into_model(
        _tiny_model(),
        adapter_paths = paths,
        weights = [0.6, 0.4],
        method = "cat",
        normalize_weights = False,
    )
    d0 = _reconstruct_deltas(
        _load_adapter_state_dict(paths[0]), _load_adapter_config(paths[0])
    )
    d1 = _reconstruct_deltas(
        _load_adapter_state_dict(paths[1]), _load_adapter_config(paths[1])
    )
    merged_state = merged.state_dict()
    key = next(key for key in base_state if key.endswith("v_proj.weight"))
    expected = base_state[key].float() + 0.6 * d0[key] + 0.4 * d1[key]
    assert torch.allclose(merged_state[key].float(), expected, atol = 2e-3)


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_every_method_merges_into_the_base(tmp_path, method):
    paths = _save_tiny_adapters(tmp_path)
    base_state = _base_state()

    merged = merge_adapters_into_model(
        _tiny_model(),
        adapter_paths = paths,
        weights = [0.5, 0.5],
        method = method,
        density = 0.5,
        seed = 7,
    )
    merged_state = merged.state_dict()
    assert any(
        not torch.allclose(merged_state[key].float(), base_state[key].float())
        for key in _touched_keys(base_state)
    ), method


def test_dare_merge_is_deterministic_for_a_seed(tmp_path):
    paths = _save_tiny_adapters(tmp_path)

    def run():
        return merge_adapters_into_model(
            _tiny_model(),
            adapter_paths = paths,
            weights = [0.5, 0.5],
            method = "dare_ties",
            density = 0.5,
            seed = 11,
        )

    first = run().state_dict()
    second = run().state_dict()
    key = next(key for key in first if key.endswith("q_proj.weight"))
    assert torch.equal(first[key], second[key])


def test_report_names_the_adapters(tmp_path):
    p1 = _save_tiny_adapter_at(tmp_path / "finance_lora" / "checkpoint-500", 5)
    p2 = _save_tiny_adapter_at(tmp_path / "math_lora" / "checkpoint-1000", 6)

    messages = []
    merge_adapters_into_model(
        _tiny_model(),
        adapter_paths = [p1, p2],
        weights = [0.6, 0.4],
        method = "linear",
        report_callback = messages.append,
    )
    joined = " ".join(messages)
    assert "finance_lora (checkpoint-500)" in joined
    assert "math_lora (checkpoint-1000)" in joined
    assert "Loading adapter: finance_lora (checkpoint-500)" in joined
    assert "Loading adapter: math_lora (checkpoint-1000)" in joined
    assert "Merge progress: adapter 1 of 2" in joined
    assert "Merge progress: adapter 2 of 2" in joined


def test_mixed_ranks_are_rejected_for_factor_space_methods(tmp_path):
    paths = _save_tiny_adapters(tmp_path, ranks = (8, 16))
    with pytest.raises(ValueError, match = "PEFT could not combine"):
        merge_adapters_into_model(
            _tiny_model(),
            adapter_paths = paths,
            weights = [0.5, 0.5],
            method = "ties",
            density = 0.5,
        )