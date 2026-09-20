# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Merge methods mergekit implements (linear/ties/dare*/task_arithmetic/della*/model_stock)
must route to the mergekit child-process engine when mergekit is installed: the export
checkpoint loader runs ``merge_adapters_via_mergekit`` and loads the merged checkpoint
back. Without mergekit, the dual-capable methods fall back to the in-memory merger while
mergekit-only methods (which ``merge_adapters_into_model`` would reject with
``Unsupported merge method``) get a clear error instead."""

import importlib.machinery
import sys
import types
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Reuse the absolute-paths stub harness: loads core/export/export.py without torch/unsloth.
from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)

MERGEKIT_ONLY = ("task_arithmetic", "della", "della_ties", "della_linear", "model_stock")


class _MergeKitUnavailableError(RuntimeError):
    pass


def _install_merge_stubs(monkeypatch, *, resolve_engine):
    """Stub unsloth.mergekit_bridge / unsloth.multi_adapter_merge with a recording fake."""

    bridge = types.ModuleType("unsloth.mergekit_bridge")
    bridge.MergeKitUnavailableError = _MergeKitUnavailableError
    bridge.normalize_method = lambda method: str(method).lower().replace("-", "_")
    bridge.resolve_engine = resolve_engine
    bridge.make_merge_output_dir = lambda: str(Path("/tmp") / "unsloth_mergekit_test")
    calls = []

    def _record_merge(**kwargs):
        calls.append(kwargs)
        return kwargs["output_dir"]

    bridge.merge_adapters_via_mergekit = _record_merge
    bridge.__spec__ = importlib.machinery.ModuleSpec("unsloth.mergekit_bridge", loader = None)
    monkeypatch.setitem(sys.modules, "unsloth.mergekit_bridge", bridge)

    multi = types.ModuleType("unsloth.multi_adapter_merge")
    multi.MERGEKIT_ONLY_METHODS = MERGEKIT_ONLY

    def _boom(*args, **kwargs):  # the in-memory engine must never see these methods
        raise AssertionError("merge_adapters_into_model must not run for mergekit-only methods")

    multi.merge_adapters_into_model = _boom
    multi.__spec__ = importlib.machinery.ModuleSpec("unsloth.multi_adapter_merge", loader = None)
    monkeypatch.setitem(sys.modules, "unsloth.multi_adapter_merge", multi)
    return calls


def _export_mod(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    # The mergekit branch imports the identifier resolver from utils.models at call
    # time; default it to "unknown base" so tests opt into a resolution explicitly.
    sys.modules["utils.models"].get_base_model_from_lora_identifier = (
        lambda *args, **kwargs: None
    )
    mod = _load_module(
        "test_core_export_backend_mergekit_merge", "core/export/export.py", monkeypatch
    )
    monkeypatch.setattr(mod, "_IS_MLX", False)
    # The stub harness sets FastVisionModel = object, which is callable, so the loader
    # branch must be forced to deterministic fakes instead.
    monkeypatch.setattr(mod, "_hf_offline", lambda: True)
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {})
    # The stub env has no peft, so the module-level import failed; provide dummies.
    monkeypatch.setattr(mod, "PeftModel", type("PeftModel", (), {}), raising = False)
    monkeypatch.setattr(mod, "PeftModelForCausalLM", type("PeftModelForCausalLM", (), {}), raising = False)
    monkeypatch.setattr(
        mod,
        "FastVisionModel",
        types.SimpleNamespace(
            from_pretrained = lambda **kwargs: (_ for _ in ()).throw(
                AssertionError(f"unexpected model load: {kwargs}")
            )
        ),
    )
    return mod


def _make_backend(mod, monkeypatch, tmp_path):
    """Adapter checkpoint at tmp_path whose base resolves to 'base/model'."""

    checkpoint = tmp_path / "checkpoint-704"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(mod, "get_base_model_from_lora", lambda path: "base/model")

    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    backend._audio_type = None
    backend.is_vision = False
    return backend, checkpoint


def test_mergekit_only_methods_route_to_the_mergekit_engine(monkeypatch, tmp_path):
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "mergekit")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)

    loaded = []
    monkeypatch.setattr(
        mod,
        "FastLanguageModel",
        types.SimpleNamespace(
            from_pretrained = lambda **kwargs: (loaded.append(kwargs) or (object(), object()))
        ),
    )

    for method in MERGEKIT_ONLY:
        # Model Stock needs 3+ adapters (see test_model_stock_with_two_adapters_is_rejected);
        # the rest merge fine with two.
        adapters = ["a", "b", "c"] if method == "model_stock" else ["a", "b"]
        merge = {"adapter_paths": adapters, "method": method}
        ok, _msg = backend.load_checkpoint(str(checkpoint), merge_adapters = merge)
        assert ok, f"{method}: {_msg}"

    assert [call["method"] for call in calls] == list(MERGEKIT_ONLY)
    for call in calls:
        assert call["base_model"] == "base/model"
        assert call["adapters"] == (["a", "b", "c"] if call["method"] == "model_stock" else ["a", "b"])
    # Every merge loaded the merged checkpoint back instead of the raw adapter.
    assert all(kwargs["model_name"] == calls[0]["output_dir"] for kwargs in loaded)
    assert len(loaded) == len(MERGEKIT_ONLY)


def test_mergekit_only_method_without_mergekit_gets_a_clear_error(monkeypatch, tmp_path):
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "legacy")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)

    ok, msg = backend.load_checkpoint(
        str(checkpoint), merge_adapters = {"adapter_paths": ["a", "b"], "method": "model_stock"}
    )
    assert not ok
    assert "mergekit" in msg
    assert "model_stock" in msg
    assert calls == []


def test_mergekit_only_method_requires_a_base_model(monkeypatch, tmp_path):
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "mergekit")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "get_base_model_from_lora", lambda path: None)

    ok, msg = backend.load_checkpoint(
        str(checkpoint), merge_adapters = {"adapter_paths": ["a", "b"], "method": "model_stock"}
    )
    assert not ok
    assert "base" in msg
    assert calls == []


DUAL_CAPABLE = ("linear", "ties", "dare_ties", "dare_linear")


def _install_in_memory_recorder(mod, monkeypatch):
    merged = []

    def _fake_load(**kwargs):
        return object(), object()

    monkeypatch.setattr(
        mod,
        "FastLanguageModel",
        types.SimpleNamespace(from_pretrained = _fake_load),
    )
    sys.modules["unsloth.multi_adapter_merge"].merge_adapters_into_model = (
        lambda model, **kwargs: (merged.append(kwargs) or model)
    )
    return merged


def test_dual_capable_methods_use_mergekit_when_available(monkeypatch, tmp_path):
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "mergekit")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)

    loaded = []
    monkeypatch.setattr(
        mod,
        "FastLanguageModel",
        types.SimpleNamespace(
            from_pretrained = lambda **kwargs: (loaded.append(kwargs) or (object(), object()))
        ),
    )

    for method in DUAL_CAPABLE:
        merge = {"adapter_paths": ["a", "b"], "method": method}
        ok, _msg = backend.load_checkpoint(str(checkpoint), merge_adapters = merge)
        assert ok, f"{method}: {_msg}"

    assert [call["method"] for call in calls] == list(DUAL_CAPABLE)
    assert all(kwargs["model_name"] == calls[0]["output_dir"] for kwargs in loaded)
    assert len(loaded) == len(DUAL_CAPABLE)


def test_dual_capable_methods_fall_back_to_in_memory_without_mergekit(monkeypatch, tmp_path):
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "legacy")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)
    merged = _install_in_memory_recorder(mod, monkeypatch)

    for method in DUAL_CAPABLE:
        ok, msg = backend.load_checkpoint(
            str(checkpoint), merge_adapters = {"adapter_paths": ["a", "b"], "method": method}
        )
        assert ok, f"{method}: {msg}"

    assert calls == []
    assert [kwargs["method"] for kwargs in merged] == list(DUAL_CAPABLE)


def test_legacy_only_methods_stay_on_the_in_memory_engine(monkeypatch, tmp_path):
    # magnitude_prune/ctm/cat have no mergekit equivalent, so the in-house engine
    # runs even when mergekit is installed.
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "legacy")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)
    merged = _install_in_memory_recorder(mod, monkeypatch)

    ok, msg = backend.load_checkpoint(
        str(checkpoint), merge_adapters = {"adapter_paths": ["a", "b"], "method": "ctm"}
    )
    assert ok, msg
    assert calls == []
    assert [kwargs["method"] for kwargs in merged] == ["ctm"]



def test_single_adapter_merge_is_rejected(monkeypatch, tmp_path):
    # One adapter + the checkpoint's own merge is a plain adapter load, not a
    # merge; the picker hides multi-merge until 2+ adapters and the backend
    # guards the same floor.
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "legacy")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)
    merged = _install_in_memory_recorder(mod, monkeypatch)

    ok, msg = backend.load_checkpoint(
        str(checkpoint), merge_adapters = {"adapter_paths": ["a"], "method": "linear"}
    )
    assert not ok
    assert "at least 2 adapters" in msg
    assert calls == []
    assert merged == []



def test_model_stock_with_two_adapters_is_rejected(monkeypatch, tmp_path):
    # mergekit's stock estimator needs three models; the picker hides it below
    # 3 adapters and the backend guards the same floor.
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "mergekit")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)
    monkeypatch.setattr(
        mod,
        "FastLanguageModel",
        types.SimpleNamespace(from_pretrained = lambda **kwargs: (object(), object())),
    )

    ok, msg = backend.load_checkpoint(
        str(checkpoint), merge_adapters = {"adapter_paths": ["a", "b"], "method": "model_stock"}
    )
    assert not ok
    assert "at least 3 adapters" in msg
    assert calls == []


def test_hub_adapter_repo_resolves_base_from_remote_config(monkeypatch, tmp_path):
    # A Hub repo id has no local adapter_config.json, so the mergekit branch must
    # fall back to the identifier resolver (the security gate's remote reader) --
    # otherwise every Hub adapter + mergekit method failed with "base model could
    # not be determined".
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "mergekit")
    mod = _export_mod(monkeypatch)

    seen = {}
    monkeypatch.setattr(
        sys.modules["utils.models"],
        "get_base_model_from_lora_identifier",
        lambda identifier, hf_token = None: seen.setdefault(
            "args", (identifier, hf_token)
        )
        and None
        or "base/model",
    )
    monkeypatch.setattr(
        mod,
        "FastLanguageModel",
        types.SimpleNamespace(from_pretrained = lambda **kwargs: (object(), object())),
    )

    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    backend._audio_type = None
    backend.is_vision = False

    ok, msg = backend.load_checkpoint(
        "org/adapter-repo",
        merge_adapters = {"adapter_paths": ["a", "b"], "method": "della_linear"},
    )
    assert ok, msg
    assert seen["args"] == ("org/adapter-repo", None)
    assert calls[0]["base_model"] == "base/model"


def test_cached_snapshot_resolves_the_base_before_the_hub(monkeypatch, tmp_path):
    # The load itself snapshot-downloads the adapter, so the cache almost always
    # has adapter_config.json: it wins over the remote reader and works offline.
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "mergekit")
    mod = _export_mod(monkeypatch)

    remote_calls = []
    monkeypatch.setattr(
        sys.modules["utils.models"],
        "get_base_model_from_lora_identifier",
        lambda *a, **k: remote_calls.append(1) or "base/model",
    )
    monkeypatch.setattr(
        mod, "_resolve_merge_base_from_cache", lambda repo_id: "cached/base"
    )
    monkeypatch.setattr(
        mod,
        "FastLanguageModel",
        types.SimpleNamespace(from_pretrained = lambda **kwargs: (object(), object())),
    )

    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    backend._audio_type = None
    backend.is_vision = False

    ok, msg = backend.load_checkpoint(
        "org/adapter-repo",
        merge_adapters = {"adapter_paths": ["a", "b"], "method": "della"},
    )
    assert ok, msg
    assert remote_calls == []
    assert calls[0]["base_model"] == "cached/base"


def test_merged_model_checkpoint_gets_a_clear_merge_base_error(monkeypatch, tmp_path):
    # A merged (non-adapter) checkpoint has no adapter_config.json anywhere; the
    # error must say so instead of the vague "could not be determined".
    _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "mergekit")
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_resolve_merge_base_from_cache", lambda repo_id: None)
    monkeypatch.setattr(
        sys.modules["utils.models"],
        "get_base_model_from_lora_identifier",
        lambda *a, **k: None,
    )

    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    backend._audio_type = None
    backend.is_vision = False

    ok, msg = backend.load_checkpoint(
        "org/merged-model",
        merge_adapters = {"adapter_paths": ["a", "b"], "method": "della"},
    )
    assert not ok
    assert "org/merged-model" in msg
    assert "merged model" in msg


def test_merge_device_reaches_the_mergekit_engine(monkeypatch, tmp_path):
    # The Export page's CPU/GPU toggle must reach merge_adapters_via_mergekit; a
    # missing/empty device stays on the safe CPU default.
    calls = _install_merge_stubs(monkeypatch, resolve_engine = lambda method: "mergekit")
    mod = _export_mod(monkeypatch)
    backend, checkpoint = _make_backend(mod, monkeypatch, tmp_path)
    monkeypatch.setattr(
        mod,
        "FastLanguageModel",
        types.SimpleNamespace(from_pretrained = lambda **kwargs: (object(), object())),
    )

    ok, msg = backend.load_checkpoint(
        str(checkpoint),
        merge_adapters = {"adapter_paths": ["a", "b"], "method": "linear", "device": "cuda"},
    )
    assert ok, msg
    ok, msg = backend.load_checkpoint(
        str(checkpoint),
        merge_adapters = {"adapter_paths": ["a", "b"], "method": "linear"},
    )
    assert ok, msg
    assert [call["device"] for call in calls] == ["cuda", "cpu"]

