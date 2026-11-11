# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for GradScaler installation on FORCE_FLOAT32 RL trainers without autocast."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_PY = REPO_ROOT / "unsloth" / "models" / "rl.py"
DECISION_PY = REPO_ROOT / "unsloth" / "models" / "decision.py"
SCALER_PY = REPO_ROOT / "unsloth" / "models" / "_forced_float32_scaler.py"
RL_SRC = RL_PY.read_text(encoding = "utf-8")


def _extract_rl_assign_literal(name: str) -> str:
    for node in ast.walk(ast.parse(RL_SRC)):
        if not isinstance(node, ast.Assign):
            continue
        if name not in [t.id for t in node.targets if isinstance(t, ast.Name)]:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        if isinstance(value, str):
            return value
    raise AssertionError(f"{name} string literal not found in rl.py")


MP_SRC = _extract_rl_assign_literal("mixed_precision")
SCALER_CHECK_SRC = _extract_rl_assign_literal("forced_float32_scaler_check")
ACCELERATOR_CHECK_SRC = _extract_rl_assign_literal("accelerator_check")


@pytest.fixture
def scaler_env(monkeypatch):
    """Load _forced_float32_scaler without requiring full unsloth package or GPU import."""
    loader_utils_stub = types.ModuleType("unsloth.models.loader_utils")
    loader_utils_stub._zero_stage = None
    loader_utils_stub._fsdp_version = 0
    loader_utils_stub.deepspeed_zero_stage = lambda: loader_utils_stub._zero_stage
    loader_utils_stub.fsdp_version_requested = lambda: loader_utils_stub._fsdp_version

    unsloth_pkg = types.ModuleType("unsloth")
    unsloth_pkg.__path__ = [str(REPO_ROOT / "unsloth")]
    models_pkg = types.ModuleType("unsloth.models")
    models_pkg.__path__ = [str(REPO_ROOT / "unsloth" / "models")]

    monkeypatch.setitem(sys.modules, "unsloth", unsloth_pkg)
    monkeypatch.setitem(sys.modules, "unsloth.models", models_pkg)
    monkeypatch.setitem(sys.modules, "unsloth.models.loader_utils", loader_utils_stub)

    spec = importlib.util.spec_from_file_location(
        "unsloth.models._forced_float32_scaler", SCALER_PY
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "unsloth.models._forced_float32_scaler", mod)
    spec.loader.exec_module(mod)
    return mod, loader_utils_stub


class _FakeParam:
    def __init__(self, device_type: str, dtype: torch.dtype = torch.float32, requires_grad: bool = True):
        self.device = torch.device(device_type)
        self.dtype = dtype
        self.requires_grad = requires_grad


class _FakeModel:
    def __init__(
        self,
        devices = ("cuda",),
        forced: bool = True,
        inner = None,
        dtype: torch.dtype = torch.float16,
    ):
        if isinstance(devices, str):
            devices = (devices,)
        self._params = [_FakeParam(d) for d in devices]
        self._unsloth_forced_float32 = forced
        self._unsloth_full_finetuning = False
        self._unsloth_user_float32 = False
        self.config = types.SimpleNamespace(dtype = dtype, torch_dtype = dtype)
        if inner is not None:
            self.model = inner

    def parameters(self):
        return iter(self._params)

    def get_input_embeddings(self):
        return types.SimpleNamespace(weight = types.SimpleNamespace(dtype = self.config.dtype))


class _FakeAccelerator:
    def __init__(self, native_amp = False, mixed_precision = "no", scaler = None, distributed_type = "NO"):
        self.native_amp = native_amp
        self.mixed_precision = mixed_precision
        self.scaler = scaler
        self.distributed_type = distributed_type
        self.clip_calls = []

    def clip_grad_norm_(self, parameters, max_norm, norm_type = 2):
        self.clip_calls.append((list(parameters), max_norm, norm_type))
        return torch.tensor(1.0)


def test_create_scaler_guards(scaler_env):
    mod, loader_utils_stub = scaler_env
    args = types.SimpleNamespace(fp16 = False, bf16 = False, deepspeed = None, fsdp = None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = mod.create_forced_float32_loss_scaler(_FakeModel("cuda", forced = True), args)
    assert scaler is not None

    # CPU-offloaded first parameter followed by CUDA parameter still detects CUDA.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        offload_scaler = mod.create_forced_float32_loss_scaler(
            _FakeModel(("cpu", "cuda"), forced = True), args
        )
    assert offload_scaler is not None

    # CPU-only or empty models never receive a GradScaler.
    assert mod.create_forced_float32_loss_scaler(_FakeModel("cpu", forced = True), args) is None
    assert mod.create_forced_float32_loss_scaler(_FakeModel((), forced = True), args) is None

    # Unforced or explicit fp16/bf16 trainers stay unchanged.
    assert mod.create_forced_float32_loss_scaler(_FakeModel("cuda", forced = False), args) is None
    assert (
        mod.create_forced_float32_loss_scaler(
            _FakeModel("cuda", forced = True),
            types.SimpleNamespace(fp16 = True, bf16 = False, deepspeed = None, fsdp = None),
        )
        is None
    )
    assert (
        mod.create_forced_float32_loss_scaler(
            _FakeModel("cuda", forced = True),
            types.SimpleNamespace(fp16 = False, bf16 = True, deepspeed = None, fsdp = None),
        )
        is None
    )

    # Already-configured accelerator scaler or distributed engines stay untouched.
    existing = object()
    assert (
        mod.create_forced_float32_loss_scaler(
            _FakeModel("cuda", forced = True), args, _FakeAccelerator(scaler = existing)
        )
        is None
    )
    assert (
        mod.create_forced_float32_loss_scaler(
            _FakeModel("cuda", forced = True),
            args,
            _FakeAccelerator(distributed_type = "FSDP"),
        )
        is None
    )
    assert (
        mod.create_forced_float32_loss_scaler(
            _FakeModel("cuda", forced = True),
            args,
            trainer = types.SimpleNamespace(is_deepspeed_enabled = True),
        )
        is None
    )
    assert (
        mod.create_forced_float32_loss_scaler(
            _FakeModel("cuda", forced = True),
            args,
            trainer = types.SimpleNamespace(_unsloth_tensor_parallel = True),
        )
        is None
    )
    assert (
        mod.create_forced_float32_loss_scaler(
            _FakeModel("cuda", forced = True),
            args,
            trainer = types.SimpleNamespace(_unsloth_pipeline_parallel = object()),
        )
        is None
    )

    loader_utils_stub._zero_stage = 2
    assert mod.create_forced_float32_loss_scaler(_FakeModel("cuda", forced = True), args) is None
    loader_utils_stub._zero_stage = None
    loader_utils_stub._fsdp_version = 2
    assert mod.create_forced_float32_loss_scaler(_FakeModel("cuda", forced = True), args) is None


def test_torch_amp_grad_scaler_legacy_fallback(scaler_env, monkeypatch):
    mod, _ = scaler_env
    args = types.SimpleNamespace(fp16 = False, bf16 = False, deepspeed = None, fsdp = None)
    monkeypatch.delattr(torch.amp, "GradScaler", raising = False)
    calls = []
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", lambda: calls.append(1) or "legacy_cuda")
    assert mod.create_forced_float32_loss_scaler(_FakeModel("cuda", forced = True), args) == "legacy_cuda"
    assert calls == [1]
    assert mod.create_forced_float32_loss_scaler(_FakeModel("xpu", forced = True), args) is None


def test_install_scaler_unscales_once_and_is_idempotent(scaler_env):
    mod, _ = scaler_env
    events = []

    class _RecordingScaler:
        def unscale_(self, opt):
            events.append(("unscale", opt))

    scaler = _RecordingScaler()
    raw_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr = 0.1)

    class _SubclassedOptWrapper(torch.optim.Optimizer):
        def __init__(self, inner):
            self.optimizer = inner

    wrapped_opt = _SubclassedOptWrapper(raw_opt)
    accel = _FakeAccelerator(native_amp = False, mixed_precision = "no")

    mod.install_forced_float32_loss_scaler(accel, scaler, lambda: wrapped_opt)
    # Installing twice must not double-wrap clip_grad_norm_.
    mod.install_forced_float32_loss_scaler(accel, scaler, lambda: wrapped_opt)
    assert accel.scaler is scaler

    accel.clip_grad_norm_([], 1.0)
    assert events == [("unscale", raw_opt)]
    assert len(accel.clip_calls) == 1

    # Native Accelerate fp16 keeps Accelerate's own unscaling before clip_grad_norm_.
    native_accel = _FakeAccelerator(native_amp = True, mixed_precision = "fp16")
    mod.install_forced_float32_loss_scaler(native_accel, scaler, lambda: raw_opt)
    assert native_accel.scaler is scaler
    native_accel.clip_grad_norm_([], 1.0)
    assert len(events) == 1  # No extra unscale_ call.


def test_rl_post_installs_scaler_before_model_propagation(scaler_env, monkeypatch):
    """Execute rl.py's mixed_precision + post-init scaler blocks end-to-end across T4 and L4."""
    mod, _ = scaler_env

    # Ensure forced_float32_scaler_check executes before accelerator_check copies self.accelerator.scaler.
    assert RL_SRC.index("RLTrainer_post += forced_float32_scaler_check") < RL_SRC.index(
        "RLTrainer_post += accelerator_check"
    )

    device_type = types.ModuleType("unsloth_zoo.device_type")
    utils = types.ModuleType("unsloth_zoo.utils")
    utils._get_dtype = lambda dt: dt if isinstance(dt, torch.dtype) else getattr(torch, str(dt))
    zoo_pkg = types.ModuleType("unsloth_zoo")
    zoo_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo_pkg)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.device_type", device_type)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.utils", utils)

    def _run_rl_init_blocks(forced: bool, bf16_supported: bool, fp16_arg: bool, bf16_arg: bool):
        device_type.device_is_bf16_supported = lambda: bf16_supported
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)
        env = {"UNSLOTH_FORCE_FLOAT32": "0", "UNSLOTH_ENABLE_FULL_FINETUNING": "0"}
        args = types.SimpleNamespace(
            fp16 = fp16_arg, bf16 = bf16_arg, mixed_precision = "no", deepspeed = None, fsdp = None
        )
        model_dtype = torch.bfloat16 if bf16_arg else torch.float16
        inner_model = _FakeModel("cuda", forced = forced, dtype = model_dtype)
        model = _FakeModel("cuda", forced = forced, inner = inner_model, dtype = model_dtype)
        accel = _FakeAccelerator(
            native_amp = False,
            mixed_precision = "fp16" if (fp16_arg and not forced) else "no",
            scaler = "native_fp16_scaler" if (fp16_arg and not forced) else None,
        )
        trainer = types.SimpleNamespace(args = args, model = model, accelerator = accel, optimizer = None)

        scope = {
            "torch": torch,
            "os": types.SimpleNamespace(environ = env),
            "args": args,
            "model": model,
            "self": trainer,
            "print": lambda *a, **k: None,
        }
        exec(MP_SRC, scope)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(SCALER_CHECK_SRC + "\n" + ACCELERATOR_CHECK_SRC, scope)
        return scope["force_float32"], trainer, model, inner_model, env

    # T4 with a FORCE_FLOAT32 model: mixed precision stays 'no', standalone GradScaler propagates to model.
    force_float32, trainer, model, inner, env = _run_rl_init_blocks(
        forced = True, bf16_supported = False, fp16_arg = True, bf16_arg = False
    )
    assert force_float32 is True
    assert trainer.args.fp16 is False and trainer.args.bf16 is False
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"
    assert isinstance(trainer.accelerator.scaler, torch.amp.GradScaler)
    assert model.accelerator_scaler is trainer.accelerator.scaler
    assert inner.accelerator_scaler is trainer.accelerator.scaler

    # L4 with a FORCE_FLOAT32 model (or loaded natively in bf16): force_float32 is False, no extra scaler.
    force_float32_l4, trainer_l4, model_l4, _, _ = _run_rl_init_blocks(
        forced = False, bf16_supported = True, fp16_arg = False, bf16_arg = True
    )
    assert force_float32_l4 is False
    assert trainer_l4.accelerator.scaler is None
    assert model_l4.accelerator_scaler is None

    # T4 non-forced model: keeps Accelerate's native fp16 scaler untouched.
    force_float32_std, trainer_std, model_std, _, _ = _run_rl_init_blocks(
        forced = False, bf16_supported = False, fp16_arg = True, bf16_arg = False
    )
    assert force_float32_std is False
    assert trainer_std.accelerator.scaler == "native_fp16_scaler"
    assert model_std.accelerator_scaler == "native_fp16_scaler"


def test_decision_delegates_to_shared_forced_float32_scaler():
    """DecisionTrainer's helpers reuse _forced_float32_scaler instead of duplicating logic."""
    decision_src = DECISION_PY.read_text(encoding = "utf-8")
    tree = ast.parse(decision_src)
    funcs = {
        node.name: ast.get_source_segment(decision_src, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_forced_float32_scaler" in funcs["_install_clef_loss_scaler"]
    assert "_forced_float32_scaler" in funcs["_clef_loss_scaler"]
    assert "torch.amp.GradScaler" not in funcs["_clef_loss_scaler"]
    assert "clip_grad_norm_" not in funcs["_install_clef_loss_scaler"]


def test_float16_frozen_chain_underflows_unscaled_but_converges_with_grad_scaler(scaler_env):
    """Reproduce the #14439/#14482 gradient underflow on float16 weights and verify GradScaler recovery."""
    mod, _ = scaler_env
    torch.manual_seed(0)

    # Simulate a float32 LoRA adapter whose output feeds INTO downstream float16 layers (like LoRA on
    # early attention/MLP blocks): small per-token loss gradients underflow to 0.0 unscaled in
    # float16 backward, while GradScaler scaling preserves gradient cosine > 0.99 vs float32.
    base_layers = [(torch.randn(16, 16) / 4.0).half() for _ in range(4)]
    x = torch.randn(8, 16, dtype = torch.float32)
    target = torch.zeros(8, 16, dtype = torch.float32)

    def _forward(adapter_a, adapter_b, float32_only = False):
        lora = torch.nn.functional.linear(torch.nn.functional.linear(x, adapter_a), adapter_b)
        h = x + lora
        if float32_only:
            for w in base_layers:
                h = torch.nn.functional.linear(h, w.float()) * 0.25
        else:
            h = h.half()
            for w in base_layers:
                h = torch.nn.functional.linear(h, w) * 0.25
            h = h.float()
        return torch.nn.functional.mse_loss(h, target) * 1e-2

    # Exact float32 reference gradient for comparison.
    ref_a = torch.nn.Parameter(torch.randn(4, 16, dtype = torch.float32) * 0.1)
    ref_b = torch.nn.Parameter(torch.randn(16, 4, dtype = torch.float32) * 0.1)
    _forward(ref_a, ref_b, float32_only = True).backward()

    # Unscaled backward through the float16 chain loses gradient signal to underflow/rounding.
    a_unscaled = torch.nn.Parameter(ref_a.detach().clone())
    b_unscaled = torch.nn.Parameter(ref_b.detach().clone())
    _forward(a_unscaled, b_unscaled).backward()
    assert a_unscaled.grad.abs().max() == 0.0 or (
        torch.nn.functional.cosine_similarity(
            a_unscaled.grad.flatten(), ref_a.grad.flatten(), dim = 0
        )
        < 0.5
    )

    # GradScaler-protected step via our helper keeps adapter gradients accurate.
    a_scaled = torch.nn.Parameter(ref_a.detach().clone())
    b_scaled = torch.nn.Parameter(ref_b.detach().clone())
    opt = torch.optim.AdamW([a_scaled, b_scaled], lr = 1e-2)
    accel = _FakeAccelerator()
    scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0)
    mod.install_forced_float32_loss_scaler(accel, scaler, lambda: opt)

    opt.zero_grad()
    loss_scaled = _forward(a_scaled, b_scaled)
    scaler.scale(loss_scaled).backward()
    accel.clip_grad_norm_([a_scaled, b_scaled], 1.0)
    scaler.step(opt)
    scaler.update()

    # After unscaling in clip_grad_norm_, the adapter gradient must stay finite and match float32 closely.
    cos = torch.nn.functional.cosine_similarity(
        a_scaled.grad.flatten(), ref_a.grad.flatten(), dim = 0
    )
    assert torch.isfinite(a_scaled.grad).all()
    assert a_scaled.grad.abs().max() > 0.0
    assert cos > 0.99, f"Expected scaled gradient cosine > 0.99, got {cos.item():.4f}"


def test_attach_scaler_initializes_preprepared_accelerated_optimizer(scaler_env):
    """If an AcceleratedOptimizer was created before RLTrainer_post, step hooks are initialized."""
    accelerate = pytest.importorskip("accelerate")
    mod, _ = scaler_env

    param = torch.nn.Parameter(torch.tensor([1.0], dtype = torch.float32))
    raw_opt = torch.optim.SGD([param], lr = 0.1)
    # Build Accelerate's wrapper without a scaler (the pre-fix mixed_precision='no' state),
    # stubbing internal state attrs instead of calling Accelerator() so CPU-only tests stay self-contained.
    acc_opt = accelerate.optimizer.AcceleratedOptimizer.__new__(accelerate.optimizer.AcceleratedOptimizer)
    acc_opt.optimizer = raw_opt
    acc_opt.scaler = None
    acc_opt.gradient_state = types.SimpleNamespace(sync_gradients = True, is_xla_gradients_synced = True)
    acc_opt.accelerator_state = types.SimpleNamespace(distributed_type = accelerate.utils.DistributedType.NO)
    acc_opt._is_overflow = False

    scaler = torch.amp.GradScaler("cpu", init_scale = 8.0)
    mod._attach_scaler_to_optimizer(acc_opt, scaler)
    assert acc_opt.scaler is scaler
    assert hasattr(acc_opt, "_optimizer_patched_step_method")

    loss = (param * 2.0).sum()
    scaler.scale(loss).backward()
    acc_opt.step()
    assert not acc_opt.step_was_skipped
    assert param.item() < 1.0
