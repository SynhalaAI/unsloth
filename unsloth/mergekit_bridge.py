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

"""mergekit merge-engine bridge for multi-adapter LoRA merging.

This module is the *isolated* seam between Unsloth and `mergekit
<https://github.com/arcee-ai/mergekit>`_ (LGPL-3.0, so it stays an optional
runtime dependency and is never vendored).  Nothing here imports torch,
transformers, peft or mergekit at module scope: the whole module is testable in
a plain Python interpreter, and mergekit only ever runs in a **child process**.

Why a child process
-------------------
mergekit declares ``transformers>=5.0,<6.0`` while Unsloth supports
``transformers>=4.51.3 ... <=5.5.0`` on a torch 2.4-2.12 matrix, and importing
it into the parent would also drag the *parent's* torch into mergekit's
``peft.PeftModel.from_pretrained`` fold.  The child process therefore selects
the engine's own ``sys.path`` and bootstraps with
``python -I -c "runpy.run_path(...)"``.  The bootstrap script is written next to
the merge config, so the parent's sys.path and ``PYTHONPATH`` are deliberately
*not* forwarded.

Method policy
-------------
``linear``, ``ties``, ``dare_ties``, ``dare_linear``, ``task_arithmetic``,
``della``/``della_ties`` (alias), ``della_linear`` and ``model_stock`` are
delegated to mergekit's own registered implementations.  ``della_ties`` is
not a separate mergekit method: mergekit's ``della`` already pairs DELLA
magnitude pruning with TIES sign election, so the alias targets ``della``.
``magnitude_prune``, ``ctm`` and ``cat`` have no mergekit equivalent
(``della`` is *not* equivalent to ``magnitude_prune``: it rescales outliers
after pruning) and exist only in ``unsloth/multi_adapter_merge.py``, so they
keep the legacy engine and this module reports ``LEGACY_ONLY_METHODS`` for
them.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MERGEKIT_METHOD_MAP",
    "LEGACY_ONLY_METHODS",
    "MergeKitUnavailableError",
    "MergeKitMergeError",
    "normalize_method",
    "mergekit_target_method",
    "resolve_engine",
    "resolve_mergekit_python",
    "mergekit_available",
    "build_merge_config",
    "write_merge_config",
    "dump_merge_config_yaml",
    "normalize_out_dtype",
    "make_merge_output_dir",
    "run_mergekit_merge",
    "merge_adapters_via_mergekit",
    "MERGE_ENGINE_ENV",
    "MERGEKIT_PYTHON_ENV",
    "MERGEKIT_SHADOW_ENV",
    "MERGEKIT_CONFIG_ENV",
]

# --------------------------------------------------------------------------
# Method mapping
# --------------------------------------------------------------------------

#: Unsloth merge method -> mergekit registered method name.
#: ``della_ties`` is an alias: mergekit registers the TIES-consensus DELLA
#: variant as ``della``, so both names target it.
MERGEKIT_METHOD_MAP: Dict[str, str] = {
    "linear": "linear",
    "ties": "ties",
    "dare_ties": "dare_ties",
    "dare_linear": "dare_linear",
    "task_arithmetic": "task_arithmetic",
    "della": "della",
    "della_ties": "della",
    "della_linear": "della_linear",
    "model_stock": "model_stock",
}

#: Methods that only the in-house engine implements.
LEGACY_ONLY_METHODS: Tuple[str, ...] = ("magnitude_prune", "ctm", "cat")

MERGE_ENGINE_ENV = "UNSLOTH_MERGE_ENGINE"
MERGEKIT_PYTHON_ENV = "UNSLOTH_MERGEKIT_PYTHON"
MERGEKIT_SHADOW_ENV = "UNSLOTH_MERGEKIT_SHADOW"
MERGEKIT_CONFIG_ENV = "UNSLOTH_MERGEKIT_CONFIG"

_ENGINES = ("auto", "mergekit", "legacy")


class MergeKitUnavailableError(RuntimeError):
    """mergekit was requested but no usable runtime could be found."""


class MergeKitMergeError(RuntimeError):
    """The mergekit child process failed to produce a merged model."""


def normalize_method(method: str) -> str:
    """Lower-case *method* and apply the core module's DARE/SVD aliases."""
    method = (method or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "dare": "dare_ties",
        "della_ties": "della",
        "task-arithmetic": "task_arithmetic",
        "task_arith": "task_arithmetic",
        "model-stock": "model_stock",
        "modelstock": "model_stock",
        "svd": "ctm",
        "magnitude": "magnitude_prune",
        "mag-prune": "magnitude_prune",
        "concatenate": "cat",
        "concat": "cat",
    }
    canonical = aliases.get(method, method)
    if canonical in MERGEKIT_METHOD_MAP:
        return canonical
    if canonical in LEGACY_ONLY_METHODS:
        return canonical
    return method


def mergekit_target_method(method: str) -> Optional[str]:
    """Return the mergekit method for *method*, or ``None`` if legacy-only."""
    return MERGEKIT_METHOD_MAP.get(normalize_method(method))


def resolve_engine(method: str, engine: Optional[str] = None) -> str:
    """Resolve ``auto``/``mergekit``/``legacy`` into a concrete engine name.

    ``auto`` picks mergekit only for a method it implements; ``mergekit`` is
    honoured only for those same methods, so an unsupported request falls back
    instead of failing after a long model load.  Anything unrecognised is
    treated as ``auto`` rather than raising, mirroring how the rest of Unsloth
    treats optional knobs.
    """

    requested = (engine or os.environ.get(MERGE_ENGINE_ENV) or "auto").strip().lower()
    if requested not in _ENGINES:
        requested = "auto"
    if requested == "legacy":
        return "legacy"
    if mergekit_target_method(method) is None:
        return "legacy"
    if requested == "mergekit":
        return "mergekit"
    return "mergekit" if mergekit_available() else "legacy"


# --------------------------------------------------------------------------
# Runtime discovery
# --------------------------------------------------------------------------

def _python_candidates() -> Sequence[str]:
    override = os.environ.get(MERGEKIT_PYTHON_ENV, "").strip()
    candidates: List[str] = []
    if override:
        candidates.append(override)
    shadow = os.environ.get(MERGEKIT_SHADOW_ENV, "").strip()
    if shadow:
        candidates.append(str(Path(shadow) / "Scripts" / "python.exe"))
        candidates.append(str(Path(shadow) / "bin" / "python"))
    candidates.append(sys.executable)
    candidates.append("python")
    return candidates


_PROBE_TEMPLATE = (
    "import importlib.util,sys\n"
    "shadow = {shadow!r}\n"
    "if shadow:\n"
    "    sys.path.insert(0, shadow)\n"
    "sys.exit(0 if importlib.util.find_spec('mergekit') else 1)\n"
)


def _shadow_dir() -> str:
    return os.environ.get(MERGEKIT_SHADOW_ENV, "").strip()


def resolve_mergekit_python() -> Optional[str]:
    """Return the configured interpreter if mergekit is importable from it.

    The probe deliberately does **not** import mergekit; ``find_spec`` is enough
    to reject an interpreter that has no mergekit, without paying the import
    cost (and the transformers >=5.0 pull-in) on every call.

    ``UNSLOTH_MERGEKIT_SHADOW`` (a directory populated by a ``--target``
    install, the same shape Studio uses for its llm-compressor shadow) is pushed
    onto ``sys.path`` for the probe and again inside the child process, so an
    engine can be pinned without installing it into the main environment.
    """

    seen = set()
    for candidate in _python_candidates():
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate != "python" and not os.path.isfile(candidate):
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", _PROBE_TEMPLATE.format(shadow = _shadow_dir())],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 20,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


def mergekit_available() -> bool:
    """True when a usable mergekit interpreter is configured and reachable."""
    return resolve_mergekit_python() is not None


_MERGEKIT_DTYPES = ("float16", "bfloat16", "float32", "float64")


def normalize_out_dtype(value: Any) -> Optional[str]:
    """Map a torch dtype or dtype string to mergekit's dtype name.

    Returns ``None`` for anything unrecognised (including ``None``) so the merge
    keeps the base checkpoint's dtype instead of writing a wrong one - mergekit's
    ``dtype_from_name`` does not understand ``str(torch.float16)``.
    """

    if value is None:
        return None
    name = str(value).rsplit(".", 1)[-1].strip().lower()
    return name if name in _MERGEKIT_DTYPES else None


def make_merge_output_dir(prefix: str = "unsloth_mergekit_") -> str:
    """Create a temp dir for a merged checkpoint, removed at interpreter exit.

    The merged checkpoint has to outlive the call (the caller loads it back) and
    on Windows a directory still mapped by the loaded model cannot be unlinked
    eagerly, so removal is deferred to shutdown instead of leaking forever.
    """

    path = tempfile.mkdtemp(prefix = prefix)
    atexit.register(shutil.rmtree, path, True)
    return path


# --------------------------------------------------------------------------
# Config building
# --------------------------------------------------------------------------

#: Loading an adapter against a base it was not trained on is a broken merge
#: rather than a surprising result, so the base is checked up front.
_BASE_MISMATCH = (
    "Unsloth: Adapter '{adapter}' was trained against base model '{adapter_base}' "
    "but the merge base is '{base}'. Merging adapters from different base models "
    "is not supported."
)


def _adapter_base_model(adapter_path: str) -> Optional[str]:
    """Best-effort read of ``base_model_name_or_path`` from adapter_config.json."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding = "utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None
    base = config.get("base_model_name_or_path")
    return base if isinstance(base, str) and base.strip() else None


def _short_name(reference: str) -> str:
    """Last path component of a model id/path, for tolerant matching."""
    return reference.rstrip("/").split("/")[-1].split("\\")[-1]


def _check_adapter_base(adapter_path: str, base_model: str, strict: bool) -> None:
    adapter_base = _adapter_base_model(adapter_path)
    if adapter_base is None:
        if strict:
            raise ValueError(
                f"Unsloth: Cannot verify the base model for adapter '{adapter_path}': "
                "adapter_config.json is missing or unreadable."
            )
        return
    # mergekit's own LoRA fold resolves the adapter, so a short-name match
    # (org/name vs name) is enough to catch the mistakes that actually happen.
    if _short_name(adapter_base) != _short_name(base_model):
        raise ValueError(
            _BASE_MISMATCH.format(
                adapter = adapter_path, adapter_base = adapter_base, base = base_model
            )
        )


def build_merge_config(
    base_model: str,
    adapters: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    method: str = "linear",
    normalize_weights: bool = True,
    density: float = 0.5,
    drop_rate: float = 0.5,
    epsilon: float = 0.15,
    task_scale: float = 1.0,
    filter_wise: bool = False,
    out_dtype: Optional[str] = None,
    verify_base: bool = True,
    strict_base_match: bool = False,
) -> Dict[str, Any]:
    """Build a mergekit YAML config for folding several LoRA adapters into a base.

    Each adapter is one source: mergekit folds the LoRA into its base and then
    merges the resulting checkpoints.  Normalising the adapter weights keeps the
    base from being counted once per adapter - ``sum(w_i) = 1`` collapses the
    weighted base terms back to the plain base - so ``normalize_weights`` must
    stay ``True`` for the result to match the in-house engine.

    ``base_model`` is emitted as the global key mergekit needs for task-vector
    arithmetic; it is intentionally *not* repeated as a plain source, which
    would otherwise count it twice.
    """

    if not adapters:
        raise ValueError("Unsloth: At least one adapter path is required.")
    if len(adapters) < 2:
        raise ValueError("Unsloth: Merging requires at least two adapters.")

    merged_method = normalize_method(method)
    target = mergekit_target_method(merged_method)
    if target is None:
        raise ValueError(
            f"Unsloth: mergekit has no equivalent for merge method '{merged_method}'. "
            f"Methods only the Unsloth engine implements: {', '.join(LEGACY_ONLY_METHODS)}."
        )

    if weights is None:
        effective = [1.0 / len(adapters)] * len(adapters)
    else:
        if len(weights) != len(adapters):
            raise ValueError(
                f"Unsloth: Number of adapter paths ({len(adapters)}) must match "
                f"number of weights ({len(weights)})."
            )
        effective = [float(weight) for weight in weights]
        total = sum(effective)
        if total == 0:
            raise ValueError("Unsloth: Adapter weights must not sum to zero.")
        if normalize_weights:
            effective = [weight / total for weight in effective]

    sources: List[Dict[str, Any]] = []
    for adapter_path, weight in zip(adapters, effective):
        if verify_base:
            _check_adapter_base(adapter_path, base_model, strict_base_match)
        parameters: Dict[str, Any] = {}
        if merged_method != "model_stock":
            parameters["weight"] = weight
        if merged_method == "ties" and density is not None:
            # mergekit's `ties` is GeneralizedTaskArithmeticMerge with the
            # magnitude sparsifier; density is its top-k fraction.
            parameters["density"] = float(density)
        if merged_method in ("dare_ties", "dare_linear") and drop_rate is not None:
            parameters["drop_rate"] = float(drop_rate)
        if merged_method in ("della", "della_linear"):
            # DELLA's stochastic magnitude pruning: `density` selects the band,
            # `epsilon` widens it into a probability ramp.
            if density is not None:
                parameters["density"] = float(density)
            if epsilon is not None:
                parameters["epsilon"] = float(epsilon)
        sources.append(
            {
                "model": base_model,
                "lora": adapter_path,
                "parameters": parameters,
            }
        )

    config: Dict[str, Any] = {
        "merge_method": target,
        "base_model": base_model,
        "models": sources,
    }
    if merged_method == "task_arithmetic" and task_scale is not None:
        # mergekit scales the summed task vector by the shared `lambda` knob.
        config["parameters"] = {"lambda": float(task_scale)}
    if merged_method == "model_stock":
        # The stock estimator takes no per-source weights; expose only the
        # optional per-filter geometry toggle at the top level.
        if filter_wise:
            config["parameters"] = {"filter_wise": True}
    if out_dtype:
        config["dtype"] = out_dtype
    return config


def write_merge_config(
    config: Dict[str, Any], directory: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Write *config* as ``merge_config.yaml``.

    Returns ``(config_path, cleanup_dir)``; ``cleanup_dir`` is the temporary
    directory that the caller must remove, or ``None`` when *directory* was
    supplied by the caller (and is therefore the caller's to keep).
    """

    if directory is None:
        work_dir = tempfile.mkdtemp(prefix = "unsloth_mergekit_")
        cleanup: Optional[str] = work_dir
    else:
        work_dir = directory
        os.makedirs(work_dir, exist_ok = True)
        cleanup = None
    config_path = os.path.join(work_dir, "merge_config.yaml")
    with open(config_path, "w", encoding = "utf-8") as handle:
        dump_merge_config_yaml(config, handle)
    return config_path, cleanup


# --------------------------------------------------------------------------
# Minimal YAML writer
# --------------------------------------------------------------------------
# mergekit needs YAML and Unsloth cannot assume PyYAML is importable in the
# engine's interpreter.  The config this module builds is a fixed shape (flat
# scalars plus one list of flat dicts), so a tiny writer keeps the dependency
# out; anything richer should go through mergekit's own pydantic schema.

_PLAIN_SCALAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_./\\@:+()\-]*$")

# YAML 1.1 resolves these to bool/null, so a plain scalar carrying one of them
# would come back out of the engine as the wrong type.
_YAML_RESERVED = frozenset(
    {"true", "false", "null", "yes", "no", "on", "off", "y", "n", "~", "none"}
)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    if text and _PLAIN_SCALAR.match(text) and text.lower() not in _YAML_RESERVED:
        return text
    return json.dumps(text)


def dump_merge_config_yaml(config: Dict[str, Any], handle) -> None:
    """Serialise *config* as the small YAML subset mergekit accepts."""
    for key, value in config.items():
        if isinstance(value, list):
            handle.write(f"{key}:\n")
            for entry in value:
                first = True
                for entry_key, entry_value in entry.items():
                    prefix = "  - " if first else "    "
                    first = False
                    if isinstance(entry_value, dict):
                        handle.write(f"{prefix}{entry_key}:\n")
                        for nested_key, nested_value in entry_value.items():
                            handle.write(f"      {nested_key}: {_yaml_scalar(nested_value)}\n")
                    else:
                        handle.write(f"{prefix}{entry_key}: {_yaml_scalar(entry_value)}\n")
        elif isinstance(value, dict):
            handle.write(f"{key}:\n")
            for nested_key, nested_value in value.items():
                handle.write(f"  {nested_key}: {_yaml_scalar(nested_value)}\n")
        else:
            handle.write(f"{key}: {_yaml_scalar(value)}\n")


# --------------------------------------------------------------------------
# Child-process runner
# --------------------------------------------------------------------------

# Written beside the merge config and executed with `python -I`, so the engine
# interpreter never imports the parent's Unsloth, torch or transformers.
_BOOTSTRAP = '''\
import os
import sys

# An engine may live in a --target directory instead of the environment itself;
# `python -I` ignores PYTHONPATH, so the shadow travels as an environment value
# the parent sets explicitly rather than through the ambient environment.
_shadow = os.environ.get("UNSLOTH_MERGEKIT_SHADOW", "").strip()
if _shadow:
    sys.path.insert(0, _shadow)

try:
    import torch
except Exception:
    pass

try:
    from pydantic_core import core_schema

    def _any_pydantic_schema(cls, *args, **kwargs):
        return core_schema.any_schema()

    if "torch" in sys.modules:
        setattr(torch.Tensor, "__get_pydantic_core_schema__", classmethod(_any_pydantic_schema))
        setattr(torch.dtype, "__get_pydantic_core_schema__", classmethod(_any_pydantic_schema))
except Exception:
    pass

import yaml
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

# In pydantic>=2.10, ConfiguredModuleArchitecture references torch types and
# requires arbitrary_types_allowed and model_rebuild() once torch is imported.
for _mod_name in ("mergekit.architecture.base", "mergekit.plan", "mergekit.architecture"):
    try:
        _mod = __import__(_mod_name, fromlist=["ConfiguredModuleArchitecture", "ConfiguredModelArchitecture"])
        for _name in ("ConfiguredModuleArchitecture", "ConfiguredModelArchitecture"):
            _cls = getattr(_mod, _name, None)
            if _cls is not None:
                if hasattr(_cls, "model_config"):
                    if isinstance(_cls.model_config, dict):
                        _cls.model_config["arbitrary_types_allowed"] = True
                    else:
                        try:
                            setattr(_cls.model_config, "arbitrary_types_allowed", True)
                        except Exception:
                            pass
                if hasattr(_cls, "model_rebuild"):
                    try:
                        _cls.model_rebuild(force=True)
                    except Exception:
                        pass
    except Exception:
        pass

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = MergeConfiguration.model_validate(yaml.safe_load(handle))

device = sys.argv[5]
run_merge(
    config,
    sys.argv[2],
    MergeOptions(
        transformers_cache = sys.argv[3] or None,
        out_shard_size = int(sys.argv[4]),
        device = device,
        cuda = device == "cuda",
        low_cpu_memory = device != "cpu",
        lazy_unpickle = False,
        copy_tokenizer = True,
        safe_serialization = True,
    ),
)
'''


def run_mergekit_merge(
    config_path: str,
    output_dir: str,
    mergekit_python: Optional[str] = None,
    device: str = "cpu",
    transformers_cache: Optional[str] = None,
    out_shard_size: int = 5_000_000_000,
    timeout: Optional[float] = None,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> str:
    """Run a prepared mergekit config in the engine interpreter.

    Raises ``MergeKitUnavailableError`` when no interpreter can be found and
    ``MergeKitMergeError`` when the child process fails; both carry the tail of
    the child's output so callers can surface a real reason instead of a bare
    exit code.

    ``runner`` is the test seam: it receives the command plus the timeout and
    working directory, and lets the whole path be exercised without mergekit.
    """

    python = mergekit_python or resolve_mergekit_python()
    if python is None:
        raise MergeKitUnavailableError(
            "Unsloth: mergekit is not importable from any configured Python "
            f"({', '.join(_python_candidates())}). Install mergekit in a dedicated "
            f"environment and point {MERGEKIT_PYTHON_ENV} at its interpreter, or use "
            "the legacy Unsloth engine."
        )
    if not os.path.isfile(config_path):
        raise MergeKitMergeError(f"Unsloth: merge config not found: {config_path}")

    os.makedirs(output_dir, exist_ok = True)
    work_dir = Path(config_path).parent
    bootstrap_path = work_dir / "unsloth_mergekit_run.py"
    bootstrap_path.write_text(_BOOTSTRAP, encoding = "utf-8")

    command = [
        python,
        "-I",
        str(bootstrap_path),
        str(config_path),
        str(output_dir),
        transformers_cache or "",
        str(int(out_shard_size)),
        "cpu" if device == "cpu" else "cuda",
    ]
    effective_timeout = float(timeout) if timeout else 3600.0

    if runner is not None:
        completed = runner(command, effective_timeout, work_dir)
    else:
        env = dict(os.environ)
        # The engine interpreter is self-contained: it must not inherit the
        # parent's PYTHONPATH or mergekit config override.
        env.pop("PYTHONPATH", None)
        env.pop(MERGEKIT_CONFIG_ENV, None)
        completed = subprocess.run(
            command,
            cwd = str(work_dir),
            env = env,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = effective_timeout,
        )

    if completed.returncode != 0:
        tail = (completed.stdout or "")[-4000:]
        raise MergeKitMergeError(
            f"Unsloth: mergekit merge failed (exit {completed.returncode}). Output:\n{tail}"
        )
    if not os.path.isfile(os.path.join(output_dir, "config.json")):
        raise MergeKitMergeError(
            f"Unsloth: mergekit reported success but {output_dir} has no config.json."
        )
    return output_dir


# --------------------------------------------------------------------------
# Convenience entry point
# --------------------------------------------------------------------------

def merge_adapters_via_mergekit(
    base_model: str,
    adapters: Sequence[str],
    output_dir: str,
    weights: Optional[Sequence[float]] = None,
    method: str = "linear",
    normalize_weights: bool = True,
    density: float = 0.5,
    drop_rate: float = 0.5,
    epsilon: float = 0.15,
    task_scale: float = 1.0,
    filter_wise: bool = False,
    device: str = "cpu",
    out_dtype: Optional[str] = None,
    transformers_cache: Optional[str] = None,
    verify_base: bool = True,
    strict_base_match: bool = False,
    mergekit_python: Optional[str] = None,
    out_shard_size: int = 5_000_000_000,
    timeout: Optional[float] = None,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    keep_config: bool = False,
) -> str:
    """Fold *adapters* into *base_model* with mergekit; return the output dir."""

    config = build_merge_config(
        base_model = base_model,
        adapters = adapters,
        weights = weights,
        method = method,
        normalize_weights = normalize_weights,
        density = density,
        drop_rate = drop_rate,
        epsilon = epsilon,
        task_scale = task_scale,
        filter_wise = filter_wise,
        out_dtype = out_dtype,
        verify_base = verify_base,
        strict_base_match = strict_base_match,
    )
    work_dir = os.path.join(output_dir, "_unsloth_mergekit") if keep_config else None
    config_path, cleanup = write_merge_config(config, work_dir)
    try:
        return run_mergekit_merge(
            config_path = config_path,
            output_dir = output_dir,
            mergekit_python = mergekit_python,
            device = device,
            transformers_cache = transformers_cache,
            out_shard_size = out_shard_size,
            timeout = timeout,
            runner = runner,
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors = True)