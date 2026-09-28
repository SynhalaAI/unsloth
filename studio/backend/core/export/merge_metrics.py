# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Weight-space interference metrics for a multi-adapter LoRA merge.

The Export page's "Test" button answers "will these adapters fight each other?"
before anything reaches a GPU. It reads only the adapter checkpoints,
reconstructs their LoRA deltas and compares them pairwise; the base model is
never loaded, so the check costs seconds of CPU instead of a model load.

Reported per adapter pair, over the modules the two share:

* ``cosine`` - cosine similarity of the two flattened deltas. Near 1 means the
  adapters push the same weights the same way (redundant, safe to add); near 0
  means they act on disjoint directions; negative means they pull against each
  other, which is what TIES sign election resolves. Computed from the low-rank
  factors through <B1 A1, B2 A2> = <B1^T B2, A1 A2^T> and
  ||B A||^2 = <B^T B, A A^T>, so no full-rank delta is materialised for it.
* ``sign_conflict_rate`` - share of weight positions where both adapters are
  non-zero and disagree in sign. 0.5 is the chance level for two independent
  adapters, so a rate well below that means they agree.
* ``norm_ratio`` - ||di|| / ||dj|| over the shared modules. A large ratio means
  one adapter dominates the merged model whatever the weights say.
* ``worst_modules`` - the shared modules with the lowest cosine, so a conflict
  can be traced to a layer instead of guessed at.

``interference`` / ``score`` / ``recommendation`` are heuristics over those
numbers, documented in ``_recommend``. They estimate interference, not
downstream loss: only an eval run measures loss. Reference points for the
metric semantics are TIES (arXiv:2306.01708) and DARE (arXiv:2311.03099); no
merge arithmetic is reimplemented here, this module only measures inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# studio/backend/core/export/merge_metrics.py -> the repository root, which is
# where the source checkout keeps unsloth/multi_adapter_merge.py.
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Standalone module name for the by-path load, kept out of the ``unsloth``
# namespace so it cannot shadow or be shadowed by the real package.
_CORE_MODULE_NAME = "studio_merge_metrics_core"
_core_module = None

# Sign scanning is the one metric that must materialise a delta, so it is
# bounded: 64M positions costs a few seconds of CPU and gives a stable rate.
# Cosine and the norms stay exact for every shared module regardless, because
# they come from the low-rank identities rather than from the expanded deltas.
DEFAULT_MAX_SIGN_ELEMENTS = 64_000_000

# Rows of a delta materialised at a time while scanning signs. 256 x in_features
# keeps the working set in the low-MB range even for a 128k-row embedding.
_SIGN_BLOCK_ROWS = 256

# Per-pair verdicts, from the two numbers that carry the signal: how opposed the
# two deltas are (cosine) and how often they disagree in sign. A linear sum is
# safe while the deltas agree; TIES trims the weakest entries and elects one
# sign per coordinate, which is what rescues an opposed pair; DARE adds a
# random drop-and-rescale before that, which pays off once conflicts are dense.
_LINEAR_MAX_COSINE = 0.5
_LINEAR_MAX_CONFLICT = 0.15
_TIES_MAX_CONFLICT = 0.30
_DARE_MIN_CONFLICT = 0.45

# Severity buckets over the same score the recommendation derives from.
_SCORE_LOW = 0.20
_SCORE_HIGH = 0.50

# How many conflicting modules a pair names, worst first.
_WORST_MODULE_COUNT = 3


def _core():
    """The merge core, imported lazily, once.

    ``unsloth/__init__`` pulls in ``_gpu_init`` and refuses to import without an
    accelerator, but ``multi_adapter_merge`` itself is only torch plus stdlib. The
    analysis is deliberately a CPU-only preflight - it never loads a model - so
    routing it through the package would make it unavailable on exactly the hosts
    where a merge wants checking before touching a GPU, and would need the
    package installed at all. Load the module straight from the source checkout
    when it is there, and fall back to the package import for a wheel install,
    where only site-packages exists.
    """
    global _core_module
    if _core_module is not None:
        return _core_module

    path = _REPO_ROOT / "unsloth" / "multi_adapter_merge.py"
    if path.is_file():
        import importlib.util
        import sys

        # Registered before exec: the module uses postponed annotations, and its
        # dataclasses resolve those through sys.modules[cls.__module__].
        spec = importlib.util.spec_from_file_location(_CORE_MODULE_NAME, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_CORE_MODULE_NAME] = module
        spec.loader.exec_module(module)
    else:
        from unsloth import multi_adapter_merge as module

    _core_module = module
    return module

def _factor_norm_sq(A, B) -> float:
    """||scaling * (B @ A)||^2 from the factors alone: ||B A||^2 = <B^T B, A A^T>."""
    import torch

    A = A.to(torch.float32)
    B = B.to(torch.float32)
    return float((B.t() @ B * (A @ A.t())).sum().item())


def _factor_inner(A1, B1, A2, B2) -> float:
    """<B1 A1, B2 A2> = <B1^T B2, A1 A2^T>, both operands (r1, r2)."""
    import torch

    A1 = A1.to(torch.float32)
    B1 = B1.to(torch.float32)
    A2 = A2.to(torch.float32)
    B2 = B2.to(torch.float32)
    return float(((B1.t() @ B2) * (A1 @ A2.t())).sum().item())


def _module_cosine(A1, B1, A2, B2, scaling1: float, scaling2: float) -> float:
    """Cosine of one module's two deltas; 0.0 when either is all zeros."""
    dot = scaling1 * scaling2 * _factor_inner(A1, B1, A2, B2)
    n1 = (scaling1**2) * _factor_norm_sq(A1, B1)
    n2 = (scaling2**2) * _factor_norm_sq(A2, B2)
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    return dot / ((n1**0.5) * (n2**0.5))


def _sign_scan(A1, B1, A2, B2, scaling1: float, scaling2: float) -> Tuple[int, int]:
    """Count (disagreements, comparable positions) over one module's delta.

    Materialises the delta one row block at a time, so a 128k-row embedding
    costs the same memory as a small projection. Only positions where both
    adapters are non-zero are comparable: a zero on one side is an untouched
    weight, not a disagreement.
    """
    import torch

    disagreements = 0
    comparable = 0
    rows = B1.shape[0]
    A1 = A1.to(torch.float32)
    A2 = A2.to(torch.float32)
    B1 = B1.to(torch.float32)
    B2 = B2.to(torch.float32)
    for start in range(0, rows, _SIGN_BLOCK_ROWS):
        end = min(start + _SIGN_BLOCK_ROWS, rows)
        d1 = scaling1 * (B1[start:end] @ A1)
        d2 = scaling2 * (B2[start:end] @ A2)
        both = (d1 != 0) & (d2 != 0)
        count = int(both.sum().item())
        if count == 0:
            continue
        comparable += count
        disagreements += int((torch.sign(d1[both]) != torch.sign(d2[both])).sum().item())
    return disagreements, comparable


def _module_name(key: str) -> str:
    """A module key for display: the core normalises to base-parameter names, which
    carry a trailing ``.weight`` the user never sees in ``target_modules``."""
    return key[: -len(".weight")] if key.endswith(".weight") else key


def _classify(score: float) -> str:
    if score < _SCORE_LOW:
        return "low"
    if score < _SCORE_HIGH:
        return "moderate"
    return "high"


def _recommend(cosine: float, conflict: float, overlap: bool) -> Dict[str, object]:
    """Map a pair's two numbers onto a method and a density.

    A heuristic, and labelled as one in the response: it estimates how much two
    adapters interfere, not how much accuracy a merge costs. The thresholds
    follow what each method is documented to fix - a weighted sum only survives
    adapters that agree, TIES rescues opposed signs, DARE additionally sheds
    redundant entries once the disagreements are dense.
    """
    if not overlap:
        return {
            "method": "linear",
            "density": 1.0,
            "reason": "The adapters share no modules, so their deltas never touch the same weights.",
        }
    if cosine <= 0.0:
        return {
            "method": "ties",
            "density": 0.5,
            "reason": (
                "The deltas point in opposite directions (cosine < 0); TIES elects one sign "
                "per weight instead of letting them cancel."
            ),
        }
    if conflict >= _DARE_MIN_CONFLICT:
        return {
            "method": "dare_ties",
            "density": 0.3,
            "reason": (
                f"{conflict:.0%} of shared weights disagree in sign; DARE drops and "
                "rescales the redundant entries before TIES elects the sign."
            ),
        }
    if conflict > _TIES_MAX_CONFLICT or cosine < _LINEAR_MAX_COSINE:
        return {
            "method": "ties",
            "density": 0.5,
            "reason": (
                "The adapters only partly agree, so trim the weakest weight changes and "
                "elect one sign before averaging."
            ),
        }
    if conflict <= _LINEAR_MAX_CONFLICT and cosine >= _LINEAR_MAX_COSINE:
        return {
            "method": "linear",
            "density": 1.0,
            "reason": (
                "The deltas agree closely, so a weighted sum keeps everything both "
                "adapters learned."
            ),
        }
    return {
        "method": "ties",
        "density": 0.7,
        "reason": (
            "Some weight changes are shared, so a plain sum would double-count them; "
            "TIES keeps one contribution per coordinate."
        ),
    }

def _pair_metrics(
    factors1: Dict[str, tuple],
    factors2: Dict[str, tuple],
    scaling1: float,
    scaling2: float,
    shared: Sequence[str],
    modules1: int,
    modules2: int,
    name1: str,
    name2: str,
    sign_budget: int,
) -> dict:
    """Metrics for one adapter pair, restricted to the modules they share.

    ``sign_budget`` is a running allowance: sign scanning stops once it is
    spent, and ``_sign_truncated`` says so, so a truncated rate is never
    presented as a full-model one.
    """
    if not shared:
        return {
            "a": name1,
            "b": name2,
            "shared_modules": 0,
            "module_overlap": 0.0,
            "cosine": 0.0,
            "sign_conflict_rate": 0.0,
            "norm_ratio": 1.0,
            "worst_modules": [],
            "_sign_truncated": False,
        }

    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    disagreements = 0
    comparable = 0
    truncated = False
    per_module = []
    for key in shared:
        A1, B1 = factors1[key]
        A2, B2 = factors2[key]
        if A1.shape != A2.shape or B1.shape != B2.shape:
            # Same module name, different shapes: the adapters disagree about the
            # layer itself, so there is nothing comparable to measure.
            continue
        dot += scaling1 * scaling2 * _factor_inner(A1, B1, A2, B2)
        norm1_sq += (scaling1**2) * _factor_norm_sq(A1, B1)
        norm2_sq += (scaling2**2) * _factor_norm_sq(A2, B2)
        module_cosine = _module_cosine(A1, B1, A2, B2, scaling1, scaling2)
        module_conflict = None
        if comparable < sign_budget:
            found, count = _sign_scan(A1, B1, A2, B2, scaling1, scaling2)
            remaining = sign_budget - comparable
            if count > remaining:
                # Out of budget: scale the count down to what the allowance
                # covers, so the rate stays an estimate of the same rate rather
                # than an over-count of a wider sample.
                found = int(found * remaining / count) if count else 0
                count = remaining
                truncated = True
            disagreements += found
            comparable += count
            module_conflict = (found / count) if count else 0.0
        per_module.append((key, module_cosine, module_conflict))

    cosine = 0.0
    if norm1_sq > 0.0 and norm2_sq > 0.0:
        cosine = dot / ((norm1_sq**0.5) * (norm2_sq**0.5))
    norm_ratio = (norm1_sq / norm2_sq) ** 0.5 if norm2_sq > 0.0 else 1.0

    ranked = sorted(per_module, key=lambda entry: entry[1])[:_WORST_MODULE_COUNT]
    worst = [
        {
            "module": _module_name(key),
            "cosine": module_cosine,
            "sign_conflict_rate": module_conflict,
        }
        for key, module_cosine, module_conflict in ranked
        if module_cosine < _LINEAR_MAX_COSINE
    ]

    return {
        "a": name1,
        "b": name2,
        "shared_modules": len(per_module),
        "module_overlap": len(per_module) / max(modules1, modules2, 1),
        "cosine": cosine,
        "sign_conflict_rate": (disagreements / comparable) if comparable else 0.0,
        "norm_ratio": norm_ratio,
        "worst_modules": worst,
        "_sign_truncated": truncated,
    }

def analyze_adapters(
    adapter_paths: Sequence[Union[str, dict]],
    *,
    weights: Optional[Sequence[float]] = None,
    normalize_weights: bool = True,
    hf_token=None,
    max_sign_elements: int = DEFAULT_MAX_SIGN_ELEMENTS,
) -> dict:
    """Measure pairwise interference between adapters without loading a model.

    ``adapter_paths`` accepts what ``MultiAdapterMergeRequest`` accepts: a local
    directory, an HF repo id, or ``{"repo_id": ..., "subfolder": ...}``. The
    weights are the ones the export would use, normalised the same way, so the
    report describes the merge the user is about to run rather than an idealised
    one. ``normalize_weights`` should stay True except when mirroring a request
    that asked for raw weights.

    Returns a JSON-ready report: per-adapter stats, per-pair metrics, the worst
    modules per pair, and a recommended method.
    """
    core = _core()
    if len(adapter_paths) < 2:
        raise ValueError("At least two adapters are required to measure interference.")

    if weights is None:
        weights = [1.0] * len(adapter_paths)
    raw_weights = [float(w) for w in weights]
    if len(raw_weights) != len(adapter_paths):
        raise ValueError("weights must match adapter_paths")
    total = sum(raw_weights)
    if total == 0:
        raise ValueError("Adapter weights must not sum to zero.")
    effective = [w / total for w in raw_weights] if normalize_weights else raw_weights

    resolved = [core._resolve_adapter_path(path, hf_token=hf_token) for path in adapter_paths]
    configs = [core._load_adapter_config(path) for path in resolved]
    # The same gate the merge itself runs, so a pair that cannot be merged is
    # reported here rather than after a GPU load.
    core._validate_adapters(configs, resolved)
    names = [
        core._adapter_display_name(raw, path)
        for raw, path in zip(adapter_paths, resolved)
    ]

    grouped: List[Dict[str, tuple]] = []
    scalings: List[float] = []
    for path, cfg in zip(resolved, configs):
        state_dict = core._load_adapter_state_dict(path)
        factors = core._group_lora_factors(state_dict)
        del state_dict
        if not factors:
            raise ValueError(f"No LoRA A/B factors found in adapter '{path}'.")
        grouped.append(factors)
        scalings.append(core._adapter_scaling(cfg))

    module_sets = [set(factors.keys()) for factors in grouped]

    # The sign budget is shared across pairs, so a 4-adapter run cannot quietly
    # do 6x the work of a 2-adapter one.
    sign_budget = max(int(max_sign_elements), 0)
    sign_truncated = False

    adapters = []
    for index, (name, cfg, factors, weight) in enumerate(
        zip(names, configs, grouped, effective)
    ):
        norm = (
            weight
            * (scalings[index] ** 2)
            * sum(_factor_norm_sq(*factors[key]) for key in module_sets[index])
        ) ** 0.5
        adapters.append(
            {
                "name": name,
                "base_model": cfg.get("base_model_name_or_path", ""),
                "rank": cfg.get("r", cfg.get("rank")),
                "lora_alpha": cfg.get("lora_alpha"),
                "scaling": scalings[index],
                "weight": raw_weights[index],
                "effective_weight": weight,
                "modules": len(module_sets[index]),
                "delta_norm": norm,
            }
        )

    pairs = []
    for i in range(len(grouped)):
        for j in range(i + 1, len(grouped)):
            pair = _pair_metrics(
                grouped[i],
                grouped[j],
                scalings[i] * effective[i],
                scalings[j] * effective[j],
                sorted(module_sets[i] & module_sets[j]),
                len(module_sets[i]),
                len(module_sets[j]),
                adapters[i]["name"],
                adapters[j]["name"],
                sign_budget,
            )
            if pair.pop("_sign_truncated", False):
                sign_truncated = True
            pairs.append(pair)

    mean_cosine = sum(pair["cosine"] for pair in pairs) / len(pairs) if pairs else 0.0
    max_conflict = max(
        (pair["sign_conflict_rate"] for pair in pairs if pair["shared_modules"] > 0),
        default=0.0,
    )
    # Opposition and disagreement each contribute half, so one opposed pair in a
    # large set cannot be drowned out by several orthogonal ones. The sign term
    # measures excess over the 0.5 chance level, rescaled to 0..1.
    score = max(
        0.0,
        min(
            1.0,
            0.5 * max(0.0, -mean_cosine) + 0.5 * max(0.0, 2.0 * max_conflict - 1.0),
        ),
    )
    recommendation = _recommend(
        mean_cosine, max_conflict, any(pair["shared_modules"] for pair in pairs)
    )

    return {
        "adapters": adapters,
        "pairs": pairs,
        "mean_cosine": mean_cosine,
        "max_sign_conflict_rate": max_conflict,
        "interference": _classify(score),
        "score": score,
        "sign_scan_truncated": sign_truncated,
        "recommendation": recommendation,
    }