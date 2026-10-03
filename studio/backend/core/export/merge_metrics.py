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
* ``sign_conflict_rate`` - how much of the disagreement is *beyond* what
  independence predicts, in 0..1. Two unrelated adapters do not disagree 0% of
  the time: each has its own positive-negative balance, and independence alone
  gives ``p1(1-p2) + (1-p1)p2``. Subtracting that removes "unrelated" from
  "opposed", so 0 now means the pair is at chance and only cancellation is
  scored. The vote itself is weighted by how much each coordinate could
  actually cancel (``min(|d1|, |d2|)``), so a disagreement on a coordinate
  worth 1e-6 no longer counts as much as one on a coordinate worth 10.
* ``norm_ratio`` - ||di|| / ||dj|| over the shared modules. A large ratio means
  one adapter dominates the merged model whatever the weights say.
* ``worst_modules`` - the shared modules with the lowest cosine, so a conflict
  can be traced to a layer instead of guessed at.
* ``dominance`` - how far the widest norm gap between two adapters reaches. This
  is a different failure from interference: two adapters can agree perfectly
  (cosine 1, conflict 0, score 0) and still leave one carrying the whole merge.
  ``norm_ratio`` was always reported for that, but nothing fed it into the
  score, so a 27x-dominant set read "low interference".

``interference`` / ``score`` / ``recommendation`` are heuristics over those
numbers, documented in ``_recommend``. They estimate interference, not
downstream loss: only an eval run measures loss. Reference points for the
metric semantics are TIES (arXiv:2306.01708) and DARE (arXiv:2311.03099); no
merge arithmetic is reimplemented here, this module only measures inputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

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

# A reconstructed delta is dense, so `!= 0` counts float noise at positions the
# adapter never moved. Those signs are random and hold the conflict rate at the
# 0.5 chance level whatever the adapters did, which is what the docstring's
# "untouched weight" filter was meant to skip. Positions below this fraction of
# the block RMS delta are treated as untouched.
_SIGN_REL_EPS = 1e-3

# Per-pair verdicts on the excess-over-chance scale, where 0 means "these two
# disagree exactly as much as independence predicts" and 1 means "they disagree
# everywhere they could". Both numbers carrying the signal are already on that
# scale: cosine is 0 at orthogonal and -1 at opposed, and conflict arrives from
# ``_excess_over_chance``, so no baseline is subtracted a second time here.
#
# A linear sum is safe while the deltas agree (low excess); TIES trims the
# weakest entries and elects one sign per coordinate, which is what rescues an
# opposed pair; DARE adds a random drop-and-rescale before that, which pays off
# only once the disagreement is dense. The old thresholds sat on raw rates
# against an assumed 0.5 baseline and read very differently once the pair's own
# sign bias is removed: 0.60 raw is a fifth of the disagreement that is left.
_LINEAR_MAX_COSINE = 0.5
_LINEAR_MAX_CONFLICT = 0.10
_TIES_MAX_CONFLICT = 0.25
_DARE_MIN_CONFLICT = 0.40

# Severity buckets over the same score the recommendation derives from. The
# score halves each of its two terms, so a realistic opposition (cosine -0.3)
# only reaches 0.15, and the old 0.50 bound left "high" unreachable barring a
# full -1.0 cosine or a 1.0 conflict rate. These bounds sit where the blended
# number actually lands: 0.10 for cosine -0.2 or a 0.55 conflict rate, 0.35 for
# cosine -0.7 or 0.85.
_SCORE_LOW = 0.10
_SCORE_HIGH = 0.35

# How many conflicting modules a pair names, worst first.
_WORST_MODULE_COUNT = 3

# Dominance is read on a log axis: the failure mode is multiplicative, so 2x
# matters to 10x the way 10x matters to 50x. 2.5 puts a 2x gap at 0.12
# (moderate, where a visibly uneven merge starts to worry) and a 10x gap at
# 0.40 (high, past the point where one adapter carries the result).
_DOMINANCE_LOG_HIGH = 2.5


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


class SignScan(NamedTuple):
    """What one module's scan found, as counts and as magnitudes.

    The counts exist for the sign budget: truncation has to say how much of the
    model was looked at, which only a position count can express. The weighted
    side is what the rate is built from - see ``_pair_metrics``.
    """

    disagreements: int
    comparable: int
    weighted_disagreements: float
    total_weight: float
    positive1: float
    positive2: float


def _sign_scan(A1, B1, A2, B2, scaling1: float, scaling2: float) -> SignScan:
    """Measure disagreement over one module's delta, by count and by magnitude.

    Materialises the delta one row block at a time, so a 128k-row embedding
    costs the same memory as a small projection. Only positions where both
    adapters move the weight are comparable: a position one adapter left alone
    is not a disagreement, and `_SIGN_REL_EPS` is what makes that distinction
    real for a dense reconstruction.

    Two things are summed that a plain count of sign flips cannot give:

    * each position's vote is weighted by ``min(|d1|, |d2|)`` - the amount that
      could actually cancel there, so a 1e-6 disagreement on a coordinate worth
      1e-6 stops being equal to a 10-point disagreement on one worth 10;
    * the sign biases ``p1``, ``p2`` (weighted positive fractions), so the
      caller can subtract what two independent adapters would have shown. A
      pair that is positive on 80% of its positions disagrees 32% of the time
      by construction, which is agreement rather than conflict.
    """
    import torch

    disagreements = 0
    comparable = 0
    weighted_disagreements = 0.0
    total_weight = 0.0
    positive1 = 0.0
    positive2 = 0.0
    rows = B1.shape[0]
    A1 = A1.to(torch.float32)
    A2 = A2.to(torch.float32)
    B1 = B1.to(torch.float32)
    B2 = B2.to(torch.float32)
    for start in range(0, rows, _SIGN_BLOCK_ROWS):
        end = min(start + _SIGN_BLOCK_ROWS, rows)
        d1 = scaling1 * (B1[start:end] @ A1)
        d2 = scaling2 * (B2[start:end] @ A2)
        # Relative to each delta's own scale, so the floor travels with the
        # adapter (a rank-4 projection and a 128k embedding differ by orders of
        # magnitude) instead of being an absolute epsilon that fits neither.
        scale = max(float(d1.pow(2).mean().sqrt()), float(d2.pow(2).mean().sqrt()))
        floor = _SIGN_REL_EPS * scale
        both = (d1.abs() > floor) & (d2.abs() > floor)
        count = int(both.sum().item())
        if count == 0:
            continue
        first = d1[both]
        second = d2[both]
        disagree = torch.sign(first) != torch.sign(second)
        # min of the magnitudes: what this coordinate could lose to cancellation.
        weight = torch.minimum(first.abs(), second.abs())

        comparable += count
        disagreements += int(disagree.sum().item())
        total_weight += float(weight.sum().item())
        weighted_disagreements += float(weight[disagree].sum().item())
        positive1 += float(weight[first > 0].sum().item())
        positive2 += float(weight[second > 0].sum().item())
    return SignScan(
        disagreements,
        comparable,
        weighted_disagreements,
        total_weight,
        positive1,
        positive2,
    )


def _mean_cosine(pairs: List[dict]) -> float:
    """Cosine across the merge, weighted by how many modules each pair shares.

    A plain average treats a 2-module overlap as equal to a 393-module one, so
    one thin pair can decide the headline for a set that behaves nothing like
    it. Pairs sharing no modules carry no cosine and no weight; an empty set
    reads 0, meaning "unrelated" rather than "aligned".
    """
    weight = sum(pair["shared_modules"] for pair in pairs)
    if not weight:
        return 0.0
    return sum(pair["cosine"] * pair["shared_modules"] for pair in pairs) / weight


def _excess_over_chance(
    weighted_disagreements: float,
    total_weight: float,
    positive1: float,
    positive2: float,
) -> float:
    """Disagreement a pair shows beyond what independence predicts, in 0..1.

    Two unrelated adapters do not disagree 0% of the time - they disagree at
    ``p1(1 - p2) + (1 - p1)p2``, where ``p`` is each one's weighted share of
    positive positions. An adapter that moves 80% of its weights upward and one
    that moves 80% downward disagree on 32% of positions while being no more
    opposed than chance, and the raw rate reports that as conflict. Subtracting
    the pair's own baseline and rescaling what is left separates "these two
    cancel" from "these two are unrelated", which the raw rate cannot.

    Zero when either delta is empty, and zero when the pair is at or below its
    baseline rather than negative: agreement is not a smaller amount of conflict.
    """
    if total_weight <= 0.0:
        return 0.0
    observed = weighted_disagreements / total_weight
    p1 = positive1 / total_weight
    p2 = positive2 / total_weight
    chance = p1 * (1.0 - p2) + (1.0 - p1) * p2
    if chance >= 1.0:
        return 0.0
    return max(0.0, (observed - chance) / (1.0 - chance))


def _module_name(key: str) -> str:
    """A module key for display: the core normalises to base-parameter names, which
    carry a trailing ``.weight`` the user never sees in ``target_modules``."""
    return key[: -len(".weight")] if key.endswith(".weight") else key


def _dominance_score(ratio: float) -> float:
    """0..1 for how far one adapter outweighs the other.

    ``ratio`` is the widest norm gap over any pair's shared modules, so 1.0
    means the adapters are even. Logarithmic rather than linear for the reason
    ``_DOMINANCE_LOG_HIGH`` gives: a linear axis reads 0 until the gap is absurd.
    """
    if ratio <= 1.0:
        return 0.0
    return min(1.0, math.log10(ratio) / _DOMINANCE_LOG_HIGH)


def _dominance(pairs: List[dict]) -> Tuple[float, str, str]:
    """The widest ``(ratio, larger, smaller)`` gap over pairs that share modules.

    Each pair's ``norm_ratio`` is already weighted the way the merge would weight
    it, so this answers "in the merge that would run, who outweighs whom".
    Pairs sharing no modules carry no ratio and are skipped rather than treated
    as even.
    """
    ratio = 1.0
    larger = smaller = ""
    for pair in pairs:
        if not pair.get("shared_modules"):
            continue
        current = float(pair.get("norm_ratio") or 0.0)
        if current <= 0.0:
            continue
        effective = current if current >= 1.0 else 1.0 / current
        if effective > ratio:
            ratio = effective
            larger, smaller = (
                (pair["a"], pair["b"]) if current >= 1.0 else (pair["b"], pair["a"])
            )
    return ratio, larger, smaller


def _score(
    mean_cosine: float,
    max_conflict: float,
    dominance_ratio: float = 1.0,
) -> float:
    """Severity in 0..1 - the worse of interference and dominance.

    ``max_conflict`` arrives already corrected for the pair's own sign bias
    (``_excess_over_chance``), so it is conflict in 0..1 and not a raw rate that
    has to have the old 0.5 chance level subtracted again. Opposition and
    disagreement each contribute half, so one opposed pair in a large set cannot
    be drowned out by several orthogonal ones.

    Dominance is combined by taking the worse axis rather than averaging the
    two: averaging would let a balanced set wash out an opposed pair, and let an
    opposed pair wash out a 27x-dominant one. The two are different failures and
    either alone is enough to send the reader looking.
    """
    interference = 0.5 * max(0.0, -mean_cosine) + 0.5 * max(0.0, max_conflict)
    return max(0.0, min(1.0, max(interference, _dominance_score(dominance_ratio))))


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

    ``conflict`` is excess over the pair's own chance level (0 means the two
    disagree exactly as independence predicts), not a raw sign-disagreement rate.
    Passing a raw rate here reads every orthogonal pair as dense conflict, which
    is how DARE came to be recommended for merges with nothing to shed.
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
                f"{conflict:.0%} of the disagreement is beyond chance; DARE drops and "
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
            "sign_positions": 0,
            "norm_ratio": 1.0,
            "worst_modules": [],
            "_sign_truncated": False,
        }

    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    comparable = 0
    weighted_disagreements = 0.0
    total_weight = 0.0
    positive1 = 0.0
    positive2 = 0.0
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
            scan = _sign_scan(A1, B1, A2, B2, scaling1, scaling2)
            remaining = sign_budget - comparable
            count = scan.comparable
            w_disc = scan.weighted_disagreements
            w_tot = scan.total_weight
            pos1 = scan.positive1
            pos2 = scan.positive2
            if count > remaining:
                # Out of budget: scale what this module contributed down to the
                # share the allowance covers, on the weighted side too, so the
                # rate stays an estimate of the same rate rather than an
                # over-count of a wider sample.
                factor = (remaining / count) if count else 0.0
                w_disc *= factor
                w_tot *= factor
                pos1 *= factor
                pos2 *= factor
                count = remaining
                truncated = True
            comparable += count
            weighted_disagreements += w_disc
            total_weight += w_tot
            positive1 += pos1
            positive2 += pos2
            # Per module, the same chance correction the pair gets, so a module
            # is never named as a conflict for disagreeing at its own baseline.
            module_conflict = _excess_over_chance(w_disc, w_tot, pos1, pos2)
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
        "sign_conflict_rate": _excess_over_chance(
            weighted_disagreements, total_weight, positive1, positive2
        ),
        # Kept for the sign budget's own reporting: how much of the model was
        # actually looked at, which only a position count can express.
        "sign_positions": comparable,
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

    # Weighted by shared modules, not a plain average: a pair that touches 393
    # modules and one that overlaps on 2 are not equally informative about where
    # a merge will land, and an unweighted mean lets the thin overlap decide the
    # headline. Pairs with no overlap contribute no cosine and no weight.
    mean_cosine = _mean_cosine(pairs)
    max_conflict = max(
        (pair["sign_conflict_rate"] for pair in pairs if pair["shared_modules"] > 0),
        default=0.0,
    )
    # The opposition and sign terms are blended inside _score; dominance is the
    # other axis, and either being bad is enough.
    dominance_ratio, dominant_adapter, dominated_adapter = _dominance(pairs)
    score = _score(mean_cosine, max_conflict, dominance_ratio)
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
        "dominance": {
            "ratio": dominance_ratio,
            "adapter": dominant_adapter,
            "against": dominated_adapter,
            "score": _dominance_score(dominance_ratio),
        },
        "sign_scan_truncated": sign_truncated,
        "recommendation": recommendation,
    }