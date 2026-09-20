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

"""Multi-adapter LoRA merging for Unsloth.

Supports merging multiple LoRA adapters into a single base model using
different strategies:

  * **linear** — weighted sum of adapter deltas (default)
  * **ties** — TIES-Merging: trim low-magnitude, elect sign, disjoint average
  * **dare_ties** — DARE (drop & rescale) followed by TIES-Merging
  * **dare_linear** — DARE followed by a plain weighted sum (no sign election)
  * **magnitude_prune** — keep the top-density magnitudes, then weighted sum
  * **ctm** — weighted sum, then truncated-SVD low-rank compression
  * **cat** — factor concatenation (PEFT-compatible; equals linear in
    weight space, lossless when exported as a merged adapter)

After merging, the model can be saved via the usual
``model.save_pretrained_merged(...)`` path.

Usage (instance method — primary API)::

    model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-1B")
    model.merge_multi_adapters(
        adapters=["path/to/adapter1", "path/to/adapter2"],
        weights=[0.7, 0.3],
        method="linear",
    )
    model.save_pretrained_merged("merged_output", tokenizer)

Usage (static convenience wrapper)::

    model, tokenizer = FastLanguageModel.merge_adapters(
        model_name="unsloth/Llama-3.2-1B",
        adapters=["path/to/adapter1", "path/to/adapter2"],
        weights=[0.7, 0.3],
    )
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch


def _resolve_adapter_path(adapter_path, hf_token=None) -> str:
    """Return a local adapter directory for a filesystem path or HF repo id."""
    subfolder = ""
    if isinstance(adapter_path, dict):
        subfolder = str(adapter_path.get("subfolder") or "").strip("/\\")
        adapter_path = adapter_path.get("repo_id", "")
    if os.path.isdir(adapter_path):
        return adapter_path

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=adapter_path,
        token=hf_token,
        allow_patterns=[
            f"{subfolder + '/' if subfolder else ''}adapter_config.json",
            f"{subfolder + '/' if subfolder else ''}adapter_model*.safetensors",
            f"{subfolder + '/' if subfolder else ''}adapter_model*.bin",
        ],
        max_workers=1,
    )
    return os.path.join(snapshot_path, subfolder) if subfolder else snapshot_path


def _adapter_display_name(raw_spec: Union[str, dict], resolved_path: Optional[str] = None) -> str:
    """Return a human-friendly display name showing both adapter and checkpoint/subfolder.

    If the adapter is from Hugging Face or specified as a dict:
        {"repo_id": "org/my-lora", "subfolder": "checkpoint-500"} -> "org/my-lora (checkpoint-500)"
        {"repo_id": "org/my-lora"} -> "org/my-lora"
    If the adapter is a filesystem path:
        ".../my-model-run/checkpoint-500" -> "my-model-run (checkpoint-500)"
        "checkpoint-500" -> "checkpoint-500"
        ".../my-model-run" -> "my-model-run"
    """
    if isinstance(raw_spec, dict):
        repo_id = str(raw_spec.get("repo_id") or "").strip()
        subfolder = str(raw_spec.get("subfolder") or "").strip("/\\")
        if repo_id and subfolder:
            return f"{repo_id} ({subfolder})"
        if repo_id:
            return repo_id
        if subfolder:
            return subfolder

    # raw_spec or resolved_path is a path string
    path_str = str(raw_spec).strip() if raw_spec else (str(resolved_path).strip() if resolved_path else "")
    if not path_str:
        return "adapter"

    p = Path(path_str)
    # Check if the folder is a checkpoint directory (e.g., 'checkpoint-500' or 'checkpoint_100')
    if p.name.lower().startswith("checkpoint") and p.parent != p and p.parent.name:
        adapter_name = p.parent.name
        checkpoint_name = p.name
        return f"{adapter_name} ({checkpoint_name})"

    # If resolved_path has checkpoint information that raw_spec didn't have
    if resolved_path:
        rp = Path(resolved_path)
        if rp.name.lower().startswith("checkpoint") and rp.parent != rp and rp.parent.name:
            adapter_name = rp.parent.name
            checkpoint_name = rp.name
            return f"{adapter_name} ({checkpoint_name})"

    return p.name or path_str


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_METHODS = (
    "linear", "ties", "dare_ties", "dare_linear", "magnitude_prune", "ctm", "cat",
    "sce", "della", "della_linear", "breadcrumbs", "breadcrumbs_ties", "multislerp",
    "model_stock",
)


@dataclass
class MultiAdapterMergeConfig:
    """Holds validated parameters for a multi-adapter merge."""

    adapter_paths: List[str]
    weights: List[float]
    method: str = "linear"
    normalize_weights: bool = True
    # TIES / DARE-TIES / magnitude-prune specific: fraction of params to keep
    # (top-k by magnitude).
    density: float = 0.5
    # DARE-specific: fraction of weight deltas to randomly drop before rescaling.
    drop_rate: float = 0.5
    # CtM / SVD specific: target low-rank for truncated SVD compression.
    target_rank: Optional[int] = None
    # Deterministic seed for reproducible dropout masking in DARE/DELLA.
    seed: int = 42
    # DELLA-specific: half-width of the per-rank probability range around
    # ``density`` (mergekit ``della_magprune`` epsilon).
    della_epsilon: float = 0.15
    # Breadcrumbs-specific: fraction of *largest* magnitudes to drop
    # (outlier removal) before pruning the smallest (mergekit
    # ``magnitude_outliers`` gamma).
    gamma: float = 0.01
    # SCE-specific: fraction of highest-variance elements to keep before the
    # sign-consensus erase step (1.0 = keep all).
    select_topk: float = 1.0

    def __post_init__(self):
        if not self.adapter_paths:
            raise ValueError("Unsloth: At least one adapter path is required.")
        if len(self.adapter_paths) != len(self.weights):
            raise ValueError(
                f"Unsloth: Number of adapter paths ({len(self.adapter_paths)}) "
                f"must match number of weights ({len(self.weights)})."
            )
        method_lower = self.method.lower().replace("-", "_")
        # Support aliases
        if method_lower in ("dare", "dare_ties"):
            method_lower = "dare_ties"
        elif method_lower in ("svd", "ctm"):
            method_lower = "ctm"
        elif method_lower in ("della_ties",):
            method_lower = "della"
        elif method_lower in ("breadcrumbs_linear",):
            method_lower = "breadcrumbs"
        elif method_lower in ("multi_slerp", "karcher"):
            method_lower = "multislerp"
        elif method_lower in ("modelstock", "stock"):
            method_lower = "model_stock"

        if method_lower not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsloth: Unknown merge method '{self.method}'. "
                f"Supported: {', '.join(SUPPORTED_METHODS)}."
            )
        self.method = method_lower
        if self.method == "model_stock" and len(self.adapter_paths) < 3:
            raise ValueError(
                f"Unsloth: model_stock requires at least 3 adapters "
                f"(got {len(self.adapter_paths)})."
            )
        if self.normalize_weights:
            total = sum(self.weights)
            if total == 0:
                raise ValueError("Unsloth: Adapter weights must not sum to zero.")
            self.weights = [w / total for w in self.weights]
        if not (0.0 < self.density <= 1.0):
            raise ValueError(
                f"Unsloth: density must be in (0, 1], got {self.density}."
            )
        if not (0.0 <= self.drop_rate < 1.0):
            raise ValueError(
                f"Unsloth: drop_rate must be in [0, 1), got {self.drop_rate}."
            )
        if self.target_rank is not None and self.target_rank <= 0:
            raise ValueError(
                f"Unsloth: target_rank must be positive, got {self.target_rank}."
            )
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(
                f"Unsloth: gamma must be in [0, 1), got {self.gamma}."
            )
        if not (0.0 < self.select_topk <= 1.0):
            raise ValueError(
                f"Unsloth: select_topk must be in (0, 1], got {self.select_topk}."
            )
        if self.method in ("della", "della_linear"):
            if not (0.0 < self.della_epsilon):
                raise ValueError(
                    f"Unsloth: della_epsilon must be positive, got {self.della_epsilon}."
                )
            if self.density - self.della_epsilon <= 0 or self.density + self.della_epsilon >= 1:
                raise ValueError(
                    f"Unsloth: density ± della_epsilon must stay within (0, 1); "
                    f"got density={self.density}, della_epsilon={self.della_epsilon}."
                )


# ---------------------------------------------------------------------------
# Adapter I/O helpers
# ---------------------------------------------------------------------------

def _load_adapter_config(adapter_path: str) -> dict:
    """Load a PEFT adapter_config.json from *adapter_path*.

    Returns the parsed dict.  Raises ``FileNotFoundError`` when the config
    cannot be located.
    """
    import json

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        # PEFT may store configs in subdirectories; try a common fallback.
        alt = os.path.join(adapter_path, "default", "adapter_config.json")
        if os.path.isfile(alt):
            cfg_path = alt
        else:
            raise FileNotFoundError(
                f"Unsloth: No adapter_config.json found in '{adapter_path}'."
            )
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_adapter_state_dict(adapter_path: str) -> Dict[str, torch.Tensor]:
    """Load the adapter weights from *adapter_path*.

    Supports safetensors (preferred) and pytorch bin formats.
    All tensors are loaded to CPU to minimise GPU memory pressure during merge.
    """
    from safetensors.torch import load_file as safetensors_load

    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(safetensors_path):
        return safetensors_load(safetensors_path, device="cpu")

    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        f"Unsloth: No adapter weights found in '{adapter_path}'. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


# ---------------------------------------------------------------------------
# Delta reconstruction  ΔW = (α / r) × (B × A)
# ---------------------------------------------------------------------------

def _normalize_module_key(base_key: str) -> str:
    """Normalise a PEFT lora module key to the base model parameter name.

    PEFT stores keys as ``base_model.model.{module_path}``; strip the prefix
    and append ``.weight`` so the result matches ``named_parameters()``.
    """
    module_key = base_key
    for prefix in ("base_model.model.", "base_model."):
        if module_key.startswith(prefix):
            module_key = module_key[len(prefix):]
            break
    # Append .weight to match the base model state dict.
    if not module_key.endswith(".weight"):
        module_key += ".weight"
    return module_key


def _delta_for_module(A: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    """Reconstruct one module's full-rank delta: ΔW = scaling × (B @ A).

    Computed on CPU in float32 to support heterogeneous ranks and preserve
    precision. Returns a tensor of shape (out_features, in_features).
    """
    return scaling * B.to(torch.float32).mm(A.to(torch.float32))


def _adapter_scaling(adapter_config: dict) -> float:
    """The α / r scaling factor recorded in a PEFT adapter config."""
    r = adapter_config.get("r", adapter_config.get("rank", 16))
    alpha = adapter_config.get("lora_alpha", r)
    return alpha / r


def _group_lora_factors(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Group a LoRA state dict's A/B factors by base model parameter name.

    The returned mapping holds REFERENCES to the state dict's tensors, so the
    caller can drop the dict wrapper without copying the (small) low-rank
    weights. Orphan A matrices without a B are skipped, matching
    ``_reconstruct_deltas``.
    """
    # Group A and B matrices by their module key.
    # PEFT keys look like: base_model.model.{path}.lora_A.weight
    a_matrices: Dict[str, torch.Tensor] = {}
    b_matrices: Dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if "lora_A" in key:
            # Derive the base key: strip .lora_A.{adapter_name}.weight or .lora_A.weight
            a_matrices[key.split(".lora_A")[0]] = tensor
        elif "lora_B" in key:
            b_matrices[key.split(".lora_B")[0]] = tensor

    grouped: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for base_key, A in a_matrices.items():
        B = b_matrices.get(base_key)
        if B is None:
            continue  # orphan A without B — skip
        grouped[_normalize_module_key(base_key)] = (A, B)
    return grouped


def _reconstruct_deltas(
    state_dict: Dict[str, torch.Tensor],
    adapter_config: dict,
) -> Dict[str, torch.Tensor]:
    """Reconstruct full-rank ΔW tensors from a LoRA state dict.

    Returns a mapping ``{module_key: ΔW}`` where *module_key* is the
    base-model parameter name (e.g. ``model.layers.0.self_attn.q_proj.weight``).
    Computation is done layer-by-layer on CPU in float32 to support
    heterogeneous ranks and preserve precision.
    """
    scaling = _adapter_scaling(adapter_config)
    deltas: Dict[str, torch.Tensor] = {}
    for module_key, (A, B) in _group_lora_factors(state_dict).items():
        # ΔW = scaling × (B @ A)  →  shape (out_features, in_features)
        deltas[module_key] = _delta_for_module(A, B, scaling)
    return deltas


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------

def _linear_merge(
    all_deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """Merge adapter deltas via weighted sum: ΔW = Σ(w_i × ΔW_i)."""
    merged: Dict[str, torch.Tensor] = {}
    all_keys: set = set()
    for d in all_deltas:
        all_keys.update(d.keys())

    for key in all_keys:
        acc = None
        for delta_dict, w in zip(all_deltas, weights):
            if key not in delta_dict:
                continue  # this adapter didn't touch this module — zero delta
            contrib = delta_dict[key] * w
            if acc is None:
                acc = contrib
            else:
                acc = acc + contrib
        if acc is not None:
            merged[key] = acc
    return merged


def _ties_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    density: float = 0.5,
) -> torch.Tensor:
    """TIES-merge ONE module's deltas (the per-key math of ``_ties_merge``).

    Steps (aligned with mergekit's ``generalized_task_arithmetic`` TIES):
      1. **Trim**: zero out the bottom (1 − density) of each adapter's delta
         by magnitude (top-k thresholding).
      2. **Elect sign**: per element, elect the dominant sign from the
         *weighted sum of deltas* — ``sign(Σ wᵢ·δᵢ)`` — so magnitude and
         weight jointly influence the vote (mergekit ``sum`` consensus).
      3. **Disjoint merge**: for each element, average only the adapters whose
         (trimmed) delta agrees with the elected sign, normalizing by the
         *sum of the aligned weights* (``Σ wᵢ``), not the count.
    """
    if len(per_adapter_deltas) == 1:
        # Single adapter: skip TIES overhead; just scale.
        return per_adapter_deltas[0] * per_adapter_weights[0]

    shape = per_adapter_deltas[0].shape

    # Step 1: Trim — top-k by magnitude per adapter.
    trimmed = []
    for delta in per_adapter_deltas:
        flat = delta.view(-1)
        k = max(1, int(density * flat.numel()))
        threshold = flat.abs().topk(k).values[-1]
        mask = flat.abs() >= threshold
        trimmed.append((flat * mask.float()).view(delta.shape))

    # Step 2: Elect sign — mergekit ``sum`` consensus: the sign of the
    # weighted sum of the (trimmed) deltas. Magnitude and weight jointly
    # influence the election.
    weighted_sum = torch.zeros(shape, dtype=torch.float32)
    for t, w in zip(trimmed, per_adapter_weights):
        weighted_sum += w * t
    elected_sign = weighted_sum.sign()
    # Where elected sign is 0 (perfect tie), default to positive.
    elected_sign[elected_sign == 0] = 1.0

    # Step 3: Disjoint merge — average aligned contributions, normalized by
    # the sum of the aligned weights (mergekit divisor), not the count.
    acc = torch.zeros(shape, dtype=torch.float32)
    weight_sum = torch.zeros(shape, dtype=torch.float32)
    for t, w in zip(trimmed, per_adapter_weights):
        aligned = (t.sign() == elected_sign) & (t != 0)
        acc += (t * w) * aligned.float()
        weight_sum += w * aligned.float()

    weight_sum = weight_sum.clamp(min=1e-8)
    merged = acc / weight_sum
    # Cleanup
    del trimmed, weighted_sum, elected_sign, acc, weight_sum
    return merged


def _ties_merge(
    all_deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    density: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Merge adapter deltas via TIES-Merging, key by key.

    The per-module math lives in ``_ties_merge_key`` so the streaming path in
    ``merge_adapters_into_model`` can reuse it without materialising every
    adapter's full-rank deltas at once.
    """
    merged: Dict[str, torch.Tensor] = {}
    all_keys: set = set()
    for d in all_deltas:
        all_keys.update(d.keys())

    for key in all_keys:
        # Collect deltas for this key, treating absent adapters as zero.
        per_adapter_deltas = []
        per_adapter_weights = []
        for delta_dict, w in zip(all_deltas, weights):
            if key in delta_dict:
                per_adapter_deltas.append(delta_dict[key])
                per_adapter_weights.append(w)
            # absent adapter → skip (treated as zero in the disjoint average)

        if not per_adapter_deltas:
            continue

        merged[key] = _ties_merge_key(per_adapter_deltas, per_adapter_weights, density)

    return merged


def _dare_ties_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    density: float = 0.5,
    drop_rate: float = 0.5,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """DARE-TIES merge ONE module's deltas.

    Steps:
      1. **DARE (Drop and Rescale)**:
         Randomly zero out deltas with probability `drop_rate` using a Bernoulli mask,
         and rescale remaining values by `1 / (1 - drop_rate)` to keep expected magnitude intact.
      2. **TIES-Merging**:
         Trim bottom `(1 - density)` of values by magnitude, elect dominant sign,
         and perform disjoint average of sign-aligned weights.
    """
    if len(per_adapter_deltas) == 1:
        return per_adapter_deltas[0] * per_adapter_weights[0]

    # Step 1: DARE mask & rescale
    rescaled_deltas = []
    scale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    for i, delta in enumerate(per_adapter_deltas):
        if drop_rate > 0.0:
            # Keep probability is (1.0 - drop_rate)
            keep_prob = 1.0 - drop_rate
            mask = torch.bernoulli(
                torch.full(delta.shape, keep_prob, dtype=torch.float32, device=delta.device),
                generator=generator,
            )
            rescaled = delta * mask * scale
        else:
            rescaled = delta
        rescaled_deltas.append(rescaled)

    # Step 2 & 3: TIES merge on rescaled deltas
    return _ties_merge_key(rescaled_deltas, per_adapter_weights, density=density)


def _dare_linear_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    drop_rate: float = 0.5,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """DARE + Linear merge ONE module's deltas (no sign election).

    Same Drop-and-Rescale step as DARE-TIES, followed by a plain weighted
    sum instead of TIES' trim/elect/disjoint-average. Suits adapter sets
    whose signs already agree, where the sign voting adds interference
    risk rather than removing it.
    """
    if len(per_adapter_deltas) == 1:
        return per_adapter_deltas[0] * per_adapter_weights[0]

    scale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    merged: Optional[torch.Tensor] = None
    for delta, weight in zip(per_adapter_deltas, per_adapter_weights):
        if drop_rate > 0.0:
            keep_prob = 1.0 - drop_rate
            mask = torch.bernoulli(
                torch.full(delta.shape, keep_prob, dtype=torch.float32, device=delta.device),
                generator=generator,
            )
            contribution = delta * mask * scale
        else:
            contribution = delta
        contribution = contribution.to(torch.float32) * weight
        merged = contribution if merged is None else merged + contribution
    assert merged is not None  # len > 1 guard above
    return merged


def _magnitude_prune_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    density: float = 0.5,
) -> torch.Tensor:
    """Magnitude-prune merge ONE module's deltas (PEFT ``magnitude_prune``).

    Keeps the top ``density`` fraction of each adapter's delta by absolute
    magnitude (zeroing the rest), then takes the weighted sum. Like TIES'
    trim stage but without sign election — cheap sparsification that
    produces compact merged deltas for sign-consistent adapter sets.
    """
    if len(per_adapter_deltas) == 1:
        return per_adapter_deltas[0] * per_adapter_weights[0]

    merged = torch.zeros_like(per_adapter_deltas[0], dtype=torch.float32)
    for delta, weight in zip(per_adapter_deltas, per_adapter_weights):
        delta32 = delta.to(torch.float32)
        if density < 1.0:
            flat_abs = delta32.abs().flatten()
            keep_num = max(1, int(flat_abs.numel() * density))
            if keep_num < flat_abs.numel():
                # kthvalue avoids torch.quantile's ~16M-element limit.
                threshold = flat_abs.kthvalue(flat_abs.numel() - keep_num + 1).values
                mask = flat_abs.view_as(delta32) >= threshold
                delta32 = torch.where(mask, delta32, torch.zeros_like(delta32))
        merged += delta32 * weight
    return merged


def _cat_merge_key(
    per_adapter_factors: List[Tuple[torch.Tensor, torch.Tensor, float, float]],
) -> torch.Tensor:
    """Concatenation merge ONE module's LoRA factors (PEFT ``combination_type="cat"``).

    ΔW = concat_j(√(w_j·s_j)·B_j) @ concat_j(√(w_j·s_j)·A_j), preserving every
    adapter's full contribution — no averaging, hence no interference. Applied
    to model weights this is numerically the weighted sum (Linear); the
    factorized concatenation matters when the merge is exported as a
    rank-extended adapter instead of into the base weights.
    """
    a_parts = []
    b_parts = []
    for A, B, scaling, weight in per_adapter_factors:
        magnitude = (abs(weight) * abs(scaling)) ** 0.5
        sign = 1.0 if weight >= 0 else -1.0
        a_parts.append(A.to(torch.float32) * magnitude)
        b_parts.append(B.to(torch.float32) * (magnitude * sign))
    return torch.cat(b_parts, dim=1).mm(torch.cat(a_parts, dim=0))


def _ctm_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    target_rank: Optional[int] = None,
) -> torch.Tensor:
    """CtM (Compress-then-Merge) via SVD orthogonal subspace projection.

    Steps:
      1. Accumulate weighted delta sum ΔW = ∑ w_i * ΔW_i in float32.
      2. If target_rank is specified and smaller than min(dim), perform truncated SVD:
         ΔW ≈ U_r @ diag(S_r) @ Vh_r
         This isolates the principal directions of task features, eliminating destructive
         cross-adapter noise and parameter interference.
    """
    if len(per_adapter_deltas) == 1:
        merged_delta = per_adapter_deltas[0] * per_adapter_weights[0]
    else:
        merged_delta = sum(d * w for d, w in zip(per_adapter_deltas, per_adapter_weights))

    if target_rank is not None and merged_delta.ndim == 2:
        m, n = merged_delta.shape
        r = min(target_rank, m, n)
        if r < min(m, n):
            # Compute thin SVD in float32 for maximum numerical stability
            U, S, Vh = torch.linalg.svd(merged_delta.to(torch.float32), full_matrices=False)
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
            # Low-rank reconstruction
            merged_delta = (U_r * S_r.unsqueeze(0)) @ Vh_r

    return merged_delta


def _dare_ties_merge(
    all_deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    density: float = 0.5,
    drop_rate: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Merge adapter deltas via DARE-TIES, key by key."""
    merged: Dict[str, torch.Tensor] = {}
    all_keys: set = set()
    for d in all_deltas:
        all_keys.update(d.keys())

    for key in all_keys:
        per_adapter_deltas = []
        per_adapter_weights = []
        for delta_dict, w in zip(all_deltas, weights):
            if key in delta_dict:
                per_adapter_deltas.append(delta_dict[key])
                per_adapter_weights.append(w)

        if not per_adapter_deltas:
            continue

        merged[key] = _dare_ties_merge_key(
            per_adapter_deltas, per_adapter_weights, density=density, drop_rate=drop_rate, seed=seed
        )

    return merged


def _ctm_merge(
    all_deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    target_rank: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Merge adapter deltas via CtM (SVD compression), key by key."""
    merged: Dict[str, torch.Tensor] = {}
    all_keys: set = set()
    for d in all_deltas:
        all_keys.update(d.keys())

    for key in all_keys:
        per_adapter_deltas = []
        per_adapter_weights = []
        for delta_dict, w in zip(all_deltas, weights):
            if key in delta_dict:
                per_adapter_deltas.append(delta_dict[key])
                per_adapter_weights.append(w)

        if not per_adapter_deltas:
            continue

        merged[key] = _ctm_merge_key(
            per_adapter_deltas, per_adapter_weights, target_rank=target_rank
        )

    return merged


def _dare_linear_merge(
    all_deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    drop_rate: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Merge adapter deltas via DARE + Linear (no sign election), key by key."""
    merged: Dict[str, torch.Tensor] = {}
    all_keys: set = set()
    for d in all_deltas:
        all_keys.update(d.keys())

    for key in all_keys:
        per_adapter_deltas = []
        per_adapter_weights = []
        for delta_dict, w in zip(all_deltas, weights):
            if key in delta_dict:
                per_adapter_deltas.append(delta_dict[key])
                per_adapter_weights.append(w)

        if not per_adapter_deltas:
            continue

        merged[key] = _dare_linear_merge_key(
            per_adapter_deltas, per_adapter_weights, drop_rate=drop_rate, seed=seed
        )

    return merged


def _magnitude_prune_merge(
    all_deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    density: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Merge adapter deltas via magnitude pruning + weighted sum, key by key."""
    merged: Dict[str, torch.Tensor] = {}
    all_keys: set = set()
    for d in all_deltas:
        all_keys.update(d.keys())

    for key in all_keys:
        per_adapter_deltas = []
        per_adapter_weights = []
    return merged


# ---------------------------------------------------------------------------
# Sparsifiers shared by DELLA / breadcrumbs (adapted from mergekit.sparsify)
# ---------------------------------------------------------------------------

def _della_magprune_tensor(
    tensor: torch.Tensor,
    density: float,
    epsilon: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """DELLA magnitude pruning (mergekit ``della_magprune``).

    Rows are ranked by magnitude; keep-probability interpolates linearly
    between ``density - epsilon`` (lowest rank) and ``density + epsilon``
    (highest rank), then a Bernoulli mask is sampled per element.
    Reference: mergekit/sparsify.py ``della_magprune``.
    """
    if density >= 1:
        return tensor
    orig_shape = tensor.shape
    work = tensor.to(torch.float32)
    if work.dim() < 2:
        work = work.unsqueeze(0)

    sorted_indices = torch.argsort(work.abs(), dim=1, descending=False)
    ranks = sorted_indices.argsort(dim=1).to(torch.float32) + 1
    min_ranks = ranks.min(dim=1, keepdim=True).values
    max_ranks = ranks.max(dim=1, keepdim=True).values
    rank_norm = ((ranks - min_ranks) / (max_ranks - min_ranks)).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    mask = torch.bernoulli(probs, generator=generator)
    return (work * mask).reshape(orig_shape)


def _magnitude_outliers_tensor(
    tensor: torch.Tensor,
    density: float,
    gamma: float = 0.01,
) -> torch.Tensor:
    """Breadcrumbs pruning (mergekit ``magnitude_outliers``).

    First removes the ``gamma`` fraction of *largest* magnitudes (outliers),
    then removes the smallest magnitudes to reach the target ``density``.
    Reference: mergekit/sparsify.py ``magnitude_outliers``; Breadcrumbs paper
    (Davari & Belilovsky, 2024, arXiv:2312.06795).
    """
    if density >= 1:
        return tensor
    num_elems = tensor.numel()
    target_n = int(density * num_elems)
    n_top = int(gamma * num_elems)
    # Reduce the outlier removal when necessary to retain the target density.
    n_bot = max(0, num_elems - target_n - n_top)

    w = tensor.abs().reshape(-1).to(torch.float32)
    indices = torch.sort(w, descending=False).indices
    mask = torch.zeros(num_elems, dtype=torch.float32)
    mask[indices[n_bot : n_bot + target_n]] = 1.0
    return tensor.to(torch.float32) * mask.reshape_as(tensor)


# ---------------------------------------------------------------------------
# New merge strategies (per-module ``*_merge_key`` functions)
# ---------------------------------------------------------------------------

def _della_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    density: float = 0.5,
    della_epsilon: float = 0.15,
    seed: Optional[int] = None,
    sign_elect: bool = True,
) -> torch.Tensor:
    """DELLA merge ONE module's deltas (mergekit ``della_magprune`` + TIES).

    Rank-based probabilistic pruning (higher-magnitude elements get a higher
    keep probability), followed by TIES sign election + disjoint average when
    ``sign_elect`` is True, or a plain weighted sum when False
    (``della_linear``).
    Reference: mergekit/sparsify.py ``della_magprune``.
    """
    if len(per_adapter_deltas) == 1:
        return per_adapter_deltas[0] * per_adapter_weights[0]

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    pruned = [
        _della_magprune_tensor(d, density, della_epsilon, generator=generator)
        for d in per_adapter_deltas
    ]
    if sign_elect:
        return _ties_merge_key(pruned, per_adapter_weights, density=1.0)
    merged = torch.zeros_like(pruned[0], dtype=torch.float32)
    for d, w in zip(pruned, per_adapter_weights):
        merged += d * w
    return merged


def _della_linear_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    density: float = 0.5,
    della_epsilon: float = 0.15,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """DELLA + linear weighted sum (no sign election)."""
    return _della_merge_key(
        per_adapter_deltas, per_adapter_weights,
        density=density, della_epsilon=della_epsilon, seed=seed,
        sign_elect=False,
    )


def _breadcrumbs_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    density: float = 0.5,
    gamma: float = 0.01,
    sign_elect: bool = False,
) -> torch.Tensor:
    """Breadcrumbs merge ONE module's deltas (mergekit ``magnitude_outliers``).

    Drops the ``gamma`` fraction of largest magnitudes (outliers) *and* the
    smallest magnitudes down to ``density``, then merges. ``sign_elect=False``
    gives ``breadcrumbs`` (weighted sum); ``True`` gives ``breadcrumbs_ties``.
    Reference: mergekit/sparsify.py ``magnitude_outliers``; Breadcrumbs paper
    (arXiv:2312.06795).
    """
    if len(per_adapter_deltas) == 1:
        return per_adapter_deltas[0] * per_adapter_weights[0]

    pruned = [
        _magnitude_outliers_tensor(d, density=density, gamma=gamma)
        for d in per_adapter_deltas
    ]
    if sign_elect:
        return _ties_merge_key(pruned, per_adapter_weights, density=1.0)
    merged = torch.zeros_like(pruned[0], dtype=torch.float32)
    for d, w in zip(pruned, per_adapter_weights):
        merged += d * w
    return merged


def _breadcrumbs_ties_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    density: float = 0.5,
    gamma: float = 0.01,
) -> torch.Tensor:
    """Breadcrumbs + TIES sign election."""
    return _breadcrumbs_merge_key(
        per_adapter_deltas, per_adapter_weights,
        density=density, gamma=gamma, sign_elect=True,
    )


def _sce_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    select_topk: float = 1.0,
) -> torch.Tensor:
    """SCE (Sign-Consensus Erasure) merge ONE module's deltas.

    1. Optional variance-based mask: keep only the ``select_topk`` fraction of
       elements with the highest cross-adapter variance.
    2. Magnitude-based per-adapter weights: ``mean(δᵢ²) / Σ mean(δⱼ²)``.
    3. Sign-consensus erase (mergekit ``sum`` consensus): drop deltas that
       disagree with the elected sign.
    4. Weighted sum normalized by the sum of surviving weights.

    Reference: mergekit/merge_methods/sce.py; SCE paper (arXiv:2408.07990).
    Note: SCE derives its own per-adapter weights from delta energies, so the
    user-supplied ``per_adapter_weights`` are intentionally unused (kept for
    interface symmetry).
    """
    if len(per_adapter_deltas) == 1:
        return per_adapter_deltas[0] * per_adapter_weights[0]

    tvs = torch.stack([d.to(torch.float32) for d in per_adapter_deltas], dim=0)

    # Step 1: variance-based element selection
    if select_topk < 1.0:
        var = torch.var(tvs, dim=0, unbiased=False)
        nonzero = torch.count_nonzero(var)
        k = int(nonzero.item() * select_topk)
        if k == 0:
            return torch.zeros_like(tvs[0])
        _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)
        sel_mask = torch.zeros_like(var)
        sel_mask.view(-1)[indices] = 1.0
        tvs = tvs * sel_mask.unsqueeze(0)

    # Step 2: magnitude-based per-adapter weights (SCE replaces user weights)
    energies = tvs.square().reshape(tvs.shape[0], -1).mean(dim=1)
    energy_sum = energies.sum()
    if energy_sum.abs() < 1e-6:
        tv_weights = torch.ones_like(energies) / energies.shape[0]
    else:
        tv_weights = energies / energy_sum

    # Step 3: sign-consensus erase (mergekit `sum` consensus)
    sign = tvs.sign()
    majority_sign = (tvs.sum(dim=0) >= 0).to(torch.float32) * 2 - 1
    erase_mask = (sign == majority_sign).to(torch.float32)

    # Step 4: weighted sum normalized by surviving weights
    while tv_weights.dim() < tvs.dim():
        tv_weights = tv_weights.unsqueeze(-1)
    erased_weights = tv_weights * erase_mask
    merged = (tvs * erased_weights).sum(dim=0)
    merged = merged / erased_weights.sum(dim=0).clamp(min=1e-6)
    return merged


def _multislerp_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Multi-SLERP merge ONE module's deltas (barycentric hypersphere interp).

    LoRA-delta adaptation of mergekit ``multislerp``: operates directly on the
    deltas (no base tensor — the "origin" of the hypersphere is the zero
    delta). Projects deltas to a unit hypersphere, interpolates in the tangent
    space at the weighted mean, projects back, and rescales by the weighted
    average of the original norms.

    Reference: mergekit/merge_methods/multislerp.py
    """
    if len(per_adapter_deltas) == 1:
        return per_adapter_deltas[0] * per_adapter_weights[0]

    tensors = torch.stack([d.to(torch.float32) for d in per_adapter_deltas], dim=0)
    weights = torch.tensor(per_adapter_weights, dtype=torch.float32)
    weights = weights / weights.sum()

    flat = tensors.view(tensors.shape[0], -1)
    norms = torch.norm(flat, dim=-1, keepdim=True)
    unit = flat / (norms + eps)

    mean = (unit * weights.view(-1, 1)).sum(0)
    mean_norm = torch.norm(mean)
    if mean_norm < eps:
        # Antipodal / balancing weights — fall back to linear interpolation.
        return (tensors * weights.view(-1, 1, 1)).sum(0)
    mean = mean / mean_norm

    dots = (unit * mean).sum(-1, keepdim=True)
    tangent = unit - dots * mean
    tangent_result = (tangent * weights.view(-1, 1)).sum(0)

    tangent_norm = torch.norm(tangent_result) + eps
    result = mean * torch.cos(tangent_norm) + tangent_result * (
        torch.sin(tangent_norm) / tangent_norm
    )
    avg_norm = (norms.squeeze(-1) * weights).sum()
    return (result * avg_norm).view(tensors.shape[1:])


def _model_stock_merge_key(
    per_adapter_deltas: List[torch.Tensor],
    per_adapter_weights: List[float],
) -> torch.Tensor:
    """Model Stock merge ONE module's deltas (mergekit ``model_stock``).

    LoRA-delta adaptation: in mergekit's formulation the base is W₀ and the
    fine-tuned models are W₀+δᵢ, so the offsets are exactly our deltas δᵢ.
    The merged result is ``t · mean(δᵢ)`` where ``t = N·cosθ / (1+(N−1)·cosθ)``
    and cosθ is the mean pairwise cosine similarity of the deltas. When all
    deltas point the same way (cosθ→1), t→1 (keep everything); when they are
    orthogonal (cosθ→0), t→0 (fall back toward the base — i.e. merge to zero
    delta). Requires ≥ 3 adapters.

    Reference: mergekit/merge_methods/model_stock.py; Model Stock paper
    (Jang et al., 2024, arXiv:2403.19522).
    Note: ``per_adapter_weights`` are not used — Model Stock is uniformly
    weighted by construction.
    """
    n = len(per_adapter_deltas)
    if n < 3:
        raise ValueError(
            f"Unsloth: model_stock requires at least 3 adapters, got {n}."
        )

    # Compute in float32 on flattened deltas (mergekit non-filter-wise mode).
    flats = [d.reshape(-1).to(torch.float32) for d in per_adapter_deltas]

    cos_thetas = []
    for idx, offset_a in enumerate(flats):
        for offset_b in flats[idx + 1:]:
            norm_product = torch.norm(offset_a) * torch.norm(offset_b)
            cos_thetas.append(
                ((offset_a * offset_b).sum() / norm_product.clamp(min=1e-6))
                .clamp(-1, 1)
            )

    cos_theta = torch.stack(cos_thetas).mean(dim=0)
    denominator = 1 + (n - 1) * cos_theta
    # At the singularity there is no finite interpolation estimate; keep the
    # base (zero delta) instead of amplifying opposing updates.
    singular = denominator.abs() < 1e-6
    if singular:
        return torch.zeros_like(per_adapter_deltas[0])
    t = (n * cos_theta) / denominator

    average = sum(flats) / n
    merged = (t * average).reshape(per_adapter_deltas[0].shape)
    return merged


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------

def _validate_adapters(
    configs: List[dict],
    adapter_paths: List[str],
) -> str:
    """Validate that all adapters are compatible for merging.

    Returns the common base model name.  Raises ``ValueError`` on
    incompatible adapters.
    """
    base_models = set()
    for cfg, path in zip(configs, adapter_paths):
        base = cfg.get("base_model_name_or_path", "")
        if base:
            base_models.add(base)

    if len(base_models) > 1:
        raise ValueError(
            f"Unsloth: Cannot merge adapters trained on different base models. "
            f"Found: {base_models}"
        )

    # Verify all are LoRA adapters (not prompt-tuning, IA3, etc.)
    for cfg, path in zip(configs, adapter_paths):
        peft_type = cfg.get("peft_type", "").upper()
        if peft_type not in ("LORA", ""):
            raise ValueError(
                f"Unsloth: Adapter at '{path}' uses '{peft_type}', "
                f"but only LoRA adapters are supported for merging."
            )

    return base_models.pop() if base_models else ""


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def merge_adapters_into_model(
    model: torch.nn.Module,
    adapter_paths: List[str],
    weights: Optional[List[float]] = None,
    method: str = "linear",
    normalize_weights: bool = True,
    density: float = 0.5,
    drop_rate: float = 0.5,
    target_rank: Optional[int] = None,
    seed: int = 42,
    della_epsilon: float = 0.15,
    gamma: float = 0.01,
    select_topk: float = 1.0,
    hf_token=None,
    report_callback=None,
) -> torch.nn.Module:
    """Merge multiple LoRA adapters into *model* in-place.

    Parameters
    ----------
    model : torch.nn.Module
        A base model (or a PeftModel whose base is to be modified).
    adapter_paths : list[str]
        Paths to PEFT adapter directories on disk.
    weights : list[float] | None
        Per-adapter merge weights.  Defaults to equal weighting.
        Note: SCE ignores these (it derives weights from delta energies).
    method : ``"linear"`` | ``"ties"`` | ``"dare_ties"`` | ``"dare_linear"`` | \
``"magnitude_prune"`` | ``"ctm"`` | ``"cat"`` | ``"sce"`` | ``"della"`` | \
``"della_linear"`` | ``"breadcrumbs"`` | ``"breadcrumbs_ties"`` | ``"multislerp"`` | \
``"model_stock"``
        Merge strategy.
    normalize_weights : bool
        If ``True``, weights are normalised to sum to 1.
    density : float
        TIES/DARE/DELLA/breadcrumbs/magnitude-prune density parameter (fraction of
        top-k params to keep).
    drop_rate : float
        DARE drop rate (fraction of non-essential weight deltas to mask). Only used
        when ``method in ("dare_ties", "dare_linear")``.
    target_rank : int | None
        Target rank for SVD low-rank compression. Only used when ``method="ctm"``.
    seed : int
        Deterministic seed for reproducible DARE/DELLA dropout masking.
    della_epsilon : float
        DELLA probability half-width around ``density`` (mergekit epsilon).
        Only used when ``method in ("della", "della_linear")``.
    gamma : float
        Breadcrumbs outlier-removal fraction (mergekit gamma).
        Only used when ``method in ("breadcrumbs", "breadcrumbs_ties")``.
    select_topk : float
        SCE variance-selection fraction (1.0 = keep all).
        Only used when ``method == "sce"``.

    Returns
    -------
    model : torch.nn.Module
        The same model, with merged ΔW applied to its base weights.  Any
        existing PEFT adapter layers are unloaded so the model is ready for
        ``save_pretrained_merged``.
    """
    if weights is None:
        weights = [1.0] * len(adapter_paths)

    resolved_adapter_paths = [
        _resolve_adapter_path(path, hf_token=hf_token) for path in adapter_paths
    ]
    config = MultiAdapterMergeConfig(
        adapter_paths=resolved_adapter_paths,
        weights=list(weights),
        method=method,
        normalize_weights=normalize_weights,
        density=density,
        drop_rate=drop_rate,
        target_rank=target_rank,
        seed=seed,
        della_epsilon=della_epsilon,
        gamma=gamma,
        select_topk=select_topk,
    )

    # Determine human-friendly display names for logging (e.g. "my-adapter (checkpoint-500)")
    display_names = [
        _adapter_display_name(raw, resolved)
        for raw, resolved in zip(adapter_paths, config.adapter_paths)
    ]

    def report(message: str) -> None:
        print(message)
        if report_callback is not None:
            report_callback(message)

    report(f"Merge: {len(config.adapter_paths)} adapters, method={config.method}")
    report(
        "Merge weights: "
        + ", ".join(
            f"{name}={weight:.4f}"
            for name, weight in zip(display_names, config.weights)
        )
    )

    # 1. Load and validate adapter configs.
    adapter_configs: List[dict] = []
    for path in config.adapter_paths:
        cfg = _load_adapter_config(path)
        adapter_configs.append(cfg)
    _validate_adapters(adapter_configs, config.adapter_paths)

    # If model is a PeftModel, get the underlying base model first.
    base_model = model
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            base_model = model.merge_and_unload()
            report("  Unloaded existing PEFT adapter layers.")
    except ImportError:
        pass

    # Linear merging is streamed so only one adapter's deltas and one layer's
    # device copy exist at a time. This avoids an O(N adapters) memory spike.
    total_adapters = len(config.adapter_paths)
    if config.method == "linear":
        model_params = dict(base_model.named_parameters())
        applied = 0
        skipped = 0
        for idx, (path, name, cfg, weight) in enumerate(
            zip(config.adapter_paths, display_names, adapter_configs, config.weights),
            start=1,
        ):
            report(f"  Loading adapter: {name} ({path})")
            pct = int((idx - 1) / total_adapters * 100)
            report(f"Merge progress: adapter {idx} of {total_adapters} ({pct}%)")
            state_dict = _load_adapter_state_dict(path)
            deltas = _reconstruct_deltas(state_dict, cfg)
            del state_dict
            delta_elements = sum(delta.numel() for delta in deltas.values())
            delta_norm = sum(
                float(delta.float().norm().item() ** 2) for delta in deltas.values()
            ) ** 0.5
            report(
                f"Merge adapter {name}: modules={len(deltas)}, elements={delta_elements}, "
                f"delta_norm={delta_norm:.6g}, effective_norm={abs(weight) * delta_norm:.6g}, "
                f"weight={weight:.4f}"
            )
            adapter_applied = 0
            adapter_skipped = 0
            for key, delta in deltas.items():
                param = model_params.get(key)
                if param is None:
                    skipped += 1
                    adapter_skipped += 1
                    continue
                # Update in place; do not materialize a second full model state dict.
                param.data.add_(delta.to(device=param.device, dtype=param.dtype), alpha=weight)
                applied += 1
                adapter_applied += 1
            report(
                f"Merge coverage {name}: applied={adapter_applied}, "
                f"skipped={adapter_skipped}"
            )
            pct_done = int(idx / total_adapters * 100)
            report(f"Merge progress: adapter {idx} of {total_adapters} ({pct_done}%)")
            del deltas
            gc.collect()
    else:
        # TIES, DARE-TIES, and CtM need every adapter's deltas for a module only WHILE
        # that module is merged, so stream: keep the small low-rank A/B factors resident
        # and reconstruct, merge, and apply one module's deltas per iteration.
        adapter_factors: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []
        scalings: List[float] = []
        for idx, (path, name, cfg) in enumerate(
            zip(config.adapter_paths, display_names, adapter_configs),
            start=1,
        ):
            report(f"  Loading adapter: {name} ({path})")
            pct = int((idx - 1) / (total_adapters * 2) * 100)
            report(f"Merge progress: loading factors {idx} of {total_adapters} ({pct}%)")
            state_dict = _load_adapter_state_dict(path)
            adapter_factors.append(_group_lora_factors(state_dict))
            del state_dict  # the factors keep references to the tensors
            scalings.append(_adapter_scaling(cfg))
            gc.collect()

        model_params = dict(base_model.named_parameters())
        # Sorted for a deterministic merge order.
        module_keys = sorted({key for factors in adapter_factors for key in factors})
        applied = 0
        skipped = 0
        total_modules = len(module_keys)
        # Report progress periodically (every 10% or at least a few steps) to avoid log spam
        step_interval = max(1, total_modules // 10)
        for mod_idx, module_key in enumerate(module_keys, start=1):
            per_adapter_deltas = []
            per_adapter_weights = []
            # Raw (A, B, scaling, weight) tuples — used by the "cat" method,
            # which composes from factors instead of reconstructed deltas.
            per_adapter_factors = []
            for factors, scaling, weight in zip(
                adapter_factors, scalings, config.weights
            ):
                pair = factors.get(module_key)
                if pair is None:
                    continue  # this adapter didn't touch this module
                A, B = pair
                per_adapter_factors.append((A, B, scaling, weight))
                per_adapter_deltas.append(_delta_for_module(A, B, scaling))
                per_adapter_weights.append(weight)
            if not per_adapter_deltas:
                continue

            if config.method == "ties":
                delta = _ties_merge_key(
                    per_adapter_deltas, per_adapter_weights, config.density
                )
            elif config.method == "dare_ties":
                delta = _dare_ties_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    density=config.density,
                    drop_rate=config.drop_rate,
                    seed=config.seed,
                )
            elif config.method == "ctm":
                delta = _ctm_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    target_rank=config.target_rank,
                )
            elif config.method == "dare_linear":
                delta = _dare_linear_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    drop_rate=config.drop_rate,
                    seed=config.seed,
                )
            elif config.method == "magnitude_prune":
                delta = _magnitude_prune_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    density=config.density,
                )
            elif config.method == "cat":
                delta = _cat_merge_key(per_adapter_factors)
            elif config.method == "della":
                delta = _della_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    density=config.density,
                    della_epsilon=config.della_epsilon,
                    seed=config.seed,
                    sign_elect=True,
                )
            elif config.method == "della_linear":
                delta = _della_linear_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    density=config.density,
                    della_epsilon=config.della_epsilon,
                    seed=config.seed,
                )
            elif config.method == "breadcrumbs":
                delta = _breadcrumbs_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    density=config.density,
                    gamma=config.gamma,
                    sign_elect=False,
                )
            elif config.method == "breadcrumbs_ties":
                delta = _breadcrumbs_ties_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    density=config.density,
                    gamma=config.gamma,
                )
            elif config.method == "sce":
                delta = _sce_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                    select_topk=config.select_topk,
                )
            elif config.method == "multislerp":
                delta = _multislerp_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                )
            elif config.method == "model_stock":
                delta = _model_stock_merge_key(
                    per_adapter_deltas,
                    per_adapter_weights,
                )
            else:
                raise ValueError(f"Unsupported merge method: {config.method}")

            del per_adapter_deltas
            param = model_params.get(module_key)
            if param is None:
                skipped += 1
                continue
            # Update in place; do not materialize a second full model state dict.
            param.data.add_(delta.to(device=param.device, dtype=param.dtype))
            applied += 1
            del delta

            if mod_idx % step_interval == 0 or mod_idx == total_modules:
                # Factor loading took 0..50%, module merge takes 50..100%
                pct = 50 + int((mod_idx / total_modules) * 50)
                report(f"Merge progress: module {mod_idx} of {total_modules} ({pct}%)")

        report(
            f"Merge {config.method.upper()}: merged_modules={len(module_keys)}"
        )
        del adapter_factors
        gc.collect()

    report(
        f"Merge complete: applied={applied}"
        f"{f', skipped={skipped}' if skipped else ''}"
    )
    return base_model
