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

SUPPORTED_METHODS = ("linear", "ties", "dare_ties", "ctm")


@dataclass
class MultiAdapterMergeConfig:
    """Holds validated parameters for a multi-adapter merge."""

    adapter_paths: List[str]
    weights: List[float]
    method: str = "linear"
    normalize_weights: bool = True
    # TIES / DARE-TIES specific: fraction of params to keep (top-k by magnitude).
    density: float = 0.5
    # DARE-specific: fraction of weight deltas to randomly drop before rescaling.
    drop_rate: float = 0.5
    # CtM / SVD specific: target low-rank for truncated SVD compression.
    target_rank: Optional[int] = None
    # Deterministic seed for reproducible dropout masking in DARE.
    seed: int = 42

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

        if method_lower not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsloth: Unknown merge method '{self.method}'. "
                f"Supported: {', '.join(SUPPORTED_METHODS)}."
            )
        self.method = method_lower
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

    Steps:
      1. **Trim**: zero out the bottom (1 − density) of each adapter's delta
         by magnitude (top-k thresholding).
      2. **Elect sign**: per element, majority-vote across adapters to pick the
         dominant sign.
      3. **Disjoint merge**: for each element, average only the adapters whose
         (trimmed) delta agrees with the elected sign.
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

    # Step 2: Elect sign — majority vote (weighted).
    # +1 for positive, −1 for negative, 0 for zero.
    sign_votes = torch.zeros(shape, dtype=torch.float32)
    for t, w in zip(trimmed, per_adapter_weights):
        sign_votes += w * t.sign()
    elected_sign = sign_votes.sign()
    # Where elected sign is 0 (perfect tie), default to positive.
    elected_sign[elected_sign == 0] = 1.0

    # Step 3: Disjoint merge — average only aligned contributions.
    acc = torch.zeros(shape, dtype=torch.float32)
    count = torch.zeros(shape, dtype=torch.float32)
    for t, w in zip(trimmed, per_adapter_weights):
        aligned = (t.sign() == elected_sign) & (t != 0)
        acc += (t * w) * aligned.float()
        count += aligned.float()

    count = count.clamp(min=1.0)
    merged = acc / count
    # Cleanup
    del trimmed, sign_votes, elected_sign, acc, count
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
    method : ``"linear"`` | ``"ties"`` | ``"dare_ties"`` | ``"ctm"``
        Merge strategy.
    normalize_weights : bool
        If ``True``, weights are normalised to sum to 1.
    density : float
        TIES/DARE-TIES density parameter (fraction of top-k params to keep). Only used
        when ``method in ("ties", "dare_ties")``.
    drop_rate : float
        DARE drop rate (fraction of non-essential weight deltas to mask). Only used
        when ``method="dare_ties"``.
    target_rank : int | None
        Target rank for SVD low-rank compression. Only used when ``method="ctm"``.
    seed : int
        Deterministic seed for reproducible dropout masking.

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
    if config.method == "linear":
        model_params = dict(base_model.named_parameters())
        applied = 0
        skipped = 0
        for path, name, cfg, weight in zip(
            config.adapter_paths, display_names, adapter_configs, config.weights
        ):
            report(f"  Loading adapter: {name} ({path})")
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
            del deltas
            gc.collect()
    else:
        # TIES, DARE-TIES, and CtM need every adapter's deltas for a module only WHILE
        # that module is merged, so stream: keep the small low-rank A/B factors resident
        # and reconstruct, merge, and apply one module's deltas per iteration.
        adapter_factors: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []
        scalings: List[float] = []
        for path, name, cfg in zip(config.adapter_paths, display_names, adapter_configs):
            report(f"  Loading adapter: {name} ({path})")
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
        for module_key in module_keys:
            per_adapter_deltas = []
            per_adapter_weights = []
            for factors, scaling, weight in zip(
                adapter_factors, scalings, config.weights
            ):
                pair = factors.get(module_key)
                if pair is None:
                    continue  # this adapter didn't touch this module
                A, B = pair
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
