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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_METHODS = ("linear", "ties")


@dataclass
class MultiAdapterMergeConfig:
    """Holds validated parameters for a multi-adapter merge."""

    adapter_paths: List[str]
    weights: List[float]
    method: str = "linear"
    normalize_weights: bool = True
    # TIES-specific: fraction of params to keep (top-k by magnitude).
    density: float = 0.5

    def __post_init__(self):
        if not self.adapter_paths:
            raise ValueError("Unsloth: At least one adapter path is required.")
        if len(self.adapter_paths) != len(self.weights):
            raise ValueError(
                f"Unsloth: Number of adapter paths ({len(self.adapter_paths)}) "
                f"must match number of weights ({len(self.weights)})."
            )
        method_lower = self.method.lower().replace("-", "_")
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
    r = adapter_config.get("r", adapter_config.get("rank", 16))
    alpha = adapter_config.get("lora_alpha", r)
    scaling = alpha / r

    # Group A and B matrices by their module key.
    # PEFT keys look like: base_model.model.{path}.lora_A.weight
    a_matrices: Dict[str, torch.Tensor] = {}
    b_matrices: Dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if "lora_A" in key:
            # Derive the base key: strip .lora_A.{adapter_name}.weight or .lora_A.weight
            base_key = key.split(".lora_A")[0]
            a_matrices[base_key] = tensor
        elif "lora_B" in key:
            base_key = key.split(".lora_B")[0]
            b_matrices[base_key] = tensor

    deltas: Dict[str, torch.Tensor] = {}
    for base_key in a_matrices:
        A = a_matrices[base_key]  # shape (r, in_features)
        B = b_matrices.get(base_key)
        if B is None:
            continue  # orphan A without B — skip
        # ΔW = scaling × (B @ A)  →  shape (out_features, in_features)
        delta = scaling * B.to(torch.float32).mm(A.to(torch.float32))
        # Normalise the key to the base model namespace.
        # PEFT stores keys as base_model.model.{module_path}; strip prefix.
        module_key = base_key
        for prefix in ("base_model.model.", "base_model."):
            if module_key.startswith(prefix):
                module_key = module_key[len(prefix):]
                break
        # Append .weight to match the base model state dict.
        if not module_key.endswith(".weight"):
            module_key += ".weight"
        deltas[module_key] = delta
        del delta  # free intermediate immediately
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


def _ties_merge(
    all_deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    density: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Merge adapter deltas via TIES-Merging.

    Steps per parameter tensor:
      1. **Trim**: zero out the bottom (1 − density) of each adapter's delta
         by magnitude (top-k thresholding).
      2. **Elect sign**: per element, majority-vote across adapters to pick the
         dominant sign.
      3. **Disjoint merge**: for each element, average only the adapters whose
         (trimmed) delta agrees with the elected sign.
    """
    merged: Dict[str, torch.Tensor] = {}
    all_keys: set = set()
    for d in all_deltas:
        all_keys.update(d.keys())

    for key in all_keys:
        # Collect deltas for this key, treating absent adapters as zero.
        per_adapter_deltas = []
        per_adapter_weights = []
        shape = None
        for delta_dict, w in zip(all_deltas, weights):
            if key in delta_dict:
                per_adapter_deltas.append(delta_dict[key])
                per_adapter_weights.append(w)
                shape = delta_dict[key].shape
            # absent adapter → skip (treated as zero in the disjoint average)

        if not per_adapter_deltas:
            continue

        if len(per_adapter_deltas) == 1:
            # Single adapter: skip TIES overhead; just scale.
            merged[key] = per_adapter_deltas[0] * per_adapter_weights[0]
            continue

        n = len(per_adapter_deltas)

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
        merged[key] = acc / count
        # Cleanup
        del trimmed, sign_votes, elected_sign, acc, count

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
    hf_token=None,
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
    method : ``"linear"`` | ``"ties"``
        Merge strategy.
    normalize_weights : bool
        If ``True``, weights are normalised to sum to 1.
    density : float
        TIES density parameter (fraction of top-k params to keep). Only used
        when ``method="ties"``.

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
    )

    print(f"Unsloth: Merging {len(config.adapter_paths)} adapters "
          f"using '{config.method}' strategy...")

    # 1. Load and validate adapter configs.
    adapter_configs: List[dict] = []
    for path in config.adapter_paths:
        cfg = _load_adapter_config(path)
        adapter_configs.append(cfg)
    _validate_adapters(adapter_configs, config.adapter_paths)

    # 2. Reconstruct deltas layer-by-layer for each adapter.
    all_deltas: List[Dict[str, torch.Tensor]] = []
    for path, cfg in zip(config.adapter_paths, adapter_configs):
        print(f"  Loading adapter: {path}")
        state_dict = _load_adapter_state_dict(path)
        deltas = _reconstruct_deltas(state_dict, cfg)
        all_deltas.append(deltas)
        del state_dict  # free raw A/B matrices
        gc.collect()

    # 3. Merge deltas.
    print(f"  Computing merged deltas ({config.method})...")
    if config.method == "linear":
        merged_deltas = _linear_merge(all_deltas, config.weights)
    elif config.method == "ties":
        merged_deltas = _ties_merge(all_deltas, config.weights, config.density)
    else:
        raise ValueError(f"Unsupported method: {config.method}")

    del all_deltas
    gc.collect()

    # 4. Apply merged deltas to the base model.
    # If model is a PeftModel, get the underlying base model first.
    base_model = model
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            base_model = model.merge_and_unload()
            print("  Unloaded existing PEFT adapter layers.")
    except ImportError:
        pass

    state_dict = base_model.state_dict()
    applied = 0
    skipped = 0
    for key, delta in merged_deltas.items():
        if key in state_dict:
            param = state_dict[key]
            device = param.device
            dtype = param.dtype
            # Add delta (computed in float32) cast to the param's dtype.
            new_val = param.to(torch.float32) + delta.to(param.device)
            state_dict[key] = new_val.to(dtype)
            applied += 1
        else:
            skipped += 1

    base_model.load_state_dict(state_dict, strict=False)
    del merged_deltas, state_dict
    gc.collect()

    print(f"  Applied merged deltas to {applied} parameters"
          f"{f' (skipped {skipped} unmatched keys)' if skipped else ''}.")
    print("Unsloth: Multi-adapter merge complete!")
    return base_model
