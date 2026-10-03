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

Merging is delegated to PEFT's official weighted-adapter combination
(``peft.tuners.lora.model.LoraModel.add_weighted_adapter``): every selected
adapter is loaded onto the base model as a named PEFT adapter, combined with
``add_weighted_adapter``, and the combined adapter is merged into the base
weights with ``merge_and_unload``. No merge arithmetic is reimplemented here,
so the numbers follow PEFT exactly.

Supported methods (all are PEFT ``combination_type`` names):

  * **linear** — weighted blend of the LoRA factors. Fast; PEFT documents it
    as an approximation, because it combines the A and B factors rather than
    the full weight deltas.
  * **svd** — exact weighted sum of the deltas, re-factorised by SVD. Precise;
    the output rank is the widest source adapter's rank unless ``target_rank``
    is given.
  * **cat** — factor concatenation. A lossless weighted sum of the deltas; the
    output adapter's rank is the sum of the source ranks.
  * **ties** — TIES: trim low magnitudes, elect a sign, disjoint merge.
  * **dare_ties** — DARE drop-and-rescale, then TIES.
  * **dare_linear** — DARE drop-and-rescale, then a weighted sum.
  * **magnitude_prune** — keep the top-density magnitudes, then weighted sum.

``density`` is PEFT's keep fraction for ``ties``, ``dare_ties``,
``dare_linear`` and ``magnitude_prune``. ``target_rank`` is PEFT's
``svd_rank`` for ``svd``. The ``linear``, ``ties``, ``dare_*`` and
``magnitude_prune`` combination types require every adapter to share the same
LoRA rank; ``svd`` and ``cat`` accept mixed ranks.

After merging, the model is a plain base model and can be saved via the usual
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


SUPPORTED_METHODS = (
    "linear", "svd", "cat", "ties", "dare_ties", "dare_linear", "magnitude_prune",
)

# Accepted aliases mapped onto a supported method (kept for backward
# compatibility with the previously shipped in-house method names).
_METHOD_ALIASES = {
    "dare": "dare_ties",
    # The in-house CtM was a weighted sum compressed by a truncated SVD, which
    # is exactly PEFT's "svd" combination type with an explicit rank.
    "ctm": "svd",
}

# PEFT combination types that read the ``density`` knob (fraction kept).
_DENSITY_METHODS = ("ties", "dare_ties", "dare_linear", "magnitude_prune")


def _peft_combination_kwargs(
    method: str,
    density: float = 0.5,
    target_rank: Optional[int] = None,
) -> dict:
    """Map a validated method and its knobs onto ``add_weighted_adapter`` args.

    Only what PEFT reads is forwarded: ``density`` for the sparsifying methods
    and ``svd_rank`` for the SVD re-factorisation. Everything else stays at
    PEFT's defaults.
    """
    kwargs: dict = {"combination_type": method}
    if method in _DENSITY_METHODS:
        kwargs["density"] = float(density)
    if method == "svd" and target_rank is not None:
        kwargs["svd_rank"] = int(target_rank)
    return kwargs


@dataclass
class MultiAdapterMergeConfig:
    """Holds validated parameters for a multi-adapter merge."""

    adapter_paths: List[str]
    weights: List[float]
    method: str = "linear"
    normalize_weights: bool = True
    # TIES / DARE / magnitude-prune: fraction of each adapter's weight deltas to
    # keep (PEFT ``density``; 1.0 keeps everything).
    density: float = 0.5
    # ``svd`` only: rank of the re-factorised output adapter (None = the widest
    # source adapter's rank).
    target_rank: Optional[int] = None
    # Deterministic seed for the DARE random drop.
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
        method_lower = _METHOD_ALIASES.get(method_lower, method_lower)
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
        if self.target_rank is not None and self.target_rank <= 0:
            raise ValueError(
                f"Unsloth: target_rank must be positive, got {self.target_rank}."
            )


# ---------------------------------------------------------------------------
# Adapter path helpers
# ---------------------------------------------------------------------------



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
    target_rank: Optional[int] = None,
    seed: int = 42,
    hf_token=None,
    report_callback=None,
) -> torch.nn.Module:
    """Merge multiple LoRA adapters into *model* using PEFT.

    Each adapter in *adapter_paths* is loaded onto *model* as a named PEFT
    adapter, combined with PEFT's ``add_weighted_adapter`` and then merged into
    the base weights with ``merge_and_unload``. The merge arithmetic is PEFT's,
    not Unsloth's.

    Parameters
    ----------
    model : torch.nn.Module
        A base model. A PeftModel is accepted too: its adapter is merged into
        the base first, so the result is base + the selected adapters.
    adapter_paths : list[str]
        Paths to PEFT adapter directories, or ``{"repo_id": ..., "subfolder":
        ...}`` dicts for Hub adapters.
    weights : list[float] | None
        Per-adapter merge weights (default: equal). Normalised to sum to one
        unless *normalize_weights* is ``False``.
    method : ``"linear"`` | ``"svd"`` | ``"cat"`` | ``"ties"`` | \
``"dare_ties"`` | ``"dare_linear"`` | ``"magnitude_prune"``
        PEFT combination type. ``"dare"`` and ``"ctm"`` are accepted as
        aliases for ``"dare_ties"`` and ``"svd"``.
    normalize_weights : bool
        Normalise the weights to sum to 1 (what PEFT recommends).
    density : float
        PEFT ``density`` for ``ties``/``dare_ties``/``dare_linear``/
        ``magnitude_prune``: the fraction of weight deltas kept (1.0 keeps
        everything).
    target_rank : int | None
        PEFT ``svd_rank`` for ``svd``: the rank of the output adapter.
    seed : int
        Seed for the DARE random drop, so a merge is reproducible.
    hf_token, report_callback :
        Hub token for private adapters; ``report_callback(message)`` receives
        the progress lines that are printed.

    Returns
    -------
    model : torch.nn.Module
        A plain base model with the merged deltas applied, ready for
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
        target_rank=target_rank,
        seed=seed,
    )

    # Human-friendly display names for logging, e.g. "my-adapter (checkpoint-500)".
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

    # 1. Load and validate the adapter configs before touching the model.
    adapter_configs: List[dict] = []
    for path in config.adapter_paths:
        cfg = _load_adapter_config(path)
        adapter_configs.append(cfg)
    _validate_adapters(adapter_configs, config.adapter_paths)

    # PEFT does the merge. Imported here so the module stays torch-only at
    # import time: the Studio interference preflight loads this file directly on
    # CPU-only hosts that need no peft.
    from peft import PeftModel

    # 2. Fold any adapter already attached to the model into the base first, so
    #    only the selected adapters are applied on top of it.
    base_model = model
    if isinstance(model, PeftModel):
        base_model = model.merge_and_unload()
        report("  Merged the model's existing PEFT adapter into the base weights.")

    # 3. Load every selected adapter onto the base as a named PEFT adapter.
    total_adapters = len(config.adapter_paths)
    adapter_names: List[str] = []
    peft_model = None
    for idx, (path, name) in enumerate(zip(config.adapter_paths, display_names)):
        adapter_name = f"merge_adapter_{idx}"
        pct = int(idx / total_adapters * 100)
        report(f"  Loading adapter: {name} ({path})")
        report(f"Merge progress: adapter {idx + 1} of {total_adapters} ({pct}%)")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                base_model, path, adapter_name=adapter_name
            )
        else:
            peft_model.load_adapter(path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
    # 4. Combine the adapters with PEFT. DARE's random drop reads the global
    #    torch RNG, so the configured seed is applied for that one call and the
    #    RNG state restored afterwards (device RNG included).
    kwargs = _peft_combination_kwargs(
        config.method, config.density, config.target_rank
    )
    report(f"Merge: combining adapters with PEFT combination_type='{config.method}'")

    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    if config.method in ("dare_ties", "dare_linear"):
        torch.manual_seed(config.seed)
    try:
        try:
            peft_model.add_weighted_adapter(
                adapter_names,
                list(config.weights),
                adapter_name="merged",
                **kwargs,
            )
        except ValueError as exc:
            # PEFT rejects, for example, mixed LoRA ranks for the factor-space
            # combinations, or two adapters saving the same modules. Say which
            # method failed instead of surfacing the bare PEFT error.
            raise ValueError(
                f"Unsloth: PEFT could not combine the adapters with method "
                f"'{config.method}': {exc}"
            ) from exc
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            except Exception:
                pass

    peft_model.set_adapter("merged")
    report("Merge progress: merging the combined adapter into the base weights")
    merged_model = peft_model.merge_and_unload()

    report(f"Merge complete: method={config.method}, adapters={total_adapters}")
    gc.collect()
    return merged_model