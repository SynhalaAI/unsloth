# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whisper decoder label budget: ``config.max_target_positions``.

``WhisperForConditionalGeneration.forward`` (huggingface/transformers PR #33196)
refuses labels longer than the decoder's positional-embedding table::

    ValueError: Labels' sequence length 453 cannot exceed the maximum allowed
    length of 448 tokens.

Rows that long are therefore dropped while the dataset is prepared - the source
dataset (often a Hub upload the user cannot or should not rewrite) is never
modified, only this run's view of it.

Resolving the limit lives here so ``trainer._preprocess_whisper_dataset`` keeps
its integration to a call and a skip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# WhisperConfig.max_target_positions: the decoder ships this many learned positions.
DEFAULT_MAX_TARGET_POSITIONS = 448


def _positive_int(value: Any) -> Optional[int]:
    """*value* as a positive int, else None (bools are not counts)."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _read_config_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding = "utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _config_json(model_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """``config.json`` for *model_name*, read from disk only.

    Never touches the network: dataset preparation must not block on the Hub.
    A local checkpoint path is read directly; a repo id is looked up in the
    active HF cache, then the library's default cache root.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    name = model_name.strip()

    direct = _read_config_json(Path(name).expanduser() / "config.json")
    if direct is not None:
        return direct

    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - no Hub in a unit-test sandbox
        return None

    roots = []
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        roots.append(str(active_hf_hub_cache()))
    except Exception:  # noqa: BLE001 - no Studio cache setting, the library default still answers
        pass
    roots.append(None)

    for root in roots:
        try:
            cached = try_to_load_from_cache(name, "config.json", cache_dir = root)
        except Exception:  # noqa: BLE001 - an unreadable cache is not a verdict, try the next root
            continue
        if not isinstance(cached, str):
            # A miss is None and a negative cache is a sentinel path, not a readable file.
            continue
        data = _read_config_json(Path(cached))
        if data is not None:
            return data
    return None


def resolve_max_target_positions(
    model: Any = None,
    model_name: Optional[str] = None,
) -> int:
    """The label budget the loaded model will enforce.

    Order: the loaded model's config (training later reads the same value), then
    the checkpoint's cached ``config.json`` - dataset preparation runs BEFORE the
    model loads in the Studio worker, so that is the usual source - then
    :data:`DEFAULT_MAX_TARGET_POSITIONS`, Whisper's own default.
    """
    limit = _positive_int(getattr(getattr(model, "config", None), "max_target_positions", None))
    if limit is not None:
        return limit
    config = _config_json(model_name)
    if config is not None:
        limit = _positive_int(config.get("max_target_positions"))
        if limit is not None:
            return limit
    return DEFAULT_MAX_TARGET_POSITIONS


def decoder_label_length(input_ids: Sequence[int], bos_token_id: Any = None) -> int:
    """Label length the collator hands the model, i.e. after its leading-BOS strip.

    ``DataCollatorSpeechSeq2SeqWithPadding`` drops a leading BOS before padding,
    and the model validates the padded batch, so counting the raw ids would skip
    rows that fit exactly once the BOS is removed.
    """
    length = len(input_ids)
    if bos_token_id is not None and length > 0 and input_ids[0] == bos_token_id:
        length -= 1
    return length


def labels_fit_decoder(
    input_ids: Sequence[int],
    limit: int,
    bos_token_id: Any = None,
) -> bool:
    """True when *input_ids* survives ``limit`` after the collator's BOS strip."""
    return decoder_label_length(input_ids, bos_token_id) <= limit