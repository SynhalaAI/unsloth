# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whisper decoder language/task for audio fine-tuning.

Whisper's tokenizer prefixes every label with ``<|startoftranscript|><|xx|><|task|>``
(the language token is built by ``WhisperProcessor.from_pretrained(language = ...)``
and lands on the decoder's inputs through the labels), so a run's language is
trained in, not chosen at inference: a Sinhala run whose labels say ``<|en|>``
only produces Sinhala after the decoder prompt it actually saw, and any
inference that asks for Sinhala - or lets Whisper detect it - starts the decoder
from a prefix the run never trained on.

The trainer used to pin that choice to English. Resolving it lives here so
``trainer.load_model`` keeps its integration to passing the values through, and an
unset language still means English, which is what every existing run trained with.
"""

from __future__ import annotations

from typing import Any, Optional

# The historical hardcode, kept as the default so an unset request is unchanged.
DEFAULT_WHISPER_LANGUAGE = "English"
DEFAULT_WHISPER_TASK = "transcribe"

# The only two tasks Whisper has tokens for: <|transcribe|> / <|translate|>.
_WHISPER_TASKS = ("transcribe", "translate")


def normalize_whisper_task(task: Optional[str]) -> str:
    """*task* as Whisper spells it, with the default for unset or unknown values."""
    if not isinstance(task, str) or not task.strip():
        return DEFAULT_WHISPER_TASK
    normalized = task.strip().lower()
    return normalized if normalized in _WHISPER_TASKS else DEFAULT_WHISPER_TASK


def _language_token_from_processor(processor: Any) -> Optional[str]:
    """The ``<|xx|>`` the processor was built with, or None if it cannot be read.

    ``WhisperProcessor.from_pretrained(language = ...)`` stores the language it
    was given in the tokenizer's ``prefix_tokens``
    (``[<|startoftranscript|>, <|xx|>, <|task|>, ...]``), which is the same list the
    labels are built from - so reading it back cannot drift from the labels.
    """
    if processor is None:
        return None
    tokenizer = getattr(processor, "tokenizer", None) or processor
    prefix = getattr(tokenizer, "prefix_tokens", None)
    if not prefix or len(prefix) < 2:
        return None
    try:
        token = tokenizer.convert_ids_to_tokens(prefix[1])
    except Exception:  # noqa: BLE001 - a processor without the helper is not a verdict
        return None
    if isinstance(token, str) and token.startswith("<|") and token.endswith("|>"):
        return token
    return None


def resolve_generation_language(language: Optional[str], processor: Any = None) -> Optional[str]:
    """``generation_config.language`` for *language*: itself, as ``<|xx|>``.

    Accepts whatever ``WhisperProcessor`` accepts - a Whisper language name
    ("Sinhala") or the token itself ("<|si|>") - and returns the token form so
    decoding starts from the prefix the labels were built with. ``None``/blank
    means "leave Whisper's own default". A processor that cannot be read falls
    back to what was passed in, or to ``<|xx|>`` for a two-letter code.
    """
    if language is None:
        return None
    text = str(language).strip()
    if not text:
        return None
    if text.startswith("<|") and text.endswith("|>"):
        return text
    token = _language_token_from_processor(processor)
    if token is not None:
        return token
    if len(text) == 2 and text.isalpha():
        return f"<|{text.lower()}|>"
    return text


def apply_whisper_decoder_language(
    model: Any,
    processor: Any,
    language: Optional[str] = None,
    task: Optional[str] = None,
) -> Optional[str]:
    """Point *model*'s decoder at the language/task its labels were built with.

    Returns the language token that was applied (for logging), or None when the
    model has no generation config to set.
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return None
    language_token = resolve_generation_language(language, processor)
    if language_token is not None:
        generation_config.language = language_token
    generation_config.task = normalize_whisper_task(task)
    return language_token
