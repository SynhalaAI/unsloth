# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OCR CER/WER metrics for OCR document-transcription training runs.

Computes Character Error Rate (CER) and Word Error Rate (WER) from model
predictions against the ground-truth transcription labels during evaluation.
Strictly best-effort: a missing ``jiwer`` install or a malformed evaluation
payload degrades to no metrics rather than failing the evaluation or the
training run. Metrics are produced only while evaluating an OCR run (the
``is_ocr_training`` path in the trainer), never on the train step hot path.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def compute_cer_wer(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """Mean CER/WER over sample pairs, one per-sample rate like the OCR benchmarks.

    Empty reference texts are handled explicitly because ``jiwer`` raises on
    them: an equal empty prediction counts as a perfect score, and any output
    where nothing was expected counts as a total error. Samples that still
    fail to score are skipped rather than poisoning the mean.

    Returns ``{}`` when ``jiwer`` is unavailable or nothing could be scored.
    """
    try:
        from jiwer import cer as _jiwer_cer
        from jiwer import wer as _jiwer_wer
    except ImportError:
        return {}

    cer_values: list[float] = []
    wer_values: list[float] = []
    for reference, hypothesis in zip(references, hypotheses):
        ref_text = (reference or "").strip()
        hyp_text = (hypothesis or "").strip()
        if ref_text == "":
            if hyp_text == "":
                cer_values.append(0.0)
                wer_values.append(0.0)
            else:
                cer_values.append(1.0)
                wer_values.append(1.0)
            continue
        try:
            cer_values.append(float(_jiwer_cer(ref_text, hyp_text)))
            wer_values.append(float(_jiwer_wer(ref_text, hyp_text)))
        except Exception:
            continue

    if not cer_values:
        return {}
    return {
        "cer": float(sum(cer_values) / len(cer_values)),
        "wer": float(sum(wer_values) / len(wer_values)),
    }


def make_ocr_metrics_fn(tokenizer: Any) -> Callable[[Any], dict[str, float]]:
    """Return a Transformers ``compute_metrics`` function for OCR runs.

    The returned function decodes greedy predictions (logits ``argmax``) and
    the masked labels back to text with the training tokenizer, then scores
    CER/WER per sample. Best-effort: any decode or scoring failure returns an
    empty dict, which HF folds into the eval logs as no metrics.
    """

    def _decode(token_ids: Any) -> list[str]:
        if token_ids is None:
            return []
        import numpy as np

        array = np.asarray(token_ids)
        if array.ndim == 0:
            array = array.reshape(1, -1)
        if array.size == 0:
            return []
        masked = array == -100
        if masked.any():
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = getattr(tokenizer, "pad_id", None)
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            if pad_id is None:
                pad_id = 0
            array = array.copy()
            array[masked] = pad_id
        try:
            return [
                (text or "").strip()
                for text in tokenizer.batch_decode(array, skip_special_tokens = True)
            ]
        except Exception:
            return []

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        try:
            if hasattr(eval_pred, "predictions"):
                raw_predictions = eval_pred.predictions
                raw_labels = getattr(eval_pred, "label_ids", None)
            elif isinstance(eval_pred, (tuple, list)) and len(eval_pred) >= 2:
                raw_predictions, raw_labels = eval_pred[0], eval_pred[1]
            else:
                return {}
            # Seq2Seq-style evals wrap (logits, past_key_values); unwrap to logits.
            if isinstance(raw_predictions, (tuple, list)) and len(raw_predictions) > 0:
                raw_predictions = raw_predictions[0]

            import numpy as np

            predictions = np.asarray(raw_predictions)
            labels = np.asarray(raw_labels)
            # Logits (batch, seq, vocab) → greedy token ids; already-decode ids are left as-is.
            if predictions.ndim == 3 and labels.ndim == 2:
                predictions = predictions.argmax(axis = -1)

            hypotheses = _decode(predictions)
            references = _decode(labels)
            if not hypotheses or not references:
                return {}
            return compute_cer_wer(references, hypotheses)
        except Exception:
            return {}

    return compute_metrics