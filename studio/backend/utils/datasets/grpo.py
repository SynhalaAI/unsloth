# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GRPO dataset detection & standardization utilities.

Supports verifiable-reward GRPO datasets in both conversational and plain-text forms:
- Conversational prompt + verifiable target, e.g.
  ``{"prompt": [{"role": "user", "content": "..."}], "answer": "B"}``
- Plain-text prompt + verifiable target, e.g.
  ``{"question": "...", "answer": "..."}`` or ``{"problem": "...", "solution": "..."}``
- Conversational prompt-only datasets (lists of user/system messages with no assistant turn).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:
    from datasets import Dataset, DatasetDict, IterableDataset
except Exception:  # noqa: BLE001
    class Dataset:  # type: ignore[no-redef]
        pass

    class DatasetDict(dict):  # type: ignore[no-redef]
        pass

    class IterableDataset:  # type: ignore[no-redef]
        pass


GRPO_PROMPT_COLUMNS = ("prompt", "question", "problem", "query", "instruction")
GRPO_ANSWER_COLUMNS = (
    "answer",
    "solution",
    "ground_truth",
    "expected_answer",
    "final_answer",
    "target",
    "reference",
)
GRPO_SYSTEM_COLUMNS = ("system", "system_prompt", "system_instruction")

_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "model", "bot"})
_NON_GRPO_SFT_COLUMNS = frozenset({"output", "completion", "response", "completions"})


def _sample_rows(dataset: Any, limit: int = 20) -> List[Dict[str, Any]]:
    if isinstance(dataset, dict):
        if "dataset" in dataset:
            return _sample_rows(dataset["dataset"], limit = limit)
        first_val = next(iter(dataset.values()), None)
        if hasattr(first_val, "column_names"):
            return _sample_rows(first_val, limit = limit)
        return [dataset]

    rows: List[Dict[str, Any]] = []
    try:
        total = min(len(dataset), limit)
        for index in range(total):
            row = dataset[index]
            if isinstance(row, dict):
                rows.append(row)
        return rows
    except Exception:  # noqa: BLE001
        pass

    try:
        for index, row in enumerate(dataset):
            if index >= limit:
                break
            if isinstance(row, dict):
                rows.append(row)
    except Exception:  # noqa: BLE001
        return []
    return rows


def _is_prompt_only_messages(value: Any) -> bool:
    """True when value is a non-empty list of chat messages containing no assistant turns."""
    if not isinstance(value, list) or len(value) == 0:
        return False
    saw_message = False
    for turn in value:
        if not isinstance(turn, dict):
            return False
        role = turn.get("role", turn.get("from"))
        content = turn.get("content", turn.get("value"))
        if role is None or content is None:
            return False
        saw_message = True
        if str(role).strip().lower() in _ASSISTANT_ROLES:
            return False
    return saw_message


def _is_valid_prompt_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return _is_prompt_only_messages(value)


def detect_grpo_columns(
    column_names: List[str] | set[str],
    sample_rows: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Optional[str]]]:
    """Detect GRPO prompt/answer/system columns without conflicting with preference or SFT datasets."""
    cols_lower = {str(c).lower(): str(c) for c in column_names}

    from .preference import detect_preference_columns

    if detect_preference_columns(column_names) is not None:
        return None

    if _NON_GRPO_SFT_COLUMNS.intersection(cols_lower.keys()):
        return None

    prompt_col = None
    for cand in GRPO_PROMPT_COLUMNS:
        if cand in cols_lower:
            prompt_col = cols_lower[cand]
            break

    if not prompt_col:
        return None

    answer_col = None
    for cand in GRPO_ANSWER_COLUMNS:
        if cand in cols_lower:
            answer_col = cols_lower[cand]
            break

    system_col = None
    for cand in GRPO_SYSTEM_COLUMNS:
        if cand in cols_lower:
            system_col = cols_lower[cand]
            break

    if sample_rows:
        prompt_values = [
            row.get(prompt_col)
            for row in sample_rows
            if isinstance(row, dict) and prompt_col in row and row.get(prompt_col) is not None
        ]
        if not prompt_values:
            return None
        for value in prompt_values:
            if isinstance(value, list) and not _is_prompt_only_messages(value):
                return None
        if not any(_is_valid_prompt_value(value) for value in prompt_values):
            return None

    if answer_col is None:
        if prompt_col.lower() != "prompt":
            return None
        if sample_rows and not any(
            _is_prompt_only_messages(row.get(prompt_col))
            for row in sample_rows
            if isinstance(row, dict)
        ):
            return None

    return {
        "prompt": prompt_col,
        "answer": answer_col,
        "system": system_col,
    }


def is_grpo_dataset(dataset: Any) -> bool:
    """Check whether a dataset or sample dict matches a GRPO prompt/verifiable-answer schema."""
    if isinstance(dataset, dict) and "dataset" in dataset:
        dataset = dataset["dataset"]

    cols = None
    if hasattr(dataset, "column_names"):
        cols = dataset.column_names
        if isinstance(cols, dict):
            cols = next(iter(cols.values()), [])
    elif isinstance(dataset, dict):
        cols = list(dataset.keys())

    if not cols:
        return False

    rows = _sample_rows(dataset)
    return detect_grpo_columns(cols, sample_rows = rows) is not None


def _normalize_message_list(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for turn in messages:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", turn.get("from", "user"))).strip().lower()
        if role == "human":
            role = "user"
        content = turn.get("content", turn.get("value", ""))
        normalized.append({"role": role, "content": str(content)})
    return normalized


def standardize_grpo_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset],
    column_mapping: Optional[Dict[str, Optional[str]]] = None,
    num_proc: Optional[int] = None,
) -> Union[Dataset, DatasetDict, IterableDataset]:
    """Standardize a GRPO dataset so it exposes `prompt` (and `answer` when present)."""
    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                split: standardize_grpo_dataset(ds, column_mapping, num_proc)
                for split, ds in dataset.items()
            }
        )

    cols = getattr(dataset, "column_names", None)
    if not cols and hasattr(dataset, "features"):
        cols = list(dataset.features.keys())
    if not cols:
        return dataset

    rows = _sample_rows(dataset)
    mapping = column_mapping or detect_grpo_columns(cols, sample_rows = rows)
    if not mapping:
        return dataset

    prompt_src = mapping.get("prompt")
    answer_src = mapping.get("answer")
    system_src = mapping.get("system")

    needs_message_norm = False
    for row in rows:
        val = row.get(prompt_src) if prompt_src else None
        if isinstance(val, list) and any(
            isinstance(turn, dict) and ("from" in turn or "value" in turn) for turn in val
        ):
            needs_message_norm = True
            break

    if (
        prompt_src == "prompt"
        and (answer_src is None or answer_src == "answer")
        and not system_src
        and not needs_message_norm
    ):
        return dataset

    def _remap_row(row: Dict[str, Any]) -> Dict[str, Any]:
        new_row = dict(row)
        prompt_val = row.get(prompt_src) if prompt_src and prompt_src in row else row.get("prompt")
        system_val = row.get(system_src) if system_src and system_src in row else None
        system_text = (
            system_val.strip() if isinstance(system_val, str) and system_val.strip() else None
        )

        if isinstance(prompt_val, list):
            messages = _normalize_message_list(prompt_val)
            if system_text and not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": system_text}, *messages]
            new_row["prompt"] = messages
        elif isinstance(prompt_val, str):
            new_row["prompt"] = prompt_val
            if system_text and "system" not in new_row:
                new_row["system"] = system_text
        elif prompt_src and prompt_src in row and prompt_src != "prompt":
            new_row["prompt"] = prompt_val

        if answer_src and answer_src in row and answer_src != "answer":
            new_row["answer"] = row[answer_src]

        return new_row

    if not system_src and not needs_message_norm and hasattr(dataset, "rename_columns"):
        rename_map: Dict[str, str] = {}
        if prompt_src and prompt_src != "prompt" and "prompt" not in cols:
            rename_map[prompt_src] = "prompt"
        if answer_src and answer_src != "answer" and "answer" not in cols:
            rename_map[answer_src] = "answer"
        if rename_map:
            try:
                return dataset.rename_columns(rename_map)
            except Exception:  # noqa: BLE001
                pass

    if hasattr(dataset, "map"):
        try:
            return dataset.map(
                _remap_row,
                num_proc = num_proc,
                desc = "Standardizing columns for GRPO",
            )
        except TypeError:
            return dataset.map(_remap_row)

    return dataset