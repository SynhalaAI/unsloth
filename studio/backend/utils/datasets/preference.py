# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Preference dataset formatting & standardization utilities for DPO and CPO training.
Supports datasets with prompt/instruction, chosen, and rejected pairs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from datasets import Dataset, DatasetDict, IterableDataset
except Exception:  # noqa: BLE001
    class Dataset:  # type: ignore[no-redef]
        pass
    class DatasetDict(dict):  # type: ignore[no-redef]
        pass
    class IterableDataset:  # type: ignore[no-redef]
        pass


PREFERENCE_CHOSEN_COLUMNS = ("chosen", "preferred", "accepted", "positive", "chosen_response")
PREFERENCE_REJECTED_COLUMNS = ("rejected", "dispreferred", "negative", "rejected_response")
PREFERENCE_PROMPT_COLUMNS = ("prompt", "instruction", "input", "question", "query", "context")
PREFERENCE_SYSTEM_COLUMNS = ("system", "system_prompt", "system_instruction")


def detect_preference_columns(column_names: List[str] | set[str]) -> Optional[Dict[str, str]]:
    """
    Detects if the column names contain preference pairs (chosen and rejected),
    and optionally a prompt and system column.
    Returns mapping of {'chosen': col, 'rejected': col, 'prompt': col (or None), 'system': col (or None)} or None.
    """
    cols_lower = {str(c).lower(): str(c) for c in column_names}

    chosen_col = None
    for cand in PREFERENCE_CHOSEN_COLUMNS:
        if cand in cols_lower:
            chosen_col = cols_lower[cand]
            break

    rejected_col = None
    for cand in PREFERENCE_REJECTED_COLUMNS:
        if cand in cols_lower:
            rejected_col = cols_lower[cand]
            break

    if not chosen_col or not rejected_col:
        return None

    prompt_col = None
    for cand in PREFERENCE_PROMPT_COLUMNS:
        if cand in cols_lower:
            prompt_col = cols_lower[cand]
            break

    system_col = None
    for cand in PREFERENCE_SYSTEM_COLUMNS:
        if cand in cols_lower:
            system_col = cols_lower[cand]
            break

    return {
        "chosen": chosen_col,
        "rejected": rejected_col,
        "prompt": prompt_col,
        "system": system_col,
    }


def is_preference_dataset(dataset: Any) -> bool:
    """Check whether a dataset or dictionary has preference (chosen/rejected) columns."""
    if hasattr(dataset, "column_names"):
        cols = dataset.column_names
        if isinstance(cols, dict):  # DatasetDict
            first_split = next(iter(cols.values()))
            cols = first_split
        return detect_preference_columns(cols or []) is not None
    if isinstance(dataset, dict):
        return detect_preference_columns(list(dataset.keys())) is not None
    return False


def standardize_preference_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset],
    column_mapping: Optional[Dict[str, str]] = None,
    num_proc: Optional[int] = None,
) -> Union[Dataset, DatasetDict, IterableDataset]:
    """
    Standardizes a preference dataset so it contains standard columns expected by
    DPOTrainer / CPOTrainer: 'prompt' (if applicable), 'chosen', and 'rejected'.
    """
    if isinstance(dataset, DatasetDict):
        return DatasetDict({
            split: standardize_preference_dataset(ds, column_mapping, num_proc)
            for split, ds in dataset.items()
        })

    cols = getattr(dataset, "column_names", None)
    if not cols and hasattr(dataset, "features"):
        cols = list(dataset.features.keys())

    if not cols:
        return dataset

    mapping = column_mapping or detect_preference_columns(cols)
    if not mapping:
        return dataset

    chosen_src = mapping.get("chosen")
    rejected_src = mapping.get("rejected")
    prompt_src = mapping.get("prompt")
    system_src = mapping.get("system")

    def _remap_row(row: Dict[str, Any]) -> Dict[str, Any]:
        new_row = dict(row)
        if chosen_src and chosen_src in row and chosen_src != "chosen":
            new_row["chosen"] = row[chosen_src]
        if rejected_src and rejected_src in row and rejected_src != "rejected":
            new_row["rejected"] = row[rejected_src]

        prompt_val = row.get(prompt_src) if prompt_src and prompt_src in row else row.get("prompt", "")
        system_val = row.get(system_src) if system_src and system_src in row else None

        if system_val and isinstance(system_val, str) and system_val.strip() and isinstance(prompt_val, str):
            new_row["prompt"] = f"{system_val.strip()}\n\n{prompt_val.strip()}"
        elif prompt_src and prompt_src in row and prompt_src != "prompt":
            new_row["prompt"] = prompt_val

        return new_row

    if hasattr(dataset, "rename_columns") or hasattr(dataset, "map"):
        rename_map = {}
        if chosen_src and chosen_src != "chosen" and "chosen" not in cols:
            rename_map[chosen_src] = "chosen"
        if rejected_src and rejected_src != "rejected" and "rejected" not in cols:
            rename_map[rejected_src] = "rejected"
        if prompt_src and prompt_src != "prompt" and "prompt" not in cols:
            rename_map[prompt_src] = "prompt"

        # Only use fast column renaming if no system prompt needs merging
        if rename_map and hasattr(dataset, "rename_columns") and not system_src:
            try:
                return dataset.rename_columns(rename_map)
            except Exception:
                pass

        if hasattr(dataset, "map"):
            try:
                return dataset.map(
                    _remap_row,
                    num_proc=num_proc,
                    desc="Standardizing preference columns for DPO/CPO",
                )
            except TypeError:
                return dataset.map(_remap_row)

    return dataset
