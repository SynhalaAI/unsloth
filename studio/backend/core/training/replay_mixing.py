# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Optional replay datasets mixed into the main training data.

Replay rows re-expose the model to older data while a new dataset is learned, slowing
catastrophic forgetting across staged fine-tunes (e.g. audio -> vision -> text). Mixing
happens AFTER formatting and AFTER the eval split is carved out, so replay rows never
enter eval and share the main dataset's final schema.

Disabled by default: a run without ``replay_datasets`` never reaches this module's mixing
path. Sources: Hugging Face Hub repo ids and local device paths; the MLX worker refuses mixing
up front and the schema refuses streaming, raw-text/CPT and preference runs.
"""

__all__ = [
    "apply_replay_mixing",
    "mix_replay_datasets",
]

import logging

logger = logging.getLogger(__name__)

_MAX_REPLAY_DATASETS = 4


def mix_replay_datasets(main_dataset, replay_datasets, probabilities, *, seed = 3407):
    """Interleave replay rows into the main dataset.

    ``first_exhausted`` stops when the main dataset runs out, so one epoch is still one
    pass over the main data with replay rows sprinkled through it.
    """
    from datasets import interleave_datasets

    replay_total = float(sum(probabilities))
    if replay_total >= 1.0:
        raise ValueError(
            f"Replay probabilities sum to {replay_total}; the main dataset must keep a share (< 1.0)."
        )
    weights = [1.0 - replay_total, * (float(p) for p in probabilities)]
    return interleave_datasets(
        [main_dataset, *replay_datasets],
        probabilities = weights,
        seed = seed,
        stopping_strategy = "first_exhausted",
    )


def _resolve_local_replay_files(paths):
    """Resolve local replay paths to concrete dataset files (parity with
    UnslothTrainer._resolve_local_files, without importing the torch-heavy trainer)."""
    import os
    from pathlib import Path
    from utils.paths import dataset_files_in_dir, resolve_dataset_path

    all_files = []
    for entry in paths:
        if os.path.isabs(entry):
            resolved = entry
        elif os.path.exists(entry):
            resolved = os.path.abspath(entry)
        else:
            resolved = str(resolve_dataset_path(entry))
        path_obj = Path(resolved)
        if path_obj.is_dir():
            all_files.extend(str(p) for p in dataset_files_in_dir(path_obj))
        else:
            all_files.append(str(path_obj))
    return all_files


def _load_local_replay(spec):
    """Load a local (device) replay dataset: HF disk directory or a json/jsonl/csv/parquet file."""
    from pathlib import Path
    from datasets import load_dataset, load_from_disk
    from utils.datasets.cells import csv_as_text_kwargs

    entry = str(spec.get("dataset") or "").strip()
    direct = Path(entry)
    if direct.is_dir() and ((direct / "dataset_info.json").exists() or (direct / "state.json").exists()):
        return load_from_disk(str(direct))
    files = _resolve_local_replay_files([entry])
    if not files:
        raise ValueError("Replay dataset: no files found at " + entry)
    first_ext = Path(files[0]).suffix.lower()
    if first_ext in (".json", ".jsonl"):
        loader = "json"
    elif first_ext == ".csv":
        loader = "csv"
    elif first_ext == ".parquet":
        loader = "parquet"
    else:
        raise ValueError("Unsupported replay dataset format: " + files[0])
    return load_dataset(
        loader, data_files = files, split = "train", **csv_as_text_kwargs(files)
    )


def _load_and_format_replay(
    spec,
    *,
    model_name,
    tokenizer,
    format_type,
    custom_format_mapping,
    is_vlm,
    hf_token,
):
    from datasets import load_dataset
    from utils.datasets import format_and_template_dataset

    name = str(spec.get("dataset") or "").strip()
    if spec.get("source", "huggingface") == "local":
        dataset = _load_local_replay(spec)
    else:
        subset = spec.get("subset") or None
        split = str(spec.get("split") or "train") or "train"
        load_kwargs = {"split": split}
        if hf_token:
            load_kwargs["token"] = hf_token
        dataset = (
            load_dataset(name, subset, **load_kwargs)
            if subset
            else load_dataset(name, **load_kwargs)
        )
    info = format_and_template_dataset(
        dataset,
        model_name = model_name,
        tokenizer = tokenizer,
        is_vlm = is_vlm,
        format_type = format_type,
        dataset_name = name,
        custom_format_mapping = custom_format_mapping,
        lazy_images = True,
    )
    if not info.get("success", True):
        errors = info.get("errors") or ["unknown formatting error"]
        raise ValueError("Replay dataset formatting failed for " + name)
    return info["dataset"]


def apply_replay_mixing(
    dataset_info,
    replay_datasets,
    *,
    model_name,
    tokenizer,
    format_type,
    custom_format_mapping = None,
    is_vlm = False,
    hf_token = None,
    seed = 3407,
    on_status = None,
):
    """Load, format and interleave replay datasets into a formatted ``dataset_info`` dict.

    No-op (same object back) without specs, so an unarmed run is byte-identical.
    """
    specs = [
        entry if isinstance(entry, dict) else entry.model_dump()
        for entry in (replay_datasets or [])
    ]
    specs = [entry for entry in specs if str(entry.get("dataset") or "").strip()]
    if not specs or not isinstance(dataset_info, dict) or dataset_info.get("dataset") is None:
        return dataset_info
    if len(specs) > _MAX_REPLAY_DATASETS:
        raise ValueError(
            f"At most {_MAX_REPLAY_DATASETS} replay datasets are supported, got {len(specs)}."
        )

    main_dataset = dataset_info["dataset"]
    if not hasattr(main_dataset, "__len__"):
        raise ValueError("Replay datasets require a map-style (non-streaming) main dataset.")

    formatted = []
    for spec in specs:
        name = str(spec.get("dataset")).strip()
        if on_status:
            on_status(f"Loading replay dataset {name}...")
        formatted.append(
            _load_and_format_replay(
                spec,
                model_name = model_name,
                tokenizer = tokenizer,
                format_type = format_type,
                custom_format_mapping = custom_format_mapping,
                is_vlm = is_vlm,
                hf_token = hf_token,
            )
        )

    probabilities = [float(spec.get("probability") or 0.0) for spec in specs]
    mixed = mix_replay_datasets(main_dataset, formatted, probabilities, seed = seed)
    main_rows = len(main_dataset)
    replay_rows = sum(len(ds) for ds in formatted)
    logger.info(
        "Replay mixing: %s main rows + %s replay rows across %s dataset(s), interleaved to %s rows",
        main_rows,
        replay_rows,
        len(formatted),
        len(mixed),
    )
    if on_status:
        on_status(f"Dataset ready ({len(mixed):,} samples, replay mixed)")
    return {
        **dataset_info,
        "dataset": mixed,
        "replay_datasets": [str(spec.get("dataset")).strip() for spec in specs],
        "replay_rows": replay_rows,
    }
