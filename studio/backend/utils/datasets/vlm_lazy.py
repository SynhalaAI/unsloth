# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read image cells without decoding them, so a VLM conversion can defer it.

A conversion that walks a ``datasets.Dataset`` triggers ``datasets`` decoding
every Image cell it touches, so the whole dataset lands in RAM as decoded PIL
objects before the trainer sees a single row. Flipping the feature to
``decode=False`` changes only how a cell is handed back - ``{"bytes", "path"}``
instead of a PIL image - and the bytes are the same ones an eager decode would
have read, so nothing about what the model trains on changes.

Both helpers here are best effort on purpose: a dataset they cannot reshape is
returned untouched, which leaves the caller on its existing eager path instead of
introducing a new failure.
"""

from __future__ import annotations

from typing import Any, Iterable

from loggers import get_logger

logger = get_logger(__name__)


def images_undecoded(features: Any) -> Any:
    """Return a copy of ``features`` with every decoding Image feature turned off.

    Nested structs and lists are walked, so an Image inside a messages column is
    found too. Anything else is returned unchanged.
    """
    import dataclasses

    try:
        from datasets import Image
    except ImportError:  # pragma: no cover - datasets is a Studio hard dependency
        return features

    if isinstance(features, Image):
        return Image(decode = False) if getattr(features, "decode", True) else features
    if isinstance(features, dict):
        return type(features)(
            {key: images_undecoded(value) for key, value in features.items()}
        )
    if isinstance(features, list):
        return [images_undecoded(value) for value in features]
    if dataclasses.is_dataclass(features) and getattr(features, "feature", None) is not None:
        return dataclasses.replace(features, feature = images_undecoded(features.feature))
    return features


def undecoded_image_columns(dataset: Any, columns: Iterable[str]) -> Any:
    """Hand back ``dataset`` with the named columns no longer decoding images.

    Nesting is handled, so a Llava ``images`` column typed ``Sequence(Image())`` is
    covered too. A column holding no Image feature is left exactly as it is: casting a
    string column to ``Image`` would rewrite each cell as ``{"path", "bytes"}`` and hide
    the URL-versus-local-path distinction the conversion branches on.

    Returns the input unchanged for anything that is not a map-style
    ``datasets.Dataset`` or cannot be reshaped.
    """
    try:
        from datasets import Dataset
    except ImportError:  # pragma: no cover - datasets is a Studio hard dependency
        return dataset
    if not isinstance(dataset, Dataset):
        return dataset
    features = getattr(dataset, "features", None)
    if not features:
        return dataset
    for column in columns:
        feature = features.get(column) if hasattr(features, "get") else None
        if feature is None:
            continue
        undecoded = images_undecoded(feature)
        if undecoded is feature or undecoded == feature:
            continue
        try:
            dataset = dataset.cast_column(column, undecoded)
        except Exception as error:  # noqa: BLE001 - an optimisation, never a gate
            logger.info(f"Keeping {column!r} decoded; cannot defer its decoding: {error}")
    return dataset


def _looks_like_messages(value: Any) -> bool:
    return isinstance(value, list) and (
        len(value) == 0
        or (isinstance(value[0], dict) and ("content" in value[0] or "role" in value[0]))
    )


def _wrap_row(row: Any, lazy_image_cls) -> Any:
    """Replace the deferred-able image parts of one row, or return it unchanged."""
    if not isinstance(row, dict):
        return row
    for column in ("messages", "conversations"):
        messages = row.get(column)
        if not _looks_like_messages(messages):
            continue
        wrapped = None
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            wrapped_content = None
            for part_index, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue
                if isinstance(part.get("image"), lazy_image_cls):
                    continue
                lazy = lazy_image_cls.from_value(part.get("image"))
                if lazy is None:
                    # Already a decoded PIL image: there is nothing left to defer
                    # and re-encoding it here would lose quality.
                    continue
                if wrapped_content is None:
                    wrapped_content = list(content)
                wrapped_content[part_index] = {**part, "image": lazy}
            if wrapped_content is None:
                continue
            if wrapped is None:
                wrapped = list(messages)
            wrapped[message_index] = {**message, "content": wrapped_content}
        if wrapped is None:
            continue
        return {**row, column: wrapped}
    return row


def wrap_lazy_images_in_messages(dataset: Any) -> list:
    """Return the rows of a messages-format dataset with their images deferred.

    Used by the branch that finds a dataset already in standard VLM messages
    format, which otherwise materialises the whole dataset as a list and decodes
    every image into RAM. Image cells are read with decoding off first, so the
    wrap sees compressed bytes rather than pixels; a cell that is already decoded
    is left exactly as it was.
    """
    from unsloth.utils.lazy_image import LazyImage

    try:
        from datasets import Dataset

        if isinstance(dataset, Dataset):
            undecoded = images_undecoded(getattr(dataset, "features", None) or {})
            try:
                dataset = dataset.cast(features = undecoded)
            except Exception as error:  # noqa: BLE001 - an optimisation, never a gate
                logger.info(f"Could not read VLM images undecoded: {error}")
    except ImportError:  # pragma: no cover - datasets is a Studio hard dependency
        pass

    return [_wrap_row(row, LazyImage) for row in dataset]
