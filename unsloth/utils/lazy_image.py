# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Deferred image decoding for vision datasets.

A VLM dataset converted for training keeps every image as a decoded PIL object,
and a decoded image costs ``width * height * 3`` bytes for as long as the run
lives. Compressed on disk that is invisible - a 1.7GB dataset of JPEGs becomes
tens of gigabytes of RGB buffers - so the process dies with a system RAM OOM
before the first optimizer step, with the model still unloaded.

``LazyImage`` stores the same input the eager path was given (the file's
compressed bytes, or its path) and decodes it when a batch is handed to the
collator, so the working set is one batch of pixels instead of the whole
dataset.

Training quality is unchanged by construction: ``to_pil`` re-issues the exact
``Image.open(...)`` / ``.convert("RGB")`` calls the eager conversion used, and
``to_rgb`` records which of those the caller had applied, so the collator sees
byte-identical pixels in the same colour mode.
"""

from __future__ import annotations

import os
from typing import Any, Optional

__all__ = [
    "LazyImage",
    "is_lazy_image",
    "resolve_lazy_images",
]

# A string cell naming one of these is not a local file, so it cannot be
# reopened later: URLs need downloading and data: URIs are inline payloads.
_REMOTE_PREFIXES = ("http://", "https://", "data:")


def _is_remote(value: str) -> bool:
    return value.startswith(_REMOTE_PREFIXES)


class LazyImage:
    """An image held as compressed bytes or a file path until a batch needs it.

    Attributes:
        path: Local path to reopen, or None when only bytes are held.
        bytes: Compressed file bytes, or None when only a path is held.
        to_rgb: Apply ``.convert("RGB")`` on decode, matching the eager
            conversion branch the value came from. Cells read from a
            ``datasets.Image`` column were already decoded by ``datasets`` with
            no colour conversion, so they keep ``to_rgb = False`` and stay
            pixel-identical to an eager run.
    """

    __slots__ = ("path", "bytes", "to_rgb")

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        bytes: Optional[bytes] = None,  # noqa: A002 - mirrors the datasets cell name
        to_rgb: bool = False,
    ) -> None:
        self.path = path
        self.bytes = bytes
        self.to_rgb = to_rgb

    @classmethod
    def from_value(cls, value: Any, *, to_rgb: bool = False) -> Optional["LazyImage"]:
        """Wrap ``value`` when it names an image that can be decoded later.

        Accepts the shapes a dataset hands out: an undecoded ``datasets.Image``
        cell (``{"bytes": ..., "path": ...}``), raw bytes, or a path to a file
        that exists. Returns None for anything that has to be decoded now - a
        decoded PIL image (nothing left to defer), a remote or inline string, or
        a path that is not there, which the caller's eager branch then reports
        exactly as it did before.
        """
        if isinstance(value, LazyImage):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            return cls(bytes = bytes(value), to_rgb = to_rgb)
        if isinstance(value, str):
            if _is_remote(value) or not os.path.isfile(value):
                return None
            # A path cell reached the eager path as a string, which that path
            # converted to RGB; record that so the decode matches it.
            return cls(path = value, to_rgb = True)
        if isinstance(value, dict):
            raw = value.get("bytes")
            path = value.get("path")
            raw = bytes(raw) if raw else None
            if not isinstance(path, str) or not path:
                path = None
            if raw is None and path is None:
                return None
            if path is not None and _is_remote(path):
                return None
            return cls(path = path, bytes = raw, to_rgb = to_rgb)
        return None

    def to_pil(self):
        """Decode now, reproducing the eager conversion this value replaced."""
        from io import BytesIO
        from PIL import Image

        if self.bytes is not None:
            image = Image.open(BytesIO(self.bytes))
            if self.to_rgb:
                return image.convert("RGB")
            # datasets.Image.decode_example calls load() and hands back the
            # image as decoded; do the same so no file handle is left open.
            image.load()
            return image
        image = Image.open(self.path)
        if self.to_rgb:
            return image.convert("RGB")
        image.load()
        return image

    def __repr__(self) -> str:
        source = self.path if self.bytes is None else f"<{len(self.bytes)} bytes>"
        return f"LazyImage({source}, to_rgb = {self.to_rgb})"


def is_lazy_image(value: Any) -> bool:
    return isinstance(value, LazyImage)


def _looks_like_messages(value: Any) -> bool:
    return isinstance(value, list) and (
        len(value) == 0
        or (isinstance(value[0], dict) and ("content" in value[0] or "role" in value[0]))
    )


def _resolve_messages(messages: list) -> list:
    """Return ``messages`` with every deferred image part decoded, in place order.

    The input is left untouched: a dataset row is fetched again for every epoch,
    so mutating it would turn the first batch's decode into the object the rest
    of the run reuses.
    """
    resolved = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        decoded_content = None
        for position, part in enumerate(content):
            if not isinstance(part, dict) or not is_lazy_image(part.get("image")):
                continue
            if decoded_content is None:
                decoded_content = list(content)
            decoded_content[position] = {
                **part,
                "image": part["image"].to_pil(),
            }
        if decoded_content is None:
            continue
        if resolved is None:
            resolved = list(messages)
        resolved[index] = {**message, "content": decoded_content}
    return messages if resolved is None else resolved


def _resolve_example(example: Any) -> Any:
    if _looks_like_messages(example):
        return _resolve_messages(example)
    if not isinstance(example, dict):
        return example
    for column in ("messages", "conversations"):
        value = example.get(column)
        if _looks_like_messages(value):
            decoded = _resolve_messages(value)
            if decoded is value:
                return example
            return {**example, column: decoded}
    return example


def resolve_lazy_images(examples: Any) -> Any:
    """Decode the deferred images of one batch before it reaches a collator.

    Accepts a batch of examples, a single example, or a bare messages list, and
    returns the input untouched when it holds no ``LazyImage`` - an eager
    dataset pays nothing for the walk.
    """
    if not isinstance(examples, list):
        return _resolve_example(examples)
    if examples and all(_looks_like_messages(example) for example in examples):
        return _resolve_messages(examples)
    return [_resolve_example(example) for example in examples]
