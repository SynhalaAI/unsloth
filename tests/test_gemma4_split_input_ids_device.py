# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A split Gemma 4 must hand the embedding card the ids its mask is built from.

transformers 5.5 builds `multimodal_mask` from `input_ids` and then combines it with the
embedding's output in one `torch.where`. The balanced plan pins the tied embedding to the
head's card, so ids left on the root hook's card mix cuda:0 with cuda:1 at the first step.
E2B/E4B reach that line on every row because their text config sets
`hidden_size_per_layer_input`, media or not.
"""

import ast
import os
import types

import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")
HELPER = "_move_gemma4_token_types_to_input_embeddings"
_HEAD_CARD = torch.device("cuda", 1)


def _move_hook():
    """vision.py imports the whole stack, so run the one helper under test on its own."""
    src = open(VISION, encoding = "utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == HELPER:
            namespace = {}
            exec(ast.get_source_segment(src, node), namespace)
            return namespace[HELPER]
    raise AssertionError(f"{HELPER} is gone from vision.py")


class _Ids:
    """Records the device it was asked to move to; `.to` is all the hook uses."""

    def __init__(self):
        self.moved_to = []

    def to(self, device):
        self.moved_to.append(device)
        return self


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type = "gemma4")
        self.seen = {}

    def get_input_embeddings(self):
        return types.SimpleNamespace(weight = types.SimpleNamespace(device = _HEAD_CARD))

    def forward(self, input_ids = None, mm_token_type_ids = None, pixel_values = None):
        self.seen = {
            "input_ids": input_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
        }


class _Outer(torch.nn.Module):
    def __init__(self, device_map):
        super().__init__()
        self.model = _Inner()
        self.hf_device_map = device_map


_SPLIT = {"model.language_model.embed_tokens": 1, "model.language_model.layers.0": 0}
_ONE_CARD = {"": 0}


def test_split_gemma4_moves_input_ids_and_token_types_to_the_embedding_card():
    model = _Outer(_SPLIT)
    _move_hook()(model)

    ids, token_types = _Ids(), _Ids()
    media = object()
    model.model(input_ids = ids, mm_token_type_ids = token_types, pixel_values = media)

    assert ids.moved_to == [_HEAD_CARD]
    assert token_types.moved_to == [_HEAD_CARD]
    # The row's own kwargs reach the model, not a copy the hook kept.
    assert model.model.seen["input_ids"] is ids
    # Only the ids the mask is built from move; media stays where its tower was placed.
    assert model.model.seen["pixel_values"] is media


def test_split_gemma4_moves_positional_input_ids():
    model = _Outer(_SPLIT)
    _move_hook()(model)

    ids = _Ids()
    model.model(ids)

    assert ids.moved_to == [_HEAD_CARD]
    assert model.model.seen["input_ids"] is ids


def test_single_card_gemma4_is_left_alone():
    model = _Outer(_ONE_CARD)
    _move_hook()(model)

    ids = _Ids()
    model.model(input_ids = ids)

    assert ids.moved_to == []
    assert not model.model._forward_pre_hooks


@pytest.mark.parametrize("model_type", ["gemma3", "llama"])
def test_other_architectures_get_no_hook(model_type):
    model = _Outer(_SPLIT)
    model.model.config = types.SimpleNamespace(model_type = model_type)
    _move_hook()(model)

    ids = _Ids()
    model.model(input_ids = ids)

    assert ids.moved_to == []
    assert not model.model._forward_pre_hooks
