# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An image dataset checked through the text-only path must fall back to the VLM detector.

The local/upload dataset check (utils.datasets.dataset_utils) mirrors the hub copy: a check
made with ``is_vlm=False`` runs the text detector, which has no notion of image columns, so
an image/text dataset would be reported as "unknown" and strand the UI on manual mapping.
"""

from utils.datasets.dataset_utils import check_dataset_format


def _sign_language_row(**extra):
    """Column layout of a real image/text dataset (SynhalaAI/AehAI-VSION-Sinhala)."""
    return {
        "image": {"bytes": b"\x89PNG-fake", "path": None},
        "concept_id": "C001",
        "concept_en": "signboard",
        "concept_si": "salakunu-puvaruva",
        "category": "traffic",
        "subcategory": "warning",
        "region": "LK",
        "recognition_level": 1,
        "priority": 5,
        "image_filename": "C001_0001.jpg",
        "image_format": "jpg",
        "instruction": "Describe this signboard in Sinhala",
        "text": "me sala text eka",
        **extra,
    }


def test_text_path_falls_back_to_vlm_detection_for_image_datasets():
    result = check_dataset_format([_sign_language_row()], is_vlm = False)

    assert result["detected_format"] == "simple_image_text"
    assert result["detected_image_column"] == "image"
    assert result["detected_text_column"] == "text"
    assert result["requires_manual_mapping"] is False
    assert result["is_image"] is True


def test_vlm_flag_still_detects_the_same_layout():
    result = check_dataset_format([_sign_language_row()], is_vlm = True)

    assert result["detected_format"] == "simple_image_text"
    assert result["detected_image_column"] == "image"
    assert result["detected_text_column"] == "text"