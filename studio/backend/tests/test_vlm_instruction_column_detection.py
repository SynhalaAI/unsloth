from hub.utils.dataset_format import check_dataset_format


def _vlm_row(**extra):
    return {"image": "sample.jpg", "text": "A caption", **extra}


def test_detects_non_empty_explicit_instruction_column():
    result = check_dataset_format([_vlm_row(instruction = "Describe it")], is_vlm = True)
    assert result["detected_instruction_column"] == "instruction"
    assert result["detected_image_column"] == "image"
    assert result["detected_text_column"] == "text"
    assert result["detected_format"] == "simple_image_text"


def test_ignores_empty_first_row_instruction_column():
    result = check_dataset_format([_vlm_row(instruction = "   ")], is_vlm = True)
    assert result["detected_instruction_column"] is None


def test_vlm_without_instruction_column_reports_none():
    result = check_dataset_format([_vlm_row()], is_vlm = True)
    assert result["detected_instruction_column"] is None


def test_detects_alternate_explicit_instruction_column():
    result = check_dataset_format([_vlm_row(prompt = "Describe it")], is_vlm = True)
    assert result["detected_instruction_column"] == "prompt"


def test_audio_dataset_does_not_detect_an_instruction_for_an_audio_vlm():
    result = check_dataset_format(
        [
            {
                "audio": {"array": [0.0], "sampling_rate": 16000},
                "text": "A transcript",
                "instruction": "Translate this audio.",
            }
        ],
        is_vlm = True,
    )

    assert result["detected_format"] == "audio"
    assert result["detected_instruction_column"] is None
    assert result["detected_audio_column"] == "audio"
    assert result["detected_text_column"] == "text"


def _sign_language_row(**extra):
    """Column layout of a real image/text dataset (SynhalaAI/AehAI-VSION-Sinhala): an
    embedded image, metadata columns, and two text columns the text detector cannot see."""
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
    # A dataset check made before a vision model was selected runs with is_vlm=False;
    # the text detector has no notion of image columns, so the VLM detector decides.
    result = check_dataset_format([_sign_language_row()], is_vlm = False)

    assert result["detected_format"] == "simple_image_text"
    assert result["detected_image_column"] == "image"
    assert result["detected_text_column"] == "text"
    assert result["requires_manual_mapping"] is False
    assert result["is_image"] is True
