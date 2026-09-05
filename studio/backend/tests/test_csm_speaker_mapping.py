# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

def test_csm_speaker_role_resolution():
    speaker_id_map: dict[str, str] = {}

    def get_role_str(raw_val) -> str:
        raw_speaker = str(raw_val).strip() if raw_val is not None else "0"
        if raw_speaker.isdigit() or (raw_speaker.startswith("-") and raw_speaker[1:].isdigit()):
            return str(int(raw_speaker))
        if raw_speaker not in speaker_id_map:
            speaker_id_map[raw_speaker] = str(len(speaker_id_map))
        return speaker_id_map[raw_speaker]

    # Test pure digits / integers
    assert get_role_str(0) == "0"
    assert get_role_str("0") == "0"
    assert get_role_str("1") == "1"
    assert get_role_str("  2  ") == "2"

    # Test None
    assert get_role_str(None) == "0"

    # Test arbitrary strings (e.g. 'Nirvana', 'speaker_a')
    assert get_role_str("Nirvana") == "0"
    assert get_role_str("Nirvana") == "0"  # deterministic for same speaker
    assert get_role_str("Nirvana") == "0"


if __name__ == "__main__":
    test_csm_speaker_role_resolution()
    print("CSM speaker mapping tests passed successfully!")

