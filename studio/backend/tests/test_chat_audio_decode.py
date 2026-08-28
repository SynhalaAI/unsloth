# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Chat audio attachments decode WAV files without torchcodec."""

import base64
import builtins
import io
import wave

import numpy as np

import routes.inference as inference_route


def test_wav_decode_does_not_require_torchaudio(monkeypatch):
    samples = (np.sin(np.arange(160) * 0.2) * 16000).astype(np.int16)
    audio_file = io.BytesIO()
    with wave.open(audio_file, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())

    original_import = builtins.__import__

    def reject_torchaudio(name, *args, **kwargs):
        if name == "torchaudio":
            raise AssertionError("WAV decoding should not import torchaudio")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torchaudio)
    decoded = inference_route._decode_audio_base64(base64.b64encode(audio_file.getvalue()).decode())

    assert decoded.dtype == np.float32
    assert decoded.shape == (160,)
    np.testing.assert_allclose(decoded, samples.astype(np.float32) / 32768, atol = 1e-4)