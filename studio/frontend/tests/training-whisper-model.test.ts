// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isWhisperTrainingModel } from "../src/features/training/lib/model-modality-inference.ts";

test("the detected whisper audio type wins over an unrelated model name", () => {
  assert.equal(
    isWhisperTrainingModel({ audioType: "whisper", identifiers: ["org/kanai-asr"] }),
    true,
  );
});

test("audio vision models are not whisper, so they get no Whisper language option", () => {
  assert.equal(
    isWhisperTrainingModel({
      audioType: "audio_vlm",
      identifiers: ["unsloth/gemma-4-E2B-it"],
    }),
    false,
  );
});

test("a Whisper repo id is recognised before the model details arrive", () => {
  assert.equal(
    isWhisperTrainingModel({
      identifiers: ["SinhaSpeech/whisper-small-sinhala-run12-best"],
    }),
    true,
  );
});

test("a local Whisper folder is recognised by its leaf name", () => {
  assert.equal(
    isWhisperTrainingModel({
      identifiers: ["C:\\models\\whisper-large-v3"],
    }),
    true,
  );
});

test("a Whisper base model marks a saved adapter as Whisper", () => {
  assert.equal(
    isWhisperTrainingModel({
      audioType: null,
      identifiers: ["org/my-adapter", "openai/whisper-small"],
    }),
    true,
  );
});

test("other audio models stay false: TTS, ASR and audio towers", () => {
  for (const identifier of [
    "sesame/csm-1b",
    "unsloth/orpheus-3b-0.1-ft",
    "nvidia/parakeet-tdt-0.6b-v2",
    "unsloth/gemma-3n-E4B-it",
  ]) {
    assert.equal(isWhisperTrainingModel({ identifiers: [identifier] }), false, identifier);
  }
});

test("a name that only contains the letters whisper is not a Whisper model", () => {
  assert.equal(
    isWhisperTrainingModel({ identifiers: ["org/whispering-woods-tts"] }),
    false,
  );
});

test("no type and no identifiers is not whisper", () => {
  assert.equal(isWhisperTrainingModel({}), false);
});
