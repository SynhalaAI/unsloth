// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const composer = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);
const recorder = readFileSync(
  new URL("../src/features/chat/model-audio-recording.ts", import.meta.url),
  "utf8",
);
const thread = readFileSync(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);
const dictationBar = readFileSync(
  new URL("../src/components/assistant-ui/chat-dictation-bar.tsx", import.meta.url),
  "utf8",
);
const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

test("model-audio recorder is visible only for audio-input models", () => {
  assert.match(composer, /activeModel\?\.hasAudioInput/);
  assert.match(composer, /aria-label=\{activeModel\?\.hasAudioInput \? "Record audio for model"/);
  assert.match(thread, /modelAcceptsAudioInput\(activeModel\)/);
  assert.match(thread, /useModelAudioRecording\(attachRecordedAudio\)/);
  assert.match(recorder, /isFinalizing/);
  assert.match(dictationBar, /modelRecording\?/);
  assert.match(thread, /<ChatDictationBar[\s\S]*modelRecording=\{isRecordingModelAudio\}/);
});

test("stopping a recording creates the existing pending-audio attachment", () => {
  assert.match(recorder, /new PcmRecorder\(stream\)/);
  assert.match(recorder, /new File\(chunks, recordedAudioName\(contentType\), \{/);
  assert.match(recorder, /contentType === "audio\/wav"\) return "recording\.wav"/);
  assert.match(thread, /setPendingAudio\(await fileToBase64\(file\), file\.name\)/);
});

test("cancelling a recording neither attaches nor sends audio", () => {
  const cancel = recorder.slice(
    recorder.indexOf("const cancel = useCallback"),
    recorder.indexOf("const start = useCallback"),
  );
  assert.match(cancel, /cancelledRef\.current = true/);
  assert.match(cancel, /recorder\.stop\(\)/);
  assert.match(cancel, /cleanup\(\)/);
  assert.match(recorder, /if \(wasCancelled\) return/);
  assert.match(recorder, /Could not stop audio recording/);
});

test("capture failures and oversized clips are reported and release resources", () => {
  assert.match(recorder, /toast\.error\("Could not start audio recording"/);
  assert.match(recorder, /getAudioSizeError\(MAX_AUDIO_SIZE \+ 1\)/);
  assert.match(recorder, /stopMicrophone\(streamRef\.current\)/);
  assert.match(recorder, /mountedRef\.current = false;[\s\S]*cancel\(\)/);
});

test("recorded pending audio uses the established audio-base64 request path", () => {
  assert.match(composer, /audio: `data:\$\{submittedAudio\.contentType\};base64,\$\{submittedAudio\.base64\}`/);
  assert.match(adapter, /audio_base64: audioBase64/);
});
