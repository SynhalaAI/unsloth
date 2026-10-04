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
  assert.match(composer, /useModelAudioRecording\(attachRecordedAudio\)/);
  // The mic gates on the shared audio-input helper: a row that omits
  // `has_audio_input` but names an audio-input type still records.
  assert.match(
    composer,
    /aria-label=\{modelAcceptsAudioInput\(activeModel\) \? "Record audio for model"/,
  );
  assert.match(thread, /modelAcceptsAudioInput\(activeModel\)/);
  assert.match(thread, /useModelAudioRecording\(attachRecordedAudio\)/);
  assert.match(recorder, /isFinalizing/);
  assert.match(dictationBar, /modelRecording\?/);
  assert.match(thread, /<ChatDictationBar[\s\S]*modelRecording=\{isRecordingModelAudio\}/);
});

test("stopping a recording attaches the clip where the send can carry it", () => {
  assert.match(recorder, /new PcmRecorder\(stream\)/);
  assert.match(recorder, /new File\(chunks, recordedAudioName\(contentType\), \{/);
  assert.match(recorder, /contentType === "audio\/wav"\) return "recording\.wav"/);
  // A composer ATTACHMENT, not the runtime store. composer().send() dispatches
  // nothing while the composer is empty, so a store-only clip produced no run at
  // all and leaked the pre-stream reservation, which then refused every later
  // submit as a running response on an idle chat.
  assert.match(thread, /await aui\.composer\(\)\.addAttachment\(file\)/);
  assert.doesNotMatch(thread, /setPendingAudio\(await fileToBase64\(file\)/);
});

test("an empty composer gives the pre-stream reservation back", () => {
  // Without this the reservation is never consumed (no run) and never released,
  // so every later submit reads it as a live run and is refused with "Wait for
  // the current response to finish" while nothing is generating.
  const sendStart = thread.indexOf("const sendReservedComposer = useCallback");
  assert.ok(sendStart > 0, "sendReservedComposer moved");
  const send = thread.slice(
    sendStart,
    thread.indexOf("const interceptSend", sendStart),
  );
  const reserve = send.indexOf("reservePreStreamRun(preStreamThreadIds, {");
  const guard = send.indexOf("if (!sentText.trim()");
  const release = send.indexOf("releasePreStreamRunReservation(reservationToken);", guard);
  const dispatched = send.indexOf("aui.composer().send();");
  assert.ok(reserve >= 0 && guard > reserve, "the guard reads the reserved composer");
  assert.ok(release > guard, "an empty composer releases the reservation");
  assert.ok(
    release < dispatched,
    "the reservation is given back before composer().send() is called",
  );
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
  // One part per clip since Chat took multiple audio files per message, so the
  // recorded clip carries its own content type into the same data URL.
  assert.match(composer, /audio: `data:\$\{clip\.contentType\};base64,\$\{clip\.base64\}`/);
  // The single-chat path stages the clip as pending audio, which the adapter
  // reads as audio_base64 for the turn.
  assert.match(adapter, /audio_base64: findLatestUserAudioBase64\(/);
});
