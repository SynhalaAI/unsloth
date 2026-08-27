// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const statusAdoption = readFileSync(
  new URL(
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
    import.meta.url,
  ),
  "utf8",
);
const composer = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);

type LoadedModelRow = {
  id: string;
  description?: string;
  hasAudioInput?: boolean;
};

test("status audio-input capability drives the header row and shared composer", () => {
  const checkpoint = "unsloth/gemma-4-E2B-it";
  const loadedActiveModel: LoadedModelRow = {
    id: checkpoint,
    description: "Vision · Audio Input",
    hasAudioInput: Boolean({ has_audio_input: true }.has_audio_input),
  };

  assert.equal(loadedActiveModel.description?.includes("Audio Input"), true);
  assert.equal(loadedActiveModel.hasAudioInput, true);
  assert.match(
    statusAdoption,
    /\[checkpointId, store\.params\.checkpoint\]\.filter\([\s\S]*\(id\): id is string/,
  );
  assert.match(statusAdoption, /hasAudioInput:\s*Boolean\(status\.has_audio_input\)/);
  assert.match(
    composer,
    /const canRecordModelAudio = activeModel\?\.hasAudioInput === true;/,
  );
  assert.match(composer, /aria-label="Record audio for model"/);
});
