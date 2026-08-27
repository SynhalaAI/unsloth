// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

type ComposerModel = {
  description?: string;
  hasAudioInput?: boolean;
};

const canRecordModelAudio = (activeModel: ComposerModel | undefined) =>
  activeModel?.hasAudioInput === true;

test("audio controls stay hidden when only the description says Audio Input", () => {
  assert.equal(
    canRecordModelAudio({
      description: "Vision · Audio Input",
      hasAudioInput: false,
    }),
    false,
  );
});

test("audio controls are visible when hasAudioInput is true regardless of description", () => {
  assert.equal(
    canRecordModelAudio({
      description: "Base",
      hasAudioInput: true,
    }),
    true,
  );
});

test("shared composer gates both audio controls on the hasAudioInput boolean", async () => {
  const source = await readFile(
    new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /const canRecordModelAudio = activeModel\?\.hasAudioInput === true;/,
  );
  assert.doesNotMatch(source, /description\?\.includes\("Audio Input"\)/);
  assert.equal(source.match(/canRecordModelAudio &&/g)?.length, 2);
});
