// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The preference trainers clamp the prompt length themselves:
//   prompt_length = max(1, min(max_prompt_length, max(1, max_length // 2)))
// so the hyperparameter row must show the value the run will actually use.

import assert from "node:assert/strict";
import test from "node:test";

import {
  effectiveMaxPromptLength,
  maxPromptLengthCap,
  maxPromptLengthPresets,
} from "../src/features/training/lib/preference-limits.ts";

test("the prompt cap is half the context length, like the trainers", () => {
  assert.equal(maxPromptLengthCap(2048), 1024);
  assert.equal(maxPromptLengthCap(4096), 2048);
  assert.equal(maxPromptLengthCap(8192), 4096);
  assert.equal(maxPromptLengthCap(1), 1);
  // A missing or malformed context length must still resolve to a value the trainer accepts.
  assert.equal(maxPromptLengthCap(0), 1);
  assert.equal(maxPromptLengthCap(Number.NaN), 1);
});

test("a requested prompt length above the cap resolves to the cap", () => {
  assert.equal(effectiveMaxPromptLength(2048, 2048), 1024);
  assert.equal(effectiveMaxPromptLength(2048, 8192), 1024);
  assert.equal(effectiveMaxPromptLength(8192, 4096), 4096);
  assert.equal(effectiveMaxPromptLength(2048, 512), 512);
  // Stale store values (0, NaN) still show a usable number.
  assert.equal(effectiveMaxPromptLength(2048, 0), 1);
  assert.equal(effectiveMaxPromptLength(2048, Number.NaN), 1024);
  assert.equal(effectiveMaxPromptLength(0, 512), 1);
});

test("the dropdown never offers a value the trainers would reduce", () => {
  assert.deepEqual(maxPromptLengthPresets(2048), [128, 256, 512, 1024]);
  assert.deepEqual(maxPromptLengthPresets(8192), [
    128,
    256,
    512,
    1024,
    2048,
    4096,
  ]);
  // A context length below the smallest preset leaves only the capped value.
  assert.deepEqual(maxPromptLengthPresets(128), []);
});
