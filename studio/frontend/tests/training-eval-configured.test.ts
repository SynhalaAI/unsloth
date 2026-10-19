// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isEvaluationConfigured } from "../src/features/training/lib/evaluation-configured.ts";

test("a positive, finite eval_steps counts as configured", () => {
  assert.equal(isEvaluationConfigured(0.1), true);
  assert.equal(isEvaluationConfigured(1), true);
  assert.equal(isEvaluationConfigured(30), true);
  // Saved run configs JSON-round-trip, so a numeric string must still read as configured.
  assert.equal(isEvaluationConfigured("0.1"), true);
});

test("disabled or unusable eval_steps does not count as configured", () => {
  assert.equal(isEvaluationConfigured(0), false);
  assert.equal(isEvaluationConfigured(0.0), false);
  assert.equal(isEvaluationConfigured(-1), false);
  assert.equal(isEvaluationConfigured(null), false);
  assert.equal(isEvaluationConfigured(undefined), false);
  assert.equal(isEvaluationConfigured(""), false);
  // A bool passes `> 0` in the backend's guard concerns, so it must be rejected explicitly.
  assert.equal(isEvaluationConfigured(true), false);
  assert.equal(isEvaluationConfigured(false), false);
  // Non-finite values never fire an eval step in the trainer.
  assert.equal(isEvaluationConfigured(Number.POSITIVE_INFINITY), false);
  assert.equal(isEvaluationConfigured(Number.NaN), false);
  assert.equal(isEvaluationConfigured("not-a-number"), false);
});
