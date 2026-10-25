// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  formatTrainingMethodLabel,
  getPreferenceMethodLabel,
  parseBackendPreferenceMethod,
} = await import("../src/features/training/lib/training-methods.ts");

test("parseBackendPreferenceMethod maps every saved preference objective", () => {
  assert.equal(parseBackendPreferenceMethod("SFT"), "sft");
  assert.equal(parseBackendPreferenceMethod("DPO"), "dpo");
  assert.equal(parseBackendPreferenceMethod("ORPO"), "orpo");
  assert.equal(parseBackendPreferenceMethod("CPO"), "cpo");
  // Saved configs are normalized but older rows may carry other casing/spacing.
  assert.equal(parseBackendPreferenceMethod(" orpo "), "orpo");
  assert.equal(parseBackendPreferenceMethod("cpo"), "cpo");
});

test("parseBackendPreferenceMethod ignores values that are not preference objectives", () => {
  assert.equal(parseBackendPreferenceMethod(undefined), null);
  assert.equal(parseBackendPreferenceMethod(null), null);
  assert.equal(parseBackendPreferenceMethod(""), null);
  assert.equal(parseBackendPreferenceMethod("LoRA/QLoRA"), null);
  assert.equal(parseBackendPreferenceMethod(42), null);
  assert.equal(parseBackendPreferenceMethod({ method: "ORPO" }), null);
});

test("getPreferenceMethodLabel falls back to null without a method", () => {
  assert.equal(getPreferenceMethodLabel("orpo"), "ORPO");
  assert.equal(getPreferenceMethodLabel("dpo"), "DPO");
  assert.equal(getPreferenceMethodLabel(null), null);
  assert.equal(getPreferenceMethodLabel(undefined), null);
});

test("formatTrainingMethodLabel appends the objective for adapter runs", () => {
  assert.equal(formatTrainingMethodLabel("qlora", "orpo"), "QLoRA · ORPO");
  assert.equal(formatTrainingMethodLabel("lora", "dpo"), "LoRA · DPO");
  assert.equal(formatTrainingMethodLabel("lora", "cpo"), "LoRA · CPO");
});

test("formatTrainingMethodLabel keeps plain labels for SFT, CPT and full finetuning", () => {
  assert.equal(formatTrainingMethodLabel("qlora", "sft"), "QLoRA");
  assert.equal(formatTrainingMethodLabel("qlora", null), "QLoRA");
  assert.equal(formatTrainingMethodLabel("qlora", undefined), "QLoRA");
  // A stale dropdown selection must not relabel a run the backend trains without ORPO.
  assert.equal(formatTrainingMethodLabel("cpt", "orpo"), "CPT");
  assert.equal(formatTrainingMethodLabel("full", "orpo"), "Full");
});
