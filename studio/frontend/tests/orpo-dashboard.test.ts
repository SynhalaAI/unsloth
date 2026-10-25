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
  usesPreferenceObjective,
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

test("formatTrainingMethodLabel keeps plain labels for SFT and CPT", () => {
  assert.equal(formatTrainingMethodLabel("qlora", "sft"), "QLoRA");
  assert.equal(formatTrainingMethodLabel("qlora", null), "QLoRA");
  assert.equal(formatTrainingMethodLabel("qlora", undefined), "QLoRA");
  // CPT keeps the dropdown selection but the backend trains it as plain SFT.
  assert.equal(formatTrainingMethodLabel("cpt", "orpo"), "CPT");
});

test("formatTrainingMethodLabel labels full finetuning preference runs", () => {
  // Full finetuning dispatches to the same DPO/ORPO/CPO trainers as LoRA and QLoRA.
  assert.equal(formatTrainingMethodLabel("full", "dpo"), "Full · DPO");
  assert.equal(formatTrainingMethodLabel("full", "orpo"), "Full · ORPO");
  assert.equal(formatTrainingMethodLabel("full", "sft"), "Full");
});

test("usesPreferenceObjective includes full finetuning but not CPT", () => {
  assert.equal(usesPreferenceObjective("full", "cpo"), true);
  assert.equal(usesPreferenceObjective("qlora", "cpo"), true);
  assert.equal(usesPreferenceObjective("full", "sft"), false);
  assert.equal(usesPreferenceObjective("cpt", "cpo"), false);
});
