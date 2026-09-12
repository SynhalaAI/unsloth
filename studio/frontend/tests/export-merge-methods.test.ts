// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { MERGE_METHODS } from "../src/features/export/constants.ts";

import { readSrc } from "./helpers/kit.ts";

const constantsSource = readSrc("features/export/constants.ts");
const exportPageSource = readSrc("features/export/export-page.tsx");

test("the merge picker lists every method the core merger supports", () => {
  // Keep in sync with SUPPORTED_METHODS in unsloth/multi_adapter_merge.py.
  assert.deepEqual(
    MERGE_METHODS.map((method) => method.value),
    ["linear", "ties", "dare_ties", "ctm"],
  );
  assert.deepEqual(
    MERGE_METHODS.map((method) => method.label),
    ["Linear", "TIES", "DARE-TIES", "CtM"],
  );
  for (const method of MERGE_METHODS) {
    assert.ok(method.description.trim().length > 0, `${method.value} needs a description`);
  }
});

test("the constants module keeps the picker type and list in one place", () => {
  assert.match(constantsSource, /export type MergeMethodType = "linear" \| "ties" \| "dare_ties" \| "ctm";/);
  assert.match(constantsSource, /export const MERGE_METHODS:/);
});

test("the export page picker renders from MERGE_METHODS, not hardcoded items", () => {
  assert.match(exportPageSource, /useState<MergeMethodType>\("linear"\)/);
  assert.match(
    exportPageSource,
    /MERGE_METHODS\.map\(\(method\) => \(\s*<SelectItem key=\{method\.value\} value=\{method\.value\}>\s*\{method\.label\}/,
  );
  // The old hardcoded pair must be gone: a future fourth method would silently miss it.
  assert.doesNotMatch(exportPageSource, /<SelectItem value="linear">Linear<\/SelectItem>/);
  assert.doesNotMatch(exportPageSource, /<SelectItem value="ties">TIES<\/SelectItem>/);
});

test("method-specific controls exist for the new strategies", () => {
  // Density now covers TIES and DARE-TIES; each new method gets its own knob.
  assert.match(exportPageSource, /\(mergeMethod === "ties" \|\| mergeMethod === "dare_ties"\)/);
  assert.match(exportPageSource, /aria-label="DARE drop rate"/);
  assert.match(exportPageSource, /aria-label="CtM target rank"/);
});

test("the merge request payload carries drop_rate and target_rank", () => {
  assert.match(exportPageSource, /drop_rate:\s*\n\s*mergeMethod === "dare_ties" &&/);
  assert.match(exportPageSource, /target_rank:\s*\n\s*mergeMethod === "ctm" &&/);
});

test("validation covers the new strategies in both canExport and the start gate", () => {
  // canExport: drop_rate must stay in [0, 1); target_rank (when set) must be >= 1.
  assert.match(
    exportPageSource,
    /mergeMethod !== "dare_ties" \|\|\s*\(Number\.isFinite\(Number\(mergeDropRate\)\) &&\s*Number\(mergeDropRate\) >= 0 &&\s*Number\(mergeDropRate\) < 1\)/,
  );
  assert.match(
    exportPageSource,
    /mergeMethod !== "ctm" \|\|\s*mergeTargetRank\.trim\(\) === "" \|\|/,
  );
  // Start gate: out-of-range values stop the run instead of shipping them.
  assert.match(exportPageSource, /Number\(mergeDropRate\) >= 1/);
  assert.match(exportPageSource, /!Number\.isInteger\(Number\(mergeTargetRank\)\)/);
});

test("the new merge states feed the runtime request deps to avoid stale sends", () => {
  assert.match(
    exportPageSource,
    /mergeMethod,\s*\n\s*mergeDensity,\s*\n\s*mergeDropRate,\s*\n\s*mergeTargetRank,/,
  );
});
