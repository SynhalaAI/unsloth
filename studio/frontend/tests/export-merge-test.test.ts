// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const apiSource = readSrc("features/export/api/export-api.ts");
const exportPageSource = readSrc("features/export/export-page.tsx");
const panelSource = readSrc("features/export/components/merge-test-panel.tsx");

test("the API posts a merge analysis without touching the export worker", () => {
  // The check reads only adapter weights, so it must be its own endpoint: it
  // cannot go through /load-checkpoint, which would load a base model first.
  assert.match(apiSource, /export async function analyzeMerge\(/);
  assert.match(apiSource, /authFetch\("\/api\/export\/merge\/analyze", \{\s*method: "POST"/);
  // A private HF adapter needs the caller's token, sent in a header so it never
  // lands in a URL or the log.
  assert.match(apiSource, /headers\["X-HF-Token"\] = params\.hf_token/);
});

test("the analysis request carries the paths and weights the merge would use", () => {
  assert.match(
    apiSource,
    /adapter_paths: params\.adapter_paths,\s*weights: params\.weights,\s*normalize_weights: params\.normalize_weights \?\? true,/,
  );
  // The report's suggested method must be typed as a method the picker can set.
  assert.match(apiSource, /method: MergeMethodType;/);
});

test("the Test button is gated on a merge that could actually run", () => {
  assert.match(exportPageSource, /const mergeTestReady = useMemo\(/);
  // Two distinct adapters, each with a path and a finite weight: a single
  // adapter has no pair, and a repeated path compares an adapter with itself.
  assert.match(exportPageSource, /const filled = adapterMergeSelections\.filter\(/);
  assert.match(exportPageSource, /if \(filled\.length < 2\) return false;/);
  assert.match(
    exportPageSource,
    /if \(new Set\(filled\.map\(\(item\) => item\.path\)\)\.size !== filled\.length\) \{\s*return false;/,
  );
  assert.match(exportPageSource, /disabled=\{!mergeTestReady \|\| mergeTestPending\}/);
});

test("the merge payload and the analysis resolve adapter paths through one helper", () => {
  // Two copies of the Local-picker-is-a-model-id resolution would be free to
  // drift, and the report would then describe a merge nobody runs.
  assert.match(exportPageSource, /function buildAdapterMergePaths\(/);
  assert.match(
    exportPageSource,
    /adapter_paths: buildAdapterMergePaths\(\s*adapterMergeSelections,\s*localMetaById,\s*\)/,
  );
  assert.match(
    exportPageSource,
    /adapter_paths: buildAdapterMergePaths\(selections, localMetaById\)/,
  );
  assert.doesNotMatch(exportPageSource, /localAdapterDir/);
});

test("a report is dropped when the adapters it describes change", () => {
  // Otherwise a verdict sits next to a different selection and reads as
  // describing it.
  assert.match(exportPageSource, /const mergeTestKey = useMemo\(/);
  assert.match(
    exportPageSource,
    /useEffect\(\(\) => \{\s*setMergeReport\(null\);\s*setMergeTestError\(null\);\s*\}, \[mergeTestKey\]\);/,
  );
});

test("the panel reports interference and can apply the recommendation", () => {
  assert.match(exportPageSource, /<MergeTestPanel/);
  assert.match(
    exportPageSource,
    /onApplyRecommendation=\{\s*handleApplyMergeRecommendation\s*\}/,
  );
  // Applying a recommendation moves the method and its density together:
  // TIES with the wrong density is the difference the recommendation is making.
  assert.match(
    exportPageSource,
    /const handleApplyMergeRecommendation = useCallback\([\s\S]*?handleMergeMethodChange\(method\);\s*setMergeDensity\(String\(density\)\);/,
  );
  // Every interference bucket is rendered, so a new one cannot be forgotten.
  for (const bucket of ["low", "moderate", "high"]) {
    assert.match(panelSource, new RegExp(`${bucket}: \\{`));
  }
});

test("the panel separates interference from accuracy", () => {
  // The numbers measure how much the adapters fight, which is not a claim
  // about the merged model; the panel has to say so where it is read.
  assert.match(panelSource, /not how accurate the merged model will be/);
  assert.match(panelSource, /report\.sign_scan_truncated/);
});