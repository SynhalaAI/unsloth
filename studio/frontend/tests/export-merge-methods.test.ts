// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { loadCheckpoint } from "../src/features/export/api/export-api.ts";
import { MERGE_METHODS, type MergeMethodType } from "../src/features/export/constants.ts";
import type { RunExportParams } from "../src/features/export/stores/export-runtime-store.ts";

import { readSrc } from "./helpers/kit.ts";

const constantsSource = readSrc("features/export/constants.ts");
const exportPageSource = readSrc("features/export/export-page.tsx");
const storeSource = readSrc("features/export/stores/export-runtime-store.ts");
const apiSource = readSrc("features/export/api/export-api.ts");

test("the merge picker lists every method the core merger supports", () => {
  // Keep in sync with SUPPORTED_METHODS in unsloth/multi_adapter_merge.py.
  assert.deepEqual(
    MERGE_METHODS.map((method) => method.value),
    [
      "linear", "ties", "dare_ties", "dare_linear", "magnitude_prune",
      "sce", "della", "della_linear", "breadcrumbs", "breadcrumbs_ties",
      "multislerp", "model_stock", "ctm", "cat",
    ],
  );
  assert.deepEqual(
    MERGE_METHODS.map((method) => method.label),
    [
      "Linear", "TIES", "DARE-TIES", "DARE-Linear", "Mag-Prune",
      "SCE", "DELLA", "DELLA-Linear", "Breadcrumbs", "Breadcrumbs-TIES",
      "Multi-SLERP", "Model Stock", "CtM", "CAT",
    ],
  );
  for (const method of MERGE_METHODS) {
    assert.ok(method.description.trim().length > 0, `${method.value} needs a description`);
    assert.ok(method.bestFor.trim().length > 0, `${method.value} needs a bestFor`);
    assert.ok(method.category, `${method.value} needs a category`);
  }
});

test("the constants module keeps the picker type and list in one place", () => {
  assert.match(
    constantsSource,
    /export type MergeMethodType =[\s\S]*"model_stock";/,
  );
  assert.match(constantsSource, /export const MERGE_METHODS:/);
  assert.match(constantsSource, /export type MergeMethodCategory/);
  assert.match(constantsSource, /MERGE_METHOD_CATEGORY_LABELS/);
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
  // Density now covers TIES, DARE-TIES, Mag-Prune, DELLA, breadcrumbs, etc.
  assert.match(
    exportPageSource,
    /MERGE_METHODS_WITH_DENSITY\.has\(mergeMethod\)/,
  );
  assert.match(exportPageSource, /aria-label="Merge density"/);
  assert.match(exportPageSource, /MERGE_METHODS_WITH_DROPOUT\.has\(mergeMethod\)/);
  assert.match(exportPageSource, /aria-label="DARE drop rate"/);
  assert.match(exportPageSource, /aria-label="CtM target rank"/);
  assert.match(exportPageSource, /MERGE_METHODS_WITH_EPSILON\.has\(mergeMethod\)/);
  assert.match(exportPageSource, /aria-label="DELLA epsilon"/);
  assert.match(exportPageSource, /MERGE_METHODS_WITH_GAMMA\.has\(mergeMethod\)/);
  assert.match(exportPageSource, /aria-label="Breadcrumbs gamma"/);
  assert.match(exportPageSource, /MERGE_METHODS_WITH_TOPK\.has\(mergeMethod\)/);
  assert.match(exportPageSource, /aria-label="SCE select top-k"/);
  // The old TIES-only density label must be gone: a future method using the
  // density knob would silently keep the misleading name.
  assert.doesNotMatch(exportPageSource, /aria-label="TIES density"/);
});

test("the merge request payload carries drop_rate and target_rank", () => {
  assert.match(
    exportPageSource,
    /MERGE_METHODS_WITH_DROPOUT\.has\(mergeMethod\) &&/,
  );
  assert.match(
    exportPageSource,
    /MERGE_METHODS_WITH_RANK\.has\(mergeMethod\) &&/,
  );
});

test("validation covers the new strategies in both canExport and the start gate", () => {
  // canExport: drop_rate must stay in [0, 1); target_rank (when set) must be >= 1.
  assert.match(
    exportPageSource,
    /!MERGE_METHODS_WITH_DROPOUT\.has\(mergeMethod\) \|\|\s*\(Number\.isFinite\(Number\(mergeDropRate\)\) &&\s*Number\(mergeDropRate\) >= 0 &&\s*Number\(mergeDropRate\) < 1\)/,
  );
  assert.match(
    exportPageSource,
    /!MERGE_METHODS_WITH_RANK\.has\(mergeMethod\) \|\|\s*mergeTargetRank\.trim\(\) === "" \|\|/,
  );
  // Start gate: out-of-range values stop the run instead of shipping them.
  assert.match(exportPageSource, /Number\(mergeDropRate\) >= 1/);
  assert.match(exportPageSource, /!Number\.isInteger\(Number\(mergeTargetRank\)\)/);
  // Density is validated in both gates via the shared knob set.
  assert.match(exportPageSource, /MERGE_METHODS_WITH_DENSITY\.has\(mergeMethod\)/);
});

test("merge parameter inputs carry hover hints", () => {
  // The number inputs are bare (no visible labels), so each one gets the UI's
  // standard InfoHint affordance explaining what value belongs in it.
  assert.match(exportPageSource, /import \{ InfoHint \} from "@\/components\/ui\/info-hint";/);
  // The method select explains itself from the shared MERGE_METHODS description.
  assert.match(
    exportPageSource,
    /<InfoHint>\s*\{MERGE_METHODS\.find\(\(method\) => method\.value === mergeMethod\)\s*\?\.description/,
  );
  for (const hintText of [
    "Fraction of each adapter's strongest weight changes to",
    "Fraction of weight changes randomly dropped before",
    "Optional SVD compression rank (≥1)",
  ]) {
    assert.ok(exportPageSource.includes(hintText), `missing hint: ${hintText}`);
  }
  // Hints sit beside their inputs (input + InfoHint wrapped together).
  assert.match(
    exportPageSource,
    /<span className="flex items-center gap-1">\s*<Input\s+type="number"\s+min="0\.01"/,
  );
});

test("the new merge states feed the runtime request deps to avoid stale sends", () => {
  assert.match(
    exportPageSource,
    /mergeMethod,\s*\n\s*mergeDensity,\s*\n\s*mergeDropRate,\s*\n\s*mergeTargetRank,/,
  );
});

test("the store and api pass-through types carry every merge field", () => {
  // The page builds the full payload (method incl. dare_ties/ctm, drop_rate,
  // target_rank); the store param and the load-checkpoint request must accept
  // all of it, not just the original linear/ties pair.
  assert.match(storeSource, /multiAdapterMerge\?: \{/);
  assert.match(storeSource, /method: MergeMethodType;/);
  assert.match(storeSource, /drop_rate\?: number;/);
  assert.match(storeSource, /target_rank\?: number;/);
  assert.match(apiSource, /method\?: MergeMethodType;/);
  assert.match(apiSource, /drop_rate\?: number;/);
  assert.match(apiSource, /target_rank\?: number;/);
  // The narrowed linear/ties-only unions must be gone from both pass-throughs.
  assert.doesNotMatch(storeSource, /method: "linear" \| "ties"/);
  assert.doesNotMatch(apiSource, /method\?: "linear" \| "ties"/);
});

test("the page-built merge payload satisfies the runtime request chain", () => {
  // Compile-time guard: `npm run typecheck` (tsconfig.test.json includes
  // tests/) fails this file if either pass-through type is narrowed again.
  // `import type` is erased under --experimental-strip-types, so the runtime
  // harness here only exercises the object shapes.
  type StoreMerge = NonNullable<RunExportParams["multiAdapterMerge"]>;
  type ApiMerge = NonNullable<Parameters<typeof loadCheckpoint>[0]["merge_adapters"]>;

  const strategies: MergeMethodType[] = MERGE_METHODS.map((method) => method.value);
  assert.equal(strategies.length, 14);
  for (const method of strategies) {
    // Mirrors export-page.tsx's mergeConfig, including the explicit-undefined
    // target_rank the non-CtM strategies produce.
    const pagePayload: StoreMerge = {
      adapter_paths: ["local/path", { repo_id: "org/repo", subfolder: "ckpt" }],
      weights: [0.6, 0.4],
      method,
      normalize_weights: true,
      density: 0.5,
      drop_rate: 0.2,
      target_rank: method === "ctm" ? 16 : undefined,
    };
    assert.equal(pagePayload.adapter_paths.length, pagePayload.weights.length);
    // The store param must flow into the load-checkpoint request unchanged.
    const apiPayload: ApiMerge = pagePayload;
    assert.equal(apiPayload.method, method);
  }
});

test("the method picker groups methods by category with a per-method info card", () => {
  // Grouped dropdown: SelectLabel category headers from MERGE_METHOD_CATEGORY_LABELS.
  assert.match(
    exportPageSource,
    /MERGE_METHOD_CATEGORY_LABELS\[category\]/,
  );
  // Info card shows the selected method's bestFor guidance.
  assert.match(
    exportPageSource,
    /MERGE_METHODS\.find\(\(method\) => method\.value === mergeMethod\)\?\.bestFor/,
  );
  // model_stock requires 3+ adapters — the picker surfaces the requirement.
  assert.match(
    exportPageSource,
    /MERGE_METHODS_MIN_3_ADAPTERS\.has\(mergeMethod\)/,
  );
  // sce / model_stock derive their own weights — the weights inputs are
  // disabled with a note instead of silently ignored.
  assert.match(
    exportPageSource,
    /MERGE_METHODS_AUTO_WEIGHTS\.has\(mergeMethod\)/,
  );
});
