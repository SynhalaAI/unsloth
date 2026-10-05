// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { loadCheckpoint } from "../src/features/export/api/export-api.ts";
import {
  MERGE_METHODS,
  MERGE_METHODS_WITH_RANK,
  type MergeMethodType,
} from "../src/features/export/constants.ts";
import type { RunExportParams } from "../src/features/export/stores/export-runtime-store.ts";

import { readSrc } from "./helpers/kit.ts";

const constantsSource = readSrc("features/export/constants.ts");
const exportPageSource = readSrc("features/export/export-page.tsx");
const storeSource = readSrc("features/export/stores/export-runtime-store.ts");
const apiSource = readSrc("features/export/api/export-api.ts");

test("the merge picker lists every method the core merger supports", () => {
  // Keep in sync with SUPPORTED_METHODS in unsloth/multi_adapter_merge.py, which
  // are PEFT's official add_weighted_adapter combination types.
  assert.deepEqual(
    MERGE_METHODS.map((method) => method.value),
    [
      "linear", "svd", "cat", "ties", "dare_ties", "dare_linear",
      "magnitude_prune", "ties_svd", "dare_ties_svd", "dare_linear_svd",
      "magnitude_prune_svd",
    ],
  );
  assert.deepEqual(
    MERGE_METHODS.map((method) => method.label),
    [
      "Linear", "SVD", "CAT", "TIES", "DARE-TIES", "DARE-Linear", "Mag-Prune",
      "TIES-SVD", "DARE-TIES-SVD", "DARE-Linear-SVD", "Mag-Prune-SVD",
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
    /export type MergeMethodType =[\s\S]*"magnitude_prune_svd";/,
  );
  assert.match(constantsSource, /export const MERGE_METHODS:/);
  assert.match(constantsSource, /export type MergeMethodCategory/);
  assert.match(constantsSource, /MERGE_METHOD_CATEGORY_LABELS/);
});

test("the export page picker renders from MERGE_METHODS, not hardcoded items", () => {
  assert.match(exportPageSource, /useState<MergeMethodType>\("linear"\)/);
  // The grouped picker still renders every entry from the shared list.
  assert.match(exportPageSource, /MERGE_METHODS\.filter\(/);
  assert.match(exportPageSource, /\.map\(\(method\) => \(/);
  assert.match(exportPageSource, /value=\{method\.value\}/);
  assert.match(exportPageSource, /\{method\.label\}/);
  // The old hardcoded pair must be gone: a future fourth method would silently miss it.
  assert.doesNotMatch(exportPageSource, /<SelectItem value="linear">Linear<\/SelectItem>/);
  assert.doesNotMatch(exportPageSource, /<SelectItem value="ties">TIES<\/SelectItem>/);
});

test("method-specific controls exist for the PEFT strategies", () => {
  // Density covers TIES, DARE-TIES, DARE-Linear, Mag-Prune and the *_svd
  // variants; every SVD combination additionally takes an output rank.
  assert.match(
    exportPageSource,
    /MERGE_METHODS_WITH_DENSITY\.has\(mergeMethod\)/,
  );
  assert.match(exportPageSource, /aria-label="Merge density"/);
  assert.match(exportPageSource, /MERGE_METHODS_WITH_RANK\.has\(mergeMethod\)/);
  assert.match(exportPageSource, /aria-label="SVD output rank"/);
  // The old TIES-only density label must be gone: a future method using the
  // density knob would silently keep the misleading name.
  assert.doesNotMatch(exportPageSource, /aria-label="TIES density"/);
  // The knobs that only fed the removed in-house methods must be gone with them.
  for (const gone of [
    "DARE drop rate",
    "DELLA epsilon",
    "Breadcrumbs gamma",
    "SCE select top-k",
  ]) {
    assert.doesNotMatch(exportPageSource, new RegExp(gone));
  }
});

test("the merge request payload carries target_rank", () => {
  assert.match(
    exportPageSource,
    /MERGE_METHODS_WITH_RANK\.has\(mergeMethod\) &&/,
  );
  assert.doesNotMatch(exportPageSource, /drop_rate:/);
});

test("validation covers the merge knobs in both canExport and the start gate", () => {
  // canExport: target_rank (when set) must be >= 1.
  assert.match(
    exportPageSource,
    /!MERGE_METHODS_WITH_RANK\.has\(mergeMethod\) \|\|\s*mergeTargetRank\.trim\(\) === "" \|\|/,
  );
  // Start gate: out-of-range values stop the run instead of shipping them.
  assert.match(exportPageSource, /!Number\.isInteger\(Number\(mergeTargetRank\)\)/);
  // Density is validated in both gates via the shared knob set.
  assert.match(exportPageSource, /MERGE_METHODS_WITH_DENSITY\.has\(mergeMethod\)/);
});

test("merge parameter inputs carry hover hints", () => {
  // The number inputs are bare (no visible labels), so each one gets the UI's
  // standard InfoHint affordance explaining what value belongs in it.
  assert.match(exportPageSource, /import \{ InfoHint \} from "@\/components\/ui\/info-hint";/);
  // The method select explains itself from the shared MERGE_METHODS metadata,
  // rendered inside the hover hint (never an inline card that resizes the
  // row when the method — and its text length — changes).
  assert.match(
    exportPageSource,
    /<InfoHint>\s*\{\(\(\) => \{\s*const selected = MERGE_METHODS\.find\(\s*\(method\) => method\.value === mergeMethod,\s*\);/,
  );
  assert.match(exportPageSource, /selected\.description/);
  for (const hintText of [
    "Fraction of each adapter's strongest weight changes to",
    "Output adapter rank for the SVD merge",
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
    /mergeMethod,\s*\n\s*mergeDensity,\s*\n\s*mergeTargetRank,/,
  );
});

test("the store and api pass-through types carry every merge field", () => {
  // The page builds the full payload (method incl. svd/ties, target_rank); the
  // store param and the load-checkpoint request must accept all of it.
  assert.match(storeSource, /multiAdapterMerge\?: \{/);
  assert.match(storeSource, /method: MergeMethodType;/);
  assert.match(storeSource, /target_rank\?: number;/);
  assert.match(apiSource, /method\?: MergeMethodType;/);
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
  assert.equal(strategies.length, 11);
  for (const method of strategies) {
    // Mirrors export-page.tsx's mergeConfig, including the explicit-undefined
    // target_rank the non-svd strategies produce.
    const pagePayload: StoreMerge = {
      adapter_paths: ["local/path", { repo_id: "org/repo", subfolder: "ckpt" }],
      weights: [0.6, 0.4],
      method,
      normalize_weights: true,
      density: 0.5,
      target_rank: MERGE_METHODS_WITH_RANK.has(method) ? 16 : undefined,
    };
    assert.equal(pagePayload.adapter_paths.length, pagePayload.weights.length);
    // The store param must flow into the load-checkpoint request unchanged.
    const apiPayload: ApiMerge = pagePayload;
    assert.equal(apiPayload.method, method);
  }
});

test("the method picker groups methods by category with a per-method hover hint", () => {
  // Two-step picker: the category dropdown lists every group order entry,
  // and the method dropdown renders the filtered list for that category.
  assert.match(
    exportPageSource,
    /MERGE_METHOD_CATEGORY_ORDER\.map\(\(category\) => \(\s*<SelectItem key=\{category\} value=\{category\}>/,
  );
  assert.match(
    exportPageSource,
    /<SelectItem[^>]*>\s*\{MERGE_METHOD_CATEGORY_LABELS\[category\]\}/,
  );
  assert.match(
    exportPageSource,
    /mergeMethodsForCategory\.map\(\(method\) => \(\s*<SelectItem key=\{method\.value\} value=\{method\.value\}>/,
  );
  assert.match(exportPageSource, /\{method\.label\}/);
  // No stale grouped markup may remain: labels must only come from the
  // category dropdown, never from inside the method list.
  assert.doesNotMatch(exportPageSource, /<SelectGroup/);
  assert.doesNotMatch(exportPageSource, /<SelectLabel/);
  // Changing the category resets the method to that group's first method.
  assert.match(
    exportPageSource,
    /const handleMergeCategoryChange[\s\S]*?setMergeMethod\(first\.value\)/,
  );
  // Importing / otherwise setting a method moves the category to match.
  assert.match(
    exportPageSource,
    /const handleMergeMethodChange[\s\S]*?setMergeCategory\(category\)/,
  );
  // The method hover hint shows the selected method's bestFor guidance.
  assert.match(
    exportPageSource,
    /MERGE_METHODS\.find\(\s*\(method\) => method\.value === mergeMethod,\s*\)/,
  );
  assert.match(exportPageSource, /selected\.bestFor/);
  // Every PEFT combination type works with a single adapter, so no method may
  // reintroduce a minimum-adapter requirement.
  assert.doesNotMatch(exportPageSource, /MERGE_METHODS_MIN_3_ADAPTERS/);
  assert.doesNotMatch(exportPageSource, /MERGE_METHODS_AUTO_WEIGHTS/);
});

test("the normalize_weights toggle drives the merge payload and the analysis", () => {
  // The Export page must let the user pick normalize_weights; a hardcoded
  // value makes the other backend mode unreachable from the UI. Default is
  // off, so the entered weights are used raw unless normalization is asked for.
  assert.match(exportPageSource, /const \[mergeNormalizeWeights, setMergeNormalizeWeights\] = useState\(false\);/);
  assert.doesNotMatch(exportPageSource, /const \[mergeNormalizeWeights, setMergeNormalizeWeights\] = useState\(true\);/);
  assert.match(exportPageSource, /aria-label="Normalize merge weights"/);
  // The payload and the preflight analysis both carry the chosen value, so the
  // report describes the merge that is about to run.
  assert.match(exportPageSource, /normalize_weights: mergeNormalizeWeights,/);
  assert.doesNotMatch(exportPageSource, /normalize_weights: true,/);
  // The state feeds the runtime request deps to avoid a stale send.
  assert.match(exportPageSource, /mergeTargetRank,\s*\n\s*mergeNormalizeWeights,/);
  // It round-trips through the saved/imported YAML config.
  assert.match(exportPageSource, /normalizeWeights: mergeNormalizeWeights,/);
  assert.match(exportPageSource, /if \(typeof config\.normalizeWeights === "boolean"\) \{/);
});

test("the normalize toggle is the last control in the merge toolbar row", () => {
  // It used to sit between the method hint and the method-specific inputs, so a
  // narrow panel left it stranded; whatever controls the selected method adds
  // has to render before it.
  const row = exportPageSource.slice(
    exportPageSource.indexOf('data-testid="merge-toolbar-row"'),
    exportPageSource.indexOf('ref={configFileInputRef}'),
  );
  assert.notEqual(row, "");
  const toolbar = exportPageSource.slice(
    exportPageSource.indexOf('ref={configFileInputRef}'),
  );
  const toggle = toolbar.indexOf('aria-label="Normalize merge weights"');
  assert.ok(toggle > 0, "the toggle must still render");
  for (const control of [
    "aria-label=\"Merge method\"",
    "aria-label=\"Merge density\"",
    "aria-label=\"SVD output rank\"",
  ]) {
    assert.ok(
      toolbar.indexOf(control) < toggle,
      `${control} must render before the normalize toggle`,
    );
  }
});

test("the merge heading and the toolbar share one non-wrapping row", () => {
  // The outer strip used to flex-wrap, so once the *_svd variants added both
  // a density and a rank knob the whole control strip dropped below the
  // Adapter merge heading. The strip stays on one line instead: the toolbar
  // row takes the leftover space (flex-1) so the heading keeps its natural
  // width, and the row scrolls its own overflow.
  const outer = "flex flex-nowrap items-center justify-between gap-3";
  assert.equal(
    exportPageSource.split(outer).length - 1,
    1,
    "the heading strip must exist exactly once",
  );
  const stripStart = exportPageSource.indexOf(outer);
  const toolbarAt = exportPageSource.indexOf('data-testid="merge-toolbar-row"');
  assert.ok(
    stripStart >= 0 && stripStart < toolbarAt,
    "the heading strip must open before the toolbar row",
  );
  const strip = exportPageSource.slice(stripStart, toolbarAt);
  assert.doesNotMatch(strip, /flex-wrap/);
  assert.ok(
    strip.indexOf("Adapter merge") > 0,
    "the heading must sit inside that strip, ahead of the controls",
  );
  const rowStart = exportPageSource.lastIndexOf(
    '<div className="flex min-w-0',
    toolbarAt,
  );
  assert.ok(rowStart > stripStart, "the toolbar row must follow the heading");
  const row = exportPageSource.slice(
    rowStart,
    exportPageSource.indexOf("ref={configFileInputRef}"),
  );
  assert.match(row, /flex min-w-0 max-w-full flex-1 flex-nowrap/);
  assert.match(row, /overflow-x-auto/);
});

test("the export page warns when a multi-adapter merge will not average", () => {
  // Raw weights summing to N blend N adapters' worth of change at once, and
  // PEFT's linear only approximates the weighted sum: both are stated in the
  // method hint so a distorted merge is not shipped blind.
  assert.match(exportPageSource, /const mergeWeightSum = adapterMergeSelections/);
  assert.match(
    exportPageSource,
    /mergeMethod === "linear" && filledAdapterCount >= 2/,
  );
  assert.match(
    exportPageSource,
    /!mergeNormalizeWeights &&\s*filledAdapterCount >= 2 &&\s*Math\.abs\(mergeWeightSum - 1\) > 0\.05/,
  );
  assert.match(
    exportPageSource,
    /turn on Normalize weights for a weighted/,
  );
});


test("a single adapter is a runnable merge, not just a multi-adapter one", () => {
  // The panel used to read as multi-adapter only, and the API schema demanded
  // two paths. Base + a single LoRA is a valid merge end to end, so the export
  // gate must accept exactly one filled adapter row.
  assert.match(
    exportPageSource,
    /adapterMergeSelections\.length >= 1 &&\s*new Set\(adapterMergeSelections\.map\(\(item\) => item\.path\)\)\.size ===/,
  );
  // No gate may reintroduce a two-adapter floor on the merge itself.
  assert.doesNotMatch(exportPageSource, /multiAdapterMerge[\s\S]{0,400}?length >= 2/);
  assert.doesNotMatch(exportPageSource, /length < 2 && multiAdapterMerge/);
  // The panel now presents one adapter as a supported amount.
  assert.match(exportPageSource, /One\s*adapter is enough/);
});

test("a single-adapter merge offers only Linear", () => {
  // PEFT falls back to linear for a single adapter, so trimming, sign election
  // and pruning never run. Offering TIES/SVD there would imply they do
  // something, so the picker locks to Linear.
  assert.match(
    exportPageSource,
    /const filledAdapterCount = adapterMergeSelections\.filter\([\s\S]*?const singleAdapterMerge = filledAdapterCount === 1;/,
  );
  // The method dropdown is disabled and lists only Linear in that state.
  assert.match(exportPageSource, /disabled=\{singleAdapterMerge\}/);
  assert.match(
    exportPageSource,
    /singleAdapterMerge\s*\?\s*MERGE_METHODS\.filter\(\s*\(method\) => method\.value === "linear",?\s*\)/,
  );
  // The category dropdown keeps its width but is disabled and invisible, so going from one to two adapters does not shift the row below.
  assert.match(exportPageSource, /disabled={singleAdapterMerge}/);
  assert.match(exportPageSource, /invisible/);
  // And the state is actually forced back to Linear, so an imported config or a
  // shrinking selection cannot leave a multi-adapter method selected.
  assert.match(
    exportPageSource,
    /if \(singleAdapterMerge && mergeMethod !== "linear"\) \{\s*setMergeMethod\("linear"\);/,
  );
  // Declaration order: singleAdapterMerge must be derived from the selection
  // state BEFORE any hook takes it as a dependency, or the dep array reads it
  // in its temporal dead zone and the page crashes on render.
  assert.ok(
    exportPageSource.indexOf("adapterMergeSelections, setAdapterMergeSelections") <
      exportPageSource.indexOf("const singleAdapterMerge = filledAdapterCount === 1;"),
  );
});