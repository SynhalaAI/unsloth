// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/export-store-resolver.mjs", import.meta.url);

const stub = await import("./helpers/export-api-stub.mjs");
const { useExportRuntimeStore } = await import(
  "../src/features/export/stores/export-runtime-store.ts"
);

// A superseded run's finally must still free the export subprocess: the
// loaded (possibly multi-adapter merged) model stays resident in RAM until
// /api/export/cleanup runs, and the old `if (isCurrent())` gate skipped it
// whenever a newer run bumped runId first — "Export complete" with the model
// still loaded.

function params(overrides: Record<string, unknown> = {}) {
  return {
    sourceMode: "model",
    checkpointPath: null,
    source: "unsloth/Qwen3-0.6B",
    modelSource: "hf",
    trustRemoteCode: false,
    exportMethod: "gguf",
    isAdapter: false,
    quantLevels: ["q4_k_m"],
    saveDirectory: "out",
    destination: "local",
    privateRepo: false,
    summary: {},
    ...overrides,
  } as unknown as Parameters<
    ReturnType<typeof useExportRuntimeStore.getState>["runExport"]
  >[0];
}

test("a superseded run still calls cleanupExport", async () => {
  stub.resetStub();
  stub.responses.set("fetchExportLogs", () => ({
    entries: [],
    cursor: 1,
    active: false,
  }));
  stub.responses.set("exportGGUF", () => ({
    success: true,
    message: "ok",
    details: { output_path: "out/model.gguf" },
  }));

  // Hold the first run inside its export phase so a second run can supersede
  // it; the first run's finally must still free the worker afterwards.
  let releaseFirstExport: (() => void) | null = null;
  const firstExportGate = new Promise<void>((resolve) => {
    releaseFirstExport = resolve;
  });
  let exportCalls = 0;
  stub.responses.set("exportGGUF", () => {
    exportCalls += 1;
    if (exportCalls === 1) return firstExportGate.then(() => ({
      success: true,
      message: "ok",
      details: { output_path: "out/model.gguf" },
    }));
    return {
      success: true,
      message: "ok",
      details: { output_path: "out/model.gguf" },
    };
  });

  const first = useExportRuntimeStore.getState().runExport(params());
  // Let the first run reach its export phase, then start a superseding run.
  await new Promise((resolve) => setTimeout(resolve, 50));
  const second = useExportRuntimeStore.getState().runExport(params());
  await second;
  releaseFirstExport!();
  await first;

  const cleanups = stub.calls.filter((c) => c.name === "cleanupExport");
  assert.ok(
    cleanups.length >= 2,
    `superseded run skipped cleanup (cleanupExport called ${cleanups.length}x for 2 runs)`,
  );
});
