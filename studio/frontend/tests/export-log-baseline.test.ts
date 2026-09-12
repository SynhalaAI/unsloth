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

// A new export run must seed its log cursor from the backend's CURRENT highest
// log seq (the baseline) instead of reconnecting with `since=null`. The
// backend's run_start_seq only advances when the load-checkpoint POST lands,
// so a since-less connect can replay the PREVIOUS run's buffered log lines
// into the new run's panel. The baseline is read BEFORE isExporting flips:
// the lifecycle hook opens the SSE/poll the instant that flag flips, with
// lastSeq in hand, so no since-less window ever exists.

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

test("a new run seeds lastSeq from the backend's current log cursor", async () => {
  stub.resetStub();
  stub.responses.set("fetchExportLogs", () => ({
    entries: [],
    cursor: 42,
    active: false,
  }));

  const flipStates: { isExporting: boolean; lastSeq: number | null }[] = [];
  const unsubscribe = useExportRuntimeStore.subscribe((state) => {
    flipStates.push({ isExporting: state.isExporting, lastSeq: state.lastSeq });
  });
  try {
    await useExportRuntimeStore.getState().runExport(params());
  } finally {
    unsubscribe();
  }

  // The baseline fetch happens before any phase POST, with no cursor (a bare
  // "what is the current highest seq?" read).
  const baselineCall = stub.calls.find((c) => c.name === "fetchExportLogs");
  assert.ok(baselineCall, "no baseline log-cursor read was made");
  assert.equal(baselineCall.args[0], null);
  const firstPost = stub.calls.find((c) => c.name === "loadCheckpoint");
  assert.ok(firstPost, "the run never reached its first phase POST");
  assert.ok(
    stub.calls.indexOf(baselineCall) < stub.calls.indexOf(firstPost),
    "the baseline read must precede the phase POSTs",
  );

  // The very first state with isExporting=true already carries the baseline:
  // there was never a flipped-but-unseeded window for a since-less connect.
  const firstFlip = flipStates.find((s) => s.isExporting);
  assert.ok(firstFlip, "isExporting never flipped");
  assert.equal(firstFlip.lastSeq, 42);
  assert.equal(useExportRuntimeStore.getState().lastSeq, 42);
});

test("a failed baseline read degrades to the previous behavior", async () => {
  stub.resetStub();
  stub.responses.set(
    "fetchExportLogs",
    () => {
      throw new Error("baseline read down");
    },
  );

  await useExportRuntimeStore.getState().runExport(params());

  // The run continues (load happened) and lastSeq falls back to null, which
  // is exactly the pre-fix seeding behavior — no new failure mode.
  assert.ok(
    stub.calls.some((c) => c.name === "loadCheckpoint"),
    "a failed baseline read must not abort the run",
  );
  assert.equal(useExportRuntimeStore.getState().lastSeq, null);
});

test("a non-numeric cursor degrades to null instead of poisoning the de-duper", async () => {
  stub.resetStub();
  stub.responses.set("fetchExportLogs", () => ({
    entries: [],
    cursor: undefined,
    active: false,
  }));

  await useExportRuntimeStore.getState().runExport(params());

  assert.equal(useExportRuntimeStore.getState().lastSeq, null);
  assert.ok(
    stub.calls.some((c) => c.name === "loadCheckpoint"),
    "the run must continue",
  );
});