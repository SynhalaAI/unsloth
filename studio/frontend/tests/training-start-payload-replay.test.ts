// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();
const { buildTrainingStartPayload } =
  await import("../src/features/training/api/mappers.ts");
const { initialTrainingConfigState } =
  await import("../src/features/training/stores/training-config-policy.ts");

const MODEL = "unsloth/gemma-4-E2B-it";

function config(patch: Partial<TrainingConfigState> = {}): TrainingConfigState {
  return {
    ...initialTrainingConfigState,
    modelType: "text",
    selectedModel: MODEL,
    datasetSource: "huggingface",
    dataset: "unsloth/test",
    ...patch,
  };
}

const ON = {
  replayDatasetEnabled: true,
  replayDataset: "org/replay-pack",
  replayDatasetSubset: null,
  replayDatasetSplit: null,
  replayDatasetProbability: 20,
} as const;

test("replay is off by default and the payload has no replay_datasets key", () => {
  assert.equal(initialTrainingConfigState.replayDatasetEnabled, false);
  const payload = buildTrainingStartPayload(config(), null);
  assert.equal(Object.hasOwn(payload, "replay_datasets"), false);
});

test("an armed replay dataset rides the payload normalized", () => {
  const payload = buildTrainingStartPayload(config(ON), null);
  assert.deepEqual(payload.replay_datasets, [
    { dataset: "org/replay-pack", subset: null, split: "train", probability: 0.2 },
  ]);
  const withOptions = buildTrainingStartPayload(
    config({
      ...ON,
      replayDatasetSubset: "dev",
      replayDatasetSplit: "validation",
      replayDatasetProbability: 95,
    }),
    null,
  );
  assert.deepEqual(withOptions.replay_datasets, [
    { dataset: "org/replay-pack", subset: "dev", split: "validation", probability: 0.95 },
  ]);
});

test("armed but unusable settings leave the payload exactly as unarmed", () => {
  const cases: Partial<TrainingConfigState>[] = [
    // Ticked but no repo id yet.
    { ...ON, replayDataset: "  " },
    // Probability outside the schema share.
    { ...ON, replayDatasetProbability: 0 },
    { ...ON, replayDatasetProbability: 96 },
    // Backend refusals mirrored in the gate.
    { ...ON, datasetStreaming: true },
    { ...ON, trainingMethod: "cpt" },
    { ...ON, preferenceTrainingMethod: "dpo" },
    { ...ON, isEmbeddingModel: true },
    { ...ON, modelType: "decision" },
  ];
  for (const patch of cases) {
    const armed = buildTrainingStartPayload(config(patch), null);
    const unarmed = buildTrainingStartPayload(
      config({ ...patch, replayDatasetEnabled: false }),
      null,
    );
    assert.equal(Object.hasOwn(armed, "replay_datasets"), false);
    assert.deepEqual(armed, unarmed);
  }
});

test("an mlx backend drops replay even when armed", () => {
  const mlx = buildTrainingStartPayload(config(ON), null, { device_backend: "mlx" } as never);
  assert.equal(Object.hasOwn(mlx, "replay_datasets"), false);
});
