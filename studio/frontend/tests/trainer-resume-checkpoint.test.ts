// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Optional resume-from-checkpoint wiring: the picker only fills one config
// field, so these two assertions are the whole contract -- the start payload
// must carry it as resume_from_checkpoint, and the CTA must say Resume Training
// only while it is set (otherwise a fresh run would silently resume).

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);
const { resolveStartTrainingButtonLabelKey } = await import(
  "../src/features/studio/wizard/start-training-cta-state.ts"
);

const CONFIG: TrainingConfigState = {
  ...initialTrainingConfigState,
  modelType: "text",
  selectedModel: "org/model",
  trainingMethod: "lora",
  datasetSource: "huggingface",
  dataset: "org/dataset",
  datasetSplit: "train",
  datasetEvalSplit: null,
  datasetStreaming: false,
};

test("an armed checkpoint rides the start payload as resume_from_checkpoint", () => {
  assert.equal(
    buildTrainingStartPayload(CONFIG, null).resume_from_checkpoint,
    null,
    "no selection means a fresh run",
  );
  // Chosen but not armed: the Resume Training switch is off, so the run is fresh.
  assert.equal(
    buildTrainingStartPayload(
      {
        ...CONFIG,
        resumeCheckpointPath: "/outputs/run/checkpoint-10",
        resumeTrainingEnabled: false,
      },
      null,
    ).resume_from_checkpoint,
    null,
    "an unarmed checkpoint must not silently resume",
  );
  assert.equal(
    buildTrainingStartPayload(
      {
        ...CONFIG,
        resumeCheckpointPath: "/outputs/run/checkpoint-10",
        resumeTrainingEnabled: true,
      },
      null,
    ).resume_from_checkpoint,
    "/outputs/run/checkpoint-10",
  );
  assert.equal(
    buildTrainingStartPayload(
      { ...CONFIG, resumeTrainingEnabled: true },
      null,
    ).resume_from_checkpoint,
    null,
    "armed with no path still starts fresh",
  );
});

test("the start CTA reads Resume Training only while a checkpoint is chosen", () => {
  const ready = {
    stopRequested: false,
    startBlocked: false,
    isLoadingModel: false,
    isCheckingDataset: false,
    hasModel: true,
    hasDataset: true,
  } as const;
  assert.equal(
    resolveStartTrainingButtonLabelKey(ready),
    "studio.training.startTraining",
  );
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...ready,
      hasResumeCheckpoint: true,
    }),
    "studio.training.resumeTraining",
  );
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...ready,
      hasDataset: false,
      hasResumeCheckpoint: true,
    }),
    "studio.training.chooseDataset",
    "an incomplete selection still asks for the missing piece",
  );
});
