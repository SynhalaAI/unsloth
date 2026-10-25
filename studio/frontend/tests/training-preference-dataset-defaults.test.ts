// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_HYPERPARAMS, LR_DEFAULT_FULL } from "../src/config/training.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { preferenceDatasetDefaultsPatch } = await import(
  "../src/features/training/lib/preference-dataset-defaults.ts"
);

const defaultState = {
  trainingMethod: "qlora",
  trainingMethodProvenance: {
    learningRateManuallySet: false,
    modelAdapterLearningRate: null,
    datasetFormatBeforeCpt: null,
    targetModulesBeforeCpt: null,
    loraRankBeforeCpt: null,
    loraAlphaBeforeCpt: null,
    loraVariantBeforeCpt: null,
    trainOnCompletionsBeforeCpt: null,
  },
  preferenceTrainingMethod: "sft",
  epochs: DEFAULT_HYPERPARAMS.epochs,
  learningRate: DEFAULT_HYPERPARAMS.learningRate,
  batchSize: DEFAULT_HYPERPARAMS.batchSize,
  gradientAccumulation: DEFAULT_HYPERPARAMS.gradientAccumulation,
} as const;

test("preference datasets receive DPO defaults when general defaults remain", () => {
  assert.deepEqual(preferenceDatasetDefaultsPatch(defaultState), {
    preferenceTrainingMethod: "dpo",
    epochs: 1,
    learningRate: 5e-6,
    batchSize: 2,
    gradientAccumulation: 4,
  });
});

test("preference defaults preserve user changes", () => {
  const customState = {
    ...defaultState,
    preferenceTrainingMethod: "orpo",
    epochs: 5,
    learningRate: 0.0001,
    batchSize: 8,
    gradientAccumulation: 2,
    trainingMethodProvenance: {
      ...defaultState.trainingMethodProvenance,
      learningRateManuallySet: true,
    },
  } as const;

  assert.deepEqual(preferenceDatasetDefaultsPatch(customState), {});
});

test("full fine-tuning uses its lower preference learning-rate default", () => {
  assert.deepEqual(
    preferenceDatasetDefaultsPatch({
      ...defaultState,
      trainingMethod: "full",
      learningRate: LR_DEFAULT_FULL,
    }),
    {
      preferenceTrainingMethod: "dpo",
      epochs: 1,
      learningRate: 5e-7,
      batchSize: 2,
      gradientAccumulation: 4,
    },
  );
});

test("preference datasets do not change defaults during CPT", () => {
  assert.deepEqual(
    preferenceDatasetDefaultsPatch({
      ...defaultState,
      trainingMethod: "cpt",
    }),
    {},
  );
});
