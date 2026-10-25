// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_HYPERPARAMS,
  LR_DEFAULT_FULL,
  LR_DEFAULT_LORA,
} from "@/config/training";
import type { TrainingConfigState } from "../types/config";

const PREFERENCE_DEFAULTS = {
  epochs: 1,
  loraLearningRate: 5e-6,
  fullLearningRate: 5e-7,
  batchSize: 2,
  gradientAccumulation: 4,
} as const;

type PreferenceDefaultsState = Pick<
  TrainingConfigState,
  | "trainingMethod"
  | "trainingMethodProvenance"
  | "preferenceTrainingMethod"
  | "epochs"
  | "learningRate"
  | "batchSize"
  | "gradientAccumulation"
>;

export function preferenceDatasetDefaultsPatch(
  state: PreferenceDefaultsState,
): Partial<TrainingConfigState> {
  if (state.trainingMethod === "cpt") {
    return {};
  }

  const patch: Partial<TrainingConfigState> = {};
  if (state.preferenceTrainingMethod === "sft") {
    patch.preferenceTrainingMethod = "dpo";
  }
  if (state.epochs === DEFAULT_HYPERPARAMS.epochs) {
    patch.epochs = PREFERENCE_DEFAULTS.epochs;
  }
  if (
    !state.trainingMethodProvenance.learningRateManuallySet &&
    state.learningRate ===
      (state.trainingMethod === "full" ? LR_DEFAULT_FULL : LR_DEFAULT_LORA)
  ) {
    patch.learningRate =
      state.trainingMethod === "full"
        ? PREFERENCE_DEFAULTS.fullLearningRate
        : PREFERENCE_DEFAULTS.loraLearningRate;
  }
  if (state.batchSize === DEFAULT_HYPERPARAMS.batchSize) {
    patch.batchSize = PREFERENCE_DEFAULTS.batchSize;
  }
  if (
    state.gradientAccumulation === DEFAULT_HYPERPARAMS.gradientAccumulation
  ) {
    patch.gradientAccumulation = PREFERENCE_DEFAULTS.gradientAccumulation;
  }
  return patch;
}
