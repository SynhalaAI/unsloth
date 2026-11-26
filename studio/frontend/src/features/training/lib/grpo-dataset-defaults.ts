// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_HYPERPARAMS,
  LR_DEFAULT_FULL,
  LR_DEFAULT_LORA,
} from "@/config/training";
import type { TrainingConfigState } from "../types/config";

const GRPO_DEFAULTS = {
  epochs: 1,
  loraLearningRate: 5e-6,
  fullLearningRate: 5e-7,
  batchSize: 1,
  gradientAccumulation: 4,
} as const;

type GrpoDefaultsState = Pick<
  TrainingConfigState,
  | "trainingMethod"
  | "trainingMethodProvenance"
  | "preferenceTrainingMethod"
  | "epochs"
  | "learningRate"
  | "batchSize"
  | "gradientAccumulation"
>;

export function grpoDatasetDefaultsPatch(
  state: GrpoDefaultsState,
): Partial<TrainingConfigState> {
  if (state.trainingMethod === "cpt") {
    return {};
  }

  const patch: Partial<TrainingConfigState> = {};
  if (state.preferenceTrainingMethod === "sft") {
    patch.preferenceTrainingMethod = "grpo";
  }
  if (state.epochs === DEFAULT_HYPERPARAMS.epochs) {
    patch.epochs = GRPO_DEFAULTS.epochs;
  }
  if (
    !state.trainingMethodProvenance.learningRateManuallySet &&
    state.learningRate ===
      (state.trainingMethod === "full" ? LR_DEFAULT_FULL : LR_DEFAULT_LORA)
  ) {
    patch.learningRate =
      state.trainingMethod === "full"
        ? GRPO_DEFAULTS.fullLearningRate
        : GRPO_DEFAULTS.loraLearningRate;
  }
  if (state.batchSize === DEFAULT_HYPERPARAMS.batchSize) {
    patch.batchSize = GRPO_DEFAULTS.batchSize;
  }
  if (
    state.gradientAccumulation === DEFAULT_HYPERPARAMS.gradientAccumulation
  ) {
    patch.gradientAccumulation = GRPO_DEFAULTS.gradientAccumulation;
  }
  return patch;
}
