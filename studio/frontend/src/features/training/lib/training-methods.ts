// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DatasetFormat,
  ModelType,
  PreferenceTrainingMethod,
  TrainingMethod,
} from "@/types/training";
import type { LoraVariant } from "../types/config";

const BACKEND_TRAINING_TYPE: Record<TrainingMethod, string> = {
  qlora: "LoRA/QLoRA",
  lora: "LoRA/QLoRA",
  full: "Full Finetuning",
  cpt: "Continued Pretraining",
};

const BACKEND_PREFERENCE_METHOD: Record<
  PreferenceTrainingMethod,
  "SFT" | "DPO" | "ORPO" | "CPO"
> = {
  sft: "SFT",
  dpo: "DPO",
  orpo: "ORPO",
  cpo: "CPO",
};

const TRAINING_METHOD_LABELS: Record<TrainingMethod, string> = {
  qlora: "QLoRA",
  lora: "LoRA",
  full: "Full",
  cpt: "CPT",
};

export function toBackendTrainingType(trainingMethod: TrainingMethod): string {
  return BACKEND_TRAINING_TYPE[trainingMethod];
}

export function toBackendPreferenceMethod(
  method: PreferenceTrainingMethod,
): "SFT" | "DPO" | "ORPO" | "CPO" {
  return BACKEND_PREFERENCE_METHOD[method];
}

export function getTrainingMethodLabel(
  trainingMethod: TrainingMethod | string,
): string {
  if (
    Object.prototype.hasOwnProperty.call(TRAINING_METHOD_LABELS, trainingMethod)
  ) {
    return TRAINING_METHOD_LABELS[trainingMethod as TrainingMethod];
  }
  return TRAINING_METHOD_LABELS.full;
}

const PREFERENCE_METHOD_LABELS: Record<PreferenceTrainingMethod, string> = {
  sft: "SFT",
  dpo: "DPO",
  orpo: "ORPO",
  cpo: "CPO",
};

/** Map a run's saved `training_method` back onto the preference objective. */
export function parseBackendPreferenceMethod(
  value: unknown,
): PreferenceTrainingMethod | null {
  if (typeof value !== "string") {
    return null;
  }
  switch (value.trim().toUpperCase()) {
    case "SFT":
      return "sft";
    case "DPO":
      return "dpo";
    case "ORPO":
      return "orpo";
    case "CPO":
      return "cpo";
    default:
      return null;
  }
}

export function getPreferenceMethodLabel(
  method: PreferenceTrainingMethod | null | undefined,
): string | null {
  return method ? PREFERENCE_METHOD_LABELS[method] : null;
}

/** "QLoRA" for an SFT run, "QLoRA · ORPO" once a preference objective is chosen. */
export function formatTrainingMethodLabel(
  trainingMethod: TrainingMethod | string,
  preferenceMethod?: PreferenceTrainingMethod | null,
): string {
  const base = getTrainingMethodLabel(trainingMethod);
  const label = getPreferenceMethodLabel(preferenceMethod);
  // LoRA, QLoRA and full finetuning all train a preference objective here; CPT keeps
  // its dropdown selection but ignores it, so it keeps the plain label.
  if (!label || !usesPreferenceObjective(trainingMethod, preferenceMethod)) {
    return base;
  }
  return `${base} · ${label}`;
}

/**
 * True when the run trains a DPO/ORPO/CPO objective, so sequence packing and response-only
 * masking do not apply: the preference trainers drop `packing` and always mask both sides
 * themselves. CPT keeps its dropdown selection but ignores it, so it counts as SFT here.
 */
export function usesPreferenceObjective(
  trainingMethod: TrainingMethod | string,
  preferenceMethod?: PreferenceTrainingMethod | null,
): boolean {
  return (
    trainingMethod !== "cpt" &&
    preferenceMethod != null &&
    preferenceMethod !== "sft"
  );
}

export function isTrainingMethodSupportedOnDevice(
  trainingMethod: TrainingMethod,
  deviceType?: string,
): boolean {
  return deviceType !== "mac" || trainingMethod !== "cpt";
}

export function isTrainingModelTypeSupportedOnDevice(
  modelType: ModelType | null,
  deviceType?: string,
): boolean {
  return (
    deviceType !== "mac" ||
    (modelType !== "audio" && modelType !== "embeddings")
  );
}

export function isTrainingLoraVariantSupportedOnDevice(
  loraVariant: LoraVariant,
  trainingMethod: TrainingMethod,
  deviceType?: string,
): boolean {
  const usesAdapter =
    trainingMethod === "qlora" ||
    trainingMethod === "lora" ||
    trainingMethod === "cpt";
  return deviceType !== "mac" || !usesAdapter || loraVariant !== "loftq";
}

export function parseBackendTrainingMethod(
  trainingType: unknown,
  loadIn4Bit: unknown,
): TrainingMethod {
  if (trainingType === "Continued Pretraining") {
    return "cpt";
  }
  if (trainingType === "LoRA/QLoRA") {
    return loadIn4Bit ? "qlora" : "lora";
  }
  return "full";
}

export function isRawTextDatasetFormat(datasetFormat: DatasetFormat): boolean {
  return datasetFormat === "raw";
}
