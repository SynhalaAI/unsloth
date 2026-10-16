// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SystemInfoResponse } from "@/hooks/use-system";
import type { TrainingConfigState } from "../types/config";

/** Whether a run can carry replay datasets: the torch worker mixes them; MLX has no mixing path,
 * and the backend refuses streaming, raw-text/CPT and preference (DPO/CPO) runs. */
export function replayDatasetSupported(
  config: Pick<
    TrainingConfigState,
    | "trainingMethod"
    | "preferenceTrainingMethod"
    | "datasetStreaming"
    | "isEmbeddingModel"
    | "modelType"
  >,
): boolean {
  if (config.trainingMethod === "cpt") return false;
  if (config.preferenceTrainingMethod !== "sft") return false;
  if (config.datasetStreaming) return false;
  if (config.isEmbeddingModel || config.modelType === "embeddings") return false;
  if (config.modelType === "decision") return false;
  return true;
}

/** Whether this config asks for replay mixing; the payload omits `replay_datasets` otherwise, so
 * an unarmed run posts byte-identical fields to what it did before the feature existed. */
export function sendsReplayDataset(
  config: TrainingConfigState,
  system: Pick<SystemInfoResponse, "device_backend"> | null = null,
): boolean {
  if (!config.replayDatasetEnabled) return false;
  if (system?.device_backend === "mlx") return false;
  if (!(config.replayDataset ?? "").trim()) return false;
  const probability = Number(config.replayDatasetProbability);
  if (!Number.isFinite(probability) || probability < 1 || probability > 95) return false;
  return replayDatasetSupported(config);
}
