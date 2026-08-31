// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingSeriesPoint } from "@/features/training";
import { type ReactElement, Suspense, lazy, useMemo } from "react";

const ChartsContent = lazy(() =>
  import("./charts-content").then((module) => ({
    default: module.ChartsContent,
  })),
);
const SKELETON_KEYS = [
  "chart-skeleton-1",
  "chart-skeleton-2",
  "chart-skeleton-3",
  "chart-skeleton-4",
];

interface ChartsSectionProps {
  currentStep: number;
  totalSteps: number;
  isTraining: boolean;
  isOcrTraining?: boolean;
  evalEnabled: boolean;
  lossHistory: TrainingSeriesPoint[];
  lrHistory: TrainingSeriesPoint[];
  gradNormHistory: TrainingSeriesPoint[];
  evalLossHistory: TrainingSeriesPoint[];
  cerHistory?: TrainingSeriesPoint[];
  werHistory?: TrainingSeriesPoint[];
}

export function ChartsSection({
  currentStep,
  totalSteps,
  isTraining,
  isOcrTraining = false,
  evalEnabled,
  lossHistory,
  lrHistory,
  gradNormHistory,
  evalLossHistory,
  cerHistory = [],
  werHistory = [],
}: ChartsSectionProps): ReactElement | null {
  const series = useMemo(
    () => ({
      currentStep,
      totalSteps,
      lossHistory: lossHistory.map((point) => ({
        step: point.step,
        loss: point.value,
      })),
      lrHistory: lrHistory.map((point) => ({
        step: point.step,
        lr: point.value,
      })),
      gradNormHistory: gradNormHistory.map((point) => ({
        step: point.step,
        gradNorm: point.value,
      })),
      evalLossHistory: evalLossHistory.map((point) => ({
        step: point.step,
        loss: point.value,
      })),
      cerHistory: cerHistory.map((point) => ({
        step: point.step,
        cer: point.value,
      })),
      werHistory: werHistory.map((point) => ({
        step: point.step,
        wer: point.value,
      })),
    }),
    [cerHistory, currentStep, evalLossHistory, gradNormHistory, lossHistory, lrHistory, totalSteps, werHistory],
  );

  if (
    series.lossHistory.length === 0 &&
    series.lrHistory.length === 0 &&
    series.gradNormHistory.length === 0 &&
    series.evalLossHistory.length === 0 &&
    series.cerHistory.length === 0 &&
    series.werHistory.length === 0 &&
    !isOcrTraining
  ) {
    return null;
  }

  return (
    <Suspense
      fallback={
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {SKELETON_KEYS.map((key) => (
            <div
              key={key}
              className="h-[280px] rounded-xl border bg-muted/30 animate-pulse"
            />
          ))}
        </div>
      }
    >
      <ChartsContent
        metrics={series}
        isTraining={isTraining}
        isOcrTraining={isOcrTraining}
        evalEnabled={evalEnabled}
      />
    </Suspense>
  );
}
