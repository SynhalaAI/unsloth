// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingSeriesPoint } from "@/features/training";
import type { PreferenceTrainingMethod } from "@/types/training";
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
  evalEnabled: boolean;
  lossHistory: TrainingSeriesPoint[];
  lrHistory: TrainingSeriesPoint[];
  gradNormHistory: TrainingSeriesPoint[];
  evalLossHistory: TrainingSeriesPoint[];
  rewardAccuracyHistory?: TrainingSeriesPoint[];
  rewardMarginHistory?: TrainingSeriesPoint[];
  evalRewardAccuracyHistory?: TrainingSeriesPoint[];
  evalRewardMarginHistory?: TrainingSeriesPoint[];
  nllLossHistory?: TrainingSeriesPoint[];
  logOddsRatioHistory?: TrainingSeriesPoint[];
  rewardsChosenHistory?: TrainingSeriesPoint[];
  rewardsRejectedHistory?: TrainingSeriesPoint[];
  preferenceMethod?: PreferenceTrainingMethod | null;
}

export function ChartsSection({
  currentStep,
  totalSteps,
  evalEnabled,
  lossHistory,
  lrHistory,
  gradNormHistory,
  evalLossHistory,
  rewardAccuracyHistory = [],
  rewardMarginHistory = [],
  evalRewardAccuracyHistory = [],
  evalRewardMarginHistory = [],
  nllLossHistory = [],
  logOddsRatioHistory = [],
  rewardsChosenHistory = [],
  rewardsRejectedHistory = [],
  preferenceMethod,
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
      rewardAccuracyHistory: rewardAccuracyHistory.map((point) => ({
        step: point.step,
        accuracy: point.value,
      })),
      rewardMarginHistory: rewardMarginHistory.map((point) => ({
        step: point.step,
        margin: point.value,
      })),
      evalRewardAccuracyHistory: evalRewardAccuracyHistory.map((point) => ({
        step: point.step,
        accuracy: point.value,
      })),
      evalRewardMarginHistory: evalRewardMarginHistory.map((point) => ({
        step: point.step,
        margin: point.value,
      })),
      nllLossHistory: nllLossHistory.map((point) => ({
        step: point.step,
        nllLoss: point.value,
      })),
      logOddsRatioHistory: logOddsRatioHistory.map((point) => ({
        step: point.step,
        logOddsRatio: point.value,
      })),
      rewardsChosenHistory: rewardsChosenHistory.map((point) => ({
        step: point.step,
        chosen: point.value,
      })),
      rewardsRejectedHistory: rewardsRejectedHistory.map((point) => ({
        step: point.step,
        rejected: point.value,
      })),
    }),
    [currentStep, evalLossHistory, evalRewardAccuracyHistory, evalRewardMarginHistory, gradNormHistory, logOddsRatioHistory, lossHistory, lrHistory, nllLossHistory, rewardAccuracyHistory, rewardMarginHistory, rewardsChosenHistory, rewardsRejectedHistory, totalSteps],
  );

  if (
    series.lossHistory.length === 0 &&
    series.lrHistory.length === 0 &&
    series.gradNormHistory.length === 0 &&
    series.evalLossHistory.length === 0 &&
    series.rewardAccuracyHistory.length === 0 &&
    series.rewardMarginHistory.length === 0 &&
    series.evalRewardAccuracyHistory.length === 0 &&
    series.evalRewardMarginHistory.length === 0 &&
    series.nllLossHistory.length === 0 &&
    series.logOddsRatioHistory.length === 0 &&
    series.rewardsChosenHistory.length === 0 &&
    series.rewardsRejectedHistory.length === 0
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
              className="h-[calc(280px*var(--ui-space-scale,1))] 4xl:h-[calc(360px*var(--ui-space-scale,1))] rounded-2xl border bg-muted/30 animate-pulse"
            />
          ))}
        </div>
      }
    >
      <ChartsContent
        metrics={series}
        evalEnabled={evalEnabled}
        preferenceMethod={preferenceMethod}
      />
    </Suspense>
  );
}
