// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PreferenceTrainingMethod } from "@/types/training";

export type TrainingPhase =
  | "idle"
  | "downloading_model"
  | "downloading_dataset"
  | "loading_model"
  | "loading_dataset"
  | "configuring"
  | "training"
  // Steps done, worker still saving. Non-terminal: 100% is not success.
  | "finalizing"
  // SFT saved; training the speculative-decoding drafter. Non-terminal.
  | "training_drafter"
  | "completed"
  | "error"
  | "stopped";

export interface TrainingStatusResponse {
  job_id: string;
  start_request_id?: string | null;
  start_request_state?: "pending" | "accepted" | "rejected" | null;
  phase: TrainingPhase;
  is_training_running: boolean;
  eval_enabled: boolean;
  message: string;
  error: string | null;
  warnings?: string[];
  details?: {
    epoch?: number;
    step?: number;
    total_steps?: number;
    loss?: number;
    learning_rate?: number;
    // null = explicit clear (run stopped without saving); absent = unchanged.
    output_dir?: string | null;
    model_download_repo_id?: string | null;
  } | null;
  metric_history?: {
    steps?: number[];
    loss?: number[];
    lr?: number[];
    grad_norm?: number[];
    grad_norm_steps?: number[];
    eval_loss?: number[];
    eval_steps?: number[];
    reward_accuracy?: number[];
    reward_accuracy_steps?: number[];
    reward_margin?: number[];
    reward_margin_steps?: number[];
    eval_reward_accuracy?: number[];
    eval_reward_accuracy_steps?: number[];
    eval_reward_margin?: number[];
    eval_reward_margin_steps?: number[];
    nll_loss?: number[];
    nll_loss_steps?: number[];
    log_odds_ratio?: number[];
    log_odds_ratio_steps?: number[];
    rewards_chosen?: number[];
    rewards_chosen_steps?: number[];
    rewards_rejected?: number[];
    rewards_rejected_steps?: number[];
  } | null;
}

export interface TrainingMetricsResponse {
  job_id: string;
  loss_history: number[];
  lr_history: number[];
  step_history: number[];
  grad_norm_history: number[];
  grad_norm_step_history: number[];
  current_loss: number | null;
  current_lr: number | null;
  current_step: number | null;
}

export interface TrainingProgressPayload {
  job_id: string;
  step: number;
  total_steps: number;
  loss: number | null;
  learning_rate: number | null;
  progress_percent: number;
  epoch: number | null;
  elapsed_seconds: number | null;
  eta_seconds: number | null;
  session_start_step?: number | null;
  grad_norm: number | null;
  num_tokens: number | null;
  eval_loss: number | null;
  rewards_chosen: number | null;
  rewards_rejected: number | null;
  rewards_accuracies: number | null;
  rewards_margins: number | null;
  eval_rewards_accuracies: number | null;
  eval_rewards_margins: number | null;
  nll_loss: number | null;
  log_odds_ratio: number | null;
}

export interface TrainingSeriesPoint {
  step: number;
  value: number;
}

export interface TrainingRuntimeState {
  jobId: string | null;
  phase: TrainingPhase;
  isTrainingRunning: boolean;
  evalEnabled: boolean;
  message: string;
  error: string | null;
  warnings: string[];
  isHydrating: boolean;
  hasHydrated: boolean;
  isStarting: boolean;
  startRequestId: string | null;
  startError: string | null;
  startModelName: string | null;
  modelDownloadRepoId: string | null;
  startDatasetName: string | null;
  startHfToken: string | null;
  startProjectName: string | null;
  startFromResume: boolean;
  sseConnected: boolean;
  firstStepReceived: boolean;
  lastEventId: number | null;
  currentStep: number;
  totalSteps: number;
  currentEpoch: number;
  // null while the latest reported loss is non-finite
  currentLoss: number | null;
  currentLearningRate: number;
  progressPercent: number;
  elapsedSeconds: number | null;
  etaSeconds: number | null;
  sessionStartStep: number;
  currentGradNorm: number | null;
  currentNumTokens: number | null;
  outputDir: string | null;
  lossHistory: TrainingSeriesPoint[];
  lrHistory: TrainingSeriesPoint[];
  gradNormHistory: TrainingSeriesPoint[];
  evalLossHistory: TrainingSeriesPoint[];
  rewardAccuracyHistory: TrainingSeriesPoint[];
  rewardMarginHistory: TrainingSeriesPoint[];
  evalRewardAccuracyHistory: TrainingSeriesPoint[];
  evalRewardMarginHistory: TrainingSeriesPoint[];
  nllLossHistory: TrainingSeriesPoint[];
  logOddsRatioHistory: TrainingSeriesPoint[];
  rewardsChosenHistory: TrainingSeriesPoint[];
  rewardsRejectedHistory: TrainingSeriesPoint[];
  resetGeneration: number;
  stopRequested: boolean;
  configureRequest: number;
  selectedHistoryRunId: string | null;
  currentRunViewActive: boolean;
}

export interface TrainingRuntimeActions {
  setStopRequested: (value: boolean) => void;
  setHydrating: (value: boolean) => void;
  setHasHydrated: (value: boolean) => void;
  tryBeginStarting: (startRequestId: string) => boolean;
  setStarting: (value: boolean) => void;
  setStartError: (value: string | null) => void;
  setStartResources: (
    modelName: string | null,
    datasetName: string | null,
    fromResume?: boolean,
    projectName?: string | null,
    hfToken?: string | null,
  ) => void;
  setSseConnected: (value: boolean) => void;
  setLastEventId: (value: number | null) => void;
  resetRuntime: () => void;
  applyStatus: (payload: TrainingStatusResponse) => void;
  applyMetrics: (payload: TrainingMetricsResponse) => void;
  applyProgress: (payload: TrainingProgressPayload, eventId?: number) => void;
  setStartPending: (
    jobId: string | null,
    message: string,
    startRequestId?: string | null,
  ) => void;
  setRuntimeError: (message: string) => void;
  setSelectedHistoryRunId: (id: string | null) => void;
  setCurrentRunViewActive: (value: boolean) => void;
}

export type TrainingRuntimeStore = TrainingRuntimeState &
  TrainingRuntimeActions;

export interface TrainingViewData {
  phase: TrainingPhase;
  currentStep: number;
  totalSteps: number;
  currentLoss: number | null;
  currentLearningRate: number | null;
  currentGradNorm: number | null;
  currentEpoch: number | null;
  currentNumTokens: number | null;
  outputDir: string | null;
  // A newer run reused this output_dir (resume), so its on-disk contents differ.
  resumedLater?: boolean;
  progressPercent: number;
  elapsedSeconds: number | null;
  etaSeconds: number | null;
  sessionStartStep?: number;
  evalEnabled: boolean;
  message: string;
  error: string | null;
  warnings: string[];
  isTrainingRunning: boolean;

  modelName: string;
  projectName: string | null;
  trainingMethod: string;
  // DPO/ORPO/CPO objective saved with the run; null/undefined for a plain SFT run.
  preferenceMethod?: PreferenceTrainingMethod | null;
  isDecision?: boolean;

  lossHistory: TrainingSeriesPoint[];
  lrHistory: TrainingSeriesPoint[];
  gradNormHistory: TrainingSeriesPoint[];
  evalLossHistory: TrainingSeriesPoint[];
  rewardAccuracyHistory: TrainingSeriesPoint[];
  rewardMarginHistory: TrainingSeriesPoint[];
  evalRewardAccuracyHistory: TrainingSeriesPoint[];
  evalRewardMarginHistory: TrainingSeriesPoint[];
  nllLossHistory: TrainingSeriesPoint[];
  logOddsRatioHistory: TrainingSeriesPoint[];
  rewardsChosenHistory: TrainingSeriesPoint[];
  rewardsRejectedHistory: TrainingSeriesPoint[];
}
