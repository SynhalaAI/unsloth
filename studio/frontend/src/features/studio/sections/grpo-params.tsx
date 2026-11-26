// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { GrpoLossType, GrpoRewardFunc } from "@/types/training";
import type { ReactElement } from "react";
import { ParamsRow, ParamsSliderRow } from "./params-section-controls";

export const GRPO_LOSS_TYPE_OPTIONS: ReadonlyArray<{
  value: GrpoLossType;
  label: string;
  hint: string;
}> = [
  {
    value: "grpo",
    label: "GRPO (Standard)",
    hint: "Standard Group Relative Policy Optimization normalizing advantages within each prompt group.",
  },
  {
    value: "bnpo",
    label: "BNPO",
    hint: "Batch-normalized policy optimization; reduces variance on small group sizes.",
  },
  {
    value: "dr_grpo",
    label: "Dr. GRPO",
    hint: "Removes length and std bias from GRPO updates to prevent runaway completion lengths.",
  },
  {
    value: "dapo",
    label: "DAPO",
    hint: "Token-level loss aggregation for long-context reasoning rollouts.",
  },
];

export const GRPO_REWARD_OPTIONS: ReadonlyArray<{
  value: GrpoRewardFunc;
  label: string;
  description: string;
}> = [
  {
    value: "answer_match",
    label: "Answer match",
    description:
      "Scores 1.0 when <answer>...</answer>, \boxed{}, or #### matches the dataset answer column.",
  },
  {
    value: "answer_format",
    label: "<answer> tag format",
    description: "Rewards properly formatted <answer>...</answer> tags.",
  },
  {
    value: "reasoning_format",
    label: "Reasoning + answer tags",
    description:
      "Rewards <think>...</think> (or <reasoning>) paired with <answer>...</answer>.",
  },
  {
    value: "conciseness",
    label: "Conciseness",
    description: "Rewards non-empty completions without repetitive loops.",
  },
];

const NUM_GENERATION_PRESETS = [2, 4, 8, 16];
const COMPLETION_LENGTH_PRESETS = [128, 256, 512, 1024, 2048, 4096];
const ITERATION_PRESETS = [1, 2, 3, 4];

export function formatGrpoLossTypeLabel(value: string | undefined): string {
  switch ((value ?? "").trim().toLowerCase()) {
    case "bnpo":
      return "BNPO";
    case "dr_grpo":
      return "Dr. GRPO";
    case "dapo":
      return "DAPO";
    default:
      return "GRPO";
  }
}

export function GrpoParams({
  isMac,
  contextLength,
  effectivePromptLength,
  numGenerations,
  maxCompletionLength,
  grpoBeta,
  grpoTemperature,
  grpoEpsilon,
  grpoNumIterations,
  grpoLossType,
  grpoRewardFuncs,
  grpoUseVllm,
  grpoKlEarlyStopThreshold,
  grpoKlEarlyStopPatience,
  onNumGenerationsChange,
  onMaxCompletionLengthChange,
  onGrpoBetaChange,
  onGrpoTemperatureChange,
  onGrpoEpsilonChange,
  onGrpoNumIterationsChange,
  onGrpoLossTypeChange,
  onGrpoRewardFuncsChange,
  onGrpoUseVllmChange,
  onGrpoKlEarlyStopThresholdChange,
  onGrpoKlEarlyStopPatienceChange,
}: {
  isMac: boolean;
  contextLength: number;
  effectivePromptLength: number;
  numGenerations: number;
  maxCompletionLength: number;
  grpoBeta: number;
  grpoTemperature: number;
  grpoEpsilon: number;
  grpoNumIterations: number;
  grpoLossType: GrpoLossType;
  grpoRewardFuncs: GrpoRewardFunc[];
  grpoUseVllm: boolean;
  grpoKlEarlyStopThreshold: number;
  grpoKlEarlyStopPatience: number;
  onNumGenerationsChange: (value: number) => void;
  onMaxCompletionLengthChange: (value: number) => void;
  onGrpoBetaChange: (value: number) => void;
  onGrpoTemperatureChange: (value: number) => void;
  onGrpoEpsilonChange: (value: number) => void;
  onGrpoNumIterationsChange: (value: number) => void;
  onGrpoLossTypeChange: (value: GrpoLossType) => void;
  onGrpoRewardFuncsChange: (value: GrpoRewardFunc[]) => void;
  onGrpoUseVllmChange: (value: boolean) => void;
  onGrpoKlEarlyStopThresholdChange: (value: number) => void;
  onGrpoKlEarlyStopPatienceChange: (value: number) => void;
}): ReactElement {
  const activeLossMeta =
    GRPO_LOSS_TYPE_OPTIONS.find((option) => option.value === grpoLossType) ??
    GRPO_LOSS_TYPE_OPTIONS[0];

  const completionCap = Math.max(16, contextLength - effectivePromptLength);
  const effectiveCompletionLength = Math.min(maxCompletionLength, completionCap);
  const completionPresets = COMPLETION_LENGTH_PRESETS.filter(
    (len) => len <= completionCap,
  );
  if (completionPresets.length === 0) {
    completionPresets.push(completionCap);
  }

  const toggleReward = (reward: GrpoRewardFunc, checked: boolean) => {
    const next = checked
      ? Array.from(new Set([...grpoRewardFuncs, reward]))
      : grpoRewardFuncs.filter((item) => item !== reward);
    if (next.length > 0) {
      onGrpoRewardFuncsChange(next);
    }
  };

  return (
    <>
      <ParamsRow
        label="Num generations (G)"
        tooltip="Number of completions sampled per prompt to compute group-relative advantages. Higher values improve gradient signal at extra rollout cost (typical: 4–8)."
      >
        <Select
          value={String(numGenerations)}
          onValueChange={(value) => onNumGenerationsChange(Number(value))}
        >
          <SelectTrigger className="w-28">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {!NUM_GENERATION_PRESETS.includes(numGenerations) && (
              <SelectItem value={String(numGenerations)}>
                {numGenerations}
              </SelectItem>
            )}
            {NUM_GENERATION_PRESETS.map((count) => (
              <SelectItem key={count} value={String(count)}>
                {count}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </ParamsRow>

      <div className="flex flex-col gap-1">
        <ParamsRow
          label="Max completion length"
          tooltip={`Maximum generated tokens per rollout. Capped at context length minus prompt length (${completionCap}).`}
        >
          <Select
            value={String(effectiveCompletionLength)}
            onValueChange={(value) =>
              onMaxCompletionLengthChange(Number(value))
            }
          >
            <SelectTrigger className="w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {!completionPresets.includes(effectiveCompletionLength) && (
                <SelectItem value={String(effectiveCompletionLength)}>
                  {effectiveCompletionLength} (Custom)
                </SelectItem>
              )}
              {completionPresets.map((length) => (
                <SelectItem key={length} value={String(length)}>
                  {length}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </ParamsRow>
        {effectiveCompletionLength < maxCompletionLength && (
          <p className="text-ui-11p5 leading-relaxed text-muted-foreground">
            {maxCompletionLength} exceeds remaining context ({completionCap});
            the run uses {effectiveCompletionLength}.
          </p>
        )}
      </div>

      <div className="flex flex-col gap-1">
        <ParamsRow
          label="Loss type"
          tooltip="GRPO objective variant passed to TRL GRPOConfig."
        >
          <Select
            value={grpoLossType}
            onValueChange={(value) =>
              onGrpoLossTypeChange(value as GrpoLossType)
            }
          >
            <SelectTrigger className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {GRPO_LOSS_TYPE_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </ParamsRow>
        <p className="text-ui-11p5 leading-relaxed text-muted-foreground">
          {activeLossMeta.hint}
        </p>
      </div>

      <ParamsSliderRow
        label="KL β"
        tooltip="KL divergence regularization penalty against the reference policy. Set to 0 for KL-free GRPO/DAPO; typical range: 0.0–0.1."
        value={grpoBeta}
        onChange={onGrpoBetaChange}
        min={0}
        max={0.5}
        step={0.01}
      />

      <ParamsSliderRow
        label="Temperature"
        tooltip="Rollout sampling temperature. Higher values increase diversity across the G generations (typical: 0.7–1.0)."
        value={grpoTemperature}
        onChange={onGrpoTemperatureChange}
        min={0.1}
        max={2.0}
        step={0.05}
      />

      <ParamsSliderRow
        label="Clip ε"
        tooltip="PPO/GRPO policy ratio clipping range (default: 0.2)."
        value={grpoEpsilon}
        onChange={onGrpoEpsilonChange}
        min={0.05}
        max={0.5}
        step={0.05}
      />

      <ParamsRow
        label="Iterations (μ)"
        tooltip="Number of policy update iterations performed per rollout batch (default: 1)."
      >
        <Select
          value={String(grpoNumIterations)}
          onValueChange={(value) => onGrpoNumIterationsChange(Number(value))}
        >
          <SelectTrigger className="w-28">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ITERATION_PRESETS.map((n) => (
              <SelectItem key={n} value={String(n)}>
                {n}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </ParamsRow>

      <div className="flex flex-col gap-2 pt-1">
        <span className="text-xs font-medium text-muted-foreground">
          Reward functions
        </span>
        <div className="flex flex-col gap-2">
          {GRPO_REWARD_OPTIONS.map((option) => {
            const checked = grpoRewardFuncs.includes(option.value);
            const isOnlySelected = checked && grpoRewardFuncs.length === 1;
            return (
              <label
                key={option.value}
                className="flex cursor-pointer items-start gap-2 text-xs"
              >
                <Checkbox
                  checked={checked}
                  disabled={isOnlySelected}
                  onCheckedChange={(value) =>
                    toggleReward(option.value, !!value)
                  }
                  className="mt-0.5"
                />
                <span className="flex flex-col">
                  <span className="font-medium text-foreground">
                    {option.label}
                  </span>
                  <span className="text-ui-11p5 text-muted-foreground">
                    {option.description}
                  </span>
                </span>
              </label>
            );
          })}
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <ParamsRow
          label="KL early stop"
          tooltip="Automatically stop GRPO if KL divergence exceeds this limit for consecutive steps (0 = off). Helps stop policy collapse early (typical: 0.5–1.0)."
        >
          <Input
            type="number"
            step="0.05"
            min="0"
            max="100"
            value={grpoKlEarlyStopThreshold}
            onChange={(event) => {
              const val = Number(event.target.value);
              onGrpoKlEarlyStopThresholdChange(
                Number.isFinite(val) && val >= 0 ? val : 0,
              );
            }}
            className="w-28 font-mono"
          />
        </ParamsRow>
        <p className="text-ui-11p5 leading-relaxed text-muted-foreground">
          {grpoKlEarlyStopThreshold > 0
            ? `Stops training if KL > ${grpoKlEarlyStopThreshold} for ${grpoKlEarlyStopPatience} consecutive step${grpoKlEarlyStopPatience === 1 ? "" : "s"} (or immediately on NaN/Inf).`
            : "0 = off. Set 0.5–1.0 to stop automatically if KL spikes and risks model collapse."}
        </p>
      </div>

      {grpoKlEarlyStopThreshold > 0 && (
        <ParamsSliderRow
          label="KL patience (steps)"
          tooltip="Number of consecutive logged steps above the KL threshold before stopping, so a single noisy step does not stop training."
          value={grpoKlEarlyStopPatience}
          onChange={onGrpoKlEarlyStopPatienceChange}
          min={1}
          max={20}
          step={1}
        />
      )}

      <div className="flex items-center gap-2 pt-1">
        <Checkbox
          id="grpoUseVllm"
          checked={grpoUseVllm && !isMac}
          disabled={isMac}
          onCheckedChange={(value) => onGrpoUseVllmChange(!!value)}
        />
        <label
          htmlFor="grpoUseVllm"
          className={`text-xs text-muted-foreground ${
            isMac ? "cursor-not-allowed opacity-60" : "cursor-pointer"
          }`}
        >
          Use vLLM fast rollouts
        </label>
      </div>
    </>
  );
}
