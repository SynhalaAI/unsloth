// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { DpoLossType, PreferenceTrainingMethod } from "@/types/training";
import type { ReactElement } from "react";
import { ParamsRow, ParamsSliderRow } from "./params-section-controls";

export const DPO_LOSS_TYPE_OPTIONS: ReadonlyArray<{
  value: DpoLossType;
  label: string;
  hint: string;
}> = [
  {
    value: "sigmoid",
    label: "Sigmoid (Standard)",
    hint: "Standard DPO logistic loss; widens the chosen-vs-rejected reward gap (best on clean data).",
  },
  {
    value: "ipo",
    label: "IPO",
    hint: "Length-normalized squared margin loss; reduces length bias and resists overfitting on small or noisy datasets.",
  },
  {
    value: "hinge",
    label: "Hinge",
    hint: "Margin-based loss; stops pushing pairs once the gap passes 1/β, limiting overfitting on noisy labels.",
  },
];

export function formatDpoLossTypeLabel(value: string | undefined): string {
  switch ((value ?? "").trim().toLowerCase()) {
    case "ipo":
      return "IPO";
    case "hinge":
      return "Hinge";
    default:
      return "Sigmoid";
  }
}

export function preferenceModeTooltip(): ReactElement {
  return (
    <div className="flex flex-col gap-1">
      <span>
        <strong>SFT:</strong> Standard supervised fine-tuning on single
        responses.
      </span>
      <span>
        <strong>DPO:</strong> Aligns chosen vs rejected pairs against the
        reference model (best after SFT).
      </span>
      <span>
        <strong>ORPO / CPO:</strong> Reference-free preference alignment
        (lower VRAM, suitable from base models).
      </span>
    </div>
  );
}

export function preferenceBetaTooltip(
  method: PreferenceTrainingMethod,
): ReactElement {
  if (method === "orpo" || method === "cpo") {
    return (
      <>
        Controls the preference penalty weight (β). Lower values allow stronger
        style shifts; typical range is 0.05–0.5.
      </>
    );
  }
  return (
    <>
      KL regularization strength (β) against the reference model. Lower values
      allow larger policy shifts; higher values keep responses closer to the
      reference model (typical: 0.05–0.5).
    </>
  );
}

export function cpoAlphaTooltip(): ReactElement {
  return (
    <>
      Weight on the chosen NLL (SFT) term in CPO loss. Keeps chosen generation
      quality stable alongside contrastive preference updates (0 disables NLL;
      default: 1.0).
    </>
  );
}

export function DpoPreferenceParams({
  dpoLossType,
  rpoAlpha,
  onDpoLossTypeChange,
  onRpoAlphaChange,
}: {
  dpoLossType: DpoLossType;
  rpoAlpha: number;
  onDpoLossTypeChange: (value: DpoLossType) => void;
  onRpoAlphaChange: (value: number) => void;
}): ReactElement {
  const activeLossMeta =
    DPO_LOSS_TYPE_OPTIONS.find((option) => option.value === dpoLossType) ??
    DPO_LOSS_TYPE_OPTIONS[0];

  return (
    <>
      <div className="flex flex-col gap-1">
        <ParamsRow
          label="Loss type"
          tooltip={
            <div className="flex flex-col gap-1">
              <span>
                <strong>Sigmoid (Standard):</strong> Classic DPO logistic loss;
                best for clean datasets.
              </span>
              <span>
                <strong>IPO:</strong> Length-normalized squared margin loss;
                prevents length bias and overfitting on noisy or small
                datasets.
              </span>
              <span>
                <strong>Hinge:</strong> Stops pushing pairs once the margin
                exceeds 1/β, reducing overfitting on noisy pairs.
              </span>
            </div>
          }
        >
          <Select
            value={dpoLossType}
            onValueChange={(value) => onDpoLossTypeChange(value as DpoLossType)}
          >
            <SelectTrigger className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {DPO_LOSS_TYPE_OPTIONS.map((option) => (
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

      <div className="flex flex-col gap-1">
        <ParamsSliderRow
          label="RPO α"
          tooltip={
            <>
              Weight for an auxiliary SFT (NLL) loss on chosen responses (0 =
              off). Helps preserve fluency, formatting, and reasoning during
              DPO (maps to{" "}
              <code className="font-mono">loss_type=[..., "sft"]</code> +{" "}
              <code className="font-mono">loss_weights</code> on TRL 0.29+, or{" "}
              <code className="font-mono">rpo_alpha</code> on older TRL).
              Typical values: 0.1–1.0.
            </>
          }
          value={rpoAlpha}
          onChange={onRpoAlphaChange}
          min={0}
          max={10}
          step={0.1}
        />
        <p className="text-ui-11p5 leading-relaxed text-muted-foreground">
          {rpoAlpha > 0
            ? `Adds ${rpoAlpha}× SFT (NLL) loss on chosen responses to anchor generation quality (RPO).`
            : "0 = standard DPO. Set 0.1–1.0 to add chosen SFT (NLL) regularization (RPO)."}
        </p>
      </div>
    </>
  );
}
