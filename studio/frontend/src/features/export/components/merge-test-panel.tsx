// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  AlertCircleIcon,
  CheckmarkCircle02Icon,
  InformationCircleIcon,
  Warning02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import type {
  MergeAnalyzeReport,
  MergeInterference,
} from "../api/export-api";
import type { MergeMethodType } from "../constants";

/** A cosine in -1..1, as a signed percentage; the sign is the whole point. */
function formatCosine(value: number): string {
  const percent = Math.round(value * 100);
  return percent > 0 ? `+${percent}%` : `${percent}%`;
}

/** A rate in 0..1 as a percentage. */
function formatRate(value: number): string {
  return `${Math.round(value * 100)}%`;
}

const INTERFERENCE_STYLE: Record<
  MergeInterference,
  { icon: typeof CheckmarkCircle02Icon; text: string; accent: string }
> = {
  low: {
    icon: CheckmarkCircle02Icon,
    text: "Low interference",
    accent: "text-emerald-600 dark:text-emerald-500",
  },
  moderate: {
    icon: Warning02Icon,
    text: "Moderate interference",
    accent: "text-amber-600 dark:text-amber-500",
  },
  high: {
    icon: AlertCircleIcon,
    text: "High interference",
    accent: "text-destructive",
  },
};

export interface MergeTestPanelProps {
  pending: boolean;
  error: string | null;
  report: MergeAnalyzeReport | null;
  /** Applies the report's suggested method and density to the merge settings. */
  onApplyRecommendation: (method: MergeMethodType, density: number) => void;
}
/**
 * The result of the Export page's Test button: how much the selected adapters
 * interfere, measured on their own weights, plus the method that fits. The
 * numbers describe interference, never accuracy - a report is a reason to pick
 * a method, not a measurement of the merged model, which is why the footer
 * says so.
 */
export function MergeTestPanel({
  pending,
  error,
  report,
  onApplyRecommendation,
}: MergeTestPanelProps) {
  if (pending) {
    return (
      <div
        className="flex items-center gap-2 rounded-md border px-3 py-2 text-xs text-muted-foreground"
        role="status"
      >
        <Spinner className="size-3.5" />
        Comparing the adapters' weights. No model is loaded.
      </div>
    );
  }

  if (error) {
    return (
      <div
        className="flex items-start gap-2 rounded-md border px-3 py-2 text-xs text-destructive"
        role="alert"
      >
        <HugeiconsIcon icon={AlertCircleIcon} className="mt-px size-3.5 shrink-0" />
        {error}
      </div>
    );
  }

  if (!report) return null;

  const severity = INTERFERENCE_STYLE[report.interference];
  const recommendation = report.recommendation;

  return (
    <div className="space-y-3 rounded-md border px-3 py-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div
          className={`flex items-center gap-1.5 text-xs font-medium ${severity.accent}`}
        >
          <HugeiconsIcon icon={severity.icon} className="size-4" />
          {severity.text}
        </div>
        <span className="text-ui-11 text-muted-foreground">
          {report.pairs.length} pair{report.pairs.length === 1 ? "" : "s"} compared
        </span>
      </div>
      <div className="grid grid-cols-1 gap-1.5 text-ui-11 sm:grid-cols-2">
        <div>
          <span className="text-muted-foreground">Mean cosine </span>
          <span className="font-medium text-foreground">
            {formatCosine(report.mean_cosine)}
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">Worst sign conflict </span>
          <span className="font-medium text-foreground">
            {formatRate(report.max_sign_conflict_rate)}
          </span>
        </div>
      </div>

      <div className="space-y-1.5">
        {report.pairs.map((pair) => (
          <div
            key={`${pair.a}::${pair.b}`}
            className="rounded-sm border border-muted-foreground/20 px-2 py-1.5 text-ui-11"
          >
            <div className="flex flex-wrap items-baseline justify-between gap-x-2">
              <span className="truncate font-medium text-foreground">
                {pair.a} + {pair.b}
              </span>
              <span className="text-muted-foreground">
                {pair.shared_modules} shared module
                {pair.shared_modules === 1 ? "" : "s"}
              </span>
            </div>
            <div className="mt-0.5 flex flex-wrap gap-x-3 text-muted-foreground">
              <span>cosine {formatCosine(pair.cosine)}</span>
              <span>sign conflict {formatRate(pair.sign_conflict_rate)}</span>
              {pair.norm_ratio > 1.01 || pair.norm_ratio < 0.99 ? (
                <span>
                  norm ratio{" "}
                  {pair.norm_ratio >= 1
                    ? `${pair.norm_ratio.toFixed(1)}x`
                    : `1 / ${(1 / pair.norm_ratio).toFixed(1)}x`}
                </span>
              ) : null}
            </div>
            {pair.worst_modules.length > 0 ? (
              <div className="mt-0.5 space-y-0.5 text-muted-foreground/80">
                {pair.worst_modules.map((module) => (
                  <div key={module.module} className="truncate">
                    {module.module} (cosine {formatCosine(module.cosine)})
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-muted-foreground/15 pt-2">
        <div className="flex min-w-0 items-start gap-1.5 text-ui-11 text-muted-foreground">
          <HugeiconsIcon
            icon={InformationCircleIcon}
            className="mt-px size-3.5 shrink-0"
          />
          <span className="min-w-0">
            {recommendation.reason}
          </span>
        </div>
        <button
          type="button"
          onClick={() =>
            onApplyRecommendation(
              recommendation.method,
              recommendation.density,
            )
          }
          className="shrink-0 text-ui-11 font-medium text-foreground underline underline-offset-2 transition-colors hover:text-foreground/70"
        >
          Use {recommendation.method} (density {recommendation.density})
        </button>
      </div>

      <p className="text-ui-11 text-muted-foreground/70">
        Measured on the adapters&apos; weights only. It says how much the
        adapters fight each other, not how accurate the merged model will be:
        run the merge and check it on held-out data before shipping it.
        {report.sign_scan_truncated
          ? " Sign-conflict rates come from a sample, so treat them as approximate."
          : ""}
      </p>
    </div>
  );
}