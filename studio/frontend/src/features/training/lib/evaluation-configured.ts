// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a saved run was configured to evaluate.
 *
 * Mirrors the backend's `evaluation_enabled` (a finite, positive `eval_steps`), so the UI
 * can treat "evaluation configured" as a property of the run rather than of whether eval
 * data has arrived yet. The eval-loss card uses this to keep a configured run from being
 * mislabeled "Evaluation not configured" while its first eval step is still pending.
 *
 * Kept dependency-free (no `@/` imports) so it runs under `node --experimental-strip-types`.
 */
export function isEvaluationConfigured(evalSteps: unknown): boolean {
  if (typeof evalSteps === "boolean") {
    return false;
  }
  const value = Number(evalSteps);
  return Number.isFinite(value) && value > 0;
}
