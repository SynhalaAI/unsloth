// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// TRL's preference trainers cap the prompt length themselves:
//   prompt_length = max(1, min(max_prompt_length, max(1, max_length // 2)))
// (studio/backend/core/training/preference_trainer.py). Offering a larger value in the UI
// would hide the fact that the run uses half the context length, so the row shows the
// value the trainer will actually use.

export const MAX_PROMPT_LENGTH_PRESETS = [
  128,
  256,
  512,
  1024,
  2048,
  4096,
  8192,
] as const;

/** The largest prompt length the trainer keeps for a given context length. */
export function maxPromptLengthCap(contextLength: number): number {
  const context = Number.isFinite(contextLength)
    ? Math.floor(contextLength)
    : 0;
  return Math.max(1, Math.floor(context / 2));
}

/** The prompt length the run uses, after the trainer's own cap. */
export function effectiveMaxPromptLength(
  contextLength: number,
  requested: number,
): number {
  const cap = maxPromptLengthCap(contextLength);
  const value = Number.isFinite(requested) ? Math.floor(requested) : cap;
  return Math.max(1, Math.min(value, cap));
}

/** The presets that fit under the cap, so the dropdown never offers a reduced value. */
export function maxPromptLengthPresets(contextLength: number): number[] {
  const cap = maxPromptLengthCap(contextLength);
  return MAX_PROMPT_LENGTH_PRESETS.filter((value) => value <= cap);
}
