// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A DPO/ORPO/CPO run ignores packing and response-only masking: the preference trainers drop
// the first and always mask both sides themselves. The UI hides both toggles, the start
// payload pins them off, and a saved config leaves them out instead of exporting values the
// trainer would discard.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();

const yaml = await import("js-yaml");
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);
const { serializeConfigToYaml } = await import(
  "../src/features/training/lib/yaml-config.ts"
);
const { countNonDefaultAdvancedSettings } = await import(
  "../src/features/studio/wizard/advanced-settings-summary.ts"
);
const { usesPreferenceObjective } = await import(
  "../src/features/training/lib/training-methods.ts"
);

const CONFIG: TrainingConfigState = {
  ...initialTrainingConfigState,
  modelType: "text",
  selectedModel: "unsloth/gemma-3-270m-it",
  trainingMethod: "qlora",
  datasetSource: "huggingface",
  dataset: "unsloth/test",
  datasetSplit: "train",
  packing: true,
  trainOnCompletions: true,
};

function savedTrainingSection(
  state: TrainingConfigState,
): Record<string, unknown> {
  const parsed = yaml.load(serializeConfigToYaml(state, false)) as {
    training: Record<string, unknown>;
  };
  return parsed.training;
}

test("only an explicit DPO/ORPO/CPO objective counts as a preference run", () => {
  for (const method of ["dpo", "orpo", "cpo"] as const) {
    assert.equal(usesPreferenceObjective("qlora", method), true, method);
    assert.equal(usesPreferenceObjective("lora", method), true, method);
    assert.equal(usesPreferenceObjective("full", method), true, method);
    // CPT keeps a stale dropdown selection but never trains it.
    assert.equal(usesPreferenceObjective("cpt", method), false, method);
  }
  assert.equal(usesPreferenceObjective("qlora", "sft"), false);
  assert.equal(usesPreferenceObjective("qlora", null), false);
  assert.equal(usesPreferenceObjective("qlora", undefined), false);
});

test("the start payload pins packing and response masking off for a preference run", () => {
  const sft = buildTrainingStartPayload(
    { ...CONFIG, preferenceTrainingMethod: "sft" },
    null,
  );
  assert.equal(sft.packing, true);
  assert.equal(sft.train_on_completions, true);

  for (const preferenceTrainingMethod of ["dpo", "orpo", "cpo"] as const) {
    const payload = buildTrainingStartPayload(
      { ...CONFIG, preferenceTrainingMethod },
      null,
    );
    assert.equal(payload.packing, false, preferenceTrainingMethod);
    assert.equal(payload.train_on_completions, false, preferenceTrainingMethod);
  }
});

test("a saved preference config leaves both toggles out of the file", () => {
  const preference = savedTrainingSection({
    ...CONFIG,
    preferenceTrainingMethod: "orpo",
  });
  assert.equal("packing" in preference, false);
  assert.equal("train_on_completions" in preference, false);

  // An SFT run still round-trips the values the user picked.
  const sft = savedTrainingSection({
    ...CONFIG,
    preferenceTrainingMethod: "sft",
  });
  assert.equal(sft.packing, true);
  assert.equal(sft.train_on_completions, true);
});

test("the advanced settings summary does not count the two dead toggles", () => {
  assert.equal(
    countNonDefaultAdvancedSettings({
      ...CONFIG,
      preferenceTrainingMethod: "orpo",
    }),
    0,
  );
  assert.equal(
    countNonDefaultAdvancedSettings({
      ...CONFIG,
      preferenceTrainingMethod: "sft",
    }),
    2,
  );
});
