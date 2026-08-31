import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { ChartsSection } from "../src/features/studio/sections/charts-section.tsx";

test("ocr training renders cer and wer cards when OCR mode is enabled", () => {
  const html = renderToStaticMarkup(
    React.createElement(ChartsSection, {
      currentStep: 10,
      totalSteps: 10,
      isTraining: true,
      evalEnabled: false,
      isOcrTraining: true,
      lossHistory: [{ step: 1, value: 2.0 }, { step: 10, value: 1.2 }],
      lrHistory: [{ step: 1, value: 0.001 }, { step: 10, value: 0.0009 }],
      gradNormHistory: [{ step: 1, value: 0.8 }, { step: 10, value: 0.6 }],
      evalLossHistory: [],
      cerHistory: [{ step: 1, value: 0.35 }, { step: 10, value: 0.12 }],
      werHistory: [{ step: 1, value: 0.65 }, { step: 10, value: 0.2 }],
    }),
  );

  assert.match(html, /CER/i);
  assert.match(html, /WER/i);
});
