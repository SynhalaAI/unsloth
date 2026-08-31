// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { CHART_CONTAINER_CLASS, DEFAULT_CHART_MARGIN, DEFAULT_Y_AXIS_WIDTH, formatAxisMetric, formatMetric, formatStepTick } from "./utils";

type Point = { step: number; accuracy: number; margin: number };

export function PreferenceMetricsChartCard({ trainData, evalData }: { trainData: Point[]; evalData: Point[] }): ReactElement {
  const data = Array.from(new Set([...trainData.map((point) => point.step), ...evalData.map((point) => point.step)])).sort((a, b) => a - b).map((step) => ({
    step,
    trainAccuracy: trainData.find((point) => point.step === step)?.accuracy ?? Number.NaN,
    evalAccuracy: evalData.find((point) => point.step === step)?.accuracy ?? Number.NaN,
    trainMargin: trainData.find((point) => point.step === step)?.margin ?? Number.NaN,
    evalMargin: evalData.find((point) => point.step === step)?.margin ?? Number.NaN,
  }));
  const values = data.flatMap((point) => [point.trainAccuracy, point.evalAccuracy, point.trainMargin, point.evalMargin]).filter(Number.isFinite);
  const config = {
    trainAccuracy: { label: "Train preference accuracy", color: "#2563eb" },
    evalAccuracy: { label: "Eval preference accuracy", color: "#16a34a" },
    trainMargin: { label: "Train reward margin", color: "#f59e0b" },
    evalMargin: { label: "Eval reward margin", color: "#dc2626" },
  } satisfies ChartConfig;
  return (
    <Card data-tour="studio-training-preference-metrics" size="sm">
      <CardHeader><CardTitle className="text-sm">DPO / CPO preference metrics</CardTitle></CardHeader>
      <CardContent>
        <ChartContainer config={config} className={CHART_CONTAINER_CLASS}>
          <LineChart data={data} syncId="train-metrics-sync" margin={DEFAULT_CHART_MARGIN}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tickLine={false} axisLine={false} tickMargin={8} fontSize={10} tickFormatter={(value) => formatStepTick(Number(value))} />
            <YAxis domain={values.length ? [Math.min(...values), Math.max(...values)] : [0, 1]} tickLine={false} axisLine={false} tickMargin={8} tickCount={5} width={DEFAULT_Y_AXIS_WIDTH} fontSize={10} tickFormatter={(value) => formatAxisMetric(Number(value))} />
            <ChartTooltip content={<ChartTooltipContent labelFormatter={(_value, payload) => `Step ${payload?.[0]?.payload?.step ?? ""}`} formatter={(value, name) => [formatMetric(Number(value)), String(name)]} />} />
            {Object.keys(config).map((key) => <Line key={key} type="linear" dataKey={key} stroke={`var(--color-${key})`} strokeWidth={1.8} dot={false} connectNulls={true} isAnimationActive={false} />)}
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}