// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { CHART_CONTAINER_CLASS, CHART_SYNC_ID, DEFAULT_CHART_MARGIN, DEFAULT_Y_AXIS_WIDTH, buildYDomain, formatAxisMetric, formatMetric, formatStepTick } from "./utils";

type Point = { step: number; accuracy: number };

/**
 * Next-token prediction accuracy on chosen completions (logged by TRL >= 0.29 / 1.x DPOTrainer).
 * Serves as the TRL 1.x generation-quality counterpart to NLL loss when using RPO / SFT loss.
 */
export function TokenAccuracyChartCard({ data }: { data: Point[] }): ReactElement {
  const values = data.map((point) => point.accuracy).filter(Number.isFinite);
  const domain = buildYDomain(values);
  const config = {
    accuracy: { label: "Token accuracy (chosen)", color: "#0d9488" },
  } satisfies ChartConfig;
  return (
    <Card data-tour="studio-training-token-accuracy" size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Chosen token accuracy</CardTitle>
        <CardDescription className="text-xs">
          Next-token accuracy on chosen responses during preference alignment
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className={CHART_CONTAINER_CLASS}>
          <LineChart data={data} syncId={CHART_SYNC_ID} margin={DEFAULT_CHART_MARGIN}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tickLine={false} axisLine={false} tickMargin={8} fontSize={10} tickFormatter={(value) => formatStepTick(Number(value))} />
            <YAxis domain={domain} tickLine={false} axisLine={false} tickMargin={8} tickCount={5} width={DEFAULT_Y_AXIS_WIDTH} fontSize={10} tickFormatter={(value) => formatAxisMetric(Number(value))} />
            <ChartTooltip content={<ChartTooltipContent labelFormatter={(_value, payload) => `Step ${payload?.[0]?.payload?.step ?? ""}`} formatter={(value, name) => [formatMetric(Number(value)), String(name)]} />} />
            <Line type="linear" dataKey="accuracy" stroke="var(--color-accuracy)" strokeWidth={1.8} dot={false} connectNulls={true} isAnimationActive={false} />
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
