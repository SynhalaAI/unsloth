// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { CHART_CONTAINER_CLASS, CHART_SYNC_ID, DEFAULT_CHART_MARGIN, DEFAULT_Y_AXIS_WIDTH, buildYDomain, formatAxisMetric, formatMetric, formatStepTick } from "./utils";

type Point = { step: number; nllLoss: number };

/**
 * Auxiliary SFT cross-entropy on chosen completions when logged without ORPO's log-odds ratio
 * (CPO runs and DPO + RPO runs on TRL < 0.29).
 */
export function NllLossChartCard({ data }: { data: Point[] }): ReactElement {
  const values = data.map((point) => point.nllLoss).filter(Number.isFinite);
  const domain = buildYDomain(values);
  const config = {
    nllLoss: { label: "NLL loss (chosen)", color: "#7c3aed" },
  } satisfies ChartConfig;
  return (
    <Card data-tour="studio-training-nll-loss" size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Chosen SFT NLL loss</CardTitle>
        <CardDescription className="text-xs">
          Cross-entropy loss on chosen responses from the RPO / CPO SFT term
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className={CHART_CONTAINER_CLASS}>
          <LineChart data={data} syncId={CHART_SYNC_ID} margin={DEFAULT_CHART_MARGIN}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tickLine={false} axisLine={false} tickMargin={8} fontSize={10} tickFormatter={(value) => formatStepTick(Number(value))} />
            <YAxis domain={domain} tickLine={false} axisLine={false} tickMargin={8} tickCount={5} width={DEFAULT_Y_AXIS_WIDTH} fontSize={10} tickFormatter={(value) => formatAxisMetric(Number(value))} />
            <ChartTooltip content={<ChartTooltipContent labelFormatter={(_value, payload) => `Step ${payload?.[0]?.payload?.step ?? ""}`} formatter={(value, name) => [formatMetric(Number(value)), String(name)]} />} />
            <Line type="linear" dataKey="nllLoss" stroke="var(--color-nllLoss)" strokeWidth={1.8} dot={false} connectNulls={true} isAnimationActive={false} />
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
