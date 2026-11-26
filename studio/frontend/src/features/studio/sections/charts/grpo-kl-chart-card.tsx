// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import type { ReactElement } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import {
  CHART_CONTAINER_CLASS,
  CHART_SYNC_ID,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  buildYDomain,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
} from "./utils";

type Point = { step: number; kl: number };

export function GrpoKlChartCard({
  data,
  klThreshold,
}: {
  data: Point[];
  klThreshold?: number | null;
}): ReactElement {
  const hasThreshold =
    klThreshold != null && Number.isFinite(klThreshold) && klThreshold > 0;
  const values = [
    ...data.map((point) => point.kl).filter(Number.isFinite),
    ...(hasThreshold ? [klThreshold] : []),
  ];
  const domain = buildYDomain(values);
  const config = {
    kl: { label: "KL divergence", color: "#2563eb" },
  } satisfies ChartConfig;

  return (
    <Card data-tour="studio-training-grpo-kl" size="sm">
      <CardHeader>
        <CardTitle className="text-sm">KL Divergence</CardTitle>
        <CardDescription className="text-xs">
          Policy drift from the reference model per step
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className={CHART_CONTAINER_CLASS}>
          <LineChart
            data={data}
            syncId={CHART_SYNC_ID}
            margin={DEFAULT_CHART_MARGIN}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              fontSize={10}
              tickFormatter={(value) => formatStepTick(Number(value))}
            />
            <YAxis
              domain={domain}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickCount={5}
              width={DEFAULT_Y_AXIS_WIDTH}
              fontSize={10}
              tickFormatter={(value) => formatAxisMetric(Number(value))}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(_value, payload) =>
                    `Step ${payload?.[0]?.payload?.step ?? ""}`
                  }
                  formatter={(value, name) => [
                    formatMetric(Number(value)),
                    String(name),
                  ]}
                />
              }
            />
            {hasThreshold && (
              <ReferenceLine
                y={klThreshold}
                stroke="#dc2626"
                strokeDasharray="4 4"
                strokeWidth={1.4}
              />
            )}
            <Line
              type="linear"
              dataKey="kl"
              stroke="var(--color-kl)"
              strokeWidth={1.8}
              dot={false}
              connectNulls={true}
              isAnimationActive={false}
            />
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
