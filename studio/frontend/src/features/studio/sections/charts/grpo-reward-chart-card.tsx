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
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
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

type Point = { step: number; reward: number; rewardStd: number };

export function GrpoRewardChartCard({
  data,
}: {
  data: Point[];
}): ReactElement {
  const values = data
    .flatMap((point) => [point.reward, point.rewardStd])
    .filter(Number.isFinite);
  const domain = buildYDomain(values);
  const hasStd = data.some((point) => Number.isFinite(point.rewardStd));
  const config = {
    reward: { label: "Mean reward", color: "#16a34a" },
    ...(hasStd ? { rewardStd: { label: "Reward std", color: "#f59e0b" } } : {}),
  } satisfies ChartConfig;

  return (
    <Card data-tour="studio-training-grpo-reward" size="sm">
      <CardHeader>
        <CardTitle className="text-sm">GRPO Reward</CardTitle>
        <CardDescription className="text-xs">
          Mean rollout reward and within-group standard deviation
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
            <Line
              type="linear"
              dataKey="reward"
              stroke="var(--color-reward)"
              strokeWidth={1.8}
              dot={false}
              connectNulls={true}
              isAnimationActive={false}
            />
            {hasStd && (
              <Line
                type="linear"
                dataKey="rewardStd"
                stroke="var(--color-rewardStd)"
                strokeWidth={1.6}
                strokeDasharray="4 3"
                dot={false}
                connectNulls={true}
                isAnimationActive={false}
              />
            )}
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
