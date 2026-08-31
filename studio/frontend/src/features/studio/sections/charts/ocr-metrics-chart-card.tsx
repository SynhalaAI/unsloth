// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { useT } from "@/i18n";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  CHART_CONTAINER_CLASS,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
} from "./utils";

export function OcrMetricsChartCard({
  data,
  domain,
  ticks,
}: {
  data: { step: number; cer: number; wer: number }[];
  domain: [number, number];
  ticks?: number[];
}): ReactElement {
  const t = useT();
  const ocrConfig = {
    cer: { label: t("studio.charts.cer"), color: "#8b5cf6" },
    wer: { label: t("studio.charts.wer"), color: "#06b6d4" },
  } satisfies ChartConfig;

  return (
    <Card data-tour="studio-ocr-metrics" size="sm">
      <CardHeader>
        <CardTitle className="text-sm">{t("studio.charts.ocrMetrics")}</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={ocrConfig} className={CHART_CONTAINER_CLASS}>
          <LineChart data={data} accessibilityLayer={true} margin={DEFAULT_CHART_MARGIN}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              type="number"
              domain={["dataMin", "dataMax"]}
              ticks={ticks}
              allowDataOverflow={true}
              allowDecimals={false}
              minTickGap={28}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              fontSize={10}
              tickFormatter={(value) => formatStepTick(Number(value))}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={domain}
              allowDataOverflow={true}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickCount={5}
              fontSize={10}
              width={DEFAULT_Y_AXIS_WIDTH}
              tickFormatter={(value) => formatAxisMetric(Number(value))}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(_value, payload) =>
                    t("studio.charts.step", {
                      step: payload?.[0]?.payload?.step ?? "",
                    })
                  }
                  formatter={(_value, name, item) => {
                    const value = Number(item?.payload?.[String(name)] ?? _value);
                    return [formatMetric(value), name === "cer" ? t("studio.charts.cer") : t("studio.charts.wer")];
                  }}
                />
              }
            />
            <Line
              type="monotone"
              dataKey="cer"
              stroke="var(--color-cer)"
              strokeWidth={2}
              dot={{ r: 3, strokeWidth: 0, fill: "#8b5cf6" }}
              activeDot={{ r: 4, strokeWidth: 0 }}
              connectNulls={true}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="wer"
              stroke="var(--color-wer)"
              strokeWidth={2}
              dot={{ r: 3, strokeWidth: 0, fill: "#06b6d4" }}
              activeDot={{ r: 4, strokeWidth: 0 }}
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
