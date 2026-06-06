"use client";

import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { RadarData } from "@/lib/types";
import { ChartTooltip } from "@/components/charts/chart-tooltip";

export function RadarCompare({ radar, height = 440 }: { radar: RadarData; height?: number }) {
  if (!radar.axes.length || !radar.series.length) {
    return (
      <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
        No hay variables numéricas para comparar.
      </div>
    );
  }

  // Pivot keyed by cluster_id (unique) — labels can collide, which would collapse series.
  const seriesKey = (clusterId: number) => `c${clusterId}`;
  const data = radar.axes.map((axis, i) => {
    const row: Record<string, string | number> = { axis };
    radar.series.forEach((s) => {
      row[seriesKey(s.cluster_id)] = s.values[i] ?? 0;
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="color-mix(in oklch, var(--border) 80%, transparent)" />
        <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
        <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
        <Tooltip content={<ChartTooltip formatter={(v) => Number(v).toFixed(2)} />} />
        <Legend
          wrapperStyle={{ fontSize: 12 }}
          iconType="circle"
          iconSize={8}
          formatter={(value) => <span className="text-muted-foreground">{value}</span>}
        />
        {radar.series.map((s) => (
          <Radar
            key={s.cluster_id}
            name={s.label}
            dataKey={seriesKey(s.cluster_id)}
            stroke={s.color}
            fill={s.color}
            fillOpacity={0.12}
            strokeWidth={2}
          />
        ))}
      </RadarChart>
    </ResponsiveContainer>
  );
}
