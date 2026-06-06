"use client";

import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ClusterSize } from "@/lib/types";
import { ChartTooltip } from "@/components/charts/chart-tooltip";

export function ClusterBar({ sizes, height = 240 }: { sizes: ClusterSize[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={sizes} margin={{ top: 16, right: 8, left: -18, bottom: 0 }}>
        <XAxis
          dataKey="label"
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          tickLine={false}
          axisLine={false}
          interval={0}
          tickFormatter={(v: string) => (v.length > 14 ? v.slice(0, 13) + "…" : v)}
        />
        <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} allowDecimals={false} />
        <Tooltip cursor={{ fill: "var(--muted)", opacity: 0.4 }} content={<ChartTooltip suffix=" personas" />} />
        <Bar dataKey="size" radius={[6, 6, 0, 0]} maxBarSize={72}>
          {sizes.map((s) => (
            <Cell key={s.cluster_id} fill={s.color} />
          ))}
          <LabelList dataKey="size" position="top" className="fill-foreground" fontSize={11} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
