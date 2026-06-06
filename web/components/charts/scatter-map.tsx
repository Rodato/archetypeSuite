"use client";

import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { ScatterPoint } from "@/lib/types";
import { paletteColor } from "@/lib/palette";
import { ChartTooltip } from "@/components/charts/chart-tooltip";

export function ScatterMap({ points, height = 440 }: { points: ScatterPoint[]; height?: number }) {
  if (!points.length) {
    return (
      <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
        No hay suficientes variables numéricas para dibujar el mapa.
      </div>
    );
  }

  // Group points by archetype, preserving cluster_id order for stable colors.
  const groups = new Map<number, { name: string; data: ScatterPoint[] }>();
  for (const p of points) {
    if (!groups.has(p.cluster_id)) groups.set(p.cluster_id, { name: p.archetype, data: [] });
    groups.get(p.cluster_id)!.data.push(p);
  }
  const ordered = [...groups.entries()].sort((a, b) => a[0] - b[0]);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 12, left: -8, bottom: 8 }}>
        <CartesianGrid stroke="color-mix(in oklch, var(--border) 70%, transparent)" />
        <XAxis
          type="number"
          dataKey="PC1"
          name="Eje 1"
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          type="number"
          dataKey="PC2"
          name="Eje 2"
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          tickLine={false}
          axisLine={false}
        />
        <ZAxis range={[60, 60]} />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          content={<ChartTooltip formatter={(v) => Number(v).toFixed(2)} />}
        />
        <Legend
          wrapperStyle={{ fontSize: 12 }}
          iconType="circle"
          iconSize={8}
          formatter={(value) => <span className="text-muted-foreground">{value}</span>}
        />
        {ordered.map(([cid, group]) => (
          <Scatter key={cid} name={group.name} data={group.data} fill={paletteColor(cid)} fillOpacity={0.78} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
