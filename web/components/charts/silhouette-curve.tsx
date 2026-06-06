"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartTooltip } from "@/components/charts/chart-tooltip";

interface Props {
  kRange: number[];
  scores: (number | null)[];
  optimalK: number;
  height?: number;
}

export function SilhouetteCurve({ kRange, scores, optimalK, height = 200 }: Props) {
  if (!kRange.length) return null;
  const data = kRange.map((k, i) => ({ k, score: scores[i] }));
  const optIdx = kRange.indexOf(optimalK);
  const optScore = optIdx >= 0 ? scores[optIdx] : null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 12, right: 12, left: -18, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="color-mix(in oklch, var(--border) 70%, transparent)" />
        <XAxis dataKey="k" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} width={42} />
        <Tooltip content={<ChartTooltip formatter={(v) => Number(v).toFixed(3)} />} />
        <Line type="monotone" dataKey="score" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 3 }} />
        {optScore != null && (
          <ReferenceDot x={optimalK} y={optScore} r={6} fill="#f59e0b" stroke="var(--card)" strokeWidth={2} />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
