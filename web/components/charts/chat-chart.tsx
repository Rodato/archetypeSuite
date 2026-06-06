"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { ChatChart, TableData } from "@/lib/types";
import { paletteColor } from "@/lib/palette";
import { BoxPlot } from "@/components/charts/box-plot";
import { ChartTooltip } from "@/components/charts/chart-tooltip";
import { DataTable } from "@/components/data-table";
import { Heatmap } from "@/components/charts/heatmap";

type Row = Record<string, string | number | null>;

function toObjects(table: TableData): Row[] {
  return table.rows.map((r) => Object.fromEntries(table.columns.map((c, i) => [c, r[i]])));
}

const AXIS = { fontSize: 11, fill: "var(--muted-foreground)" } as const;
const gridStroke = "color-mix(in oklch, var(--border) 70%, transparent)";

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

/** True if a sample of non-null values in `col` are all numeric. */
function isNumericCol(rows: Row[], col?: string | null): boolean {
  if (!col) return false;
  const sample = rows.map((r) => r[col]).filter((v) => v !== null && v !== undefined && v !== "").slice(0, 12);
  return sample.length > 0 && sample.every((v) => Number.isFinite(Number(v)));
}

export function ChatChartView({ chart, height = 260 }: { chart: ChatChart; height?: number }) {
  const { type } = chart;

  if (type === "heatmap" && chart.x_labels && chart.y_labels && chart.z) {
    return <Heatmap xLabels={chart.x_labels} yLabels={chart.y_labels} z={chart.z} />;
  }

  if (!chart.data || type === "table" || type === "none") {
    return chart.data ? <DataTable table={chart.data} maxHeight={260} /> : null;
  }

  const data = toObjects(chart.data);
  const x = chart.x ?? chart.data.columns[0];
  const y = chart.y ?? chart.data.columns[1];
  const color = chart.color ?? undefined;
  // If the LLM picks a chart type incompatible with the actual data, show the table instead.
  const fallback = <DataTable table={chart.data} maxHeight={260} />;

  if (type === "pie") {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie data={data} dataKey={y} nameKey={x} innerRadius={48} outerRadius={84} paddingAngle={2} strokeWidth={0}>
            {data.map((_, i) => (
              <Cell key={i} fill={paletteColor(i)} />
            ))}
          </Pie>
          <Tooltip content={<ChartTooltip />} />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  if (type === "histogram") {
    const vals = data.map((d) => Number(d[x])).filter((v) => Number.isFinite(v));
    if (!vals.length) return null;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const bins = Math.min(12, Math.max(5, Math.round(Math.sqrt(vals.length))));
    const w = (max - min) / bins || 1;
    const hist = Array.from({ length: bins }, (_, i) => ({
      bin: `${(min + i * w).toFixed(0)}`,
      count: 0,
    }));
    vals.forEach((v) => {
      const idx = Math.min(bins - 1, Math.floor((v - min) / w));
      hist[idx].count++;
    });
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={hist} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke={gridStroke} />
          <XAxis dataKey="bin" tick={AXIS} tickLine={false} axisLine={false} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} />
          <Tooltip cursor={{ fill: "var(--muted)", opacity: 0.4 }} content={<ChartTooltip />} />
          <Bar dataKey="count" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (type === "scatter") {
    if (!isNumericCol(data, x) || !isNumericCol(data, y)) return fallback;
    return (
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid stroke={gridStroke} />
          <XAxis type="number" dataKey={x} name={x} tick={AXIS} tickLine={false} axisLine={false} />
          <YAxis type="number" dataKey={y} name={y} tick={AXIS} tickLine={false} axisLine={false} />
          <ZAxis range={[50, 50]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<ChartTooltip />} />
          <Scatter data={data} fill="var(--chart-1)" fillOpacity={0.7} />
        </ScatterChart>
      </ResponsiveContainer>
    );
  }

  if (type === "box") {
    // A real box needs a numeric value column with several points per category.
    // Aggregated/single-column results (e.g. one mean per group) can't form a box → table.
    if (!y || !isNumericCol(data, y)) return fallback;
    // Compute quartiles client-side per category from tidy {x, y}.
    const byCat = new Map<string, number[]>();
    data.forEach((d) => {
      const key = String(d[x]);
      const v = Number(d[y]);
      if (Number.isFinite(v)) byCat.set(key, [...(byCat.get(key) ?? []), v]);
    });
    const maxCount = Math.max(0, ...[...byCat.values()].map((a) => a.length));
    if (byCat.size === 0 || maxCount < 2) return fallback;
    const groups = [...byCat.entries()].map(([label, vals], i) => {
      const s = [...vals].sort((a, b) => a - b);
      return {
        cluster_id: i,
        label,
        color: paletteColor(i),
        min: s[0],
        q1: quantile(s, 0.25),
        median: quantile(s, 0.5),
        q3: quantile(s, 0.75),
        max: s[s.length - 1],
      };
    });
    return <BoxPlot groups={groups} height={height} />;
  }

  // bar / line — optionally grouped by `color`
  if (!chart.data.columns.includes(x) || !chart.data.columns.includes(y)) return fallback;
  const grouped = color && color !== x;
  let chartData = data;
  let seriesKeys = [y];
  if (grouped) {
    const xVals = [...new Set(data.map((d) => String(d[x])))];
    const colorVals = [...new Set(data.map((d) => String(d[color])))];
    seriesKeys = colorVals;
    chartData = xVals.map((xv) => {
      const row: Row = { [x]: xv };
      colorVals.forEach((cv) => {
        const match = data.find((d) => String(d[x]) === xv && String(d[color]) === cv);
        row[cv] = match ? Number(match[y]) : 0;
      });
      return row;
    });
  }

  const ChartComp = type === "line" ? LineChart : BarChart;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ChartComp data={chartData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke={gridStroke} />
        <XAxis dataKey={x} tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} />
        <Tooltip cursor={{ fill: "var(--muted)", opacity: 0.4 }} content={<ChartTooltip />} />
        {seriesKeys.map((key, i) =>
          type === "line" ? (
            <Line key={key} type="monotone" dataKey={key} stroke={paletteColor(i)} strokeWidth={2} dot={{ r: 3 }} />
          ) : (
            <Bar key={key} dataKey={key} fill={paletteColor(i)} radius={[4, 4, 0, 0]} maxBarSize={56} />
          ),
        )}
      </ChartComp>
    </ResponsiveContainer>
  );
}
