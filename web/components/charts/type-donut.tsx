"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { AlertTriangle, CheckCircle2 } from "lucide-react";
import type { Donut } from "@/lib/types";
import { ChartTooltip } from "@/components/charts/chart-tooltip";

export function TypeDonut({ donut }: { donut: Donut }) {
  return (
    <div>
      <div className="relative mx-auto h-[176px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={donut.segments}
              dataKey="value"
              nameKey="label"
              innerRadius={56}
              outerRadius={82}
              paddingAngle={donut.segments.length > 1 ? 2 : 0}
              strokeWidth={0}
              startAngle={90}
              endAngle={-270}
              isAnimationActive
            >
              {donut.segments.map((s) => (
                <Cell key={s.label} fill={s.color} />
              ))}
            </Pie>
            <Tooltip content={<ChartTooltip suffix=" columnas" />} />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold tabular-nums">{donut.n_cols}</span>
          <span className="text-[11px] text-muted-foreground">columnas</span>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap justify-center gap-x-4 gap-y-1.5">
        {donut.segments.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="size-2 rounded-full" style={{ background: s.color }} />
            {s.label} <span className="font-medium text-foreground">{s.value}</span>
          </span>
        ))}
      </div>

      <div className="mt-3 flex items-center justify-center gap-1.5 text-xs">
        {donut.has_missing ?? donut.missing_pct > 0 ? (
          <span className="inline-flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
            <AlertTriangle className="size-3.5" />
            {donut.missing_pct >= 0.1
              ? `${donut.missing_pct}% de valores vacíos — los imputaremos`
              : "Hay algunos valores vacíos — los imputaremos"}
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="size-3.5" />
            Sin valores faltantes
          </span>
        )}
      </div>
    </div>
  );
}
