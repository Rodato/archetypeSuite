"use client";

import type { BoxGroup } from "@/lib/types";

/** Horizontal-friendly vertical box plot rendered as SVG. */
export function BoxPlot({ groups, height = 320 }: { groups: BoxGroup[]; height?: number }) {
  if (!groups.length) return null;

  const lo = Math.min(...groups.map((g) => g.min));
  const hi = Math.max(...groups.map((g) => g.max));
  const pad = (hi - lo) * 0.08 || 1;
  const domLo = lo - pad;
  const domHi = hi + pad;

  const plotH = height - 48;
  const y = (v: number) => plotH - ((v - domLo) / (domHi - domLo)) * plotH + 8;

  const n = groups.length;
  const slot = 100 / n; // percent width per group
  const boxW = Math.min(slot * 0.5, 14); // in %

  const ticks = 4;
  const tickVals = Array.from({ length: ticks + 1 }, (_, i) => domLo + ((domHi - domLo) * i) / ticks);

  return (
    <div className="w-full" style={{ height }}>
      <svg width="100%" height={height} className="overflow-visible">
        {/* gridlines */}
        {tickVals.map((tv, i) => (
          <g key={i}>
            <line x1="0%" x2="100%" y1={y(tv)} y2={y(tv)} stroke="var(--border)" strokeWidth={1} />
            <text x={2} y={y(tv) - 3} fontSize={10} fill="var(--muted-foreground)">
              {tv.toLocaleString("es", { maximumFractionDigits: 1 })}
            </text>
          </g>
        ))}

        {groups.map((g, i) => {
          const cx = slot * i + slot / 2;
          return (
            <g key={g.cluster_id}>
              {/* whisker */}
              <line x1={`${cx}%`} x2={`${cx}%`} y1={y(g.max)} y2={y(g.min)} stroke={g.color} strokeWidth={1.5} />
              <line x1={`${cx - boxW / 2}%`} x2={`${cx + boxW / 2}%`} y1={y(g.max)} y2={y(g.max)} stroke={g.color} strokeWidth={1.5} />
              <line x1={`${cx - boxW / 2}%`} x2={`${cx + boxW / 2}%`} y1={y(g.min)} y2={y(g.min)} stroke={g.color} strokeWidth={1.5} />
              {/* box */}
              <rect
                x={`${cx - boxW}%`}
                width={`${boxW * 2}%`}
                y={y(g.q3)}
                height={Math.max(y(g.q1) - y(g.q3), 1)}
                rx={3}
                fill={g.color}
                fillOpacity={0.22}
                stroke={g.color}
                strokeWidth={1.5}
              />
              {/* median */}
              <line x1={`${cx - boxW}%`} x2={`${cx + boxW}%`} y1={y(g.median)} y2={y(g.median)} stroke={g.color} strokeWidth={2.5} />
              <title>{`${g.label}\nmín ${g.min} · Q1 ${g.q1} · mediana ${g.median} · Q3 ${g.q3} · máx ${g.max}`}</title>
              {/* label */}
              <text x={`${cx}%`} y={height - 6} fontSize={10} fill="var(--muted-foreground)" textAnchor="middle">
                <tspan>{g.label.length > 12 ? g.label.slice(0, 11) + "…" : g.label}</tspan>
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
