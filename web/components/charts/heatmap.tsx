"use client";

/** Correlation-style heatmap (RdBu, domain [-1,1]). Pure CSS grid, no chart lib. */
export function Heatmap({
  xLabels,
  yLabels,
  z,
}: {
  xLabels: string[];
  yLabels: string[];
  z: (number | null)[][];
}) {
  const color = (v: number | null) => {
    if (v == null) return "var(--muted)";
    // RdBu_r: -1 blue, 0 white-ish, +1 red
    const t = Math.max(-1, Math.min(1, v));
    if (t >= 0) {
      const a = t;
      return `color-mix(in oklch, #ef4444 ${Math.round(a * 80)}%, var(--card))`;
    }
    const a = -t;
    return `color-mix(in oklch, #3b82f6 ${Math.round(a * 80)}%, var(--card))`;
  };

  return (
    <div className="scroll-slim overflow-auto">
      <table className="border-separate border-spacing-0.5 text-[11px]">
        <thead>
          <tr>
            <th className="sticky left-0 bg-card" />
            {xLabels.map((x) => (
              <th key={x} className="px-1 pb-1 text-center font-medium text-muted-foreground">
                <span className="inline-block max-w-[60px] truncate align-bottom">{x}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {yLabels.map((y, i) => (
            <tr key={y}>
              <th className="sticky left-0 bg-card pr-2 text-right font-medium text-muted-foreground">
                <span className="inline-block max-w-[90px] truncate align-middle">{y}</span>
              </th>
              {xLabels.map((_, j) => {
                const v = z[i]?.[j] ?? null;
                return (
                  <td
                    key={j}
                    className="h-9 w-12 rounded text-center tabular-nums"
                    style={{ background: color(v), color: v != null && Math.abs(v) > 0.6 ? "#fff" : "var(--foreground)" }}
                    title={`${y} × ${xLabels[j]}: ${v ?? "—"}`}
                  >
                    {v != null ? v.toFixed(2) : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
