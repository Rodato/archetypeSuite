import type { TableData } from "@/lib/types";
import { cn } from "@/lib/utils";

function fmt(v: string | number | null): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    return Number.isInteger(v) ? v.toLocaleString("es") : v.toLocaleString("es", { maximumFractionDigits: 2 });
  }
  return String(v);
}

export function DataTable({ table, maxHeight }: { table: TableData; maxHeight?: number }) {
  return (
    <div
      className={cn("scroll-slim overflow-auto rounded-lg border", maxHeight && "")}
      style={maxHeight ? { maxHeight } : undefined}
    >
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-muted/80 backdrop-blur">
          <tr>
            {table.columns.map((c) => (
              <th key={c} className="whitespace-nowrap px-3 py-2 text-left text-xs font-semibold text-muted-foreground">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="border-t last:border-b-0 hover:bg-muted/40">
              {row.map((cell, j) => (
                <td key={j} className="whitespace-nowrap px-3 py-1.5 tabular-nums">
                  {fmt(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
