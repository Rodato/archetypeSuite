"use client";

import type { ReactNode } from "react";

interface TooltipPayloadItem {
  name?: string | number;
  value?: string | number;
  color?: string;
  payload?: Record<string, unknown>;
}

interface Props {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string | number;
  suffix?: string;
  formatter?: (v: number | string, name?: string | number) => ReactNode;
}

/** Themed tooltip shared across all recharts charts. */
export function ChartTooltip({ active, payload, label, suffix = "", formatter }: Props) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-popover px-2.5 py-1.5 text-xs shadow-md">
      {label != null && label !== "" && (
        <div className="mb-1 font-medium text-foreground">{label}</div>
      )}
      <div className="flex flex-col gap-0.5">
        {payload.map((item, i) => (
          <div key={i} className="flex items-center gap-1.5 text-muted-foreground">
            {item.color && <span className="size-2 rounded-[3px]" style={{ background: item.color }} />}
            {item.name != null && <span>{item.name}:</span>}
            <span className="font-medium tabular-nums text-foreground">
              {formatter ? formatter(item.value ?? "", item.name) : `${item.value}${suffix}`}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
