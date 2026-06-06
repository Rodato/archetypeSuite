import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

const STEPS = ["Datos", "Analizar", "Arquetipos"];

/** current is 1-based. Steps before `current` are done. */
export function WizardProgress({ current }: { current: 1 | 2 | 3 }) {
  return (
    <div className="mb-6">
      <div className="mb-2 flex items-baseline gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Paso {current} de 3
        </span>
        <span className="text-sm font-semibold">{STEPS[current - 1]}</span>
      </div>
      <div className="flex items-center gap-2">
        {STEPS.map((label, i) => {
          const n = i + 1;
          const state = n < current ? "done" : n === current ? "current" : "pending";
          return (
            <div key={label} className="flex flex-1 items-center gap-2">
              <div
                className={cn(
                  "h-1.5 flex-1 rounded-full transition-all duration-500",
                  state === "done" && "bg-emerald-500",
                  state === "current" && "bg-primary",
                  state === "pending" && "bg-secondary",
                )}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] text-muted-foreground">
        {STEPS.map((label, i) => (
          <span
            key={label}
            className={cn("inline-flex items-center gap-1", i + 1 <= current && "text-foreground")}
          >
            {i + 1 < current && <Check className="size-3 text-emerald-500" />}
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
