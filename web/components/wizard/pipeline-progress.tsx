"use client";

import { AnimatePresence, motion } from "motion/react";
import { Check, Loader2, X } from "lucide-react";
import type { ProgressStep } from "@/lib/types";
import { cn } from "@/lib/utils";

export function PipelineProgress({
  steps,
  message,
}: {
  steps: ProgressStep[];
  message?: string;
}) {
  const done = steps.filter((s) => s.status === "done").length;
  const pct = steps.length ? Math.round((done / steps.length) * 100) : 0;

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <div className="text-sm font-medium text-muted-foreground">
          {message || "Procesando…"}
        </div>
        <div className="text-sm font-semibold tabular-nums text-primary">{pct}%</div>
      </div>
      <div className="mb-5 h-1.5 overflow-hidden rounded-full bg-secondary">
        <motion.div
          className="h-full rounded-full bg-primary"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        />
      </div>

      <ul className="space-y-1">
        {steps.map((step) => (
          <li
            key={step.key}
            className={cn(
              "flex items-center gap-3 rounded-lg px-2 py-2 transition-colors",
              step.status === "running" && "bg-accent/50",
            )}
          >
            <Marker status={step.status} />
            <span
              className={cn(
                "text-sm transition-colors",
                step.status === "done" && "text-muted-foreground",
                step.status === "running" && "font-medium text-foreground",
                step.status === "pending" && "text-muted-foreground/60",
                step.status === "failed" && "font-medium text-red-600 dark:text-red-400",
              )}
            >
              {step.label}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Marker({ status }: { status: ProgressStep["status"] }) {
  return (
    <span
      className={cn(
        "flex size-5 shrink-0 items-center justify-center rounded-full",
        status === "done" && "bg-emerald-500 text-white",
        status === "running" && "border border-primary bg-accent text-primary [animation:var(--animate-pulse-ring)]",
        status === "pending" && "border border-border bg-secondary",
        status === "failed" && "bg-red-500 text-white",
      )}
    >
      <AnimatePresence mode="wait">
        {status === "done" && (
          <motion.span key="d" initial={{ scale: 0 }} animate={{ scale: 1 }}>
            <Check className="size-3" strokeWidth={3} />
          </motion.span>
        )}
        {status === "running" && <Loader2 className="size-3 animate-spin" />}
        {status === "failed" && <X className="size-3" strokeWidth={3} />}
        {status === "pending" && <span className="size-1 rounded-full bg-muted-foreground/50" />}
      </AnimatePresence>
    </span>
  );
}
