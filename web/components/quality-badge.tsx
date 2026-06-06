import type { Quality } from "@/lib/types";
import { GRADE_CLASS } from "@/lib/palette";
import { cn } from "@/lib/utils";

export function QualityBadge({ quality, size = "md" }: { quality: Quality; size?: "sm" | "md" | "lg" }) {
  const dims = {
    sm: "size-7 text-sm rounded-lg",
    md: "size-10 text-lg rounded-xl",
    lg: "size-16 text-3xl rounded-2xl",
  }[size];
  return (
    <span
      className={cn("inline-flex items-center justify-center font-bold tabular-nums", GRADE_CLASS[quality.color], dims)}
      title={`${quality.label}${quality.score != null ? ` · ${quality.score.toFixed(2)}` : ""}`}
    >
      {quality.grade}
    </span>
  );
}
