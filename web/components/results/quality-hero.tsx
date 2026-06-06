import type { Quality } from "@/lib/types";
import { GRADE_TEXT } from "@/lib/palette";
import { QualityBadge } from "@/components/quality-badge";
import { cn } from "@/lib/utils";

export function QualityHero({ quality }: { quality: Quality }) {
  return (
    <div className="flex items-center gap-4">
      <QualityBadge quality={quality} size="lg" />
      <div className="min-w-0">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Calidad de la segmentación
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-lg font-semibold">{quality.label}</span>
          {quality.score != null && (
            <span className={cn("text-sm font-semibold tabular-nums", GRADE_TEXT[quality.color])}>
              {quality.score.toFixed(2)}
            </span>
          )}
        </div>
        <p className="mt-0.5 text-sm text-muted-foreground">{quality.description}</p>
      </div>
    </div>
  );
}
