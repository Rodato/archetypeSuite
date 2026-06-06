import Link from "next/link";
import { ArrowUpRight, Layers, Rows3 } from "lucide-react";
import type { RunSummary } from "@/lib/types";
import { formatDate } from "@/lib/palette";
import { QualityBadge } from "@/components/quality-badge";
import { Card } from "@/components/ui/card";

export function RunCard({ run }: { run: RunSummary }) {
  return (
    <Link href={`/runs/${run.id}`} className="group block">
      <Card className="relative h-full gap-0 overflow-hidden p-5 transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-lg">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate font-semibold tracking-tight">{run.file_name}</div>
            <div className="mt-0.5 text-xs text-muted-foreground">{formatDate(run.created_at)}</div>
          </div>
          <QualityBadge quality={run.quality} size="sm" />
        </div>

        {run.dataset_context && (
          <p className="mt-3 line-clamp-2 text-sm text-muted-foreground">{run.dataset_context}</p>
        )}

        <div className="mt-4 flex flex-wrap gap-1.5">
          {run.archetype_labels.slice(0, 4).map((label, i) => (
            <span
              key={i}
              className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
            >
              {label}
            </span>
          ))}
          {run.archetype_labels.length > 4 && (
            <span className="rounded-full px-1.5 py-0.5 text-xs text-muted-foreground">
              +{run.archetype_labels.length - 4}
            </span>
          )}
        </div>

        <div className="mt-4 flex items-center gap-4 border-t pt-3 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <Layers className="size-3.5" /> {run.n_archetypes} arquetipos
          </span>
          <span className="inline-flex items-center gap-1">
            <Rows3 className="size-3.5" /> {run.n_rows} filas
          </span>
          <ArrowUpRight className="ml-auto size-4 text-muted-foreground/50 transition-all group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-primary" />
        </div>
      </Card>
    </Link>
  );
}
