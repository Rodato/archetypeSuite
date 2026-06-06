"use client";

import { useState } from "react";
import { AlertCircle, Check, Loader2, Sparkles, Wand2 } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { IMPORTANCE_META } from "@/lib/palette";
import type { ColumnExclusion, ColumnRec, Importance } from "@/lib/types";
import { useWizard } from "@/lib/wizard-store";
import { cn } from "@/lib/utils";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

function ImportanceBadge({ importance }: { importance: Importance }) {
  const meta = IMPORTANCE_META[importance];
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span
            className={cn(
              "cursor-default rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
              meta.className,
            )}
          />
        }
      >
        {meta.label}
      </TooltipTrigger>
      <TooltipContent>{meta.hint}</TooltipContent>
    </Tooltip>
  );
}

function ColRow({
  name,
  importance,
  reason,
  checked,
  onToggle,
}: {
  name: string;
  importance?: Importance;
  reason?: string;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className={cn(
        "flex w-full items-center gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors",
        checked ? "border-primary/40 bg-accent/50" : "border-transparent hover:bg-muted/60",
      )}
    >
      <span
        className={cn(
          "flex size-4 shrink-0 items-center justify-center rounded border transition-colors",
          checked ? "border-primary bg-primary text-primary-foreground" : "border-muted-foreground/40",
        )}
      >
        {checked && <Check className="size-3" strokeWidth={3} />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2">
          <span className="truncate font-medium">{name}</span>
          {importance && <ImportanceBadge importance={importance} />}
        </span>
        {reason && <span className="mt-0.5 block truncate text-xs text-muted-foreground">{reason}</span>}
      </span>
    </button>
  );
}

export function ColumnSelector() {
  const { dataset, context, suggestion, selectedColumns, setSuggestion, setSelectedColumns } = useWizard();
  const [loading, setLoading] = useState(false);

  if (!dataset) return null;

  const runSuggest = async () => {
    setLoading(true);
    try {
      const s = await api.suggestColumns(dataset.dataset_id, context);
      setSuggestion(s);
    } catch (e) {
      toast.error((e as Error).message || "No pudimos sugerir variables");
    } finally {
      setLoading(false);
    }
  };

  if (!suggestion) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-6 text-center">
        <p className="text-sm text-muted-foreground">
          Deja que la IA elija las variables que mejor diferencian los grupos — descartando IDs, fechas y texto libre.
        </p>
        <Button onClick={runSuggest} disabled={loading} className="gap-2">
          {loading ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
          Sugerir variables
        </Button>
      </div>
    );
  }

  const toggle = (name: string) => {
    setSelectedColumns(
      selectedColumns.includes(name)
        ? selectedColumns.filter((c) => c !== name)
        : [...selectedColumns, name],
    );
  };

  const recommended: ColumnRec[] = suggestion.column_recommendation.selected_columns;
  const recommendedNames = new Set(recommended.map((c) => c.name));
  const excludedMap = new Map<string, string>(
    suggestion.column_recommendation.excluded_columns.map((c: ColumnExclusion) => [c.name, c.reason]),
  );
  const others = suggestion.available_columns.filter((c) => !recommendedNames.has(c));
  const dropped = suggestion.static_filter_result.dropped;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <div className="inline-flex items-center gap-1.5 text-sm font-medium">
          <Sparkles className="size-4 text-primary" />
          La IA recomendó {recommended.length} de {suggestion.available_columns.length} columnas
        </div>
        <span className="text-xs text-muted-foreground">
          {selectedColumns.length} seleccionadas
        </span>
      </div>

      {suggestion.llm_error && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
          <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
          IA no disponible — te mostramos todas las columnas para que elijas tú.
        </div>
      )}

      <div className="scroll-slim max-h-[300px] overflow-y-auto pr-1">
        <Accordion multiple defaultValue={["rec"]} className="w-full">
          <AccordionItem value="rec" className="border-b-0">
            <AccordionTrigger className="py-2 text-sm">
              Recomendadas <span className="text-muted-foreground">({recommended.length})</span>
            </AccordionTrigger>
            <AccordionContent className="flex flex-col gap-1 pb-2">
              {recommended.map((c) => (
                <ColRow
                  key={c.name}
                  name={c.name}
                  importance={c.importance}
                  reason={c.reason}
                  checked={selectedColumns.includes(c.name)}
                  onToggle={() => toggle(c.name)}
                />
              ))}
            </AccordionContent>
          </AccordionItem>

          {others.length > 0 && (
            <AccordionItem value="others" className="border-b-0">
              <AccordionTrigger className="py-2 text-sm">
                Otras disponibles <span className="text-muted-foreground">({others.length})</span>
              </AccordionTrigger>
              <AccordionContent className="flex flex-col gap-1 pb-2">
                {others.map((name) => (
                  <ColRow
                    key={name}
                    name={name}
                    reason={excludedMap.get(name)}
                    checked={selectedColumns.includes(name)}
                    onToggle={() => toggle(name)}
                  />
                ))}
              </AccordionContent>
            </AccordionItem>
          )}

          {dropped.length > 0 && (
            <AccordionItem value="dropped" className="border-b-0">
              <AccordionTrigger className="py-2 text-sm">
                Filtros automáticos previos <span className="text-muted-foreground">({dropped.length})</span>
              </AccordionTrigger>
              <AccordionContent className="flex flex-col gap-1 pb-2">
                {dropped.map((d) => (
                  <div
                    key={d.column}
                    className="flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-muted-foreground"
                  >
                    <span className="size-4 shrink-0 rounded border border-dashed border-muted-foreground/40" />
                    <span className="min-w-0 flex-1">
                      <span className="truncate font-medium line-through decoration-muted-foreground/40">
                        {d.column}
                      </span>
                      <span className="mt-0.5 block truncate text-xs">{d.reason}</span>
                    </span>
                  </div>
                ))}
              </AccordionContent>
            </AccordionItem>
          )}
        </Accordion>
      </div>

      <Button variant="ghost" size="sm" className="self-start text-xs text-muted-foreground" onClick={runSuggest} disabled={loading}>
        {loading ? <Loader2 className="size-3.5 animate-spin" /> : <Wand2 className="size-3.5" />}
        Re-sugerir
      </Button>
    </div>
  );
}
