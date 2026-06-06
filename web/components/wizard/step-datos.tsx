"use client";

import { ArrowRight, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { COPY } from "@/lib/copy";
import { useWizard } from "@/lib/wizard-store";
import { BrandMark } from "@/components/brand-mark";
import { DataChat } from "@/components/chat/data-chat";
import { TypeDonut } from "@/components/charts/type-donut";
import { DataTable } from "@/components/data-table";
import { ColumnSelector } from "@/components/wizard/column-selector";
import { UploadZone } from "@/components/wizard/upload-zone";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{children}</div>
  );
}

export function StepDatos({ autoSample }: { autoSample?: boolean }) {
  const { dataset, context, setContext, setDataset } = useWizard();

  if (!dataset) {
    return (
      <Card className="relative overflow-hidden border-primary/10 bg-gradient-to-br from-accent/40 to-card p-8 sm:p-12">
        <div className="mx-auto max-w-lg text-center">
          <BrandMark size="lg" className="mx-auto mb-5" />
          <h2 className="text-2xl font-bold tracking-tight">{COPY.product_tagline}</h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
            Sube un CSV o Excel donde cada fila sea una persona, cliente o unidad de análisis.
          </p>
          <div className="mx-auto mt-7 max-w-md">
            <UploadZone autoSample={autoSample} />
          </div>
        </div>
      </Card>
    );
  }

  const profile = dataset.profile;
  const firstCat = profile.categorical_columns[0];
  const firstNum = profile.numeric_columns[0];
  const suggestions = [
    "¿Hay valores faltantes?",
    firstCat ? `¿Cuántos hay por ${firstCat}?` : null,
    firstNum && firstCat ? `${firstNum} promedio por ${firstCat}` : firstNum ? `Distribución de ${firstNum}` : null,
  ].filter(Boolean) as string[];

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[1.4fr_1.8fr_2fr]">
        <Card className="p-5">
          <div className="flex items-center justify-between">
            <Eyebrow>Tipos de variables</Eyebrow>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 gap-1 text-xs text-muted-foreground"
              onClick={() => setDataset(null)}
            >
              <RefreshCw className="size-3" /> Cambiar
            </Button>
          </div>
          <TypeDonut donut={dataset.donut} />
        </Card>

        <Card className="p-5">
          <Eyebrow>Contexto</Eyebrow>
          <div className="mb-2 text-sm font-semibold">¿Qué representa este dataset?</div>
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="Ejemplo: Clientes de una tienda de ropa online; queremos entender patrones de compra…"
            className="scroll-slim h-[150px] w-full resize-none rounded-lg border bg-background p-3 text-sm outline-none transition-colors focus:border-primary focus:ring-2 focus:ring-primary/15"
          />
          <p className="mt-2 text-xs text-muted-foreground">
            Ayuda a la IA a nombrar y describir mejor los arquetipos.
          </p>
        </Card>

        <Card className="p-5">
          <Eyebrow>Variables a usar</Eyebrow>
          <ColumnSelector />
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <Card className="flex flex-col p-5">
          <Eyebrow>Vista previa · {dataset.n_rows} filas</Eyebrow>
          <DataTable table={dataset.preview} />
        </Card>

        <Card className="flex h-[420px] flex-col p-5">
          <Eyebrow>Pregunta sobre tus datos</Eyebrow>
          <DataChat
            ask={(q, history) => api.chatDataset(dataset.dataset_id, { question: q, context, history })}
            suggestions={suggestions}
          />
        </Card>
      </div>

      <div className="flex justify-end">
        <ContinueButton />
      </div>
    </div>
  );
}

function ContinueButton() {
  const setStep = useWizard((s) => s.setStep);
  return (
    <Button size="lg" className="gap-2" onClick={() => setStep(2)}>
      Continuar <ArrowRight className="size-4" />
    </Button>
  );
}
