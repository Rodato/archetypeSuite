"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { CheckCircle2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { COPY } from "@/lib/copy";
import { useWizard } from "@/lib/wizard-store";
import { AppShell } from "@/components/app-shell";
import { WizardProgress } from "@/components/wizard/wizard-progress";
import { StepDatos } from "@/components/wizard/step-datos";
import { StepAnalizar } from "@/components/wizard/step-analizar";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

function CompletedCard({ runId, onRestart }: { runId: string | null; onRestart: () => void }) {
  return (
    <Card className="mx-auto max-w-lg p-8 text-center sm:p-12">
      <CheckCircle2 className="mx-auto mb-4 size-10 text-emerald-500" />
      <h2 className="text-xl font-semibold tracking-tight">{COPY.analysis_done}</h2>
      <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">
        Este análisis ya terminó. Puedes volver a los resultados o empezar uno nuevo.
      </p>
      <div className="mt-6 flex items-center justify-center gap-3">
        <Button variant="outline" onClick={onRestart}>
          Empezar de nuevo
        </Button>
        {runId && (
          <Link href={`/runs/${runId}`} className={buttonVariants()}>
            {COPY.go_to_results}
          </Link>
        )}
      </div>
    </Card>
  );
}

function NewWizard() {
  const params = useSearchParams();
  const paramsKey = params.toString();
  const wantsSample = params.get("sample") === "1";
  const wantsFresh = params.get("fresh") === "1" || wantsSample;
  const { step, dataset, lastRunId, setDataset, reset } = useWizard();
  const [autoLoading, setAutoLoading] = useState(wantsSample);
  const [hydrated, setHydrated] = useState(false);
  const initialized = useRef(false);

  useEffect(() => {
    const loadSample = () => {
      setAutoLoading(true);
      api
        .loadSample()
        .then(setDataset)
        .catch((e: Error) => toast.error(e.message || "No pudimos cargar el dataset de ejemplo"))
        .finally(() => setAutoLoading(false));
    };

    if (!initialized.current) {
      initialized.current = true;
      if (wantsFresh) {
        // ?fresh=1 / ?sample=1 → empezar de cero (reset escribe sessionStorage).
        reset();
        setHydrated(true);
      } else {
        // F5 o back-navigation: restaurar dataset/contexto/selección persistidos.
        void Promise.resolve(useWizard.persist.rehydrate()).finally(() => setHydrated(true));
      }
      if (wantsSample) loadSample();
      return;
    }
    // Navegación in-place (p. ej. header "Nuevo análisis" estando ya en /new).
    if (wantsFresh) reset();
    if (wantsSample) loadSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramsKey]);

  if (!hydrated) {
    return (
      <AppShell>
        <div className="py-24" />
      </AppShell>
    );
  }

  return (
    <AppShell>
      <WizardProgress current={step} />
      {autoLoading && !dataset ? (
        <div className="flex flex-col items-center justify-center gap-3 py-24 text-muted-foreground">
          <Loader2 className="size-6 animate-spin" />
          Cargando dataset de ejemplo…
        </div>
      ) : step === 1 ? (
        <StepDatos autoSample={wantsSample} />
      ) : step === 2 ? (
        <StepAnalizar />
      ) : (
        <CompletedCard runId={lastRunId} onRestart={reset} />
      )}
    </AppShell>
  );
}

export default function NewPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-muted-foreground">Cargando…</div>}>
      <NewWizard />
    </Suspense>
  );
}
