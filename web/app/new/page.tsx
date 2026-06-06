"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useWizard } from "@/lib/wizard-store";
import { AppShell } from "@/components/app-shell";
import { WizardProgress } from "@/components/wizard/wizard-progress";
import { StepDatos } from "@/components/wizard/step-datos";
import { StepAnalizar } from "@/components/wizard/step-analizar";

function NewWizard() {
  const params = useSearchParams();
  const wantsSample = params.get("sample") === "1";
  const { step, dataset, setDataset, reset } = useWizard();
  const [autoLoading, setAutoLoading] = useState(wantsSample);
  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    reset();
    if (wantsSample) {
      api
        .loadSample()
        .then(setDataset)
        .catch((e: Error) => toast.error(e.message || "No pudimos cargar el dataset de ejemplo"))
        .finally(() => setAutoLoading(false));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      ) : (
        <StepAnalizar />
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
