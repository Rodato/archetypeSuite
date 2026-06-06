"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "motion/react";
import { AlertTriangle, ArrowLeft, Play, Sparkles } from "lucide-react";
import { streamAnalyze } from "@/lib/api";
import { COPY } from "@/lib/copy";
import type { ProgressStep } from "@/lib/types";
import { useWizard } from "@/lib/wizard-store";
import { PipelineProgress } from "@/components/wizard/pipeline-progress";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const INITIAL_STEPS: ProgressStep[] = [
  { key: "ingest", label: "Cargar datos", status: "pending" },
  { key: "profile", label: "Entender el dataset", status: "pending" },
  { key: "column_selection", label: "Filtrar columnas", status: "pending" },
  { key: "preprocess", label: "Limpiar y normalizar", status: "pending" },
  { key: "optimize_k", label: "Buscar k óptimo", status: "pending" },
  { key: "cluster", label: "Encontrar patrones", status: "pending" },
  { key: "interpret", label: "Generar arquetipos", status: "pending" },
  { key: "refinement", label: "Refinar narrativa", status: "pending" },
];

type Status = "idle" | "running" | "error";

export function StepAnalizar() {
  const router = useRouter();
  const { dataset, context, suggestion, selectedColumns, setStep } = useWizard();
  const [status, setStatus] = useState<Status>("idle");
  const [steps, setSteps] = useState<ProgressStep[]>(INITIAL_STEPS);
  const [message, setMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);

  // Tear down any in-flight stream when the component unmounts (redirect / nav away).
  useEffect(() => () => abortRef.current?.(), []);

  if (!dataset) return null;

  const run = () => {
    abortRef.current?.(); // close any previous stream before starting a new one (retry/volver)
    setStatus("running");
    setError(null);
    setSteps(INITIAL_STEPS);
    setMessage("Iniciando…");

    const fastPath = suggestion && selectedColumns.length > 0;
    abortRef.current = streamAnalyze(
      dataset.dataset_id,
      {
        context,
        selected_columns: fastPath ? selectedColumns : null,
        static_filter_result: fastPath ? suggestion.static_filter_result : undefined,
        column_recommendation: fastPath ? suggestion.column_recommendation : undefined,
      },
      (ev) => {
        if (ev.type === "progress") {
          setSteps(ev.steps);
          setMessage(ev.message);
        } else if (ev.type === "done") {
          router.push(`/runs/${ev.run_id}`);
        } else if (ev.type === "error") {
          setStatus("error");
          setError(ev.message);
          setSteps((prev) =>
            prev.map((s) => (s.key === ev.failed_step ? { ...s, status: "failed" } : s)),
          );
        }
      },
      (err) => {
        if (err) {
          setStatus("error");
          setError(err.message);
        }
      },
    );
  };

  if (status === "idle") {
    return (
      <Card className="relative overflow-hidden p-8 sm:p-12">
        <div className="grid-bg pointer-events-none absolute inset-0" />
        <div className="relative mx-auto max-w-lg text-center">
          <div className="mx-auto mb-5 flex size-12 items-center justify-center rounded-2xl bg-accent text-primary">
            <Sparkles className="size-6" />
          </div>
          <h2 className="text-2xl font-bold tracking-tight">Generar arquetipos</h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
            La IA detecta cuántos grupos hay en tu dataset, los caracteriza y les asigna un nombre basado en
            patrones de comportamiento. Suele tomar 1–2 minutos.
          </p>
          {context && (
            <div className="mx-auto mt-4 max-w-md rounded-lg border bg-muted/40 px-3 py-2 text-left text-xs text-muted-foreground">
              <span className="font-medium text-foreground">Contexto:</span> {context}
            </div>
          )}
          <div className="mt-7 flex items-center justify-center gap-3">
            <Button variant="outline" size="lg" className="gap-2" onClick={() => setStep(1)}>
              <ArrowLeft className="size-4" /> Atrás
            </Button>
            <Button size="lg" className="gap-2" onClick={run}>
              <Play className="size-4" /> {COPY.analyze_button}
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="mx-auto max-w-xl p-6 sm:p-8">
      {status === "error" ? (
        <div className="text-center">
          <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-2xl bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400">
            <AlertTriangle className="size-6" />
          </div>
          <h3 className="text-lg font-semibold">{COPY.error_pipeline_interrupted}</h3>
          <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">{error}</p>
          <div className="mt-5 flex justify-center gap-3">
            <Button
              variant="outline"
              onClick={() => {
                abortRef.current?.();
                setStatus("idle");
              }}
            >
              Volver
            </Button>
            <Button onClick={run}>{COPY.retry}</Button>
          </div>
        </div>
      ) : (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <PipelineProgress steps={steps} message={message} />
        </motion.div>
      )}
    </Card>
  );
}
