"use client";

import { useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  BarChart3,
  Download,
  FileText,
  HelpCircle,
  Info,
  MessagesSquare,
  Sparkles,
  Table2,
  Trash2,
  UserSearch,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { Archetype, GroupProfile } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { formatDate } from "@/lib/palette";
import { AppShell } from "@/components/app-shell";
import { WizardProgress } from "@/components/wizard/wizard-progress";
import { ArchetypeCard } from "@/components/results/archetype-card";
import { QualityHero } from "@/components/results/quality-hero";
import { ClusterBar } from "@/components/charts/cluster-bar";
import { ScatterMap } from "@/components/charts/scatter-map";
import { RadarCompare } from "@/components/charts/radar";
import { BoxPlot } from "@/components/charts/box-plot";
import { SilhouetteCurve } from "@/components/charts/silhouette-curve";
import { DataChat } from "@/components/chat/data-chat";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

function Eyebrow({ children }: { children: React.ReactNode }) {
  return <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{children}</div>;
}

export default function RunPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const qc = useQueryClient();
  const { data: run, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["run", id],
    queryFn: () => api.getRun(id),
  });

  const del = useMutation({
    mutationFn: () => api.deleteRun(id),
    onSuccess: () => {
      toast.success("Análisis eliminado");
      router.push("/");
      qc.invalidateQueries({ queryKey: ["runs"] });
      // Sin esto, volver con el botón "atrás" dentro del staleTime muestra el run borrado.
      qc.removeQueries({ queryKey: ["run", id] });
    },
    onError: () => {
      toast.error("No se pudo eliminar el análisis. Inténtalo de nuevo.");
    },
  });

  const boxCols = run ? Object.keys(run.charts.box) : [];
  const [boxCol, setBoxCol] = useState<string | null>(null);
  const activeBoxCol = boxCol ?? boxCols[0] ?? null;

  // Perfilado a demanda
  const [profileDesc, setProfileDesc] = useState("");
  const prof = useMutation({
    mutationFn: () => api.profileGroup(id, profileDesc.trim()),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["run", id] });
      setProfileDesc("");
      toast.success("Perfil creado");
    },
    onError: (e: Error) => toast.error(e.message || "No pudimos perfilar ese grupo."),
  });
  const delProf = useMutation({
    mutationFn: (profileId: string) => api.deleteProfile(id, profileId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["run", id] }),
    onError: () => toast.error("No se pudo eliminar el perfil."),
  });

  if (isLoading) {
    return (
      <AppShell>
        <div className="space-y-4">
          <Skeleton className="h-28 rounded-2xl" />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-44 rounded-2xl" />
            ))}
          </div>
        </div>
      </AppShell>
    );
  }

  if (isError || !run) {
    // El 404 del backend trae "Análisis no encontrado."; cualquier otra cosa es
    // un problema de conexión — son mensajes (y acciones) distintos.
    const notFound = !isError || ((error as Error)?.message ?? "").includes("no encontrado");
    return (
      <AppShell>
        <div className="flex flex-col items-center gap-3 py-24 text-center">
          <div className="font-semibold">
            {notFound ? "No encontramos este análisis." : "No pudimos conectar con el servidor."}
          </div>
          {!notFound && (
            <p className="max-w-sm text-sm text-muted-foreground">
              El análisis sigue guardado. Revisa que el backend esté activo y reintenta.
            </p>
          )}
          <div className="flex items-center gap-3">
            {!notFound && (
              <Button variant="outline" onClick={() => refetch()}>
                Reintentar
              </Button>
            )}
            <Link href="/" className={buttonVariants()}>
              Volver al inicio
            </Link>
          </div>
        </div>
      </AppShell>
    );
  }

  const chatSuggestions = (() => {
    const out: string[] = [];
    if (run.archetypes.length >= 2) out.push(`Diferencias clave entre ${run.archetypes[0].label} y ${run.archetypes[1].label}`);
    if (run.columns.numeric[0]) out.push(`¿Qué arquetipo tiene mayor ${run.columns.numeric[0]} promedio?`);
    if (run.columns.categorical[0]) out.push(`Distribución de ${run.columns.categorical[0]} por arquetipo`);
    return out;
  })();

  return (
    <AppShell>
      <WizardProgress current={3} />

      {/* Header */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Link href="/" aria-label="Volver" className={buttonVariants({ variant: "ghost", size: "icon" })}>
            <ArrowLeft className="size-4" />
          </Link>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">{run.file_name}</h1>
            <p className="text-xs text-muted-foreground">
              {formatDate(run.created_at)} · {run.n_rows} filas · {run.archetypes.length} arquetipos
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Popover>
            <PopoverTrigger className={cn(buttonVariants({ variant: "outline", size: "sm" }), "gap-1.5")}>
              <Download className="size-4" /> Descargar
            </PopoverTrigger>
            <PopoverContent align="end" className="w-60">
              <div className="grid gap-1">
                <DownloadLink href={api.exportUrl(run.id, "archetypes.csv")} icon={<Table2 className="size-4" />} label="Arquetipos (CSV)" />
                <DownloadLink href={api.exportUrl(run.id, "labeled.csv")} icon={<Table2 className="size-4" />} label="Datos etiquetados (CSV)" />
                <DownloadLink href={api.exportUrl(run.id, "report.md")} icon={<FileText className="size-4" />} label="Reporte (Markdown)" />
              </div>
            </PopoverContent>
          </Popover>
          <Dialog>
            <DialogTrigger
              aria-label="Eliminar"
              className={cn(buttonVariants({ variant: "ghost", size: "icon" }))}
            >
              <Trash2 className="size-4 text-muted-foreground" />
            </DialogTrigger>
            <DialogContent showCloseButton={false}>
              <DialogHeader>
                <DialogTitle>¿Eliminar este análisis?</DialogTitle>
                <DialogDescription>
                  Se borrará «{run.file_name}» de Mis análisis. Esta acción no se puede deshacer.
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <DialogClose render={<Button variant="outline" />}>Cancelar</DialogClose>
                <Button variant="destructive" onClick={() => del.mutate()} disabled={del.isPending}>
                  {del.isPending ? "Eliminando…" : "Eliminar"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Quality + sizes */}
      <Card className="p-5">
        <div className="grid gap-6 lg:grid-cols-[2fr_3fr]">
          <div className="flex flex-col justify-center">
            <QualityHero quality={run.quality} />
          </div>
          <div>
            <Eyebrow>Tamaño de cada arquetipo</Eyebrow>
            <ClusterBar sizes={run.cluster_sizes} />
          </div>
        </div>
      </Card>

      {/* Archetype cards */}
      <section className="mt-6">
        <div className="mb-3 flex items-center gap-2">
          <h2 className="text-lg font-semibold tracking-tight">Arquetipos</h2>
          <Popover>
            <PopoverTrigger
              className="text-muted-foreground transition-colors hover:text-foreground"
              aria-label="Qué es un arquetipo"
            >
              <HelpCircle className="size-4" />
            </PopoverTrigger>
            <PopoverContent className="w-80 text-sm text-muted-foreground">
              Un <strong className="text-foreground">arquetipo</strong> es una{" "}
              <strong className="text-foreground">hipótesis comportamental</strong>: un patrón de conductas, tensiones y
              barreras que aparece en un grupo — no un retrato de persona. Sirve para leer comportamientos y abrir mejores
              preguntas de diseño, no para etiquetar a nadie.
            </PopoverContent>
          </Popover>
        </div>
        {run.summary && (
          <p className="mb-4 max-w-3xl text-sm leading-relaxed text-muted-foreground">{run.summary}</p>
        )}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {run.archetypes.map((a, i) => (
            <ArchetypeCard key={a.cluster_id} archetype={a} index={i} runId={run.id} />
          ))}
        </div>
      </section>

      {/* Perfilado a demanda */}
      <section className="mt-6">
        <Card className="p-5">
          <div className="mb-1 flex items-center gap-2">
            <UserSearch className="size-4 text-primary" />
            <h2 className="text-base font-semibold tracking-tight">Perfilar un grupo</h2>
          </div>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Describe un grupo en tus palabras y la IA construye su hipótesis comportamental con la
            misma metodología — útil aunque el grupo no coincida con ningún arquetipo.
          </p>
          <form
            className="mt-3 flex flex-col gap-2 sm:flex-row"
            onSubmit={(e) => {
              e.preventDefault();
              if (profileDesc.trim().length >= 3 && !prof.isPending) prof.mutate();
            }}
          >
            <Input
              value={profileDesc}
              onChange={(e) => setProfileDesc(e.target.value)}
              placeholder='Ej.: "quienes usan redes de madrugada y nunca toman pausas"'
              maxLength={500}
              disabled={prof.isPending}
              className="flex-1"
            />
            <Button type="submit" disabled={prof.isPending || profileDesc.trim().length < 3}>
              {prof.isPending ? "Perfilando…" : "Perfilar"}
            </Button>
          </form>
          {prof.isPending && (
            <p className="mt-2 text-xs text-muted-foreground">
              Traduciendo el grupo a filtros y construyendo la hipótesis… suele tomar ~20 s.
            </p>
          )}
          {(run.custom_profiles ?? []).length > 0 && (
            <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {(run.custom_profiles ?? []).map((p, i) => (
                <div key={p.id} className="relative">
                  <ArchetypeCard archetype={profileToArchetype(p, i)} index={i} />
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`Eliminar perfil ${p.label}`}
                    className="absolute -right-2 -top-2 rounded-full border bg-background shadow-sm"
                    onClick={() => delProf.mutate(p.id)}
                    disabled={delProf.isPending}
                  >
                    <X className="size-3.5" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </Card>
      </section>

      {/* Explore */}
      <section className="mt-6 grid gap-4 lg:grid-cols-[3fr_1.1fr]">
        <Card className="p-5">
          <Eyebrow>Explorar</Eyebrow>
          <Tabs defaultValue="map">
            <TabsList className="mb-4">
              <TabsTrigger value="map" className="gap-1.5">
                <Sparkles className="size-3.5" /> Mapa
              </TabsTrigger>
              <TabsTrigger value="compare" className="gap-1.5">
                <BarChart3 className="size-3.5" /> Comparar
              </TabsTrigger>
              <TabsTrigger value="variable" className="gap-1.5">
                <Table2 className="size-3.5" /> Por variable
              </TabsTrigger>
              <TabsTrigger value="chat" className="gap-1.5">
                <MessagesSquare className="size-3.5" /> Conversar
              </TabsTrigger>
            </TabsList>

            {/* keepMounted: Base UI desmonta los paneles inactivos por defecto — sin esto,
                cambiar de tab borra el historial del chat y re-anima los charts. */}
            <TabsContent value="map" keepMounted>
              <p className="mb-2 text-xs text-muted-foreground">
                Cada punto es una fila de tu dataset, coloreada por arquetipo. Las filas parecidas aparecen cerca.
              </p>
              <ScatterMap points={run.charts.scatter} />
            </TabsContent>

            <TabsContent value="compare" keepMounted>
              <p className="mb-2 text-xs text-muted-foreground">
                Compara los arquetipos en varias variables a la vez. Valores normalizados de 0 a 1 para poder compararse.
              </p>
              <RadarCompare radar={run.charts.radar} />
            </TabsContent>

            <TabsContent value="variable" keepMounted>
              <div className="mb-3 flex items-center justify-between gap-3">
                <p className="text-xs text-muted-foreground">Cómo se distribuye una variable entre los arquetipos.</p>
                {boxCols.length > 0 && (
                  <Select value={activeBoxCol ?? undefined} onValueChange={setBoxCol}>
                    <SelectTrigger className="w-44" size="sm">
                      <SelectValue placeholder="Variable" />
                    </SelectTrigger>
                    <SelectContent>
                      {boxCols.map((c) => (
                        <SelectItem key={c} value={c}>
                          {c}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
              {activeBoxCol ? (
                <BoxPlot groups={run.charts.box[activeBoxCol]} />
              ) : (
                <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
                  No hay variables numéricas para mostrar.
                </div>
              )}
            </TabsContent>

            <TabsContent value="chat" keepMounted>
              <div className="flex h-[440px] flex-col">
                <DataChat
                  ask={(q, history, onTool) =>
                    api.chatRunStream(run.id, { question: q, context: run.dataset_context, history }, onTool)
                  }
                  suggestions={chatSuggestions}
                  placeholder="Pregunta sobre los arquetipos…"
                />
              </div>
            </TabsContent>
          </Tabs>
        </Card>

        {/* Methodology */}
        <Card className="h-fit p-5">
          <Eyebrow>Metodología</Eyebrow>
          <div className="text-sm font-semibold">¿Por qué {run.optimal_k} arquetipos?</div>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            {run.k_analysis.flat_k_curve ? (
              <>
                Probamos varios números de grupos y <strong className="text-foreground">ninguno separa claramente
                mejor que otro</strong> — tus datos no marcan una división nítida. En esos casos usamos la
                segmentación más simple (<strong className="text-foreground">{run.optimal_k}</strong> grupos) y
                tratamos los arquetipos con cautela extra.
              </>
            ) : (
              <>
                Probamos dividir tus datos en distintos números de grupos y medimos qué tan bien separados quedan.{" "}
                <strong className="text-foreground">{run.optimal_k}</strong> es el punto donde cada grupo es
                distintivo y suficientemente grande para nombrarlo.
              </>
            )}
          </p>
          <div className="mt-3">
            <SilhouetteCurve
              kRange={run.k_analysis.k_range}
              scores={run.k_analysis.silhouette_scores}
              optimalK={run.k_analysis.optimal_k}
            />
          </div>
        </Card>
      </section>

      {/* Advanced */}
      <section className="mt-6">
        <Accordion>
          <AccordionItem value="adv" className="rounded-2xl border bg-card px-5">
            <AccordionTrigger className="text-sm">
              <span className="inline-flex items-center gap-2">
                <Info className="size-4 text-muted-foreground" /> Detalles técnicos
              </span>
            </AccordionTrigger>
            <AccordionContent>
              <div className="grid gap-4 pb-2 sm:grid-cols-2">
                <div>
                  <div className="mb-2 text-xs font-semibold">Métricas</div>
                  <dl className="space-y-1 text-sm">
                    <Metric label="Silhouette" value={run.metrics.silhouette_score} digits={3} />
                    <Metric label="Calinski–Harabasz" value={run.metrics.calinski_harabasz_score} digits={1} />
                    <Metric label="Davies–Bouldin" value={run.metrics.davies_bouldin_score} digits={3} />
                    <Metric label="Algoritmo" value={run.advanced.selected_algorithm} />
                    <Metric label="Iteraciones de refinamiento" value={run.advanced.refinement_count} />
                  </dl>
                </div>
                <div>
                  {/* El algoritmo es KMeans fijo por diseño (determinismo) — no hay "selección"
                      que explicar; la decisión real del LLM es el refinamiento. */}
                  {run.advanced.refinement_reason && (
                    <>
                      <div className="mb-2 text-xs font-semibold">Decisión de refinamiento</div>
                      <p className="text-sm text-muted-foreground">{run.advanced.refinement_reason}</p>
                    </>
                  )}
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </section>
    </AppShell>
  );
}

/** Adapta un perfil a demanda a la forma que espera ArchetypeCard (misma visual, sin edición). */
function profileToArchetype(p: GroupProfile, i: number): Archetype {
  return {
    cluster_id: -(i + 1),
    label: p.label,
    description: p.description,
    comportamiento_principal: p.comportamiento_principal,
    microcomportamientos: p.microcomportamientos,
    barreras: p.barreras,
    habilitadores: p.habilitadores,
    oportunidades_accion: p.oportunidades_accion,
    nivel_cautela: p.nivel_cautela,
    cautela_reason: p.cautela_reason,
    size: p.n,
    prevalence: p.share,
    color: "#8B5CF6",
  };
}

function DownloadLink({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
  return (
    <a
      href={href}
      className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition-colors hover:bg-muted"
    >
      {icon} {label}
    </a>
  );
}

function Metric({ label, value, digits }: { label: string; value: number | string | null; digits?: number }) {
  const display =
    value === null || value === undefined
      ? "—"
      : typeof value === "number" && digits !== undefined
        ? value.toFixed(digits)
        : String(value);
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium tabular-nums">{display}</dd>
    </div>
  );
}
