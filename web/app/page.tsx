"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Database, MessagesSquare, Sparkles, Wand2 } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { BrandMark } from "@/components/brand-mark";
import { RunCard } from "@/components/run-card";
import { buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { COPY } from "@/lib/copy";
import { cn } from "@/lib/utils";

const STEPS = [
  { icon: Database, title: "Trae tus datos", desc: "Sube un CSV o Excel donde cada fila sea una persona o unidad." },
  { icon: Wand2, title: "La IA analiza", desc: "Limpia, agrupa y valida los patrones de comportamiento sin que toques una métrica." },
  { icon: MessagesSquare, title: "Conversa", desc: "Explora arquetipos narrativos, mapas y un chat que entiende tus datos." },
];

export default function HomePage() {
  const { data, isLoading } = useQuery({ queryKey: ["runs"], queryFn: api.listRuns });
  const runs = data?.runs ?? [];

  return (
    <AppShell>
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl border bg-card px-6 py-14 sm:px-12 sm:py-20">
        <div className="grid-bg pointer-events-none absolute inset-0" />
        <div className="relative mx-auto max-w-2xl text-center">
          <BrandMark size="lg" className="mx-auto mb-6 animate-[float_6s_ease-in-out_infinite]" />
          <div className="mb-4 inline-flex items-center gap-1.5 rounded-full border bg-secondary/60 px-3 py-1 text-xs font-medium text-muted-foreground">
            <Sparkles className="size-3.5 text-primary" />
            Clustering agéntico con IA
          </div>
          <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl">
            Descubre los <span className="gradient-text">arquetipos</span> que esconden tus datos
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-balance text-lg text-muted-foreground">
            {COPY.product_tagline}
          </p>
          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link href="/new" className={cn(buttonVariants({ size: "lg" }), "gap-2")}>
              {COPY.nav_new} <ArrowRight className="size-4" />
            </Link>
            <Link href="/new?sample=1" className={buttonVariants({ size: "lg", variant: "outline" })}>
              {COPY.use_sample}
            </Link>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="mt-6 grid gap-4 sm:grid-cols-3">
        {STEPS.map((s, i) => (
          <div key={i} className="rounded-2xl border bg-card p-5">
            <div className="flex size-9 items-center justify-center rounded-xl bg-accent text-accent-foreground">
              <s.icon className="size-[18px]" />
            </div>
            <div className="mt-3 font-semibold tracking-tight">
              <span className="mr-1.5 text-muted-foreground/60">{i + 1}.</span>
              {s.title}
            </div>
            <p className="mt-1 text-sm text-muted-foreground">{s.desc}</p>
          </div>
        ))}
      </section>

      {/* Mis análisis */}
      <section className="mt-10">
        <div className="mb-4 flex items-end justify-between">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">{COPY.nav_history}</h2>
            <p className="text-sm text-muted-foreground">Tus análisis quedan guardados localmente.</p>
          </div>
          {runs.length > 0 && <span className="text-sm text-muted-foreground">{runs.length} en total</span>}
        </div>

        {isLoading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-44 rounded-2xl" />
            ))}
          </div>
        ) : runs.length === 0 ? (
          <div className="flex flex-col items-center rounded-2xl border border-dashed py-16 text-center">
            <BrandMark size="md" className="mb-4 opacity-70" />
            <div className="font-semibold">{COPY.empty_history_title}</div>
            <p className="mt-1 max-w-xs text-sm text-muted-foreground">{COPY.empty_history_desc}</p>
            <Link href="/new?sample=1" className={cn(buttonVariants(), "mt-5 gap-2")}>
              {COPY.use_sample} <ArrowRight className="size-4" />
            </Link>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {runs.map((run) => (
              <RunCard key={run.id} run={run} />
            ))}
          </div>
        )}
      </section>
    </AppShell>
  );
}
