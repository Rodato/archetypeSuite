"use client";

import { useState } from "react";
import { motion } from "motion/react";
import {
  Activity,
  AlertCircle,
  ChevronDown,
  Compass,
  ShieldAlert,
  Sparkles,
  Sprout,
  Target,
  Users,
} from "lucide-react";
import type { Archetype, CautionLevel } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";

const CAUTION_META: Record<CautionLevel, { label: string; cls: string }> = {
  baja: { label: "Cautela baja", cls: "grade-green" },
  media: { label: "Cautela media", cls: "grade-orange" },
  alta: { label: "Cautela alta", cls: "grade-red" },
};

export function ArchetypeCard({ archetype, index }: { archetype: Archetype; index: number }) {
  const [open, setOpen] = useState(false);
  const a = archetype;

  const micro = a.microcomportamientos ?? [];
  const barreras = a.barreras ?? [];
  const habilitadores = a.habilitadores ?? [];
  const oportunidades = a.oportunidades_accion ?? [];
  const legacyChars = a.key_characteristics ?? [];
  const legacyDiffs = a.differentiators ?? [];

  const hasBehavioral =
    !!a.comportamiento_principal ||
    micro.length > 0 ||
    barreras.length > 0 ||
    habilitadores.length > 0 ||
    oportunidades.length > 0;
  const hasDetails = hasBehavioral || legacyChars.length > 0 || legacyDiffs.length > 0;

  const caution = a.nivel_cautela ? CAUTION_META[a.nivel_cautela] : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      <Card className="group relative h-full gap-0 overflow-hidden p-5 transition-all hover:shadow-md">
        <span className="absolute inset-x-0 top-0 h-1" style={{ background: a.color }} />

        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span
              className="flex size-8 items-center justify-center rounded-lg text-sm font-bold text-white"
              style={{ background: a.color }}
            >
              {index + 1}
            </span>
            <h3 className="text-base font-semibold leading-tight tracking-tight">{a.label}</h3>
          </div>
          {caution && (
            <span
              className={cn(
                "inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold",
                caution.cls,
              )}
              title={a.cautela_reason || caution.label}
            >
              <ShieldAlert className="size-3" />
              {a.nivel_cautela}
            </span>
          )}
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full bg-accent px-2 py-0.5 text-xs font-medium text-accent-foreground">
            <Users className="size-3" /> {a.size} {a.size === 1 ? "fila" : "filas"}
          </span>
          <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
            {a.prevalence}% del total
          </span>
        </div>

        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{a.description}</p>

        {hasDetails && (
          <>
            <button
              onClick={() => setOpen((v) => !v)}
              className="mt-3 inline-flex items-center gap-1 self-start text-xs font-medium text-primary transition-colors hover:text-primary/80"
            >
              {open ? "Ocultar detalles" : "Ver detalles"}
              <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
            </button>

            <motion.div
              initial={false}
              animate={{ height: open ? "auto" : 0, opacity: open ? 1 : 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden"
            >
              <div className="mt-3 space-y-3 border-t pt-3">
                {hasBehavioral ? (
                  <>
                    {a.comportamiento_principal && (
                      <div>
                        <div className="mb-1 inline-flex items-center gap-1.5 text-xs font-semibold">
                          <Activity className="size-3.5" /> Comportamiento principal
                        </div>
                        <p className="text-sm text-muted-foreground">{a.comportamiento_principal}</p>
                      </div>
                    )}
                    {micro.length > 0 && <Detail icon={<Sparkles className="size-3.5" />} title="Microcomportamientos" items={micro} />}
                    {barreras.length > 0 && <Detail icon={<AlertCircle className="size-3.5" />} title="Barreras probables" items={barreras} />}
                    {habilitadores.length > 0 && <Detail icon={<Sprout className="size-3.5" />} title="Habilitadores" items={habilitadores} />}
                    {oportunidades.length > 0 && <Detail icon={<Compass className="size-3.5" />} title="Oportunidades de acción" items={oportunidades} />}
                  </>
                ) : (
                  <>
                    {legacyChars.length > 0 && <Detail icon={<Sparkles className="size-3.5" />} title="Características clave" items={legacyChars} />}
                    {legacyDiffs.length > 0 && <Detail icon={<Target className="size-3.5" />} title="Qué lo diferencia" items={legacyDiffs} />}
                  </>
                )}
                {a.cautela_reason && (
                  <p className="border-t pt-2 text-xs italic text-muted-foreground">{a.cautela_reason}</p>
                )}
              </div>
            </motion.div>
          </>
        )}
      </Card>
    </motion.div>
  );
}

function Detail({ icon, title, items }: { icon: React.ReactNode; title: string; items: string[] }) {
  return (
    <div>
      <div className="mb-1.5 inline-flex items-center gap-1.5 text-xs font-semibold text-foreground">
        {icon} {title}
      </div>
      <ul className="space-y-1">
        {items.map((it, i) => (
          <li key={i} className="flex gap-2 text-sm text-muted-foreground">
            <span className="mt-1.5 size-1 shrink-0 rounded-full bg-muted-foreground/50" />
            {it}
          </li>
        ))}
      </ul>
    </div>
  );
}
