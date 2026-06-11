"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import {
  Activity,
  AlertCircle,
  BadgeCheck,
  ChevronDown,
  Compass,
  Pencil,
  ShieldAlert,
  Sparkles,
  Sprout,
  Target,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { Archetype, ArchetypePatch, CautionLevel } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const CAUTION_META: Record<CautionLevel, { label: string; cls: string }> = {
  baja: { label: "Cautela baja", cls: "grade-green" },
  media: { label: "Cautela media", cls: "grade-orange" },
  alta: { label: "Cautela alta", cls: "grade-red" },
};

export function ArchetypeCard({
  archetype,
  index,
  runId,
}: {
  archetype: Archetype;
  index: number;
  runId?: string;
}) {
  const [open, setOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
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
          <div className="flex shrink-0 items-center gap-1">
            {a.validated && (
              <span
                className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold grade-green"
                title={a.curated_at ? `Validado por el equipo · ${a.curated_at.slice(0, 10)}` : "Validado por el equipo"}
              >
                <BadgeCheck className="size-3" /> validado
              </span>
            )}
            {caution && (
              <span
                className={cn(
                  "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold",
                  caution.cls,
                )}
                title={a.cautela_reason || caution.label}
              >
                <ShieldAlert className="size-3" />
                {a.nivel_cautela}
              </span>
            )}
            {runId && (
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Editar ${a.label}`}
                className="text-muted-foreground hover:text-foreground"
                onClick={() => setEditOpen(true)}
              >
                <Pencil className="size-3.5" />
              </Button>
            )}
          </div>
        </div>

        {runId && (
          <EditArchetypeDialog
            runId={runId}
            archetype={a}
            open={editOpen}
            onOpenChange={setEditOpen}
          />
        )}

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

function linesToList(s: string): string[] {
  return s
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

function seedForm(a: Archetype) {
  return {
    label: a.label,
    description: a.description ?? "",
    comportamiento: a.comportamiento_principal ?? "",
    micro: (a.microcomportamientos ?? []).join("\n"),
    barreras: (a.barreras ?? []).join("\n"),
    habilitadores: (a.habilitadores ?? []).join("\n"),
    oportunidades: (a.oportunidades_accion ?? []).join("\n"),
    cautelaReason: a.cautela_reason ?? "",
    validated: a.validated ?? false,
  };
}

function EditArchetypeDialog({
  runId,
  archetype,
  open,
  onOpenChange,
}: {
  runId: string;
  archetype: Archetype;
  open: boolean;
  onOpenChange: (o: boolean) => void;
}) {
  const qc = useQueryClient();
  const a = archetype;
  const [form, setForm] = useState(() => seedForm(a));

  const handleOpenChange = (next: boolean) => {
    if (next) setForm(seedForm(a)); // re-sembrar con la versión vigente al abrir
    onOpenChange(next);
  };

  const save = useMutation({
    mutationFn: () =>
      api.updateArchetype(runId, a.cluster_id, {
        label: form.label.trim() || a.label,
        description: form.description.trim(),
        comportamiento_principal: form.comportamiento.trim(),
        microcomportamientos: linesToList(form.micro),
        barreras: linesToList(form.barreras),
        habilitadores: linesToList(form.habilitadores),
        oportunidades_accion: linesToList(form.oportunidades),
        cautela_reason: form.cautelaReason.trim(),
        validated: form.validated,
      } satisfies ArchetypePatch),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["run", runId] });
      toast.success("Arquetipo actualizado");
      onOpenChange(false);
    },
    onError: () => toast.error("No se pudo guardar el arquetipo. Inténtalo de nuevo."),
  });

  const set = (key: keyof ReturnType<typeof seedForm>) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => setForm((f) => ({ ...f, [key]: e.target.value }));

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent showCloseButton={false} className="max-h-[85dvh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Editar arquetipo</DialogTitle>
          <DialogDescription>
            El arquetipo es una hipótesis propuesta por la IA — afínala con lo que tu equipo sabe.
            El nivel de cautela no es editable: lo fija la calidad estadística del agrupamiento.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4">
          <Field label="Nombre">
            <Input value={form.label} onChange={set("label")} maxLength={120} />
          </Field>
          <Field label="Descripción del patrón">
            <Textarea value={form.description} onChange={set("description")} rows={3} />
          </Field>
          <Field label="Comportamiento principal">
            <Textarea value={form.comportamiento} onChange={set("comportamiento")} rows={2} />
          </Field>
          <Field label="Microcomportamientos" hint="Uno por línea.">
            <Textarea value={form.micro} onChange={set("micro")} rows={3} />
          </Field>
          <Field label="Barreras probables" hint="Una por línea.">
            <Textarea value={form.barreras} onChange={set("barreras")} rows={3} />
          </Field>
          <Field label="Habilitadores" hint="Uno por línea.">
            <Textarea value={form.habilitadores} onChange={set("habilitadores")} rows={3} />
          </Field>
          <Field label="Oportunidades de acción" hint="Una por línea.">
            <Textarea value={form.oportunidades} onChange={set("oportunidades")} rows={3} />
          </Field>
          <Field label="Nota de cautela">
            <Textarea value={form.cautelaReason} onChange={set("cautelaReason")} rows={2} />
          </Field>

          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={form.validated}
              onChange={(e) => setForm((f) => ({ ...f, validated: e.target.checked }))}
              className="size-4 accent-[var(--primary)]"
            />
            <span className="inline-flex items-center gap-1 font-medium">
              <BadgeCheck className="size-4 text-emerald-500" /> Validado por el equipo
            </span>
          </label>
        </div>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>Cancelar</DialogClose>
          <Button onClick={() => save.mutate()} disabled={save.isPending}>
            {save.isPending ? "Guardando…" : "Guardar cambios"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1.5">
      <Label className="text-xs font-semibold">
        {label}
        {hint && <span className="ml-1 font-normal text-muted-foreground">({hint})</span>}
      </Label>
      {children}
    </div>
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
