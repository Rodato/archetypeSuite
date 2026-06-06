"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

export function BackendStatus() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 20_000,
  });

  const state = isLoading
    ? { dot: "bg-amber-400 animate-pulse", text: "Conectando…" }
    : isError
      ? { dot: "bg-red-500", text: "Backend desconectado" }
      : data?.has_api_key
        ? { dot: "bg-emerald-500", text: "Backend listo" }
        : { dot: "bg-amber-500", text: "Sin API key" };

  return (
    <div
      className="hidden items-center gap-1.5 rounded-full border bg-card px-2.5 py-1 text-xs text-muted-foreground sm:flex"
      title={state.text}
    >
      <span className={cn("size-1.5 rounded-full", state.dot)} />
      {state.text}
    </div>
  );
}
