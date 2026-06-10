"use client";

import Link from "next/link";
import { AlertTriangle } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="flex size-12 items-center justify-center rounded-2xl bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400">
        <AlertTriangle className="size-6" />
      </div>
      <h1 className="text-xl font-semibold tracking-tight">Algo no salió bien</h1>
      <p className="max-w-sm text-sm text-muted-foreground">
        Ocurrió un error inesperado al mostrar esta pantalla. Puedes reintentar o volver al inicio.
      </p>
      {error.digest && (
        <p className="text-xs text-muted-foreground/60">Código: {error.digest}</p>
      )}
      <div className="flex items-center gap-3">
        <Button variant="outline" onClick={reset}>
          Reintentar
        </Button>
        <Link href="/" className={buttonVariants()}>
          Volver al inicio
        </Link>
      </div>
    </div>
  );
}
