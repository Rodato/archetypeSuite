import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="text-5xl font-bold tracking-tight text-muted-foreground/40">404</div>
      <h1 className="text-xl font-semibold tracking-tight">Esta página no existe</h1>
      <p className="max-w-sm text-sm text-muted-foreground">
        El enlace puede estar roto o la página fue movida.
      </p>
      <Link href="/" className={buttonVariants()}>
        Volver al inicio
      </Link>
    </div>
  );
}
