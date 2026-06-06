import Link from "next/link";
import { Plus } from "lucide-react";
import type { ReactNode } from "react";
import { BackendStatus } from "@/components/backend-status";
import { BrandWordmark } from "@/components/brand-mark";
import { ThemeToggle } from "@/components/theme-toggle";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-40 border-b glass">
        <div className="mx-auto flex h-14 w-full max-w-[1400px] items-center justify-between gap-4 px-4 sm:px-6">
          <Link href="/" className="transition-opacity hover:opacity-80">
            <BrandWordmark />
          </Link>
          <div className="flex items-center gap-2">
            <BackendStatus />
            <ThemeToggle />
            <Link href="/new" className={cn(buttonVariants({ size: "sm" }), "gap-1.5")}>
              <Plus className="size-4" />
              <span className="hidden sm:inline">Nuevo análisis</span>
            </Link>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6 sm:py-8">{children}</main>
    </div>
  );
}
