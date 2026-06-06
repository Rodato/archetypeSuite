import type { GradeColor } from "./types";

// Categorical color cycle — mirrors api/transform.BRAND_PALETTE exactly.
export const BRAND_PALETTE = [
  "#4f46e5", "#f59e0b", "#10b981", "#ef4444",
  "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
];

export function paletteColor(i: number): string {
  return BRAND_PALETTE[((i % BRAND_PALETTE.length) + BRAND_PALETTE.length) % BRAND_PALETTE.length];
}

export const GRADE_CLASS: Record<GradeColor, string> = {
  green: "grade-green",
  orange: "grade-orange",
  red: "grade-red",
  gray: "grade-gray",
};

export const GRADE_TEXT: Record<GradeColor, string> = {
  green: "text-emerald-600 dark:text-emerald-400",
  orange: "text-amber-600 dark:text-amber-400",
  red: "text-red-600 dark:text-red-400",
  gray: "text-slate-500 dark:text-slate-400",
};

export const IMPORTANCE_META: Record<
  "high" | "medium" | "low",
  { label: string; hint: string; className: string }
> = {
  high: {
    label: "Alta",
    hint: "Clave para diferenciar",
    className: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  },
  medium: {
    label: "Media",
    hint: "Aporta al análisis",
    className: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  },
  low: {
    label: "Baja",
    hint: "Útil con pocas columnas",
    className: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
  },
};

export function formatDate(iso: string): string {
  try {
    return new Intl.DateTimeFormat("es", {
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}
