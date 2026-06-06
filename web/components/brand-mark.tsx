import { cn } from "@/lib/utils";

const SIZES = {
  sm: "size-7 rounded-lg text-sm",
  md: "size-9 rounded-xl text-lg",
  lg: "size-14 rounded-2xl text-2xl",
} as const;

export function BrandMark({ size = "md", className }: { size?: keyof typeof SIZES; className?: string }) {
  return (
    <span className={cn("brand-mark", SIZES[size], className)} aria-hidden>
      ◆
    </span>
  );
}

export function BrandWordmark({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <BrandMark size="sm" />
      <span className="text-[15px] tracking-tight text-muted-foreground">
        <strong className="font-bold text-foreground">Archetype</strong> Suite
      </span>
    </div>
  );
}
