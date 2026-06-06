"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { FileUp, Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { COPY } from "@/lib/copy";
import { useWizard } from "@/lib/wizard-store";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export function UploadZone({ autoSample = false }: { autoSample?: boolean }) {
  const setDataset = useWizard((s) => s.setDataset);
  const [busy, setBusy] = useState<"upload" | "sample" | null>(null);

  const handleUpload = useCallback(
    async (file: File) => {
      setBusy("upload");
      try {
        const ds = await api.uploadDataset(file);
        setDataset(ds);
        toast.success(`${COPY.file_loaded}: ${ds.file_name}`);
      } catch (e) {
        toast.error((e as Error).message || COPY.error_load_file);
      } finally {
        setBusy(null);
      }
    },
    [setDataset],
  );

  const loadSample = useCallback(async () => {
    setBusy("sample");
    try {
      const ds = await api.loadSample();
      setDataset(ds);
      toast.success("Cargamos el dataset de ejemplo");
    } catch (e) {
      toast.error((e as Error).message || COPY.error_load_file);
    } finally {
      setBusy(null);
    }
  }, [setDataset]);

  const onDrop = useCallback(
    (files: File[]) => {
      if (files[0]) handleUpload(files[0]);
    },
    [handleUpload],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
    },
    maxFiles: 1,
    disabled: busy !== null,
  });

  return (
    <div className="flex flex-col items-center">
      <div
        {...getRootProps()}
        className={cn(
          "flex w-full cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-all",
          isDragActive ? "border-primary bg-accent/60" : "border-border hover:border-primary/50 hover:bg-accent/30",
          busy && "pointer-events-none opacity-70",
        )}
      >
        <input {...getInputProps()} />
        <div className="mb-3 flex size-12 items-center justify-center rounded-2xl bg-accent text-primary">
          {busy === "upload" ? <Loader2 className="size-6 animate-spin" /> : <FileUp className="size-6" />}
        </div>
        <div className="font-medium">{COPY.upload_cta}</div>
        <div className="mt-1 text-xs text-muted-foreground">{COPY.upload_hint}</div>
      </div>

      <div className="my-4 flex w-full items-center gap-3 text-xs text-muted-foreground">
        <span className="h-px flex-1 bg-border" />
        o
        <span className="h-px flex-1 bg-border" />
      </div>

      <Button variant="outline" className="gap-2" onClick={loadSample} disabled={busy !== null}>
        {busy === "sample" ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4 text-primary" />}
        {COPY.use_sample}
      </Button>
      {autoSample && busy === "sample" && (
        <span className="mt-2 text-xs text-muted-foreground">{COPY.loading}</span>
      )}
    </div>
  );
}
