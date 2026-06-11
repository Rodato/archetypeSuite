// Typed client for the Archetype Suite FastAPI backend.
import type {
  AnalyzeEvent,
  Archetype,
  ArchetypePatch,
  DatasetResponse,
  QAResult,
  RunRecord,
  RunSummary,
  SuggestResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Error ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || body.message || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async health(): Promise<{ status: string; has_api_key: boolean }> {
    return jsonOrThrow(await fetch(`${API_BASE}/api/health`, { cache: "no-store" }));
  },

  async uploadDataset(file: File): Promise<DatasetResponse> {
    const form = new FormData();
    form.append("file", file);
    return jsonOrThrow(await fetch(`${API_BASE}/api/datasets/upload`, { method: "POST", body: form }));
  },

  async loadSample(): Promise<DatasetResponse> {
    return jsonOrThrow(await fetch(`${API_BASE}/api/datasets/sample`, { method: "POST" }));
  },

  async getDataset(id: string): Promise<DatasetResponse> {
    return jsonOrThrow(await fetch(`${API_BASE}/api/datasets/${id}`, { cache: "no-store" }));
  },

  async suggestColumns(id: string, context: string): Promise<SuggestResponse> {
    return jsonOrThrow(
      await fetch(`${API_BASE}/api/datasets/${id}/suggest-columns`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context }),
      }),
    );
  },

  async chatDataset(
    id: string,
    payload: { question: string; context?: string; history?: { role: string; text: string }[] },
  ): Promise<QAResult> {
    return jsonOrThrow(
      await fetch(`${API_BASE}/api/datasets/${id}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
  },

  async listRuns(): Promise<{ runs: RunSummary[] }> {
    return jsonOrThrow(await fetch(`${API_BASE}/api/runs`, { cache: "no-store" }));
  },

  async getRun(id: string): Promise<RunRecord> {
    return jsonOrThrow(await fetch(`${API_BASE}/api/runs/${id}`, { cache: "no-store" }));
  },

  async deleteRun(id: string): Promise<{ ok: boolean }> {
    return jsonOrThrow(await fetch(`${API_BASE}/api/runs/${id}`, { method: "DELETE" }));
  },

  async updateArchetype(runId: string, clusterId: number, patch: ArchetypePatch): Promise<Archetype> {
    return jsonOrThrow(
      await fetch(`${API_BASE}/api/runs/${runId}/archetypes/${clusterId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      }),
    );
  },

  async chatRun(
    id: string,
    payload: { question: string; context?: string; history?: { role: string; text: string }[] },
  ): Promise<QAResult> {
    return jsonOrThrow(
      await fetch(`${API_BASE}/api/runs/${id}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
  },

  exportUrl(runId: string, kind: "archetypes.csv" | "labeled.csv" | "report.md"): string {
    return `${API_BASE}/api/runs/${runId}/export/${kind}`;
  },
};

export interface AnalyzeBody {
  context?: string;
  selected_columns?: string[] | null;
  static_filter_result?: unknown;
  column_recommendation?: unknown;
}

/**
 * Stream the analysis pipeline over SSE. Calls `onEvent` for every parsed event.
 * Returns a function that aborts the stream.
 */
export function streamAnalyze(
  datasetId: string,
  body: AnalyzeBody,
  onEvent: (e: AnalyzeEvent) => void,
  onClose?: (err?: Error) => void,
): () => void {
  const controller = new AbortController();

  // Watchdog: si el stream deja de emitir (proxy colgado, red caída sin cierre),
  // abortamos con un error explicable en vez de dejar el checklist girando para siempre.
  // 180s > peor caso de un nodo LLM con reintentos.
  const STALL_MS = 180_000;
  let stalled = false;
  let stallTimer: ReturnType<typeof setTimeout> | undefined;
  const armStall = () => {
    clearTimeout(stallTimer);
    stallTimer = setTimeout(() => {
      stalled = true;
      controller.abort();
    }, STALL_MS);
  };

  (async () => {
    try {
      armStall();
      const res = await fetch(`${API_BASE}/api/datasets/${datasetId}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        // El backend manda un detail útil ("Dataset no encontrado o expirado…") — no lo tires.
        let detail = `No se pudo iniciar el análisis (HTTP ${res.status}).`;
        try {
          const errBody = await res.json();
          if (errBody.detail) detail = errBody.detail;
        } catch {
          /* cuerpo no-JSON */
        }
        throw new Error(detail);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        armStall();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";
        for (const chunk of chunks) {
          const line = chunk.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          try {
            onEvent(JSON.parse(line.slice(5).trim()) as AnalyzeEvent);
          } catch {
            /* ignore malformed line */
          }
        }
      }
      onClose?.();
    } catch (err) {
      if (stalled) {
        onClose?.(new Error("El análisis dejó de responder. Revisa tu conexión y vuelve a intentarlo."));
        return;
      }
      if ((err as Error).name === "AbortError") return;
      onClose?.(err as Error);
    } finally {
      clearTimeout(stallTimer);
    }
  })();

  return () => controller.abort();
}
