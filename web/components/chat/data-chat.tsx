"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, SendHorizontal, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { ChatTurn, QAResult } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ChatChartView } from "@/components/charts/chat-chart";
import { DataTable } from "@/components/data-table";
import { Button } from "@/components/ui/button";

interface Props {
  ask: (question: string, history: { role: string; text: string }[]) => Promise<QAResult>;
  suggestions?: string[];
  placeholder?: string;
}

export function DataChat({ ask, suggestions = [], placeholder = "Pregunta sobre tus datos…" }: Props) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, busy]);

  const historyForLlm = (list: ChatTurn[]) =>
    list
      .map((t) => ({ role: t.role, text: t.role === "user" ? t.text ?? "" : t.result?.narrative ?? "" }))
      .filter((h) => h.text);

  const send = async (question: string, originalForClarification?: string) => {
    if (!question.trim() || busy) return;
    const userTurn: ChatTurn = { role: "user", text: question };
    const base = [...turns, userTurn];
    setTurns(base);
    setInput("");
    setBusy(true);
    try {
      const result = await ask(question, historyForLlm(turns));
      setTurns([
        ...base,
        { role: "assistant", result, originalQuestion: originalForClarification ?? question },
      ]);
    } catch (e) {
      setTurns([
        ...base,
        {
          role: "assistant",
          result: {
            narrative: (e as Error).message || "No pude responder.",
            operation: "error",
            error: "error",
            clarification: null,
            table: null,
            chart: null,
          },
        },
      ]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scrollRef} className="scroll-slim min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center gap-3 py-6 text-center">
            <div className="flex size-9 items-center justify-center rounded-xl bg-accent text-primary">
              <Sparkles className="size-4" />
            </div>
            <p className="max-w-xs text-sm text-muted-foreground">
              Pregúntame lo que quieras sobre tus datos. Entiendo lenguaje natural y recuerdo el contexto.
            </p>
            {suggestions.length > 0 && (
              <div className="flex flex-wrap justify-center gap-2">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="rounded-full border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {turns.map((turn, i) => (
          <ChatMessage key={i} turn={turn} onClarify={(opt, original) => send(`${original} (en ${opt.toLowerCase()})`, original)} />
        ))}

        {busy && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" /> Pensando…
          </div>
        )}
      </div>

      <form
        className="mt-3 flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={placeholder}
          disabled={busy}
          className="h-9 flex-1 rounded-lg border bg-background px-3 text-sm outline-none transition-colors focus:border-primary focus:ring-2 focus:ring-primary/15"
        />
        <Button type="submit" size="icon" disabled={busy || !input.trim()} aria-label="Enviar">
          <SendHorizontal className="size-4" />
        </Button>
      </form>
    </div>
  );
}

function ChatMessage({
  turn,
  onClarify,
}: {
  turn: ChatTurn;
  onClarify: (option: string, original: string) => void;
}) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground">
          {turn.text}
        </div>
      </div>
    );
  }

  const r = turn.result!;
  return (
    <div className="flex justify-start">
      <div className="max-w-full space-y-2 rounded-2xl rounded-bl-sm bg-muted px-3 py-2">
        <div className="prose-chat text-foreground">
          <ReactMarkdown>{r.narrative}</ReactMarkdown>
        </div>

        {r.clarification && (
          <div className="flex flex-wrap gap-2 pt-1">
            {r.clarification.options.map((opt) => (
              <button
                key={opt}
                onClick={() => onClarify(opt, turn.originalQuestion ?? "")}
                className="rounded-full border bg-card px-3 py-1 text-xs font-medium transition-colors hover:border-primary/50 hover:text-primary"
              >
                {opt}
              </button>
            ))}
          </div>
        )}

        {r.chart && (
          <div className="rounded-lg border bg-card p-2">
            <ChatChartView chart={r.chart} />
          </div>
        )}
        {!r.chart && r.table && <DataTable table={r.table} maxHeight={220} />}
      </div>
    </div>
  );
}
