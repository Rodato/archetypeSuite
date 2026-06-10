import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { DatasetResponse, SuggestResponse } from "./types";

interface WizardState {
  step: 1 | 2 | 3;
  dataset: DatasetResponse | null;
  context: string;
  suggestion: SuggestResponse | null;
  selectedColumns: string[];
  lastRunId: string | null;

  setStep: (step: 1 | 2 | 3) => void;
  setDataset: (d: DatasetResponse | null) => void;
  setContext: (c: string) => void;
  setSuggestion: (s: SuggestResponse | null) => void;
  setSelectedColumns: (cols: string[]) => void;
  setLastRunId: (id: string | null) => void;
  reset: () => void;
}

const initialState = {
  step: 1 as const,
  dataset: null,
  context: "",
  suggestion: null,
  selectedColumns: [],
  lastRunId: null,
};

export const useWizard = create<WizardState>()(
  persist(
    (set) => ({
      ...initialState,

      setStep: (step) => set({ step }),
      setDataset: (dataset) => set({ dataset, suggestion: null, selectedColumns: [] }),
      setContext: (context) => set({ context }),
      setSuggestion: (suggestion) =>
        set({
          suggestion,
          selectedColumns: suggestion
            ? suggestion.column_recommendation.selected_columns.map((c) => c.name)
            : [],
        }),
      setSelectedColumns: (selectedColumns) => set({ selectedColumns }),
      setLastRunId: (lastRunId) => set({ lastRunId }),
      reset: () => set({ ...initialState }),
    }),
    {
      name: "archetype-wizard",
      version: 1,
      // sessionStorage: sobrevive F5 y back-navigation, muere al cerrar la pestaña.
      storage: createJSONStorage(() => sessionStorage),
      // /new rehidrata manualmente tras montar — el HTML prerenderizado siempre es
      // el paso 1 y rehidratar en SSR provocaría mismatch de hidratación.
      skipHydration: true,
    },
  ),
);
