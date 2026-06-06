import { create } from "zustand";
import type { DatasetResponse, SuggestResponse } from "./types";

interface WizardState {
  step: 1 | 2;
  dataset: DatasetResponse | null;
  context: string;
  suggestion: SuggestResponse | null;
  selectedColumns: string[];

  setStep: (step: 1 | 2) => void;
  setDataset: (d: DatasetResponse | null) => void;
  setContext: (c: string) => void;
  setSuggestion: (s: SuggestResponse | null) => void;
  setSelectedColumns: (cols: string[]) => void;
  reset: () => void;
}

export const useWizard = create<WizardState>((set) => ({
  step: 1,
  dataset: null,
  context: "",
  suggestion: null,
  selectedColumns: [],

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
  reset: () => set({ step: 1, dataset: null, context: "", suggestion: null, selectedColumns: [] }),
}));
