// TypeScript mirror of the FastAPI contracts (see api/transform.py, api/routers/*).

export type Grade = "A" | "B" | "C" | "D" | "—";
export type GradeColor = "green" | "orange" | "red" | "gray";

export interface TableData {
  columns: string[];
  rows: (string | number | null)[][];
}

export interface DonutSegment {
  label: string;
  value: number;
  color: string;
}
export interface Donut {
  segments: DonutSegment[];
  n_cols: number;
  n_rows: number;
  missing_pct: number;
  missing_cells?: number;
  has_missing?: boolean;
}

export interface ProfileColumn {
  name: string;
  dtype: string;
  n_missing: number;
  pct_missing: number;
  n_unique: number;
  is_numeric: boolean;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  median?: number | null;
  top_categories?: Record<string, number>;
}
export interface Profile {
  n_rows: number;
  n_cols: number;
  columns: ProfileColumn[];
  numeric_columns: string[];
  categorical_columns: string[];
  correlation_matrix: Record<string, Record<string, number | null>>;
}

export interface DatasetResponse {
  dataset_id: string;
  file_name: string;
  n_rows: number;
  n_cols: number;
  donut: Donut;
  preview: TableData;
  profile: Profile;
}

export type Importance = "high" | "medium" | "low";
export interface ColumnRec {
  name: string;
  reason: string;
  importance: Importance;
}
export interface ColumnExclusion {
  name: string;
  reason: string;
}
export interface ColumnRecommendation {
  selected_columns: ColumnRec[];
  excluded_columns: ColumnExclusion[];
  summary: string;
}
export interface StaticFilterReport {
  kept: string[];
  dropped: { column: string; reason: string }[];
  datetime_extracted: { original: string; new: string }[];
}
export interface SuggestResponse {
  available_columns: string[];
  static_filter_result: StaticFilterReport;
  column_recommendation: ColumnRecommendation;
  llm_error: string | null;
}

// ---- Chat ----
export type ChartType =
  | "bar" | "pie" | "histogram" | "box" | "scatter" | "line" | "heatmap" | "table" | "none";

export interface ChatChart {
  type: ChartType;
  x?: string | null;
  y?: string | null;
  color?: string | null;
  data?: TableData;
  // heatmap-only
  x_labels?: string[];
  y_labels?: string[];
  z?: (number | null)[][];
}
export interface QAResult {
  narrative: string;
  operation: string;
  error: string | null;
  clarification: { question: string; options: string[] } | null;
  table: TableData | null;
  chart: ChatChart | null;
}
export interface ChatTurn {
  role: "user" | "assistant";
  text?: string;
  result?: QAResult;
  originalQuestion?: string;
}

// ---- Analyze SSE ----
export type StepStatus = "pending" | "running" | "done" | "failed";
export interface ProgressStep {
  key: string;
  label: string;
  status: StepStatus;
}
export interface ProgressEvent {
  type: "progress";
  steps: ProgressStep[];
  running: string | null;
  message: string;
  completed_count: number;
  total: number;
}
export interface DoneEvent {
  type: "done";
  run_id: string;
  summary: RunSummary;
}
export interface ErrorEvent {
  type: "error";
  error_type: string;
  message: string;
  failed_step?: string | null;
}
export type AnalyzeEvent = ProgressEvent | DoneEvent | ErrorEvent;

// ---- Runs ----
export interface Quality {
  grade: Grade;
  label: string;
  description?: string;
  color: GradeColor;
  score: number | null;
}
export interface RunSummary {
  id: string;
  created_at: string;
  file_name: string;
  dataset_context: string;
  n_rows: number;
  n_cols: number;
  n_archetypes: number;
  optimal_k: number | null;
  quality: Quality;
  archetype_labels: string[];
}
export type CautionLevel = "baja" | "media" | "alta";

export interface Archetype {
  cluster_id: number;
  label: string;
  description: string;
  // Behavioral layer (methodology §4) — optional for backward compat with old runs.
  comportamiento_principal?: string;
  microcomportamientos?: string[];
  barreras?: string[];
  habilitadores?: string[];
  oportunidades_accion?: string[];
  nivel_cautela?: CautionLevel;
  cautela_reason?: string;
  // Legacy fields.
  key_characteristics?: string[];
  differentiators?: string[];
  size: number;
  prevalence: number;
  color: string;
  // Curación humana.
  validated?: boolean;
  curated_at?: string | null;
}

/** Campos editables vía PATCH /runs/{id}/archetypes/{cluster_id} (nivel_cautela no es editable). */
export interface ArchetypePatch {
  label?: string;
  description?: string;
  comportamiento_principal?: string;
  microcomportamientos?: string[];
  barreras?: string[];
  habilitadores?: string[];
  oportunidades_accion?: string[];
  cautela_reason?: string;
  validated?: boolean;
}
/** Perfil de grupo definido por el usuario (perfilado a demanda). */
export interface GroupProfile {
  id: string;
  created_at: string;
  origin: "user_defined";
  group_description: string;
  interpretation: string;
  filters: unknown[];
  n: number;
  share: number;
  label: string;
  description: string;
  comportamiento_principal: string;
  microcomportamientos: string[];
  barreras: string[];
  habilitadores: string[];
  oportunidades_accion: string[];
  nivel_cautela: CautionLevel;
  cautela_reason: string;
}

export interface ClusterSize {
  cluster_id: number;
  label: string;
  size: number;
  color: string;
}
export interface ScatterPoint {
  PC1: number;
  PC2: number;
  cluster_id: number;
  archetype: string;
}
export interface RadarSeries {
  cluster_id: number;
  label: string;
  color: string;
  values: number[];
  raw_values: number[];
}
export interface RadarData {
  axes: string[];
  series: RadarSeries[];
}
export interface BoxGroup {
  cluster_id: number;
  label: string;
  color: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
}
export interface RunRecord {
  custom_profiles?: GroupProfile[];
  id: string;
  created_at: string;
  file_name: string;
  dataset_context: string;
  n_rows: number;
  n_cols: number;
  optimal_k: number;
  n_clusters: number;
  quality: Quality;
  archetypes: Archetype[];
  summary: string;
  cluster_sizes: ClusterSize[];
  metrics: {
    silhouette_score: number | null;
    calinski_harabasz_score: number | null;
    davies_bouldin_score: number | null;
    n_clusters: number;
  };
  k_analysis: { k_range: number[]; silhouette_scores: (number | null)[]; optimal_k: number; forced_k_min?: boolean };
  charts: { scatter: ScatterPoint[]; radar: RadarData; box: Record<string, BoxGroup[]> };
  columns: { numeric: string[]; categorical: string[] };
  labels: number[];
  raw_data: Record<string, (string | number | null)[]>;
  advanced: {
    selected_algorithm: string;
    selection_reasoning: string;
    refinement_reason: string;
    refinement_count: number;
  };
}
