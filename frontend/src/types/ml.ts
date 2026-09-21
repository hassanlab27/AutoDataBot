export type TrainingMode = "quick" | "standard" | "extended" | "custom";
export type EnginePreference = "all" | "autogluon" | "flaml" | "sklearn_gbdt";

export interface AutoMLConfig {
  dataset_id: string;
  target: string;
  problem_type: string;
  training_mode: TrainingMode;
  time_limit?: number;
  primary_metric: string;
  random_state: number;
  max_cpus?: number;
  presets?: string;
  engine_preference: EnginePreference;
}

export interface RunStatus {
  run_id: string;
  dataset_id: string;
  status: "created" | "queued" | "running" | "evaluating" | "completed" | "failed" | "cancelled";
  current_stage: string;
  current_engine?: string;
  progress_pct: number;
  start_time: number;
  end_time?: number;
  elapsed_seconds: number;
  error_message?: string;
  cancellation_requested: boolean;
}

export interface ModelResult {
  model_id: string;
  model_name: string;
  engine: string;
  problem_type: string;
  validation_metrics: Record<string, number>;
  validation_primary_score: number;
  test_metrics?: Record<string, number>;
  test_primary_score?: number;
  generalization_gap?: number;
  training_time_seconds: number;
  status: "success" | "failed" | "timeout";
  error_message?: string;
  diagnostic_label?: string;
  diagnostic_notes: string[];
  is_naive_baseline: boolean;
  is_winner: boolean;
  model_path?: string;
  hyperparameters?: Record<string, any>;
}

export interface RunMetricsPayload {
  run_id: string;
  primary_metric: string;
  metric_direction: "higher" | "lower";
  naive_baseline_score?: number;
  winner_id?: string;
  winner_validation_metrics?: Record<string, number>;
  winner_test_metrics?: Record<string, number>;
  generalization_gap?: number;
  diagnostic_label?: string;
  diagnostic_notes?: string[];
}

export interface RunSummaryPayload {
  run_id: string;
  dataset_id: string;
  target: string;
  problem_type: string;
  primary_metric: string;
  total_models_trained: number;
  successful_models: number;
  failed_models: number;
  winning_model: string;
  winning_engine: string;
  validation_score?: number;
  test_score?: number;
  generalization_gap?: number;
  diagnostic_label?: string;
  created_at: string;
}

export interface RunDetails {
  run_id: string;
  status: RunStatus;
  config: Record<string, any>;
  leaderboard: ModelResult[];
  metrics: RunMetricsPayload;
  summary: RunSummaryPayload;
}
