export interface ExportManifest {
  application: string;
  app_version: string;
  report_version: string;
  run_id: string;
  dataset_id: string;
  created_at: string;
  problem_type: string;
  selected_model: string;
  winning_engine: string;
  validation_score: number | null;
  test_score: number | null;
  files_included: string[];
  files_excluded: string[];
}

export interface RunSummaryData {
  run_id: string;
  dataset_id: string;
  dataset_name?: string;
  problem_type?: string;
  target?: string;
  selected_model?: string;
  winning_engine?: string;
  validation_score?: number | null;
  test_score?: number | null;
  train_score?: number | null;
  primary_metric?: string;
  duration_seconds?: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at?: string;
}
