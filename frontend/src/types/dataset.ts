export interface ColumnSummary {
  name: string;
  raw_dtype: string;
  inferred_dtype: string;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  unique_percentage: number;
  is_constant: boolean;
  is_high_cardinality: boolean;
  sample_values: any[];
}

export interface DatasetStructureSummary {
  row_count: number;
  column_count: number;
  memory_usage_bytes: number;
  memory_usage_human: string;
  duplicate_rows_count: number;
  duplicate_rows_percentage: number;
  constant_columns_count: number;
  columns: ColumnSummary[];
}

export interface QualityCheck {
  name: string;
  status: 'pass' | 'warning' | 'fail';
  summary: string;
  description: string;
  affected_columns: string[];
}

export interface SuspiciousColumn {
  column_name: string;
  issue_type: 'id_like' | 'timestamp_like' | 'constant' | 'high_cardinality' | 'high_missingness';
  severity: 'warning' | 'info';
  description: string;
}

export interface DatasetQualityReport {
  quality_score: number;
  rating: 'Excellent' | 'Good' | 'Fair' | 'Needs Attention';
  checks: QualityCheck[];
  suspicious_columns: SuspiciousColumn[];
  score_breakdown: Record<string, number>;
}

export interface RepairAction {
  action_type: string;
  description: string;
  details: Record<string, any>;
}

export interface IngestionReport {
  original_filename: string;
  file_size_bytes: number;
  encoding_used: string;
  delimiter_used: string;
  total_raw_rows: number;
  total_raw_columns: number;
  skipped_bad_lines: number;
  actions: RepairAction[];
}

export interface UploadResponse {
  dataset_id: string;
  original_filename: string;
  rows_count: number;
  columns_count: number;
  summary: DatasetStructureSummary;
  quality: DatasetQualityReport;
  ingestion: IngestionReport;
}

export interface PreviewResponse {
  dataset_id: string;
  total_rows: number;
  returned_rows: number;
  columns: string[];
  inferred_dtypes: Record<string, string>;
  rows: Record<string, any>[];
}

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, any>;
}
