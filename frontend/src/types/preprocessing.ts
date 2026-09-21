export interface TargetDiagnostics {
  valid_count: number;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  id_warning?: string | null;
  classification?: {
    classes_count: number;
    classes: Array<{ class_label: string; count: number; percentage: number }>;
    minority_class_percentage: number;
    is_imbalanced: boolean;
    imbalance_warning?: string | null;
  };
  regression?: {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
    q1: number;
    q3: number;
    iqr: number;
    potential_extreme_values_count: number;
  };
}

export interface TargetInspectResponse {
  target: string;
  inferred_dtype: string;
  problem_type: 'binary_classification' | 'multiclass_classification' | 'regression' | 'ambiguous';
  confidence: 'high' | 'medium' | 'low';
  detection_reason: string;
  diagnostics: TargetDiagnostics;
}

export interface FeatureAuditItem {
  name: string;
  inferred_dtype: string;
  raw_dtype: string;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  unique_percentage: number;
  flags: string[];
  status: 'included' | 'flagged_exclude' | 'review_recommended';
  recommended_action: 'include' | 'exclude' | 'review';
  reason?: string | null;
}

export interface FeatureAuditResponse {
  target: string;
  total_candidate_features: number;
  recommended_included_count: number;
  recommended_excluded_count: number;
  review_recommended_count: number;
  features: FeatureAuditItem[];
}

export interface PrepareDatasetRequest {
  target: string;
  problem_type: string;
  selected_features?: string[];
  excluded_features?: string[];
  test_size: number;
  random_state: number;
  numeric_imputation: string;
  categorical_imputation: string;
  categorical_encoding: string;
  scaling: string;
  extract_datetime: boolean;
}

export interface PreprocessingSummaryResponse {
  status: string;
  dataset_id: string;
  created_at?: string;
  target?: string;
  problem_type?: string;
  original_features_count: number;
  selected_features_count: number;
  excluded_features_count: number;
  final_features_count: number;
  numeric_features_count: number;
  categorical_features_count: number;
  generated_datetime_features_count: number;
  one_hot_generated_features_count: number;
  feature_names_out: string[];
  split: {
    train_rows: number;
    test_rows: number;
    train_percentage: number;
    test_percentage: number;
    split_strategy: string;
    random_state: number;
    target_distribution?: Record<string, any>;
    train_class_distribution?: Record<string, any>;
    test_class_distribution?: Record<string, any>;
  };
  config?: {
    test_size: number;
    random_state: number;
    numeric_imputation: string;
    categorical_imputation: string;
    categorical_encoding: string;
    scaling: string;
    datetime_features: boolean;
  };
}

export interface PreprocessingPreviewResponse {
  status: string;
  dataset_id: string;
  total_training_rows: number;
  returned_rows: number;
  columns: string[];
  rows: Array<Record<string, any>>;
}

export interface PrepareDatasetResponse {
  status: string;
  dataset_id: string;
  target: string;
  problem_type: string;
  config: Record<string, any>;
  metadata: PreprocessingSummaryResponse;
  preview: {
    total_rows: number;
    preview_rows: Array<Record<string, any>>;
    columns: string[];
  };
}
