export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
  matrix_normalized: number[][];
  total_samples: number;
}

export interface CurveData {
  x: number[];
  y: number[];
  thresholds?: number[];
  auc?: number;
  average_precision?: number;
}

export interface ThresholdPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface CalibrationData {
  prob_pred: number[];
  prob_true: number[];
  brier_score?: number | null;
}

export interface ClassificationPerformance {
  problem_type: 'binary_classification' | 'multiclass_classification';
  metrics: Record<string, number | null>;
  confusion_matrix: ConfusionMatrixData;
  per_class_metrics: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  roc_curve?: CurveData | null;
  pr_curve?: CurveData | null;
  threshold_analysis?: ThresholdPoint[] | null;
  calibration?: CalibrationData | null;
}

export interface RegressionPerformance {
  problem_type: 'regression';
  metrics: {
    mae: number;
    rmse: number;
    r2: number;
    median_absolute_error: number;
    explained_variance?: number | null;
    mape?: number | null;
  };
  actual_vs_predicted: {
    actual: number[];
    predicted: number[];
    sample_size: number;
  };
  residuals: {
    mean: number;
    median: number;
    std: number;
    quantiles: Record<string, number>;
    histogram: {
      counts: number[];
      bin_edges: number[];
    };
    residuals: number[];
    predicted: number[];
    sample_size: number;
  };
  diagnostics_notes: string[];
}

export interface WorstPrediction {
  id: number;
  actual: string | number;
  predicted: string | number;
  probability?: number | null;
  is_correct?: boolean;
  residual?: number;
  absolute_error?: number;
  features: Record<string, any>;
}

export interface ClassificationErrorAnalysis {
  misclassification_count: number;
  total_samples: number;
  error_rate: number;
  false_positives?: number | null;
  false_negatives?: number | null;
  most_confused_pairs: Array<{ actual: string; predicted: string; count: number }>;
  worst_predictions: WorstPrediction[];
}

export interface RegressionErrorAnalysis {
  total_samples: number;
  worst_predictions: WorstPrediction[];
  largest_absolute_errors: number[];
}

export interface GeneralizationDiagnostics {
  label: string;
  validation_score: number | null;
  test_score: number | null;
  generalization_gap: number | null;
  train_score?: number | null;
  train_val_gap?: number | null;
  notes: string[];
}

export interface ComparisonModelItem {
  model_id: string;
  model_name: string;
  engine: string;
  is_winner: boolean;
  is_naive_baseline: boolean;
  validation_score: number | null;
  test_score: number | null;
  generalization_gap: number | null;
  training_time_seconds: number;
  inference_time_seconds: number | null;
  primary_metric: string;
  validation_metrics: Record<string, number | null>;
  test_metrics: Record<string, number | null>;
}

export interface ComparisonSummary {
  primary_metric: string;
  metric_direction: 'higher' | 'lower';
  models: ComparisonModelItem[];
  chart_data: {
    model_names: string[];
    validation_scores: (number | null)[];
    test_scores: (number | null)[];
  };
}

export interface EvaluationOverview {
  run_id: string;
  model_name: string;
  engine: string;
  problem_type: string;
  primary_metric: string;
  validation_score: number | null;
  test_score: number | null;
  generalization_gap: number | null;
  training_time_seconds: number;
  prediction_time_seconds: number;
  performance: ClassificationPerformance | RegressionPerformance;
  error_analysis: ClassificationErrorAnalysis | RegressionErrorAnalysis;
  diagnostics: GeneralizationDiagnostics;
  comparison: ComparisonSummary;
}
