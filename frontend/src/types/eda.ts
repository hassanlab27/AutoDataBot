export interface EDAOverview {
  rows: number;
  columns: number;
  numeric_columns: number;
  categorical_columns: number;
  boolean_columns: number;
  datetime_columns: number;
  text_columns: number;
  missing_cells: number;
  missing_percentage: number;
  duplicate_rows: number;
  duplicate_percentage: number;
  memory_usage_bytes: number;
  memory_usage_human: string;
  constant_columns: string[];
  high_cardinality_columns: string[];
  column_classification: {
    numeric: string[];
    categorical: string[];
    boolean: string[];
    datetime: string[];
    text: string[];
  };
}

export interface NumericalStat {
  column: string;
  count: number;
  valid_count: number;
  missing_count: number;
  missing_percentage: number;
  mean: number | null;
  median: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  q1: number | null;
  q3: number | null;
  iqr: number | null;
  skewness: number | null;
  kurtosis: number | null;
}

export interface CategoryFrequency {
  category: string;
  count: number;
  percentage: number;
}

export interface CategoricalStat {
  column: string;
  count: number;
  valid_count: number;
  missing_count: number;
  missing_percentage: number;
  unique_values: number;
  mode: string | null;
  mode_count: number;
  mode_percentage: number;
  top_categories: CategoryFrequency[];
  other_count: number;
  other_percentage: number;
}

export interface MissingColumn {
  column: string;
  missing_count: number;
  missing_percentage: number;
  present_count: number;
  present_percentage: number;
}

export interface PlotlyFigurePayload {
  chart_type: string;
  title?: string;
  recommended_type?: string;
  points_plotted?: number;
  sampled?: boolean;
  data: any[];
  layout: Record<string, any>;
  box_plot?: {
    chart_type: string;
    data: any[];
    layout: Record<string, any>;
  };
}

export interface MissingAnalysis {
  total_cells: number;
  total_missing_cells: number;
  missing_percentage: number;
  columns_with_missing_count: number;
  columns_with_missing: string[];
  severe_missing_columns: string[];
  columns: MissingColumn[];
  chart: PlotlyFigurePayload;
}

export interface StrongCorrelationPair {
  feature_a: string;
  feature_b: string;
  correlation: number;
  abs_correlation: number;
  relationship_strength: string;
  direction: string;
}

export interface CorrelationAnalysis {
  columns: string[];
  matrix: number[][];
  total_numeric_columns: number;
  columns_included: number;
  truncated: boolean;
  selection_strategy: string;
  strong_correlations: StrongCorrelationPair[];
  heatmap: PlotlyFigurePayload;
  disclaimer: string;
}

export interface OutlierStat {
  column: string;
  method: string;
  multiplier: number;
  q1: number | null;
  q3: number | null;
  iqr: number | null;
  lower_bound: number | null;
  upper_bound: number | null;
  potential_outliers_count: number;
  potential_outliers_percentage: number;
  lower_outliers_count: number;
  upper_outliers_count: number;
  min_val: number | null;
  max_val: number | null;
  has_outliers: boolean;
  disclaimer: string;
}

export interface DistributionResponse {
  column: string;
  dtype: string;
  stats?: NumericalStat | CategoricalStat;
  outliers?: OutlierStat;
  temporal?: {
    column: string;
    total_count: number;
    valid_dates_count: number;
    missing_count: number;
    min_date: string;
    max_date: string;
    date_range_days: number;
  };
  histogram?: PlotlyFigurePayload;
  box_plot?: PlotlyFigurePayload;
  chart?: PlotlyFigurePayload;
}

export interface TargetAnalysisResponse {
  target_column: string;
  target_dtype: string;
  profile: {
    type: 'numeric' | 'categorical';
    stats: any;
    outliers?: any;
    charts?: any;
    chart?: any;
    is_imbalanced?: boolean;
    imbalance_warning?: string | null;
    top_associated_features?: Array<{
      feature: string;
      correlation: number;
      abs_correlation: number;
    }>;
  };
  disclaimer: string;
}
