export interface FeatureImportanceItem {
  feature: string;
  original_feature?: string | null;
  importance: number;
}

export interface FeatureImportanceResponse {
  status: 'available' | 'unavailable';
  source: 'model_feature_importance' | 'permutation_importance';
  features: FeatureImportanceItem[];
  reason?: string | null;
  disclaimer: string;
}

export interface ShapSummaryItem {
  feature: string;
  mean_abs_shap: number;
}

export interface ShapBeeswarmPoint {
  feature: string;
  shap_value: number;
  feature_value: number;
}

export interface ShapSummaryResponse {
  status: 'completed' | 'running' | 'queued' | 'unavailable' | 'failed';
  explainer_type?: string;
  sample_size?: number;
  mean_abs_shap: ShapSummaryItem[];
  beeswarm_sample?: ShapBeeswarmPoint[];
  features: string[];
  reason?: string | null;
  disclaimer: string;
}

export interface ShapDependenceResponse {
  status: 'available' | 'unavailable';
  feature: string;
  feature_values: number[];
  shap_values: number[];
  reason?: string | null;
  disclaimer: string;
}

export interface LocalAttributionItem {
  feature: string;
  original_feature?: string | null;
  feature_value: any;
  shap_value: number;
}

export interface LocalExplanationResponse {
  status: 'available' | 'unavailable';
  prediction_id: number;
  actual: any;
  predicted: any;
  probability?: number | null;
  base_value?: number;
  attributions: LocalAttributionItem[];
  reason?: string | null;
  disclaimer: string;
}

export interface ShapStatusResponse {
  status: 'completed' | 'running' | 'queued' | 'unavailable' | 'failed';
  progress_pct?: number;
  stage?: string;
  reason?: string | null;
}
