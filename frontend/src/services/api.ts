import {
  UploadResponse,
  DatasetStructureSummary,
  DatasetQualityReport,
  PreviewResponse,
  ApiError
} from '../types/dataset';
import {
  EDAOverview,
  NumericalStat,
  CategoricalStat,
  MissingAnalysis,
  CorrelationAnalysis,
  OutlierStat,
  DistributionResponse,
  PlotlyFigurePayload,
  TargetAnalysisResponse
} from '../types/eda';
import {
  TargetInspectResponse,
  FeatureAuditResponse,
  PrepareDatasetRequest,
  PrepareDatasetResponse,
  PreprocessingSummaryResponse,
  PreprocessingPreviewResponse
} from '../types/preprocessing';
import {
  AutoMLConfig,
  RunStatus,
  RunDetails,
  ModelResult
} from '../types/ml';
import {
  EvaluationOverview,
  ClassificationPerformance,
  RegressionPerformance,
  ClassificationErrorAnalysis,
  RegressionErrorAnalysis,
  GeneralizationDiagnostics,
  ComparisonSummary
} from '../types/evaluation';
import {
  FeatureImportanceResponse,
  ShapSummaryResponse,
  ShapStatusResponse,
  ShapDependenceResponse,
  LocalExplanationResponse
} from '../types/explainability';

const BASE_URL = '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorData: ApiError;
    try {
      errorData = await res.json();
    } catch {
      errorData = {
        error: 'NETWORK_ERROR',
        message: `HTTP error ${res.status}: ${res.statusText}`
      };
    }
    throw errorData;
  }
  return res.json();
}

export const api = {
  async checkHealth(): Promise<{ status: string; version: string; app: string }> {
    const res = await fetch(`${BASE_URL}/health`);
    return handleResponse(res);
  },

  async uploadDataset(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${BASE_URL}/datasets/upload`, {
      method: 'POST',
      body: formData
    });
    return handleResponse<UploadResponse>(res);
  },

  async getDatasetSummary(datasetId: string): Promise<DatasetStructureSummary> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/summary`);
    return handleResponse<DatasetStructureSummary>(res);
  },

  async getDatasetQuality(datasetId: string): Promise<DatasetQualityReport> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/quality`);
    return handleResponse<DatasetQualityReport>(res);
  },

  async getDatasetPreview(datasetId: string, limit: number = 20): Promise<PreviewResponse> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/preview?limit=${limit}`);
    return handleResponse<PreviewResponse>(res);
  },

  // EDA API endpoints
  async getEDAOverview(datasetId: string): Promise<EDAOverview> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/overview`);
    return handleResponse<EDAOverview>(res);
  },

  async getEDANumerical(datasetId: string): Promise<NumericalStat[]> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/numerical`);
    return handleResponse<NumericalStat[]>(res);
  },

  async getEDACategorical(datasetId: string): Promise<CategoricalStat[]> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/categorical`);
    return handleResponse<CategoricalStat[]>(res);
  },

  async getEDAMissing(datasetId: string): Promise<MissingAnalysis> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/missing`);
    return handleResponse<MissingAnalysis>(res);
  },

  async getEDACorrelation(datasetId: string): Promise<CorrelationAnalysis> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/correlation`);
    return handleResponse<CorrelationAnalysis>(res);
  },

  async getEDAOutliers(datasetId: string): Promise<OutlierStat[]> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/outliers`);
    return handleResponse<OutlierStat[]>(res);
  },

  async getEDADistribution(datasetId: string, columnName: string): Promise<DistributionResponse> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/distribution/${encodeURIComponent(columnName)}`);
    return handleResponse<DistributionResponse>(res);
  },

  async createEDAChart(
    datasetId: string,
    x: string,
    y?: string,
    chartType: string = 'auto'
  ): Promise<PlotlyFigurePayload> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/chart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x, y, chart_type: chartType })
    });
    return handleResponse<PlotlyFigurePayload>(res);
  },

  async getEDATargetAnalysis(datasetId: string, targetCol: string): Promise<TargetAnalysisResponse> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/eda/target-analysis?target=${encodeURIComponent(targetCol)}`);
    return handleResponse<TargetAnalysisResponse>(res);
  },

  // ----------------------------------------------------------------
  // Phase 3: Preprocessing & ML Dataset Preparation
  // ----------------------------------------------------------------
  async inspectTarget(datasetId: string, targetCol: string): Promise<TargetInspectResponse> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/preprocessing/target-inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target: targetCol })
    });
    return handleResponse<TargetInspectResponse>(res);
  },

  async auditFeatures(datasetId: string, targetCol: string): Promise<FeatureAuditResponse> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/preprocessing/features-inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target: targetCol })
    });
    return handleResponse<FeatureAuditResponse>(res);
  },

  async prepareDataset(datasetId: string, config: PrepareDatasetRequest): Promise<PrepareDatasetResponse> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/preprocessing/prepare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return handleResponse<PrepareDatasetResponse>(res);
  },

  async getPreprocessingSummary(datasetId: string): Promise<PreprocessingSummaryResponse> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/preprocessing/summary`);
    return handleResponse<PreprocessingSummaryResponse>(res);
  },

  async getPreprocessingPreview(datasetId: string, limit: number = 20): Promise<PreprocessingPreviewResponse> {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/preprocessing/preview?limit=${limit}`);
    return handleResponse<PreprocessingPreviewResponse>(res);
  },

  // AutoML API endpoints
  async startRun(config: AutoMLConfig): Promise<{ success: boolean; message: string; data: any }> {
    const res = await fetch(`${BASE_URL}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return handleResponse<{ success: boolean; message: string; data: any }>(res);
  },

  async getRun(runId: string): Promise<{ success: boolean; data: RunDetails }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}`);
    return handleResponse<{ success: boolean; data: RunDetails }>(res);
  },

  async getRunStatus(runId: string): Promise<{ success: boolean; data: RunStatus }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/status`);
    return handleResponse<{ success: boolean; data: RunStatus }>(res);
  },

  async getRunLeaderboard(runId: string): Promise<{ success: boolean; run_id: string; leaderboard: ModelResult[] }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/leaderboard`);
    return handleResponse<{ success: boolean; run_id: string; leaderboard: ModelResult[] }>(res);
  },

  async cancelRun(runId: string): Promise<{ success: boolean; message: string }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/cancel`, {
      method: 'POST'
    });
    return handleResponse<{ success: boolean; message: string }>(res);
  },

  async getDatasetRuns(datasetId: string): Promise<{ success: boolean; dataset_id: string; runs: any[] }> {
    const res = await fetch(`${BASE_URL}/runs/dataset/${datasetId}`);
    return handleResponse<{ success: boolean; dataset_id: string; runs: any[] }>(res);
  },

  // Evaluation & Diagnostics API endpoints
  async getEvaluationOverview(runId: string): Promise<{ success: boolean; data: EvaluationOverview }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation`);
    return handleResponse<{ success: boolean; data: EvaluationOverview }>(res);
  },

  async getClassificationEvaluation(runId: string): Promise<{ success: boolean; data: ClassificationPerformance }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation/classification`);
    return handleResponse<{ success: boolean; data: ClassificationPerformance }>(res);
  },

  async getRegressionEvaluation(runId: string): Promise<{ success: boolean; data: RegressionPerformance }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation/regression`);
    return handleResponse<{ success: boolean; data: RegressionPerformance }>(res);
  },

  async getErrorAnalysis(runId: string, limit: number = 20): Promise<{ success: boolean; data: ClassificationErrorAnalysis | RegressionErrorAnalysis }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation/errors?limit=${limit}`);
    return handleResponse<{ success: boolean; data: ClassificationErrorAnalysis | RegressionErrorAnalysis }>(res);
  },

  async getCalibration(runId: string): Promise<{ success: boolean; data: any }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation/calibration`);
    return handleResponse<{ success: boolean; data: any }>(res);
  },

  async getDiagnostics(runId: string): Promise<{ success: boolean; data: GeneralizationDiagnostics }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation/diagnostics`);
    return handleResponse<{ success: boolean; data: GeneralizationDiagnostics }>(res);
  },

  async getModelComparison(runId: string): Promise<{ success: boolean; data: ComparisonSummary }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation/comparison`);
    return handleResponse<{ success: boolean; data: ComparisonSummary }>(res);
  },

  async getNativeFeatureImportance(runId: string): Promise<{ success: boolean; data: FeatureImportanceResponse }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation/importance`);
    return handleResponse<{ success: boolean; data: FeatureImportanceResponse }>(res);
  },

  async getPermutationImportance(runId: string, nRepeats: number = 5): Promise<{ success: boolean; data: FeatureImportanceResponse }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/evaluation/permutation?n_repeats=${nRepeats}`);
    return handleResponse<{ success: boolean; data: FeatureImportanceResponse }>(res);
  },

  // Explainability (SHAP) API endpoints
  async triggerShap(runId: string, sampleSize: number = 500): Promise<{ success: boolean; message: string; run_id: string; status: string }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/explainability/shap?sample_size=${sampleSize}`, {
      method: 'POST'
    });
    return handleResponse<{ success: boolean; message: string; run_id: string; status: string }>(res);
  },

  async getShapStatus(runId: string): Promise<{ success: boolean; run_id: string; data: ShapStatusResponse }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/explainability/shap/status`);
    return handleResponse<{ success: boolean; run_id: string; data: ShapStatusResponse }>(res);
  },

  async getShapSummary(runId: string, maxFeatures: number = 20): Promise<{ success: boolean; run_id: string; data: ShapSummaryResponse }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/explainability/shap?max_features=${maxFeatures}`);
    return handleResponse<{ success: boolean; run_id: string; data: ShapSummaryResponse }>(res);
  },

  async getShapDependence(runId: string, feature: string): Promise<{ success: boolean; run_id: string; data: ShapDependenceResponse }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/explainability/dependence?feature=${encodeURIComponent(feature)}`);
    return handleResponse<{ success: boolean; run_id: string; data: ShapDependenceResponse }>(res);
  },

  async getLocalExplanation(runId: string, predictionId: number): Promise<{ success: boolean; run_id: string; data: LocalExplanationResponse }> {
    const res = await fetch(`${BASE_URL}/runs/${runId}/explainability/local/${predictionId}`);
    return handleResponse<{ success: boolean; run_id: string; data: LocalExplanationResponse }>(res);
  }
};
