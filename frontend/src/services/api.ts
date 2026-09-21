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
  }
};
