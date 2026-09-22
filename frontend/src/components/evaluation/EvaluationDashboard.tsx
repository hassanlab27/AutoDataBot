import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import { EvaluationOverview } from '../../types/evaluation';
import { EvaluationHeader } from './EvaluationHeader';
import { ClassificationPerformanceView } from './ClassificationPerformanceView';
import { RegressionPerformanceView } from './RegressionPerformanceView';
import { ErrorAnalysisView } from './ErrorAnalysisView';
import { ExplainabilityView } from './ExplainabilityView';
import { ModelComparisonView } from './ModelComparisonView';
import { ErrorBoundary } from '../common/ErrorBoundary';
import { BarChart3, AlertOctagon, Brain, GitCompare, RefreshCw, AlertCircle } from 'lucide-react';

interface EvaluationDashboardProps {
  datasetId: string;
  initialRunId?: string | null;
}

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  datasetId,
  initialRunId
}) => {
  const [runs, setRuns] = useState<any[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(initialRunId || null);
  const [evaluation, setEvaluation] = useState<EvaluationOverview | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Sub-tab selection
  const [activeSubTab, setActiveSubTab] = useState<'performance' | 'errors' | 'explainability' | 'comparison'>('performance');
  const [jumpToPredictionId, setJumpToPredictionId] = useState<number | null>(null);

  // 1. Fetch available runs for dataset on mount
  useEffect(() => {
    let isMounted = true;
    async function loadRuns() {
      try {
        const res = await api.getDatasetRuns(datasetId);
        if (isMounted && res.success && res.runs.length > 0) {
          setRuns(res.runs);
          if (!activeRunId) {
            // Find latest completed run
            const completedRun = res.runs.find((r: any) => r.status === 'completed') || res.runs[0];
            setActiveRunId(completedRun.run_id);
          }
        } else if (isMounted) {
          setIsLoading(false);
        }
      } catch (err) {
        console.error('Error loading dataset runs:', err);
        if (isMounted) setIsLoading(false);
      }
    }
    loadRuns();
    return () => { isMounted = false; };
  }, [datasetId]);

  // 2. Fetch full evaluation overview whenever activeRunId changes
  useEffect(() => {
    if (!activeRunId) return;

    let isMounted = true;
    async function loadEvaluation() {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const res = await api.getEvaluationOverview(activeRunId!);
        if (isMounted && res.success && res.data) {
          setEvaluation(res.data);
        }
      } catch (err: any) {
        if (isMounted) {
          setErrorMsg(err?.message || 'Failed to load evaluation results for this run.');
        }
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    loadEvaluation();
    return () => { isMounted = false; };
  }, [activeRunId]);

  const handleExplainPrediction = (predictionId: number) => {
    setJumpToPredictionId(predictionId);
    setActiveSubTab('explainability');
  };

  if (isLoading && !evaluation) {
    return (
      <div className="p-12 flex flex-col items-center justify-center space-y-4 text-center">
        <RefreshCw size={36} className="text-indigo-400 animate-spin" />
        <h3 className="text-lg font-bold text-white">Loading Model Evaluation & Diagnostics...</h3>
        <p className="text-xs text-slate-400">Loading performance metrics, confusion matrices, and attributions</p>
      </div>
    );
  }

  if (errorMsg || !evaluation) {
    return (
      <div className="p-8 bg-slate-900/60 border border-slate-800 rounded-2xl text-center max-w-xl mx-auto my-8">
        <AlertCircle size={40} className="text-amber-400 mx-auto mb-3" />
        <h3 className="text-base font-bold text-white mb-1">Evaluation Unavailable</h3>
        <p className="text-xs text-slate-400 mb-4">
          {errorMsg || 'No completed training runs found for this dataset. Please run AutoML training first in Phase 4.'}
        </p>
        {runs.length > 0 && (
          <div className="text-xs text-slate-400">
            Available runs:{' '}
            <select
              value={activeRunId || ''}
              onChange={(e) => setActiveRunId(e.target.value)}
              className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1"
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>{r.run_id} ({r.status})</option>
              ))}
            </select>
          </div>
        )}
      </div>
    );
  }

  const isClassification = evaluation.problem_type
    ? !evaluation.problem_type.toLowerCase().includes('regression')
    : true;

  return (
    <div className="space-y-6">
      {/* Header Spotlight */}
      <ErrorBoundary fallbackTitle="Evaluation Header Unavailable">
        <EvaluationHeader
          evaluation={evaluation}
          runs={runs}
          activeRunId={activeRunId!}
          onSelectRun={(rId) => setActiveRunId(rId)}
        />
      </ErrorBoundary>

      {/* Sub-navigation Tabs */}
      <div className="flex flex-wrap items-center gap-2 bg-slate-950/70 p-1.5 rounded-xl border border-slate-800">
        <button
          onClick={() => setActiveSubTab('performance')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
            activeSubTab === 'performance'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
          }`}
        >
          <BarChart3 size={15} />
          <span>Model Performance</span>
        </button>

        <button
          onClick={() => setActiveSubTab('errors')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
            activeSubTab === 'errors'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
          }`}
        >
          <AlertOctagon size={15} />
          <span>Error Analysis</span>
        </button>

        <button
          onClick={() => setActiveSubTab('explainability')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
            activeSubTab === 'explainability'
              ? 'bg-gradient-to-r from-pink-600 to-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
          }`}
        >
          <Brain size={15} />
          <span>Feature Importance & SHAP</span>
        </button>

        <button
          onClick={() => setActiveSubTab('comparison')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
            activeSubTab === 'comparison'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
          }`}
        >
          <GitCompare size={15} />
          <span>Model Comparison</span>
        </button>
      </div>

      {/* Tab Contents */}
      <ErrorBoundary fallbackTitle="Tab Content Display Error" fallbackMessage="Could not render this evaluation view due to an unexpected format. Other tabs are available.">
        {activeSubTab === 'performance' && isClassification && (
          <ClassificationPerformanceView
            performance={evaluation.performance as any}
          />
        )}

        {activeSubTab === 'performance' && !isClassification && (
          <RegressionPerformanceView
            performance={evaluation.performance as any}
          />
        )}

        {activeSubTab === 'errors' && (
          <ErrorAnalysisView
            errorAnalysis={evaluation.error_analysis}
            problemType={evaluation.problem_type}
            onExplainPrediction={handleExplainPrediction}
          />
        )}

        {activeSubTab === 'explainability' && (
          <ExplainabilityView
            runId={activeRunId!}
            problemType={evaluation.problem_type}
            initialPredictionId={jumpToPredictionId}
          />
        )}

        {activeSubTab === 'comparison' && (
          <ModelComparisonView
            comparison={evaluation.comparison}
          />
        )}
      </ErrorBoundary>
    </div>
  );
};
