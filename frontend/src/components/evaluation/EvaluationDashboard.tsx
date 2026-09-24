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
import './evaluation.css';
import { BarChart3, AlertOctagon, Brain, GitCompare, RefreshCw, AlertCircle } from 'lucide-react';

interface EvaluationDashboardProps {
  datasetId: string;
  initialRunId?: string | null;
  onNavigateToReport?: (runId: string) => void;
}

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  datasetId,
  initialRunId,
  onNavigateToReport
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
      <div className="eval-card" style={{ padding: '3.5rem 2rem', textAlign: 'center', maxWidth: '600px', margin: '2rem auto' }}>
        <div style={{ display: 'inline-flex', padding: '1rem', borderRadius: '50%', background: 'rgba(99, 102, 241, 0.12)', marginBottom: '1.25rem' }}>
          <RefreshCw size={36} className="text-indigo-400 spin" />
        </div>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.5rem' }}>
          Analyzing Model Performance & Generalization...
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--eval-text-muted)', margin: 0 }}>
          Computing confusion matrices, ROC/PR curves, probability calibration, and error distributions.
        </p>
      </div>
    );
  }

  if (errorMsg || !evaluation) {
    return (
      <div className="eval-card" style={{ padding: '2.5rem', textAlign: 'center', maxWidth: '580px', margin: '2rem auto' }}>
        <AlertCircle size={44} className="text-amber-400" style={{ margin: '0 auto 1rem auto' }} />
        <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.5rem' }}>
          Model Evaluation Unavailable
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--eval-text-muted)', marginBottom: '1.5rem', lineHeight: 1.5 }}>
          {errorMsg || 'No completed training runs found for this dataset. Train candidate models in the Prepare Dataset section to generate diagnostic evaluations.'}
        </p>
        {runs.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', fontSize: '0.825rem', color: 'var(--eval-text-muted)' }}>
            <span>Available runs:</span>
            <select
              value={activeRunId || ''}
              onChange={(e) => setActiveRunId(e.target.value)}
              className="select-input"
              style={{ padding: '0.45rem 0.85rem', fontSize: '0.8rem' }}
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
    <div className="eval-container">
      {/* 1. Header Spotlight Card */}
      <ErrorBoundary fallbackTitle="Evaluation Header Unavailable">
        <EvaluationHeader
          evaluation={evaluation}
          runs={runs}
          activeRunId={activeRunId!}
          onSelectRun={(rId) => setActiveRunId(rId)}
          onNavigateToReport={onNavigateToReport}
        />
      </ErrorBoundary>

      {/* 2. Sub-navigation Tabs */}
      <nav
        className="eval-nav-bar"
        role="tablist"
        aria-label="Model Evaluation Sections"
      >
        <button
          role="tab"
          id="eval-tab-performance"
          aria-selected={activeSubTab === 'performance'}
          aria-controls="eval-panel-performance"
          onClick={() => setActiveSubTab('performance')}
          className={`eval-nav-btn ${activeSubTab === 'performance' ? 'eval-nav-btn-active' : ''}`}
        >
          <BarChart3 size={16} />
          <span>Model Performance</span>
        </button>

        <button
          role="tab"
          id="eval-tab-errors"
          aria-selected={activeSubTab === 'errors'}
          aria-controls="eval-panel-errors"
          onClick={() => setActiveSubTab('errors')}
          className={`eval-nav-btn ${activeSubTab === 'errors' ? 'eval-nav-btn-active' : ''}`}
        >
          <AlertOctagon size={16} />
          <span>Error Analysis</span>
        </button>

        <button
          role="tab"
          id="eval-tab-explainability"
          aria-selected={activeSubTab === 'explainability'}
          aria-controls="eval-panel-explainability"
          onClick={() => setActiveSubTab('explainability')}
          className={`eval-nav-btn ${activeSubTab === 'explainability' ? 'eval-nav-btn-explain' : ''}`}
        >
          <Brain size={16} />
          <span>Feature Importance & SHAP</span>
        </button>

        <button
          role="tab"
          id="eval-tab-comparison"
          aria-selected={activeSubTab === 'comparison'}
          aria-controls="eval-panel-comparison"
          onClick={() => setActiveSubTab('comparison')}
          className={`eval-nav-btn ${activeSubTab === 'comparison' ? 'eval-nav-btn-active' : ''}`}
        >
          <GitCompare size={16} />
          <span>Model Comparison</span>
        </button>
      </nav>

      {/* 3. Tab Contents */}
      <ErrorBoundary fallbackTitle="Tab Content Display Error" fallbackMessage="Could not render this evaluation view due to an unexpected format. Other tabs remain functional.">
        <div id={`eval-panel-${activeSubTab}`} role="tabpanel" aria-labelledby={`eval-tab-${activeSubTab}`}>
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
        </div>
      </ErrorBoundary>
    </div>
  );
};
