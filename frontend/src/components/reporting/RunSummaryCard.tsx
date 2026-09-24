import React from 'react';
import { 
  FileText, 
  Download, 
  Archive, 
  Sparkles, 
  Brain, 
  CheckCircle2, 
  AlertCircle
} from 'lucide-react';
import { api } from '../../services/api';
import { RunSummaryData } from '../../types/report';

interface RunSummaryCardProps {
  summary: RunSummaryData;
  rowsCount?: number | string;
  columnsCount?: number | string;
  onViewEvaluation?: () => void;
  onViewExplainability?: () => void;
  onOpenReportModal?: () => void;
}

export const RunSummaryCard: React.FC<RunSummaryCardProps> = ({
  summary,
  rowsCount,
  columnsCount,
  onViewEvaluation,
  onViewExplainability,
  onOpenReportModal
}) => {
  const isCompleted = summary.status === 'completed';
  const isFailed = summary.status === 'failed';

  const formatScore = (score?: number | null) => {
    if (score === null || score === undefined) return '—';
    return typeof score === 'number' ? score.toFixed(4) : score;
  };

  return (
    <div className="reporting-card">
      <div className="reporting-header-row">
        <div>
          <h3 className="reporting-title">
            <CheckCircle2 size={22} className="text-emerald-400" />
            <span>Executive Run Summary</span>
            <span className={`badge ${isCompleted ? 'badge-success' : isFailed ? 'badge-failed' : 'badge-warning'}`} style={{ marginLeft: '0.5rem' }}>
              {summary.status.toUpperCase()}
            </span>
          </h3>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#9ca3af' }}>
            Run ID: <code style={{ color: '#e0e7ff' }}>{summary.run_id}</code>
          </p>
        </div>

        {/* Primary Action Buttons */}
        <div className="reporting-actions-bar">
          {isCompleted && (
            <>
              {onOpenReportModal && (
                <button
                  className="btn btn-primary"
                  onClick={onOpenReportModal}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.825rem' }}
                >
                  <FileText size={16} /> View Final Report
                </button>
              )}

              <a
                href={api.getExportZipUrl(summary.run_id)}
                download={`autodatabot_${summary.run_id}.zip`}
                className="btn btn-outline"
                style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.825rem', color: '#fff', borderColor: 'var(--accent-purple)' }}
                title="Download verified zip bundle with all artifacts, models, metrics, and manifest"
              >
                <Archive size={16} className="text-purple-400" /> Export Bundle (ZIP)
              </a>

              <a
                href={api.getReportPdfUrl(summary.run_id)}
                download={`autodatabot_${summary.run_id}_report.pdf`}
                className="btn btn-outline"
                style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.825rem', color: '#fff' }}
                title="Download PDF report"
              >
                <Download size={16} /> PDF Report
              </a>
            </>
          )}

          {onViewEvaluation && isCompleted && (
            <button
              className="btn btn-outline"
              onClick={onViewEvaluation}
              style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.825rem', color: '#fff' }}
            >
              <Sparkles size={16} className="text-indigo-400" /> View Evaluation
            </button>
          )}

          {onViewExplainability && isCompleted && (
            <button
              className="btn btn-outline"
              onClick={onViewExplainability}
              style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.825rem', color: '#fff' }}
            >
              <Brain size={16} className="text-pink-400" /> View SHAP
            </button>
          )}
        </div>
      </div>

      {/* Grid of Key Metrics */}
      <div className="summary-metrics-grid">
        <div className="summary-metric-card">
          <div className="summary-metric-label">Dataset / Dimensions</div>
          <div className="summary-metric-value" style={{ fontSize: '1.15rem' }}>
            {rowsCount ? `${rowsCount} rows` : 'Loaded'}
          </div>
          <div className="summary-metric-subtext">
            {columnsCount ? `${columnsCount} columns` : summary.dataset_id}
          </div>
        </div>

        <div className="summary-metric-card">
          <div className="summary-metric-label">Target & Problem</div>
          <div className="summary-metric-value" style={{ fontSize: '1.15rem' }}>
            {summary.target || 'Target'}
          </div>
          <div className="summary-metric-subtext">
            {summary.problem_type ? summary.problem_type.replace('_', ' ') : 'Tabular Task'}
          </div>
        </div>

        <div className="summary-metric-card">
          <div className="summary-metric-label">Winning Model</div>
          <div className="summary-metric-value" style={{ fontSize: '1.15rem', color: '#818cf8' }}>
            {summary.selected_model || '—'}
          </div>
          <div className="summary-metric-subtext">
            Engine: {summary.winning_engine || 'AutoML'}
          </div>
        </div>

        <div className="summary-metric-card">
          <div className="summary-metric-label">Validation ({summary.primary_metric || 'Metric'})</div>
          <div className="summary-metric-value" style={{ color: '#34d399' }}>
            {formatScore(summary.validation_score)}
          </div>
          <div className="summary-metric-subtext">Holdout split score</div>
        </div>

        <div className="summary-metric-card">
          <div className="summary-metric-label">Final Test Score</div>
          <div className="summary-metric-value" style={{ color: '#60a5fa' }}>
            {formatScore(summary.test_score)}
          </div>
          <div className="summary-metric-subtext">External isolated test set</div>
        </div>

        <div className="summary-metric-card">
          <div className="summary-metric-label">Training Duration</div>
          <div className="summary-metric-value" style={{ fontSize: '1.15rem' }}>
            {summary.duration_seconds !== undefined ? `${summary.duration_seconds.toFixed(1)}s` : '—'}
          </div>
          <div className="summary-metric-subtext">Total elapsed time</div>
        </div>
      </div>

      {!isCompleted && (
        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 'var(--radius-md)', padding: '1rem', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <AlertCircle size={20} />
          <div>
            <strong>Run Incomplete:</strong> Reports, exports, and final evaluation are only accessible once training finishes successfully.
          </div>
        </div>
      )}
    </div>
  );
};
