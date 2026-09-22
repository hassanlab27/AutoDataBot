import React, { useState } from 'react';
import {
  ClassificationErrorAnalysis,
  RegressionErrorAnalysis,
  WorstPrediction
} from '../../types/evaluation';
import { AlertOctagon, Sparkles, ChevronRight } from 'lucide-react';

interface ErrorAnalysisViewProps {
  errorAnalysis: ClassificationErrorAnalysis | RegressionErrorAnalysis;
  problemType: string;
  onExplainPrediction?: (predictionId: number) => void;
}

export const ErrorAnalysisView: React.FC<ErrorAnalysisViewProps> = ({
  errorAnalysis,
  problemType,
  onExplainPrediction
}) => {
  const isClassification = problemType.includes('classification');
  const clsErrors = isClassification ? (errorAnalysis as ClassificationErrorAnalysis) : null;
  const regErrors = !isClassification ? (errorAnalysis as RegressionErrorAnalysis) : null;

  const worstPredictions = errorAnalysis?.worst_predictions || [];
  const totalSamples = errorAnalysis?.total_samples ?? 0;

  const [selectedRow, setSelectedRow] = useState<WorstPrediction | null>(
    worstPredictions.length > 0 ? worstPredictions[0] : null
  );

  const formatScore = (val?: number | null) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(4);
  };

  const largestErrors = regErrors?.largest_absolute_errors || [];

  return (
    <div className="eval-container">
      {/* Error Analysis Summary Cards */}
      <div className="eval-grid-4">
        <div className="eval-stat-card">
          <span className="eval-stat-label">Evaluated Samples</span>
          <span className="eval-stat-val eval-stat-val-indigo">{totalSamples}</span>
          <span className="eval-stat-subtext">Held-out Test Partition</span>
        </div>

        {isClassification && clsErrors && (
          <>
            <div className="eval-stat-card">
              <span className="eval-stat-label">Misclassifications</span>
              <span className="eval-stat-val eval-stat-val-rose">
                {clsErrors.misclassification_count ?? 0}
              </span>
              <span className="eval-stat-subtext">Incorrect Test Labels</span>
            </div>

            <div className="eval-stat-card">
              <span className="eval-stat-label">Test Error Rate</span>
              <span className="eval-stat-val eval-stat-val-rose">
                {clsErrors.error_rate !== undefined && clsErrors.error_rate !== null
                  ? `${(clsErrors.error_rate * 100).toFixed(2)}%`
                  : '—'}
              </span>
              <span className="eval-stat-subtext">1.0 − Accuracy</span>
            </div>

            {clsErrors.false_positives !== null && clsErrors.false_positives !== undefined && (
              <div className="eval-stat-card">
                <span className="eval-stat-label">FP / FN Breakdown</span>
                <span className="eval-stat-val" style={{ fontSize: '1.25rem' }}>
                  FP: <b style={{ color: '#fcd34d' }}>{clsErrors.false_positives}</b> | FN: <b style={{ color: '#fb7185' }}>{clsErrors.false_negatives}</b>
                </span>
                <span className="eval-stat-subtext">Type I vs Type II Errors</span>
              </div>
            )}
          </>
        )}

        {!isClassification && regErrors && (
          <>
            <div className="eval-stat-card">
              <span className="eval-stat-label">Max Absolute Error</span>
              <span className="eval-stat-val eval-stat-val-rose">
                {largestErrors.length > 0 ? formatScore(largestErrors[0]) : '—'}
              </span>
              <span className="eval-stat-subtext">Worst Single Sample Error</span>
            </div>

            <div className="eval-stat-card">
              <span className="eval-stat-label">Top 5 Mean Error</span>
              <span className="eval-stat-val eval-stat-val-rose">
                {largestErrors.length > 0
                  ? formatScore(
                      largestErrors.slice(0, 5).reduce((a, b) => a + b, 0) /
                        Math.min(5, largestErrors.length)
                    )
                  : '—'}
              </span>
              <span className="eval-stat-subtext">Average of Top 5 Errors</span>
            </div>
          </>
        )}
      </div>

      {/* Most Confused Pairs (Classification Only) */}
      {isClassification && clsErrors && (clsErrors.most_confused_pairs || []).length > 0 && (
        <div className="eval-card">
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertOctagon size={18} style={{ color: '#f59e0b' }} />
                Most Confused Class Pairs
              </h3>
              <p className="eval-card-desc">Specific target labels that the model frequently mistakes for one another</p>
            </div>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
            {(clsErrors.most_confused_pairs || []).map((pair, idx) => (
              <div key={idx} className="eval-chip">
                <div>
                  <span style={{ color: 'var(--eval-text-muted)' }}>Actual: </span>
                  <span style={{ fontWeight: 700, color: '#fb7185' }}>{pair.actual}</span>
                </div>
                <ChevronRight size={14} style={{ color: 'var(--eval-text-dim)' }} />
                <div>
                  <span style={{ color: 'var(--eval-text-muted)' }}>Predicted: </span>
                  <span style={{ fontWeight: 700, color: '#a5b4fc' }}>{pair.predicted}</span>
                </div>
                <span className="eval-chip-count">{pair.count} cases</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top 20 Worst Predictions Table */}
      <div className="eval-card">
        <div className="eval-card-header">
          <div>
            <h3 className="eval-card-title">Worst Test Predictions</h3>
            <p className="eval-card-desc">
              {isClassification
                ? 'Ranked by highest model confidence in incorrect predictions (most overconfident mistakes)'
                : 'Ranked by largest absolute difference between actual and predicted target'}
            </p>
          </div>
          <span className="eval-badge eval-badge-indigo">
            Top {worstPredictions.length} Outliers
          </span>
        </div>

        <div className="eval-table-container">
          <table className="eval-table">
            <thead>
              <tr>
                <th>Test Row #</th>
                <th>Actual Value</th>
                <th>Predicted Value</th>
                {isClassification && <th>Confidence / Prob</th>}
                {!isClassification && <th>Residual</th>}
                {!isClassification && <th>Absolute Error</th>}
                <th style={{ textAlign: 'right' }}>Diagnostic Action</th>
              </tr>
            </thead>
            <tbody>
              {worstPredictions.map((pred) => {
                const isSelected = selectedRow?.id === pred.id;
                return (
                  <tr
                    key={pred.id}
                    onClick={() => setSelectedRow(pred)}
                    className={isSelected ? 'eval-row-selected' : ''}
                    style={{ cursor: 'pointer' }}
                    tabIndex={0}
                    role="button"
                    aria-label={`Inspect features for test row ${pred.id}`}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        setSelectedRow(pred);
                      }
                    }}
                  >
                    <td style={{ fontWeight: 700, color: '#94a3b8', fontFamily: 'monospace' }}>#{pred.id}</td>
                    <td style={{ fontWeight: 700, color: '#34d399', fontFamily: 'monospace' }}>{String(pred.actual)}</td>
                    <td style={{ fontWeight: 700, color: '#fb7185', fontFamily: 'monospace' }}>{String(pred.predicted)}</td>
                    {isClassification && (
                      <td style={{ color: '#cbd5e1', fontFamily: 'monospace' }}>
                        {pred.probability !== null && pred.probability !== undefined
                          ? `${(pred.probability * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                    )}
                    {!isClassification && (
                      <td style={{ color: '#a5b4fc', fontFamily: 'monospace' }}>
                        {pred.residual !== undefined ? (pred.residual > 0 ? `+${formatScore(pred.residual)}` : formatScore(pred.residual)) : '—'}
                      </td>
                    )}
                    {!isClassification && (
                      <td style={{ color: '#fb7185', fontWeight: 700, fontFamily: 'monospace' }}>
                        {formatScore(pred.absolute_error)}
                      </td>
                    )}
                    <td style={{ textAlign: 'right' }}>
                      {onExplainPrediction && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onExplainPrediction(pred.id);
                          }}
                          className="eval-btn-primary eval-btn-sm"
                          style={{ marginLeft: 'auto' }}
                          title={`Explain sample #${pred.id} using SHAP`}
                        >
                          <Sparkles size={12} />
                          Explain with SHAP
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Selected Prediction Feature Inspector */}
        {selectedRow && selectedRow.features && (
          <div className="eval-inspector-box">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span>🔍</span> Features for Prediction #{selectedRow.id}
              </div>
              <span style={{ fontSize: '0.8rem', color: 'var(--eval-text-muted)' }}>
                Actual: <b style={{ color: '#34d399' }}>{String(selectedRow.actual)}</b> | Predicted: <b style={{ color: '#fb7185' }}>{String(selectedRow.predicted)}</b>
              </span>
            </div>
            <div className="eval-inspector-grid">
              {Object.entries(selectedRow.features).map(([feat, val]) => (
                <div key={feat} className="eval-inspector-item">
                  <div className="eval-inspector-label" title={feat}>{feat}</div>
                  <div className="eval-inspector-val" title={String(val)}>{String(val)}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
