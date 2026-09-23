import React from 'react';
import { RegressionPerformance } from '../../types/evaluation';
import { PlotlyChart } from '../eda/PlotlyChart';
import { CheckCircle2, TrendingUp } from 'lucide-react';
import { formatMetricScore, formatPercentValue } from '../../utils/evaluationFormatters';

interface RegressionPerformanceViewProps {
  performance: RegressionPerformance;
}

export const RegressionPerformanceView: React.FC<RegressionPerformanceViewProps> = ({
  performance
}) => {
  const {
    metrics = {} as any,
    actual_vs_predicted = { actual: [], predicted: [] },
    residuals = { residuals: [], predicted: [], mean: 0, std: 0 },
    diagnostics_notes = [],
    human_summary,
    accuracy_within_10_pct,
    accuracy_within_20_pct,
    r2_pct
  } = performance || {};

  // 1. Actual vs Predicted Scatter Plot
  const actList = actual_vs_predicted?.actual || [];
  const predList = actual_vs_predicted?.predicted || [];
  const minVal = (actList.length > 0 && predList.length > 0)
    ? Math.min(...actList, ...predList)
    : 0;
  const maxVal = (actList.length > 0 && predList.length > 0)
    ? Math.max(...actList, ...predList)
    : 1;

  const avpChartData = [
    {
      x: [minVal, maxVal],
      y: [minVal, maxVal],
      mode: 'lines',
      name: 'Ideal Fit (Actual = Predicted)',
      line: { dash: 'dash', color: '#64748b', width: 2 }
    },
    {
      x: actList,
      y: predList,
      mode: 'markers',
      name: 'Held-out Test Samples',
      type: 'scatter',
      marker: {
        color: '#6366f1',
        size: 6,
        opacity: 0.75,
        line: { color: '#818cf8', width: 0.5 }
      }
    }
  ];

  const r2Display = r2_pct || formatMetricScore('r2', metrics?.r2);

  const avpLayout = {
    title: { text: `Actual vs. Predicted Target Fit (R²: ${r2Display})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Actual Target Values', gridcolor: '#1e293b' },
    yaxis: { title: 'Predicted Model Values', gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.25, font: { color: '#cbd5e1', size: 10 } },
    height: 380,
    margin: { l: 55, r: 25, t: 40, b: 65 }
  };

  // 2. Residual Distribution Histogram
  const resList = residuals?.residuals || [];

  const resHistData = [
    {
      x: resList,
      type: 'histogram',
      nbinsx: 25,
      name: 'Residual Frequency',
      marker: {
        color: '#3b82f6',
        line: { color: '#1e3a8a', width: 1 }
      }
    }
  ];

  const resHistLayout = {
    title: { text: `Residual Error Distribution (Mean: ±${Number(residuals?.mean || 0).toFixed(2)}, Std: ±${Number(residuals?.std || 0).toFixed(2)})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Residual Error (Actual - Predicted)', gridcolor: '#1e293b' },
    yaxis: { title: 'Sample Count', gridcolor: '#1e293b' },
    shapes: [
      {
        type: 'line',
        x0: 0,
        x1: 0,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: { color: '#ef4444', width: 2, dash: 'dot' }
      }
    ],
    height: 380,
    margin: { l: 55, r: 25, t: 40, b: 65 }
  };

  // Plain-Language Interpretation Text
  const acc10 = accuracy_within_10_pct !== undefined && accuracy_within_10_pct !== null
    ? formatPercentValue(accuracy_within_10_pct)
    : (human_summary?.accuracy_within_10_pct || null);

  const acc20 = accuracy_within_20_pct !== undefined && accuracy_within_20_pct !== null
    ? formatPercentValue(accuracy_within_20_pct)
    : (human_summary?.accuracy_within_20_pct || null);

  const plainSummaryText = human_summary?.r2_explained
    ? `${human_summary.r2_explained}. Average margin of error is ${human_summary.average_error_mae || formatMetricScore('mae', metrics.mae)}.`
    : `This model explains ${r2Display} of all variance in the target variable with an average margin of error of ${formatMetricScore('mae', metrics.mae)}.`;

  return (
    <div className="eval-container">
      {/* Plain-Language Performance Summary Banner */}
      <div className="eval-notice-box" style={{ background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.25)', padding: '1rem 1.25rem' }}>
        <CheckCircle2 size={20} className="text-emerald-400" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 800, color: '#f8fafc', fontSize: '0.95rem' }}>
              Plain-Language Regression Accuracy Summary
            </span>
            <span className="eval-badge eval-badge-emerald">
              Fit Quality (R²): {r2Display}
            </span>
            {acc10 && (
              <span className="eval-badge eval-badge-indigo">
                ±10% Accuracy Rate: {acc10}
              </span>
            )}
          </div>
          <div style={{ color: '#cbd5e1', fontSize: '0.85rem', lineHeight: '1.4' }}>
            {plainSummaryText}
            {acc10 && ` Furthermore, ${acc10} of all test predictions fall within a strict ±10% margin of error.`}
          </div>
        </div>
      </div>

      {/* Primary Metrics Grid with Refined Plain-Language Values */}
      <div className="eval-grid-metrics">
        {/* R² Score */}
        <div className="eval-stat-card" style={{ borderLeft: '4px solid #10b981' }}>
          <span className="eval-stat-label">Model Fit Quality (R²)</span>
          <span className="eval-stat-val eval-stat-val-green">{r2Display}</span>
          <span className="eval-stat-subtext">Percentage of target variation explained</span>
        </div>

        {/* MAE */}
        <div className="eval-stat-card" style={{ borderLeft: '4px solid #6366f1' }}>
          <span className="eval-stat-label">Average Error (MAE)</span>
          <span className="eval-stat-val eval-stat-val-indigo">{formatMetricScore('mae', metrics.mae)}</span>
          <span className="eval-stat-subtext">Typical difference from actual values</span>
        </div>

        {/* RMSE */}
        <div className="eval-stat-card">
          <span className="eval-stat-label">Outlier Penalty (RMSE)</span>
          <span className="eval-stat-val">{formatMetricScore('rmse', metrics.rmse)}</span>
          <span className="eval-stat-subtext">Error sensitive to large misses</span>
        </div>

        {/* Accuracy within 10% */}
        {acc10 && (
          <div className="eval-stat-card">
            <span className="eval-stat-label">Accuracy (±10% Margin)</span>
            <span className="eval-stat-val eval-stat-val-green">{acc10}</span>
            <span className="eval-stat-subtext">Predictions within 10% tolerance</span>
          </div>
        )}

        {/* Accuracy within 20% */}
        {acc20 && (
          <div className="eval-stat-card">
            <span className="eval-stat-label">Accuracy (±20% Margin)</span>
            <span className="eval-stat-val">{acc20}</span>
            <span className="eval-stat-subtext">Predictions within 20% tolerance</span>
          </div>
        )}

        {/* MAPE Percentage Error */}
        <div className="eval-stat-card">
          <span className="eval-stat-label">Percentage Error (MAPE)</span>
          <span className="eval-stat-val">
            {metrics.mape !== null && metrics.mape !== undefined ? `${(metrics.mape * 100).toFixed(1)}%` : '—'}
          </span>
          <span className="eval-stat-subtext">Relative percentage deviation</span>
        </div>
      </div>

      {/* Row 1: Actual vs Predicted & Residual Distribution */}
      <div className="eval-grid-2">
        {/* Actual vs Predicted */}
        <div className="eval-card">
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Actual vs. Predicted Target</h3>
              <p className="eval-card-desc">Tightly aligned points along the dashed diagonal confirm high prediction accuracy</p>
            </div>
            <span className="eval-badge eval-badge-emerald">R² {r2Display}</span>
          </div>
          <PlotlyChart data={avpChartData} layout={avpLayout} style={{ width: '100%', height: '380px' }} />
        </div>

        {/* Residual Distribution */}
        <div className="eval-card">
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Residual Distribution Histogram</h3>
              <p className="eval-card-desc">Unbiased model errors should center around zero with a bell-shaped distribution</p>
            </div>
            <span className="eval-badge eval-badge-purple">Mean ±{Number(residuals?.mean || 0).toFixed(2)}</span>
          </div>
          <PlotlyChart data={resHistData} layout={resHistLayout} style={{ width: '100%', height: '380px' }} />
        </div>
      </div>

      {/* Explanatory Diagnostics Notes */}
      {diagnostics_notes && diagnostics_notes.length > 0 && (
        <div className="eval-notice-box" style={{ marginTop: '1.25rem' }}>
          <TrendingUp size={16} className="text-emerald-400" style={{ marginTop: '0.15rem', flexShrink: 0 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontWeight: 700, color: '#f8fafc', fontSize: '0.85rem' }}>Regression Reliability Insights:</span>
            {diagnostics_notes.map((note, idx) => (
              <div key={idx} style={{ color: '#cbd5e1', fontSize: '0.8rem' }}>• {note}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
