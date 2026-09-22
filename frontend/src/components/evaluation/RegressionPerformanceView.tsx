import React from 'react';
import { RegressionPerformance } from '../../types/evaluation';
import { PlotlyChart } from '../eda/PlotlyChart';
import { Info } from 'lucide-react';

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
    diagnostics_notes = []
  } = performance || {};

  const formatScore = (val?: number | null) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(4);
  };

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

  const avpLayout = {
    title: { text: `Actual vs. Predicted (R²: ${formatScore(metrics?.r2)})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Actual Target Values', gridcolor: '#1e293b' },
    yaxis: { title: 'Predicted Model Values', gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.25, font: { color: '#cbd5e1', size: 10 } },
    height: 380,
    margin: { l: 55, r: 25, t: 40, b: 65 }
  };

  // 2. Residual Distribution Histogram
  const resList = residuals?.residuals || [];
  const resPredList = residuals?.predicted || [];

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
    title: { text: `Residual Distribution (Mean: ${formatScore(residuals?.mean)}, Std: ${formatScore(residuals?.std)})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Residual (Actual - Predicted)', gridcolor: '#1e293b' },
    yaxis: { title: 'Count', gridcolor: '#1e293b' },
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

  // 3. Residuals vs Predicted Scatter Plot
  const resVsPredData = [
    {
      x: resPredList,
      y: resList,
      mode: 'markers',
      name: 'Residuals',
      type: 'scatter',
      marker: {
        color: '#a855f7',
        size: 6,
        opacity: 0.75,
        line: { color: '#c084fc', width: 0.5 }
      }
    }
  ];

  const resVsPredLayout = {
    title: { text: 'Residuals vs. Predicted Values', font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Predicted Model Values', gridcolor: '#1e293b' },
    yaxis: { title: 'Residual (Actual - Predicted)', gridcolor: '#1e293b' },
    shapes: [
      {
        type: 'line',
        x0: minVal,
        x1: maxVal,
        y0: 0,
        y1: 0,
        line: { color: '#64748b', width: 1.5, dash: 'dash' }
      }
    ],
    height: 380,
    margin: { l: 55, r: 25, t: 40, b: 65 }
  };

  return (
    <div className="eval-container">
      {/* Primary Metrics Grid */}
      <div className="eval-grid-metrics">
        <div className="eval-stat-card">
          <span className="eval-stat-label">MAE</span>
          <span className="eval-stat-val eval-stat-val-indigo">{formatScore(metrics.mae)}</span>
          <span className="eval-stat-subtext">Mean Absolute Error</span>
        </div>
        <div className="eval-stat-card">
          <span className="eval-stat-label">RMSE</span>
          <span className="eval-stat-val eval-stat-val-indigo">{formatScore(metrics.rmse)}</span>
          <span className="eval-stat-subtext">Root Mean Squared</span>
        </div>
        <div className="eval-stat-card">
          <span className="eval-stat-label">R² Score</span>
          <span className="eval-stat-val eval-stat-val-green">{formatScore(metrics.r2)}</span>
          <span className="eval-stat-subtext">Goodness of Fit</span>
        </div>
        <div className="eval-stat-card">
          <span className="eval-stat-label">Median AE</span>
          <span className="eval-stat-val">{formatScore(metrics.median_absolute_error)}</span>
          <span className="eval-stat-subtext">Outlier Robust Error</span>
        </div>
        <div className="eval-stat-card">
          <span className="eval-stat-label">Expl. Variance</span>
          <span className="eval-stat-val">{formatScore(metrics.explained_variance)}</span>
          <span className="eval-stat-subtext">Variance Fraction</span>
        </div>
        <div className="eval-stat-card">
          <span className="eval-stat-label">MAPE</span>
          <span className="eval-stat-val">
            {metrics.mape !== null && metrics.mape !== undefined ? `${(metrics.mape * 100).toFixed(2)}%` : '—'}
          </span>
          <span className="eval-stat-subtext">Percentage Error</span>
        </div>
      </div>

      {/* Row 1: Actual vs Predicted & Residual Distribution */}
      <div className="eval-grid-2">
        {/* Actual vs Predicted */}
        <div className="eval-card">
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Actual vs. Predicted Target</h3>
              <p className="eval-card-desc">Tightly aligned points along the dashed 45° diagonal confirm high prediction accuracy</p>
            </div>
            <span className="eval-badge eval-badge-indigo">R² {formatScore(metrics?.r2)}</span>
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
            <span className="eval-badge eval-badge-purple">Mean {formatScore(residuals?.mean)}</span>
          </div>
          <PlotlyChart data={resHistData} layout={resHistLayout} style={{ width: '100%', height: '380px' }} />
        </div>
      </div>

      {/* Row 2: Residuals vs Predicted & Diagnostics */}
      <div className="eval-grid-2">
        {/* Residuals vs Predicted */}
        <div className="eval-card">
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Residuals vs. Predicted Values</h3>
              <p className="eval-card-desc">Tests for heteroscedasticity (error variance drifting across target scales)</p>
            </div>
          </div>
          <PlotlyChart data={resVsPredData} layout={resVsPredLayout} style={{ width: '100%', height: '380px' }} />
        </div>

        {/* Residual Diagnostics & Quantiles */}
        <div className="eval-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="eval-card-header">
              <div>
                <h3 className="eval-card-title">Residual Diagnostics & Quantiles</h3>
                <p className="eval-card-desc">Quantile boundaries and statistical heuristic warnings</p>
              </div>
            </div>

            {/* Quantiles summary table */}
            {residuals && (residuals as any).quantiles && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(80px, 1fr))', gap: '0.65rem', marginBottom: '1.25rem' }}>
                {Object.entries((residuals as any).quantiles).map(([q, val]) => (
                  <div key={q} className="eval-stat-card-sm">
                    <div className="label">{q}</div>
                    <div className="value">{formatScore(val as number)}</div>
                  </div>
                ))}
              </div>
            )}

            {/* Diagnostic Notes */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
              <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', fontWeight: 800, letterSpacing: '0.05em', color: 'var(--eval-text-muted)' }}>
                Diagnostic Observations
              </div>
              {diagnostics_notes.length > 0 ? (
                diagnostics_notes.map((note, idx) => (
                  <div key={idx} className="eval-notice-box">
                    <Info size={16} style={{ color: '#818cf8', flexShrink: 0, marginTop: '2px' }} />
                    <span>{note}</span>
                  </div>
                ))
              ) : (
                <div className="eval-notice-box">
                  <Info size={16} style={{ color: '#34d399', flexShrink: 0, marginTop: '2px' }} />
                  <span>No severe residual anomalies, skewness, or extreme outliers detected in the test partition.</span>
                </div>
              )}
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', paddingTop: '0.85rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)', fontSize: '0.75rem', color: 'var(--eval-text-dim)' }}>
            Statistical Notice: Diagnostics report automated heuristic flags on held-out test data and do not substitute for custom domain checks.
          </div>
        </div>
      </div>
    </div>
  );
};
