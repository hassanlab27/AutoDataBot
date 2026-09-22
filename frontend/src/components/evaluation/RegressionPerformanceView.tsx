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
    <div className="space-y-8 animate-fadeIn">
      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 text-center">
          <div className="text-[11px] font-mono text-slate-400 uppercase">MAE</div>
          <div className="text-xl font-mono font-bold text-white mt-1">{formatScore(metrics.mae)}</div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 text-center">
          <div className="text-[11px] font-mono text-slate-400 uppercase">RMSE</div>
          <div className="text-xl font-mono font-bold text-white mt-1">{formatScore(metrics.rmse)}</div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 text-center">
          <div className="text-[11px] font-mono text-slate-400 uppercase">R² Score</div>
          <div className="text-xl font-mono font-bold text-white mt-1">{formatScore(metrics.r2)}</div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 text-center">
          <div className="text-[11px] font-mono text-slate-400 uppercase">Median AE</div>
          <div className="text-xl font-mono font-bold text-white mt-1">{formatScore(metrics.median_absolute_error)}</div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 text-center">
          <div className="text-[11px] font-mono text-slate-400 uppercase">Expl. Var.</div>
          <div className="text-xl font-mono font-bold text-white mt-1">{formatScore(metrics.explained_variance)}</div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 text-center">
          <div className="text-[11px] font-mono text-slate-400 uppercase">MAPE</div>
          <div className="text-xl font-mono font-bold text-white mt-1">
            {metrics.mape !== null && metrics.mape !== undefined ? `${(metrics.mape * 100).toFixed(2)}%` : '—'}
          </div>
        </div>
      </div>

      {/* Row 1: Actual vs Predicted & Residual Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Actual vs Predicted */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="pb-3 border-b border-slate-800 mb-4">
            <h3 className="text-base font-bold text-white">Actual vs. Predicted Target</h3>
            <p className="text-xs text-slate-400">Samples clustering tightly along the diagonal indicate strong calibration</p>
          </div>
          <div className="flex-1">
            <PlotlyChart data={avpChartData} layout={avpLayout} style={{ width: '100%', height: '380px' }} />
          </div>
        </div>

        {/* Residual Distribution */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="pb-3 border-b border-slate-800 mb-4">
            <h3 className="text-base font-bold text-white">Residual Distribution Histogram</h3>
            <p className="text-xs text-slate-400">Unbiased errors should center around zero without strong skew</p>
          </div>
          <div className="flex-1">
            <PlotlyChart data={resHistData} layout={resHistLayout} style={{ width: '100%', height: '380px' }} />
          </div>
        </div>
      </div>

      {/* Row 2: Residuals vs Predicted & Diagnostics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Residuals vs Predicted */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="pb-3 border-b border-slate-800 mb-4">
            <h3 className="text-base font-bold text-white">Residuals vs. Predicted Values</h3>
            <p className="text-xs text-slate-400">Checks for heteroscedasticity (variance changes across prediction magnitudes)</p>
          </div>
          <div className="flex-1">
            <PlotlyChart data={resVsPredData} layout={resVsPredLayout} style={{ width: '100%', height: '380px' }} />
          </div>
        </div>

        {/* Residual Diagnostics & Quantiles */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="pb-3 border-b border-slate-800 mb-4">
              <h3 className="text-base font-bold text-white">Residual Diagnostics & Quantiles</h3>
              <p className="text-xs text-slate-400">Heuristic pattern indicators on test error distributions</p>
            </div>

            {/* Quantiles summary table */}
            {residuals && (residuals as any).quantiles && (
              <div className="grid grid-cols-5 gap-2 text-center mb-6">
                {Object.entries((residuals as any).quantiles).map(([q, val]) => (
                  <div key={q} className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                    <div className="text-[10px] font-mono text-slate-400 uppercase">{q}</div>
                    <div className="text-sm font-mono font-bold text-slate-200 mt-0.5">{formatScore(val as number)}</div>
                  </div>
                ))}
              </div>
            )}

            {/* Diagnostic Notes */}
            <div className="space-y-2">
              <div className="text-xs uppercase font-bold text-slate-400 tracking-wider">Pattern Notes</div>
              {diagnostics_notes.map((note, idx) => (
                <div key={idx} className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl text-xs text-slate-300 flex items-start gap-2.5">
                  <Info size={14} className="text-indigo-400 mt-0.5 shrink-0" />
                  <span>{note}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500">
            Statistical Note: Diagnostics flag heuristic signals and do not constitute formal mathematical proof of assumption violation.
          </div>
        </div>
      </div>
    </div>
  );
};
