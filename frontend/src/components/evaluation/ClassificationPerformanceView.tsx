import React, { useState } from 'react';
import { ClassificationPerformance } from '../../types/evaluation';
import { PlotlyChart } from '../eda/PlotlyChart';
import { Sliders } from 'lucide-react';

interface ClassificationPerformanceViewProps {
  performance: ClassificationPerformance;
}

export const ClassificationPerformanceView: React.FC<ClassificationPerformanceViewProps> = ({
  performance
}) => {
  const [cmMode, setCmMode] = useState<'counts' | 'percentages'>('counts');
  const [selectedThreshold, setSelectedThreshold] = useState<number>(0.5);

  const {
    metrics = {},
    confusion_matrix,
    per_class_metrics,
    roc_curve,
    pr_curve,
    threshold_analysis,
    calibration
  } = performance || {};

  const formatScore = (val?: number | null) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(4);
  };

  // Safe Confusion Matrix Heatmap Data
  const cmMatrix = confusion_matrix?.matrix || [];
  const cmNorm = confusion_matrix?.matrix_normalized || (confusion_matrix as any)?.normalized_matrix || cmMatrix;
  const cmLabels = confusion_matrix?.labels || [];

  const cmValues = cmMode === 'counts'
    ? cmMatrix
    : cmNorm.map((row: number[]) => (row || []).map((v: number) => Number(((v || 0) * 100).toFixed(1))));

  const cmText = cmMode === 'counts'
    ? cmMatrix.map((row: number[]) => (row || []).map((v: number) => `${v}`))
    : cmNorm.map((row: number[], i: number) =>
        (row || []).map((v: number, j: number) => `${((v || 0) * 100).toFixed(1)}%\n(${cmMatrix[i]?.[j] ?? 0})`)
      );

  const cmChartData = [
    {
      z: cmValues,
      x: cmLabels.map(l => `Pred: ${l}`),
      y: cmLabels.map(l => `Act: ${l}`),
      text: cmText,
      texttemplate: '%{text}',
      textfont: { color: '#ffffff', size: 14, family: 'monospace' },
      type: 'heatmap',
      hoverongaps: false,
      colorscale: [
        [0, '#0f172a'],
        [0.5, '#3b82f6'],
        [1, '#6366f1']
      ],
      showscale: true,
      colorbar: {
        title: cmMode === 'counts' ? 'Count' : 'Percent (%)',
        titleside: 'right',
        tickfont: { color: '#94a3b8' }
      }
    }
  ];

  const cmLayout = {
    title: { text: `Confusion Matrix (${cmMode === 'counts' ? 'Sample Counts' : 'Row Percentages'})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Predicted Label', tickfont: { color: '#cbd5e1' }, side: 'bottom' },
    yaxis: { title: 'Actual Label', tickfont: { color: '#cbd5e1' }, autorange: 'reversed' },
    height: 380,
    margin: { l: 80, r: 40, t: 40, b: 60 }
  };

  // Safe ROC Curve Data
  const rocX = roc_curve?.x || (roc_curve as any)?.fpr || [];
  const rocY = roc_curve?.y || (roc_curve as any)?.tpr || [];

  const rocChartData = roc_curve && rocX.length > 0 ? [
    {
      x: [0, 1],
      y: [0, 1],
      mode: 'lines',
      name: 'Random Chance (AUC = 0.50)',
      line: { dash: 'dash', color: '#64748b', width: 1.5 }
    },
    {
      x: rocX,
      y: rocY,
      mode: 'lines',
      name: `Model ROC (AUC = ${formatScore(roc_curve.auc)})`,
      line: { color: '#6366f1', width: 2.5 }
    }
  ] : [];

  const rocLayout = {
    title: { text: `ROC Curve (AUC: ${formatScore(roc_curve?.auc)})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'False Positive Rate (1 - Specificity)', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    yaxis: { title: 'True Positive Rate (Sensitivity / Recall)', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.25, font: { color: '#cbd5e1', size: 10 } },
    height: 380,
    margin: { l: 55, r: 25, t: 40, b: 65 }
  };

  // Safe PR Curve Data
  const prX = pr_curve?.x || (pr_curve as any)?.recall || [];
  const prY = pr_curve?.y || (pr_curve as any)?.precision || [];

  const prChartData = pr_curve && prX.length > 0 ? [
    {
      x: prX,
      y: prY,
      mode: 'lines',
      name: `PR Curve (AP = ${formatScore(pr_curve.average_precision)})`,
      line: { color: '#10b981', width: 2.5 },
      fill: 'tozeroy',
      fillcolor: 'rgba(16, 185, 129, 0.1)'
    }
  ] : [];

  const prLayout = {
    title: { text: `Precision-Recall Curve (AP: ${formatScore(pr_curve?.average_precision)})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Recall', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    yaxis: { title: 'Precision', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.25, font: { color: '#cbd5e1', size: 10 } },
    height: 380,
    margin: { l: 55, r: 25, t: 40, b: 65 }
  };

  // Safe Calibration Curve Data
  const calPred = calibration?.prob_pred || (calibration as any)?.prob_predicted || [];
  const calTrue = calibration?.prob_true || [];

  const calibrationChartData = calibration && calPred.length > 0 ? [
    {
      x: [0, 1],
      y: [0, 1],
      mode: 'lines',
      name: 'Perfect Calibration',
      line: { dash: 'dash', color: '#64748b', width: 1.5 }
    },
    {
      x: calPred,
      y: calTrue,
      mode: 'lines+markers',
      name: `Model Calibration (Brier = ${formatScore(calibration.brier_score)})`,
      line: { color: '#a855f7', width: 2 },
      marker: { size: 6, color: '#c084fc' }
    }
  ] : [];

  const calibrationLayout = {
    title: { text: `Probability Calibration (Brier: ${formatScore(calibration?.brier_score)})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Mean Predicted Probability', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    yaxis: { title: 'Fraction of Positives', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.25, font: { color: '#cbd5e1', size: 10 } },
    height: 380,
    margin: { l: 55, r: 25, t: 40, b: 65 }
  };

  // Safe per-class metrics handling
  const perClassItems: Array<{ classLabel: string; precision?: number; recall?: number; f1?: number; support?: number }> = [];
  if (per_class_metrics && typeof per_class_metrics === 'object') {
    Object.entries(per_class_metrics).forEach(([cls, cm]) => {
      perClassItems.push({
        classLabel: cls,
        precision: (cm as any)?.precision,
        recall: (cm as any)?.recall,
        f1: (cm as any)?.f1,
        support: (cm as any)?.support
      });
    });
  } else if (Array.isArray((performance as any)?.per_class)) {
    ((performance as any).per_class as any[]).forEach((item: any) => {
      perClassItems.push({
        classLabel: item?.class_label !== undefined ? String(item.class_label) : (item?.class !== undefined ? String(item.class) : ''),
        precision: item?.precision,
        recall: item?.recall,
        f1: item?.f1,
        support: item?.support
      });
    });
  }

  // Find closest threshold row for slider
  const closestThresholdPoint = (threshold_analysis && threshold_analysis.length > 0)
    ? threshold_analysis.reduce((prev, curr) =>
        Math.abs(curr.threshold - selectedThreshold) < Math.abs(prev.threshold - selectedThreshold) ? curr : prev
      , threshold_analysis[0])
    : null;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
        {Object.entries(metrics).map(([key, val]) => (
          <div key={key} className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 text-center">
            <div className="text-[11px] font-mono text-slate-400 uppercase truncate">
              {key.replace('_', ' ')}
            </div>
            <div className="text-xl font-mono font-bold text-white mt-1">
              {formatScore(val)}
            </div>
          </div>
        ))}
      </div>

      {/* Row 1: Confusion Matrix & ROC Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
            <div>
              <h3 className="text-base font-bold text-white">Confusion Matrix</h3>
              <p className="text-xs text-slate-400">Actual vs predicted label distributions on held-out test data</p>
            </div>
            <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
              <button
                onClick={() => setCmMode('counts')}
                className={`px-2.5 py-1 rounded font-medium transition-colors ${
                  cmMode === 'counts' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Counts
              </button>
              <button
                onClick={() => setCmMode('percentages')}
                className={`px-2.5 py-1 rounded font-medium transition-colors ${
                  cmMode === 'percentages' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Percentages
              </button>
            </div>
          </div>

          <div className="flex-1">
            <PlotlyChart data={cmChartData} layout={cmLayout} style={{ width: '100%', height: '380px' }} />
          </div>

          {/* Per-class breakdown badges */}
          {perClassItems.length > 0 && (
            <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap gap-2 text-xs">
              {perClassItems.map((cm) => (
                <div key={cm.classLabel} className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300 flex items-center gap-2">
                  <span className="font-bold text-indigo-400">Class {cm.classLabel}:</span>
                  <span>Prec {formatScore(cm.precision)}</span>
                  <span className="text-slate-600">|</span>
                  <span>Rec {formatScore(cm.recall)}</span>
                  <span className="text-slate-600">|</span>
                  <span>F1 {formatScore(cm.f1)}</span>
                  <span className="text-slate-500 text-[10px]">({cm.support ?? 0} samples)</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ROC Curve Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="pb-3 border-b border-slate-800 mb-4">
            <h3 className="text-base font-bold text-white">Receiver Operating Characteristic (ROC)</h3>
            <p className="text-xs text-slate-400">Tradeoff between true positive rate and false positive rate</p>
          </div>
          {roc_curve ? (
            <div className="flex-1">
              <PlotlyChart data={rocChartData} layout={rocLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
              ROC curve unavailable for this model (no probability predictions available).
            </div>
          )}
        </div>
      </div>

      {/* Row 2: Precision-Recall Curve & Probability Calibration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Precision-Recall Curve */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="pb-3 border-b border-slate-800 mb-4">
            <h3 className="text-base font-bold text-white">Precision-Recall Curve</h3>
            <p className="text-xs text-slate-400">Particularly informative for imbalanced positive class detection</p>
          </div>
          {pr_curve ? (
            <div className="flex-1">
              <PlotlyChart data={prChartData} layout={prLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
              Precision-Recall curve unavailable for this model.
            </div>
          )}
        </div>

        {/* Probability Calibration Curve */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="pb-3 border-b border-slate-800 mb-4">
            <h3 className="text-base font-bold text-white">Probability Calibration Curve</h3>
            <p className="text-xs text-slate-400">Assesses whether predicted probabilities reflect empirical frequencies</p>
          </div>
          {calibration ? (
            <div className="flex-1">
              <PlotlyChart data={calibrationChartData} layout={calibrationLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
              Probability calibration unavailable for this model.
            </div>
          )}
        </div>
      </div>

      {/* Threshold Exploration Tool (Binary Classification Only) */}
      {threshold_analysis && threshold_analysis.length > 0 && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800 mb-6">
            <div>
              <div className="flex items-center gap-2">
                <Sliders size={18} className="text-indigo-400" />
                <h3 className="text-base font-bold text-white">Classification Threshold Analysis</h3>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Explore how selecting different decision thresholds impacts precision, recall, and F1.
              </p>
            </div>
            <div className="p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-xl text-xs text-amber-300 max-w-md">
              <span className="font-bold">⚠️ Diagnostic Analysis Only:</span> This tool is for tradeoff inspection. The final model decision boundary has not been modified.
            </div>
          </div>

          {/* Interactive Slider & Spotlight */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5 mb-6">
            <div className="flex items-center justify-between gap-4 mb-4">
              <label htmlFor="threshold-range" className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Decision Threshold: <span className="font-mono text-indigo-400 text-sm">{selectedThreshold.toFixed(2)}</span>
              </label>
              <span className="text-xs text-slate-500 font-mono">Range: [0.05 — 0.95]</span>
            </div>

            <input
              id="threshold-range"
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={selectedThreshold}
              onChange={(e) => setSelectedThreshold(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg mb-6"
            />

            {closestThresholdPoint && (
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
                  <div className="text-xs text-slate-400 uppercase font-mono">Precision</div>
                  <div className="text-2xl font-mono font-bold text-emerald-400 mt-1">
                    {formatScore(closestThresholdPoint.precision)}
                  </div>
                </div>
                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
                  <div className="text-xs text-slate-400 uppercase font-mono">Recall</div>
                  <div className="text-2xl font-mono font-bold text-blue-400 mt-1">
                    {formatScore(closestThresholdPoint.recall)}
                  </div>
                </div>
                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
                  <div className="text-xs text-slate-400 uppercase font-mono">F1-Score</div>
                  <div className="text-2xl font-mono font-bold text-purple-400 mt-1">
                    {formatScore(closestThresholdPoint.f1)}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
