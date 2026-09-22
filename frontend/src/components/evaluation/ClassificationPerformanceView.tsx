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
        title: cmMode === 'counts' ? 'Sample Count' : 'Row %',
        titleside: 'right',
        tickfont: { color: '#94a3b8' }
      }
    }
  ];

  const cmLayout = {
    title: {
      text: `Confusion Matrix (${cmMode === 'counts' ? 'Held-Out Sample Counts' : 'Row Normalization %'})`,
      font: { color: '#f8fafc', size: 14 }
    },
    xaxis: { title: 'Predicted Class', tickfont: { color: '#cbd5e1' }, side: 'bottom' },
    yaxis: { title: 'Actual True Class', tickfont: { color: '#cbd5e1' }, autorange: 'reversed' },
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
      name: 'Random Baseline (AUC = 0.50)',
      line: { dash: 'dash', color: '#64748b', width: 1.5 }
    },
    {
      x: rocX,
      y: rocY,
      mode: 'lines',
      name: `Trained Model (AUC = ${formatScore(roc_curve.auc)})`,
      line: { color: '#6366f1', width: 2.5 }
    }
  ] : [];

  const rocLayout = {
    title: { text: `ROC Curve (AUC: ${formatScore(roc_curve?.auc)})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'False Positive Rate (1 - Specificity)', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    yaxis: { title: 'True Positive Rate (Sensitivity / Recall)', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.25, font: { color: '#cbd5e1', size: 11 } },
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
      name: `PR Curve (Avg Prec = ${formatScore(pr_curve.average_precision)})`,
      line: { color: '#10b981', width: 2.5 },
      fill: 'tozeroy',
      fillcolor: 'rgba(16, 185, 129, 0.1)'
    }
  ] : [];

  const prLayout = {
    title: { text: `Precision-Recall Curve (AP: ${formatScore(pr_curve?.average_precision)})`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Recall', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    yaxis: { title: 'Precision', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.25, font: { color: '#cbd5e1', size: 11 } },
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
    yaxis: { title: 'Observed Fraction of Positives', range: [-0.02, 1.02], gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.25, font: { color: '#cbd5e1', size: 11 } },
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
    <div className="eval-container">
      {/* 1. Primary Metrics Ribbon */}
      <section aria-label="Key Performance Metrics">
        <div className="eval-grid-metrics">
          {Object.entries(metrics).map(([key, val]) => (
            <div key={key} className="eval-stat-card">
              <div className="eval-stat-label">
                <span>{key.replace(/_/g, ' ')}</span>
              </div>
              <div className="eval-stat-val" style={{ fontSize: '1.45rem' }}>
                {formatScore(val)}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 2. Confusion Matrix & ROC Curve */}
      <section className="eval-grid-2">
        {/* Confusion Matrix Card */}
        <div className="eval-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Confusion Matrix</h3>
              <p className="eval-card-desc">Actual vs predicted label distributions on held-out test data</p>
            </div>

            {/* Counts vs Percentages Toggle */}
            <div className="eval-segmented-control" role="group" aria-label="Confusion Matrix display mode">
              <button
                type="button"
                onClick={() => setCmMode('counts')}
                className={`eval-segmented-btn ${cmMode === 'counts' ? 'eval-segmented-btn-active' : ''}`}
              >
                Counts
              </button>
              <button
                type="button"
                onClick={() => setCmMode('percentages')}
                className={`eval-segmented-btn ${cmMode === 'percentages' ? 'eval-segmented-btn-active' : ''}`}
              >
                Percentages
              </button>
            </div>
          </div>

          <div style={{ flex: 1, minHeight: '380px' }}>
            <PlotlyChart data={cmChartData} layout={cmLayout} style={{ width: '100%', height: '380px' }} />
          </div>

          {/* Per-class Breakdown Badges */}
          {perClassItems.length > 0 && (
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '0.6rem',
              marginTop: '1rem',
              paddingTop: '0.85rem',
              borderTop: '1px solid rgba(255, 255, 255, 0.08)'
            }}>
              {perClassItems.map((cm) => (
                <div
                  key={cm.classLabel}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    background: 'rgba(10, 15, 29, 0.8)',
                    padding: '0.4rem 0.75rem',
                    borderRadius: '8px',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    fontSize: '0.75rem',
                    fontFamily: 'var(--font-mono, monospace)',
                    color: '#cbd5e1'
                  }}
                >
                  <span style={{ fontWeight: 700, color: '#818cf8' }}>Class {cm.classLabel}:</span>
                  <span>Prec <b>{formatScore(cm.precision)}</b></span>
                  <span style={{ color: '#475569' }}>|</span>
                  <span>Rec <b>{formatScore(cm.recall)}</b></span>
                  <span style={{ color: '#475569' }}>|</span>
                  <span>F1 <b>{formatScore(cm.f1)}</b></span>
                  <span style={{ color: '#64748b', fontSize: '0.7rem' }}>({cm.support ?? 0} samples)</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ROC Curve Card */}
        <div className="eval-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Receiver Operating Characteristic (ROC)</h3>
              <p className="eval-card-desc">Tradeoff between true positive rate and false positive rate</p>
            </div>
            {roc_curve?.auc !== undefined && (
              <span className="eval-badge eval-badge-indigo">
                AUC: {formatScore(roc_curve.auc)}
              </span>
            )}
          </div>

          {roc_curve ? (
            <div style={{ flex: 1, minHeight: '380px' }}>
              <PlotlyChart data={rocChartData} layout={rocLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--eval-text-dim)', fontSize: '0.875rem' }}>
              ROC curve unavailable for this model.
            </div>
          )}
        </div>
      </section>

      {/* 3. Precision-Recall Curve & Probability Calibration */}
      <section className="eval-grid-2">
        {/* Precision-Recall Curve */}
        <div className="eval-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Precision-Recall Curve</h3>
              <p className="eval-card-desc">Informative benchmark for minority and imbalanced class detection</p>
            </div>
            {pr_curve?.average_precision !== undefined && (
              <span className="eval-badge eval-badge-emerald">
                AP: {formatScore(pr_curve.average_precision)}
              </span>
            )}
          </div>

          {pr_curve ? (
            <div style={{ flex: 1, minHeight: '380px' }}>
              <PlotlyChart data={prChartData} layout={prLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--eval-text-dim)', fontSize: '0.875rem' }}>
              Precision-Recall curve unavailable for this model.
            </div>
          )}
        </div>

        {/* Probability Calibration Curve */}
        <div className="eval-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Probability Calibration Curve</h3>
              <p className="eval-card-desc">Reliability check comparing predicted probabilities with true empirical rates</p>
            </div>
            {calibration?.brier_score !== undefined && (
              <span className="eval-badge eval-badge-purple">
                Brier: {formatScore(calibration.brier_score)}
              </span>
            )}
          </div>

          {calibration ? (
            <div style={{ flex: 1, minHeight: '380px' }}>
              <PlotlyChart data={calibrationChartData} layout={calibrationLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--eval-text-dim)', fontSize: '0.875rem' }}>
              Probability calibration unavailable for this model.
            </div>
          )}
        </div>
      </section>

      {/* 4. Threshold Exploration Tool (Binary Classification) */}
      {threshold_analysis && threshold_analysis.length > 0 && (
        <section className="eval-card">
          <div className="eval-card-header">
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sliders size={18} className="text-indigo-400" />
                <h3 className="eval-card-title">Interactive Decision Threshold Exploration</h3>
              </div>
              <p className="eval-card-desc">
                Evaluate trade-offs between precision, recall, and F1 across candidate decision cutoffs.
              </p>
            </div>
            <div className="eval-badge eval-badge-amber" style={{ fontSize: '0.7rem' }}>
              Diagnostic Tool Only
            </div>
          </div>

          {/* Interactive Slider & Spotlight */}
          <div className="eval-slider-container">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '0.5rem' }}>
              <label htmlFor="threshold-slider-input" style={{ fontSize: '0.85rem', fontWeight: 700, color: '#f8fafc' }}>
                Decision Threshold: <span style={{ fontFamily: 'var(--font-mono, monospace)', color: '#818cf8', fontSize: '1rem', marginLeft: '0.35rem' }}>{selectedThreshold.toFixed(2)}</span>
              </label>
              <span style={{ fontSize: '0.75rem', color: 'var(--eval-text-dim)', fontFamily: 'var(--font-mono, monospace)' }}>
                Range: [0.05 — 0.95]
              </span>
            </div>

            <input
              id="threshold-slider-input"
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={selectedThreshold}
              onChange={(e) => setSelectedThreshold(parseFloat(e.target.value))}
              className="eval-slider"
              aria-label="Decision Threshold"
            />

            {closestThresholdPoint && (
              <div className="eval-grid-3" style={{ marginTop: '1.25rem' }}>
                <div className="eval-stat-card" style={{ textAlign: 'center' }}>
                  <div className="eval-stat-label" style={{ justifyContent: 'center' }}>Precision</div>
                  <div className="eval-stat-val eval-stat-val-green">
                    {formatScore(closestThresholdPoint.precision)}
                  </div>
                  <div className="eval-stat-subtext">True Positives / Predicted Positives</div>
                </div>

                <div className="eval-stat-card" style={{ textAlign: 'center' }}>
                  <div className="eval-stat-label" style={{ justifyContent: 'center' }}>Recall / Sensitivity</div>
                  <div className="eval-stat-val eval-stat-val-indigo">
                    {formatScore(closestThresholdPoint.recall)}
                  </div>
                  <div className="eval-stat-subtext">True Positives / Actual Positives</div>
                </div>

                <div className="eval-stat-card" style={{ textAlign: 'center' }}>
                  <div className="eval-stat-label" style={{ justifyContent: 'center' }}>F1-Score</div>
                  <div className="eval-stat-val" style={{ color: '#c084fc' }}>
                    {formatScore(closestThresholdPoint.f1)}
                  </div>
                  <div className="eval-stat-subtext">Harmonic Mean of Precision & Recall</div>
                </div>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
};
