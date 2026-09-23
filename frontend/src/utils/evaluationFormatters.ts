/**
 * evaluationFormatters.ts
 * Clean, human-understandable number and percentage formatters for AutoDataBot Model Evaluation.
 */

export function isPercentageMetric(metricName?: string): boolean {
  if (!metricName) return false;
  const m = metricName.toLowerCase();
  return (
    m.includes('accuracy') ||
    m.includes('f1') ||
    m.includes('precision') ||
    m.includes('recall') ||
    m.includes('auc') ||
    m.includes('r2') ||
    m.includes('variance') ||
    m.includes('mape') ||
    m.includes('pct') ||
    m.includes('rate')
  );
}

export function formatMetricScore(metricName: string, val?: number | null): string {
  if (val === undefined || val === null || isNaN(val)) return '—';

  const m = metricName.toLowerCase();

  // Negative R2 check
  if (m === 'r2' && val < 0) {
    return `${(val * 100).toFixed(1)}% (Poor Fit)`;
  }

  // Percentage-based metrics
  if (isPercentageMetric(metricName)) {
    const pct = Math.abs(val) <= 1.0 ? val * 100 : val;
    return `${pct.toFixed(1)}%`;
  }

  // Regression error metrics (MAE, RMSE, MedAE)
  if (m.includes('mae') || m.includes('rmse') || m.includes('error') || m.includes('deviation')) {
    return `±${Number(val).toFixed(2)}`;
  }

  // Generic decimal score
  if (Math.abs(val) < 0.001 && val !== 0) {
    return Number(val).toExponential(2);
  }
  return Number(val).toFixed(3);
}

export function formatPercentValue(val?: number | null): string {
  if (val === undefined || val === null || isNaN(val)) return '—';
  const pct = Math.abs(val) <= 1.0 ? val * 100 : val;
  return `${pct.toFixed(1)}%`;
}

export function getGeneralizationGapAssessment(gap?: number | null): {
  text: string;
  statusText: string;
  badgeClass: string;
} {
  if (gap === undefined || gap === null || isNaN(gap)) {
    return { text: '—', statusText: 'No data', badgeClass: 'eval-badge-indigo' };
  }

  const gapPct = Math.abs(gap) <= 1.0 ? gap * 100 : gap;
  const sign = gapPct > 0 ? `+${gapPct.toFixed(1)}%` : `${gapPct.toFixed(1)}%`;

  if (Math.abs(gapPct) <= 3.0) {
    return {
      text: sign,
      statusText: 'Excellent Generalization (Minimal Gap)',
      badgeClass: 'eval-badge-emerald'
    };
  }
  if (Math.abs(gapPct) <= 7.0) {
    return {
      text: sign,
      statusText: 'Healthy Fit (Normal Test Drift)',
      badgeClass: 'eval-badge-indigo'
    };
  }
  if (Math.abs(gapPct) <= 15.0) {
    return {
      text: sign,
      statusText: 'Moderate Drop (Watch for Overfitting)',
      badgeClass: 'eval-badge-amber'
    };
  }
  return {
    text: sign,
    statusText: 'Large Gap (Overfitting Detected)',
    badgeClass: 'eval-badge-rose'
  };
}

export function getMetricExplanatoryText(metricName: string): string {
  const m = metricName.toLowerCase();
  if (m === 'accuracy') return 'Percentage of total cases predicted correctly';
  if (m.includes('f1')) return 'Balanced harmonic score combining precision & recall';
  if (m.includes('precision')) return 'When model flags a positive, how often it is truly correct';
  if (m.includes('recall')) return 'Proportion of all actual positive cases captured by the model';
  if (m.includes('roc_auc')) return 'Separation quality between positive and negative classes';
  if (m.includes('pr_auc') || m.includes('average_precision')) return 'Precision across all decision thresholds';
  if (m === 'r2') return 'Proportion of variance explained by model (100% = perfect)';
  if (m.includes('mae')) return 'Average absolute difference between predictions and truth';
  if (m.includes('rmse')) return 'Root Mean Squared Error (heavily penalizes large outliers)';
  if (m.includes('median_absolute_error')) return 'Median absolute error (outlier-resistant)';
  if (m.includes('mape')) return 'Mean percentage error relative to actual magnitude';
  return 'Core performance metric for model validation';
}
