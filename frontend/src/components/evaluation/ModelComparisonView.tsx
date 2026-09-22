import React from 'react';
import { ComparisonSummary } from '../../types/evaluation';
import { PlotlyChart } from '../eda/PlotlyChart';
import { Info } from 'lucide-react';

interface ModelComparisonViewProps {
  comparison: ComparisonSummary;
}

export const ModelComparisonView: React.FC<ModelComparisonViewProps> = ({
  comparison
}) => {
  const {
    models = [],
    chart_data = { model_names: [], validation_scores: [], test_scores: [] },
    primary_metric = 'score'
  } = comparison || {};

  const formatScore = (val?: number | null) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(4);
  };

  const modelNames = chart_data?.model_names || (chart_data as any)?.models || [];
  const valScores = chart_data?.validation_scores || [];
  const testScores = chart_data?.test_scores || [];

  // Grouped Bar Chart Data (Validation vs Held-out Test)
  const groupedChartData = [
    {
      x: modelNames,
      y: valScores,
      name: `Validation ${primary_metric}`,
      type: 'bar',
      marker: { color: '#6366f1' }
    },
    {
      x: modelNames,
      y: testScores,
      name: `Held-out Test ${primary_metric}`,
      type: 'bar',
      marker: { color: '#10b981' }
    }
  ];

  const groupedChartLayout = {
    title: { text: `Generalization Comparison: Validation vs. Held-out Test (${primary_metric})`, font: { color: '#f8fafc', size: 14 } },
    barmode: 'group',
    xaxis: { tickfont: { color: '#cbd5e1', size: 11 }, tickangle: -25 },
    yaxis: { title: primary_metric, gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.3, font: { color: '#cbd5e1', size: 11 } },
    height: 400,
    margin: { l: 55, r: 25, t: 40, b: 90 }
  };

  return (
    <div className="eval-container">
      {/* Informational Header Notice */}
      <div className="eval-notice-box">
        <Info size={18} style={{ color: '#818cf8', flexShrink: 0, marginTop: '2px' }} />
        <div>
          <span style={{ fontWeight: 800, color: '#ffffff' }}>Generalization Diagnostics & Benchmark Stability: </span>
          This view visualizes how candidate models performed during cross-validated training compared against untouched, held-out test benchmarks.
          Phase 4 selected the final winning estimator based on generalization criteria without lookahead bias.
        </div>
      </div>

      {/* Validation vs Test Grouped Bar Chart */}
      <div className="eval-card">
        <div className="eval-card-header">
          <div>
            <h3 className="eval-card-title">Generalization Comparison: Validation vs. Held-out Test</h3>
            <p className="eval-card-desc">Comparing benchmark scores across all architectures to detect overfitting</p>
          </div>
          <span className="eval-badge eval-badge-purple">Metric: {primary_metric}</span>
        </div>
        <PlotlyChart data={groupedChartData} layout={groupedChartLayout} style={{ width: '100%', height: '400px' }} />
      </div>

      {/* Complete Comparison Table */}
      <div className="eval-card">
        <div className="eval-card-header">
          <div>
            <h3 className="eval-card-title">Candidate Models Comparison Table</h3>
            <p className="eval-card-desc">All trained candidate models sorted by validation benchmark performance</p>
          </div>
          <span className="eval-badge eval-badge-indigo">{models.length} Models Trained</span>
        </div>

        <div className="eval-table-container">
          <table className="eval-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Engine</th>
                <th>Validation {primary_metric}</th>
                <th>Held-out Test {primary_metric}</th>
                <th>Generalization Gap</th>
                <th>Training Duration</th>
                <th style={{ textAlign: 'right' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.model_id}
                  className={m.is_winner ? 'eval-row-selected' : ''}
                >
                  <td style={{ fontWeight: 700, color: m.is_winner ? '#ffffff' : '#e2e8f0' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {m.is_winner && <span title="Winning Model" style={{ fontSize: '1rem' }}>👑</span>}
                      {m.is_naive_baseline && <span title="Naive Baseline" style={{ fontSize: '1rem' }}>⚓</span>}
                      <span>{m.model_name}</span>
                    </div>
                  </td>
                  <td style={{ textTransform: 'uppercase', fontSize: '0.75rem', color: 'var(--eval-text-muted)', fontFamily: 'monospace' }}>
                    {m.engine}
                  </td>
                  <td style={{ fontWeight: 700, color: '#f8fafc', fontFamily: 'monospace' }}>
                    {formatScore(m.validation_score)}
                  </td>
                  <td style={{ fontWeight: 700, color: '#34d399', fontFamily: 'monospace' }}>
                    {formatScore(m.test_score)}
                  </td>
                  <td style={{ color: '#a5b4fc', fontFamily: 'monospace' }}>
                    {m.generalization_gap !== null && m.generalization_gap !== undefined
                      ? (m.generalization_gap > 0 ? `+${formatScore(m.generalization_gap)}` : formatScore(m.generalization_gap))
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--eval-text-muted)', fontFamily: 'monospace' }}>
                    {m.training_time_seconds !== undefined && m.training_time_seconds !== null ? `${Number(m.training_time_seconds).toFixed(2)}s` : '—'}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    {m.is_winner ? (
                      <span className="eval-badge eval-badge-amber">
                        Selected Winner
                      </span>
                    ) : m.is_naive_baseline ? (
                      <span className="eval-badge eval-badge-indigo">
                        Baseline
                      </span>
                    ) : (
                      <span className="eval-badge" style={{ background: 'rgba(255, 255, 255, 0.05)', color: '#94a3b8' }}>
                        Candidate
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
