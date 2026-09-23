import React from 'react';
import { ComparisonSummary } from '../../types/evaluation';
import { PlotlyChart } from '../eda/PlotlyChart';
import { Sparkles, CheckCircle2 } from 'lucide-react';
import { formatMetricScore, getGeneralizationGapAssessment, isPercentageMetric } from '../../utils/evaluationFormatters';

interface ModelComparisonViewProps {
  comparison: ComparisonSummary;
}

export const ModelComparisonView: React.FC<ModelComparisonViewProps> = ({
  comparison
}) => {
  const {
    models = [],
    chart_data = { validation_scores: [], test_scores: [] },
    primary_metric = 'score'
  } = comparison || {};

  const modelNames = chart_data?.model_names || (chart_data as any)?.models || [];
  const valScores = chart_data?.validation_scores || [];
  const trainScores = chart_data?.train_scores || [];
  const testScores = chart_data?.test_scores || [];

  const isPct = isPercentageMetric(primary_metric);

  // Grouped Bar Chart Data (Training vs Validation vs Held-out Test)
  const groupedChartData: any[] = [];

  if (trainScores.length > 0 && trainScores.some(s => s !== null && s !== undefined)) {
    groupedChartData.push({
      x: modelNames,
      y: trainScores.map(s => (s !== null && s !== undefined ? (isPct && Math.abs(s) <= 1.0 ? Number((s * 100).toFixed(1)) : s) : null)),
      name: `Training ${primary_metric}${isPct ? ' (%)' : ''}`,
      type: 'bar',
      marker: { color: '#818cf8' }
    });
  }

  groupedChartData.push(
    {
      x: modelNames,
      y: valScores.map(s => (s !== null && s !== undefined ? (isPct && Math.abs(s) <= 1.0 ? Number((s * 100).toFixed(1)) : s) : null)),
      name: `Validation ${primary_metric}${isPct ? ' (%)' : ''}`,
      type: 'bar',
      marker: { color: '#6366f1' }
    },
    {
      x: modelNames,
      y: testScores.map(s => (s !== null && s !== undefined ? (isPct && Math.abs(s) <= 1.0 ? Number((s * 100).toFixed(1)) : s) : null)),
      name: `Held-out Test ${primary_metric}${isPct ? ' (%)' : ''}`,
      type: 'bar',
      marker: { color: '#10b981' }
    }
  );

  const groupedChartLayout = {
    title: { text: `Model Comparison: Training, Validation & Held-out Test (${primary_metric})`, font: { color: '#f8fafc', size: 14 } },
    barmode: 'group',
    xaxis: { tickfont: { color: '#cbd5e1', size: 11 }, tickangle: -25 },
    yaxis: { title: isPct ? `${primary_metric} (%)` : primary_metric, gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: -0.35, font: { color: '#cbd5e1', size: 11 } },
    height: 420,
    margin: { l: 55, r: 25, t: 40, b: 100 }
  };

  return (
    <div className="eval-container">
      {/* Informational Header Notice */}
      <div className="eval-notice-box" style={{ background: 'rgba(99, 102, 241, 0.08)', borderColor: 'rgba(99, 102, 241, 0.25)' }}>
        <Sparkles size={18} style={{ color: '#818cf8', flexShrink: 0, marginTop: '2px' }} />
        <div>
          <span style={{ fontWeight: 800, color: '#ffffff' }}>Authenticity Benchmark & Iterative Model Tuning: </span>
          Models were trained and tuned targeting the <b style={{ color: '#34d399' }}>&gt;= 80% authenticity threshold</b>.
          Comparing Training vs Testing metrics directly exposes whether candidate architectures achieved true generalization without overfitting.
        </div>
      </div>

      {/* Validation vs Test Grouped Bar Chart */}
      <div className="eval-card">
        <div className="eval-card-header">
          <div>
            <h3 className="eval-card-title">Generalization Comparison Across Architectures</h3>
            <p className="eval-card-desc">Training vs Validation vs Held-Out Test performance</p>
          </div>
          <span className="eval-badge eval-badge-purple">Metric: {primary_metric}</span>
        </div>
        <PlotlyChart data={groupedChartData} layout={groupedChartLayout} style={{ width: '100%', height: '420px' }} />
      </div>

      {/* Complete Comparison Table */}
      <div className="eval-card">
        <div className="eval-card-header">
          <div>
            <h3 className="eval-card-title">Candidate Models Leaderboard & Tuning Rounds</h3>
            <p className="eval-card-desc">All candidate estimators evaluated across tuning rounds and test partitions</p>
          </div>
          <span className="eval-badge eval-badge-indigo">{models.length} Models Evaluated</span>
        </div>

        <div className="eval-table-container">
          <table className="eval-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Tuning Stage</th>
                <th>Engine</th>
                <th>Training {primary_metric}</th>
                <th>Validation {primary_metric}</th>
                <th>Held-out Test {primary_metric}</th>
                <th>Generalization Gap</th>
                <th>Duration</th>
                <th style={{ textAlign: 'right' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const gapObj = getGeneralizationGapAssessment(m.generalization_gap);
                const testMeetsBenchmark = m.test_score !== null && m.test_score !== undefined && m.test_score >= 0.80;

                return (
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

                    <td>
                      {m.tuning_round && m.tuning_round > 1 ? (
                        <span className="eval-badge eval-badge-indigo" style={{ fontSize: '0.65rem' }}>
                          Round {m.tuning_round} ({m.tuning_stage || 'Tuned'})
                        </span>
                      ) : (
                        <span className="eval-badge eval-badge-dim" style={{ fontSize: '0.65rem' }}>
                          Round 1 (Initial)
                        </span>
                      )}
                    </td>

                    <td style={{ textTransform: 'uppercase', fontSize: '0.75rem', color: 'var(--eval-text-muted)', fontFamily: 'monospace' }}>
                      {m.engine}
                    </td>

                    <td style={{ fontWeight: 600, color: '#818cf8', fontFamily: 'monospace' }}>
                      {formatMetricScore(primary_metric, m.train_score)}
                    </td>

                    <td style={{ fontWeight: 600, color: '#f8fafc', fontFamily: 'monospace' }}>
                      {formatMetricScore(primary_metric, m.validation_score)}
                    </td>

                    <td style={{ fontWeight: 700, color: testMeetsBenchmark ? '#34d399' : '#38bdf8', fontFamily: 'monospace' }}>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                        {formatMetricScore(primary_metric, m.test_score)}
                        {testMeetsBenchmark && <CheckCircle2 size={12} className="text-emerald-400" />}
                      </span>
                    </td>

                    <td style={{ color: '#a5b4fc', fontFamily: 'monospace' }}>
                      <span className={`eval-badge ${gapObj.badgeClass}`} style={{ fontSize: '0.7rem' }}>
                        {gapObj.text}
                      </span>
                    </td>

                    <td style={{ color: 'var(--eval-text-muted)', fontFamily: 'monospace', fontSize: '0.75rem' }}>
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
                        <span style={{ fontSize: '0.75rem', color: 'var(--eval-text-dim)' }}>
                          Candidate
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
