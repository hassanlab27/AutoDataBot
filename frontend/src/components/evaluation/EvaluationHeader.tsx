import React from 'react';
import { EvaluationOverview } from '../../types/evaluation';
import { Clock, Zap, Award, Sparkles, AlertTriangle, CheckCircle2, ShieldAlert, Target } from 'lucide-react';
import { formatMetricScore, getGeneralizationGapAssessment } from '../../utils/evaluationFormatters';

interface EvaluationHeaderProps {
  evaluation: EvaluationOverview;
  runs: any[];
  activeRunId: string;
  onSelectRun: (runId: string) => void;
}

export const EvaluationHeader: React.FC<EvaluationHeaderProps> = ({
  evaluation,
  runs,
  activeRunId,
  onSelectRun
}) => {
  const getDiagnosticBadgeDetails = (label?: string) => {
    switch (label) {
      case 'Normal / Well-fit':
      case 'No obvious gap':
        return {
          className: 'eval-badge eval-badge-emerald',
          icon: <CheckCircle2 size={13} />,
          text: label || 'Normal / Well-fit'
        };
      case 'Moderate generalization gap':
      case 'Potential overfitting indicator':
      case 'Possible Overfitting':
        return {
          className: 'eval-badge eval-badge-amber',
          icon: <AlertTriangle size={13} />,
          text: label || 'Possible Overfitting'
        };
      case 'Large generalization gap':
      case 'Severe Overfitting':
        return {
          className: 'eval-badge eval-badge-rose',
          icon: <ShieldAlert size={13} />,
          text: label || 'Severe Overfitting'
        };
      case 'Potential underfitting indicator':
      case 'Underfitting':
        return {
          className: 'eval-badge eval-badge-amber',
          icon: <AlertTriangle size={13} />,
          text: label || 'Underfitting'
        };
      case 'Suspiciously High Performance':
        return {
          className: 'eval-badge eval-badge-purple',
          icon: <Sparkles size={13} />,
          text: 'Near-Perfect Score (Potential Leakage)'
        };
      default:
        return {
          className: 'eval-badge eval-badge-indigo',
          icon: <Award size={13} />,
          text: label || 'Evaluated Model'
        };
    }
  };

  const modelName = evaluation.model_name || (evaluation as any).winning_model?.model_name || 'Model';
  const engine = evaluation.engine || (evaluation as any).winning_model?.engine || 'sklearn';
  const problemType = evaluation.problem_type || (evaluation as any).winning_model?.problem_type || 'classification';
  const primaryMetric = evaluation.primary_metric || 'accuracy';

  const trainScore = evaluation.train_score ?? evaluation.winning_model?.train_score ?? evaluation.diagnostics?.train_score;
  const valScore = evaluation.validation_score ?? evaluation.winning_model?.validation_score;
  const testScore = evaluation.test_score ?? evaluation.winning_model?.test_score;
  const genGap = evaluation.generalization_gap ?? evaluation.winning_model?.generalization_gap;
  const trainTime = evaluation.training_time_seconds ?? evaluation.winning_model?.training_time_seconds;
  const predTime = evaluation.prediction_time_seconds ?? evaluation.winning_model?.prediction_time_seconds;

  const tuningHistory = evaluation.tuning_history ?? evaluation.winning_model?.tuning_history;
  const isBenchmarkAchieved = (testScore !== null && testScore !== undefined && testScore >= 0.80) ||
                             (valScore !== null && valScore !== undefined && valScore >= 0.80) ||
                             (tuningHistory?.target_achieved ?? false);

  const diagBadge = getDiagnosticBadgeDetails(evaluation.diagnostics?.label);
  const gapAssessment = getGeneralizationGapAssessment(genGap);

  return (
    <header className="eval-hero-header" aria-label="Model Evaluation Overview">
      {/* Decorative ambient lighting */}
      <div className="eval-card-glow" />

      {/* Top Banner: Winner & Run Selector */}
      <div className="eval-hero-top">
        <div className="eval-title-group">
          <div className="eval-subtitle" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <Award size={16} className="text-amber-400" />
            <span>Winning Model Diagnostics & Authenticity Assessment</span>

            {/* Continuous Auto-Tuning & Authenticity Badge */}
            {isBenchmarkAchieved ? (
              <span className="eval-badge eval-badge-emerald" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', marginLeft: '0.5rem' }}>
                <CheckCircle2 size={12} />
                <span>Authentic Model (&gt;= 80% Benchmark Achieved)</span>
              </span>
            ) : (
              <span className="eval-badge eval-badge-indigo" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', marginLeft: '0.5rem' }}>
                <Target size={12} />
                <span>Continuous Auto-Tuning ({tuningHistory?.rounds_run || 1} Rounds)</span>
              </span>
            )}
          </div>

          <h2 className="eval-title">
            <span>{modelName}</span>
            <span className="eval-badge eval-badge-indigo">
              Engine: {engine}
            </span>
            <span className="eval-badge eval-badge-purple">
              {String(problemType).replace(/_/g, ' ')}
            </span>
          </h2>
        </div>

        {/* Run Selector (if multiple training runs exist) */}
        {runs && runs.length > 1 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <label htmlFor="eval-run-select" style={{ fontSize: '0.8rem', color: 'var(--eval-text-muted)', fontWeight: 600 }}>
              Select Run:
            </label>
            <select
              id="eval-run-select"
              value={activeRunId}
              onChange={(e) => onSelectRun(e.target.value)}
              className="select-input"
              style={{ padding: '0.45rem 0.85rem', fontSize: '0.8rem' }}
              aria-label="Select evaluation run"
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} ({r.winning_model || 'Run'})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Spotlight Key Metrics Grid: Training vs Testing Side-by-Side */}
      <div className="eval-grid-4" style={{ marginTop: '1.5rem' }}>
        {/* 1. Training Accuracy / Benchmark */}
        <div className="eval-stat-card" style={{ borderLeft: '4px solid #6366f1' }}>
          <div className="eval-stat-label">
            <span>Training Accuracy ({primaryMetric})</span>
            <span className="eval-badge eval-badge-indigo" style={{ fontSize: '0.65rem', padding: '0.15rem 0.45rem' }}>
              Training Set
            </span>
          </div>
          <div className="eval-stat-val" style={{ color: '#818cf8' }}>
            {formatMetricScore(primaryMetric, trainScore ?? valScore)}
          </div>
          <div className="eval-stat-subtext">
            Performance learned across the training partition
          </div>
        </div>

        {/* 2. Held-Out Testing Accuracy */}
        <div className="eval-stat-card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)', borderLeft: '4px solid #10b981' }}>
          <div className="eval-stat-label">
            <span>Testing Accuracy ({primaryMetric})</span>
            <span className="eval-badge eval-badge-emerald" style={{ fontSize: '0.65rem', padding: '0.15rem 0.45rem' }}>
              Held-Out Test Set
            </span>
          </div>
          <div className="eval-stat-val eval-stat-val-green">
            {formatMetricScore(primaryMetric, testScore)}
          </div>
          <div className="eval-stat-subtext">
            Authentic score on unseen held-out records
          </div>
        </div>

        {/* 3. Generalization Gap */}
        <div className="eval-stat-card">
          <div className="eval-stat-label">
            <span>Generalization Gap</span>
            <span className={`eval-badge ${gapAssessment.badgeClass}`} style={{ fontSize: '0.65rem', padding: '0.15rem 0.45rem' }}>
              Drop vs Train
            </span>
          </div>
          <div className="eval-stat-val eval-stat-val-indigo">
            {gapAssessment.text}
          </div>
          <div className="eval-stat-subtext">
            {gapAssessment.statusText}
          </div>
        </div>

        {/* 4. Diagnostic Assessment & Latency */}
        <div className="eval-stat-card" style={{ justifyContent: 'space-between' }}>
          <div>
            <div className="eval-stat-label" style={{ marginBottom: '0.5rem' }}>
              <span>Diagnostic Assessment</span>
            </div>
            <span className={diagBadge.className}>
              {diagBadge.icon}
              <span>{diagBadge.text}</span>
            </span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.75rem',
            paddingTop: '0.6rem',
            marginTop: '0.5rem',
            borderTop: '1px solid rgba(255, 255, 255, 0.08)',
            fontSize: '0.75rem',
            fontFamily: 'var(--font-mono, monospace)',
            color: 'var(--eval-text-muted)'
          }}>
            <span title="Training Duration" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
              <Clock size={12} className="text-indigo-400" />
              <span>{trainTime !== undefined && trainTime !== null ? Number(trainTime).toFixed(2) : '—'}s train</span>
            </span>
            <span title="Inference Latency" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
              <Zap size={12} className="text-emerald-400" />
              <span>{predTime !== undefined && predTime !== null ? Number(predTime).toFixed(3) : '—'}s infer</span>
            </span>
          </div>
        </div>
      </div>

      {/* Continuous Tuning Banner / Notes */}
      {tuningHistory && (
        <div className="eval-notice-box" style={{ marginTop: '1.25rem', background: 'rgba(99, 102, 241, 0.08)', borderColor: 'rgba(99, 102, 241, 0.25)' }}>
          <Sparkles size={16} className="text-indigo-400" style={{ marginTop: '0.15rem', flexShrink: 0 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <div style={{ fontWeight: 700, color: '#f8fafc', fontSize: '0.85rem' }}>
              Continuous Auto-Tuning Optimization Report:
            </div>
            <div style={{ color: '#cbd5e1', fontSize: '0.8rem' }}>
              {tuningHistory.summary_text || `Executed ${tuningHistory.rounds_run} iterative tuning rounds targeting the 80% benchmark.`}
            </div>
            {tuningHistory.round_scores && tuningHistory.round_scores.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.25rem' }}>
                {tuningHistory.round_scores.map((rs) => (
                  <span key={rs.round} className={`eval-badge ${rs.target_achieved ? 'eval-badge-emerald' : 'eval-badge-indigo'}`} style={{ fontSize: '0.7rem' }}>
                    Round {rs.round} ({rs.stage}): {(rs.best_score * 100).toFixed(1)}% {rs.target_achieved && '✓'}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Diagnostics Explanatory Notes */}
      {evaluation.diagnostics?.notes && evaluation.diagnostics.notes.length > 0 && (
        <div className="eval-notice-box" style={{ marginTop: '1rem' }}>
          <Sparkles size={16} className="text-emerald-400" style={{ marginTop: '0.15rem', flexShrink: 0 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            {evaluation.diagnostics.notes.map((note, idx) => (
              <div key={idx} style={{ color: '#e2e8f0', fontSize: '0.8rem' }}>• {note}</div>
            ))}
          </div>
        </div>
      )}
    </header>
  );
};
