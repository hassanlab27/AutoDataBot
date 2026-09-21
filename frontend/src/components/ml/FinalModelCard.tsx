import React from 'react';
import { ModelResult } from '../../types/ml';

interface FinalModelCardProps {
  winner: ModelResult;
  primaryMetric: string;
  naiveBaselineScore?: number;
}

export const FinalModelCard: React.FC<FinalModelCardProps> = ({
  winner,
  primaryMetric,
  naiveBaselineScore
}) => {
  const formatScore = (val?: number) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(4);
  };

  const getDiagnosticBadge = (label?: string) => {
    switch (label) {
      case 'Normal / Well-fit':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'Possible Overfitting':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'Severe Overfitting':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      case 'Underfitting':
        return 'bg-orange-500/20 text-orange-300 border-orange-500/40';
      case 'Suspiciously High Performance':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/40';
      case 'Worse than Baseline':
        return 'bg-red-500/20 text-red-300 border-red-500/40';
      default:
        return 'bg-slate-700/40 text-slate-300 border-slate-600/40';
    }
  };

  // Calculate percentage improvement over naive baseline if available
  let baselineDiffText: string | null = null;
  if (naiveBaselineScore !== undefined && naiveBaselineScore !== null && winner.validation_primary_score !== undefined) {
    const diff = winner.validation_primary_score - naiveBaselineScore;
    if (Math.abs(naiveBaselineScore) > 1e-4) {
      const pct = (diff / Math.abs(naiveBaselineScore)) * 100;
      baselineDiffText = `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}% vs Naive Baseline (${formatScore(naiveBaselineScore)})`;
    } else {
      baselineDiffText = `${diff >= 0 ? '+' : ''}${diff.toFixed(4)} vs Naive Baseline`;
    }
  }

  return (
    <div className="bg-gradient-to-br from-slate-900 via-indigo-950/20 to-slate-900 border-2 border-indigo-500/40 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
      {/* Decorative Glow */}
      <div className="absolute top-0 right-0 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header Banner */}
      <div className="flex flex-wrap items-start justify-between gap-4 pb-5 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xl">👑</span>
            <span className="text-xs uppercase font-bold tracking-wider text-amber-400">
              Selected Winning Model
            </span>
          </div>
          <h2 className="text-2xl font-black text-white flex items-center gap-3">
            {winner.model_name}
            <span className="text-xs uppercase px-3 py-1 rounded-full font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Engine: {winner.engine}
            </span>
          </h2>
        </div>

        {/* Diagnostic Label Badge */}
        {winner.diagnostic_label && (
          <div className="flex flex-col items-end">
            <div className="text-xs text-slate-400 mb-1">Generalization Diagnosis</div>
            <span className={`px-3.5 py-1 rounded-full text-xs font-bold border uppercase tracking-wider ${getDiagnosticBadge(winner.diagnostic_label)}`}>
              {winner.diagnostic_label}
            </span>
          </div>
        )}
      </div>

      {/* Metrics Spotlight Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 my-6">
        {/* Validation Score */}
        <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">
            Validation {primaryMetric}
          </div>
          <div className="text-3xl font-mono font-black text-white mt-1">
            {formatScore(winner.validation_primary_score)}
          </div>
          {baselineDiffText && (
            <div className="text-xs text-emerald-400 mt-2 font-medium">
              {baselineDiffText}
            </div>
          )}
        </div>

        {/* Held-Out Test Score */}
        <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 font-medium uppercase tracking-wider flex items-center gap-1.5">
            <span>Held-Out Test {primaryMetric}</span>
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 font-semibold">
              Untouched Test Partition
            </span>
          </div>
          <div className="text-3xl font-mono font-black text-slate-100 mt-1">
            {formatScore(winner.test_primary_score)}
          </div>
          <div className="text-xs text-slate-400 mt-2">
            Evaluated strictly after model selection
          </div>
        </div>

        {/* Generalization Gap */}
        <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">
            Generalization Gap
          </div>
          <div className="text-3xl font-mono font-black text-indigo-300 mt-1">
            {winner.generalization_gap !== undefined && winner.generalization_gap !== null
              ? (winner.generalization_gap > 0 ? `+${formatScore(winner.generalization_gap)}` : formatScore(winner.generalization_gap))
              : '—'}
          </div>
          <div className="text-xs text-slate-400 mt-2">
            Performance difference (Val vs Test)
          </div>
        </div>
      </div>

      {/* Diagnostic Notes */}
      {winner.diagnostic_notes && winner.diagnostic_notes.length > 0 && (
        <div className="bg-slate-950/40 border border-slate-800/80 rounded-xl p-4 mb-6">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
            <span>🔍</span> Diagnostic Assessment
          </div>
          <ul className="space-y-1.5 text-xs text-slate-300">
            {winner.diagnostic_notes.map((note, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-indigo-400 font-bold">•</span>
                <span>{note}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Secondary Metrics Breakdown */}
      {winner.test_metrics && Object.keys(winner.test_metrics).length > 0 && (
        <div>
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
            Complete Test Set Evaluation Metrics
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2.5">
            {Object.entries(winner.test_metrics).map(([metricKey, val]) => (
              <div
                key={metricKey}
                className="bg-slate-950/60 border border-slate-800/80 px-3 py-2 rounded-lg text-center"
              >
                <div className="text-[11px] uppercase font-mono text-slate-400 truncate">
                  {metricKey}
                </div>
                <div className="text-sm font-mono font-bold text-slate-200 mt-0.5">
                  {formatScore(val)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
