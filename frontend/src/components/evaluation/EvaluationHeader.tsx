import React from 'react';
import { EvaluationOverview } from '../../types/evaluation';
import { Clock } from 'lucide-react';

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
  const getDiagnosticBadge = (label?: string) => {
    switch (label) {
      case 'Normal / Well-fit':
      case 'No obvious gap':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'Moderate generalization gap':
      case 'Potential overfitting indicator':
      case 'Possible Overfitting':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'Large generalization gap':
      case 'Severe Overfitting':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      case 'Potential underfitting indicator':
      case 'Underfitting':
        return 'bg-orange-500/20 text-orange-300 border-orange-500/40';
      default:
        return 'bg-slate-700/40 text-slate-300 border-slate-600/40';
    }
  };

  const formatScore = (val?: number | null) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(4);
  };

  const modelName = evaluation.model_name || (evaluation as any).winning_model?.model_name || 'Model';
  const engine = evaluation.engine || (evaluation as any).winning_model?.engine || 'sklearn';
  const problemType = evaluation.problem_type || (evaluation as any).winning_model?.problem_type || 'classification';
  const valScore = evaluation.validation_score ?? (evaluation as any).winning_model?.validation_score;
  const testScore = evaluation.test_score ?? (evaluation as any).winning_model?.test_score;
  const genGap = evaluation.generalization_gap ?? (evaluation as any).winning_model?.generalization_gap;
  const trainTime = evaluation.training_time_seconds ?? (evaluation as any).winning_model?.training_time_seconds;
  const predTime = evaluation.prediction_time_seconds ?? (evaluation as any).winning_model?.prediction_time_seconds;

  return (
    <div className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden mb-6">
      {/* Background ambient glow */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Top row: Title + Run Selector */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xl">🏆</span>
            <span className="text-xs uppercase font-bold tracking-wider text-indigo-400">
              Phase 5 Model Evaluation & Diagnostics
            </span>
          </div>
          <h2 className="text-2xl font-black text-white flex items-center gap-3">
            {modelName}
            <span className="text-xs uppercase px-3 py-1 rounded-full font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Engine: {engine}
            </span>
            <span className="text-xs uppercase px-3 py-1 rounded-full font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30">
              {String(problemType).replace('_', ' ')}
            </span>
          </h2>
        </div>

        {/* Run selector dropdown */}
        {runs.length > 1 && (
          <div className="flex items-center gap-2">
            <label htmlFor="eval-run-select" className="text-xs text-slate-400 font-medium">Evaluation Run:</label>
            <select
              id="eval-run-select"
              value={activeRunId}
              onChange={(e) => onSelectRun(e.target.value)}
              className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-500"
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

      {/* Overview Spotlight Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
        {/* Validation Score */}
        <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 uppercase font-semibold tracking-wider">
            Validation {evaluation.primary_metric}
          </div>
          <div className="text-2xl font-black font-mono text-white mt-1">
            {formatScore(valScore)}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            Calculated on validation partition
          </div>
        </div>

        {/* Held-out Test Score */}
        <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 uppercase font-semibold tracking-wider flex items-center gap-1.5">
            <span>Held-out Test {evaluation.primary_metric}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">
              Test Set
            </span>
          </div>
          <div className="text-2xl font-black font-mono text-emerald-400 mt-1">
            {formatScore(testScore)}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            Untouched held-out test data
          </div>
        </div>

        {/* Generalization Gap */}
        <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 uppercase font-semibold tracking-wider">
            Generalization Gap
          </div>
          <div className="text-2xl font-black font-mono text-indigo-300 mt-1">
            {genGap !== undefined && genGap !== null
              ? (genGap > 0 ? `+${formatScore(genGap)}` : formatScore(genGap))
              : '—'}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            Validation vs Test performance
          </div>
        </div>

        {/* Diagnostics & Timing */}
        <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="text-xs text-slate-400 uppercase font-semibold tracking-wider mb-1.5">
              Diagnostic Assessment
            </div>
            <span className={`px-2.5 py-1 rounded-full text-xs font-bold border inline-block ${getDiagnosticBadge(evaluation.diagnostics?.label)}`}>
              {evaluation.diagnostics?.label || 'Evaluated'}
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-400 font-mono mt-2 pt-2 border-t border-slate-800/80">
            <span title="Training Duration" className="flex items-center gap-1">
              <Clock size={12} /> {trainTime !== undefined && trainTime !== null ? Number(trainTime).toFixed(2) : '—'}s train
            </span>
            <span title="Inference Latency" className="flex items-center gap-1">
              ⚡ {predTime !== undefined && predTime !== null ? Number(predTime).toFixed(3) : '—'}s test
            </span>
          </div>
        </div>
      </div>

      {/* Diagnostics notes banner */}
      {evaluation.diagnostics?.notes && evaluation.diagnostics.notes.length > 0 && (
        <div className="mt-4 p-3 bg-indigo-950/20 border border-indigo-500/20 rounded-xl text-xs text-slate-300 flex items-start gap-2.5">
          <span className="text-indigo-400 text-sm mt-0.5">ℹ️</span>
          <div className="space-y-1">
            {evaluation.diagnostics.notes.map((note, idx) => (
              <div key={idx}>{note}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
