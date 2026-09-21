import React from 'react';
import { ModelResult } from '../../types/ml';

interface LeaderboardTableProps {
  leaderboard: ModelResult[];
  primaryMetric: string;
}

export const LeaderboardTable: React.FC<LeaderboardTableProps> = ({
  leaderboard,
  primaryMetric
}) => {
  const formatScore = (val?: number) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(4);
  };

  const getEngineBadgeClass = (engine: string) => {
    switch (engine.toLowerCase()) {
      case 'autogluon':
        return 'bg-amber-500/10 text-amber-300 border-amber-500/30';
      case 'flaml':
        return 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30';
      case 'lightgbm':
        return 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30';
      case 'xgboost':
        return 'bg-blue-500/10 text-blue-300 border-blue-500/30';
      case 'catboost':
        return 'bg-yellow-500/10 text-yellow-300 border-yellow-500/30';
      case 'sklearn':
        return 'bg-purple-500/10 text-purple-300 border-purple-500/30';
      default:
        return 'bg-slate-700/30 text-slate-300 border-slate-600/30';
    }
  };

  const getGapBadgeClass = (gap?: number) => {
    if (gap === undefined || gap === null) return 'text-slate-500';
    if (gap > 0.15) return 'text-rose-400 bg-rose-500/10 border-rose-500/30 font-semibold';
    if (gap > 0.08) return 'text-amber-400 bg-amber-500/10 border-amber-500/30 font-semibold';
    return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30 font-semibold';
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <span>🏆</span> Model Leaderboard
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Ranked deterministically by validation <span className="font-mono text-indigo-400">{primaryMetric}</span>.
          </p>
        </div>
        <div className="text-xs text-slate-400 font-mono">
          Total Candidates: <span className="text-white font-bold">{leaderboard.length}</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="bg-slate-950/60 text-xs uppercase font-semibold text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-3 px-4 w-12 text-center">Rank</th>
              <th className="py-3 px-4">Model Name</th>
              <th className="py-3 px-4">Engine</th>
              <th className="py-3 px-4 text-right">Val {primaryMetric}</th>
              <th className="py-3 px-4 text-right">Test {primaryMetric}</th>
              <th className="py-3 px-4 text-right">Gen. Gap</th>
              <th className="py-3 px-4 text-right">Time (s)</th>
              <th className="py-3 px-4 text-center">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {leaderboard.map((model, idx) => {
              const isWinner = model.is_winner;
              const isNaive = model.is_naive_baseline;

              return (
                <tr
                  key={model.model_id}
                  className={`transition-colors ${
                    isWinner
                      ? 'bg-indigo-950/30 hover:bg-indigo-950/40 border-l-4 border-l-amber-400'
                      : 'hover:bg-slate-800/40'
                  }`}
                >
                  {/* Rank */}
                  <td className="py-3 px-4 text-center font-mono font-bold text-xs">
                    {isWinner ? (
                      <span className="text-amber-400 flex items-center justify-center gap-1">
                        👑 1
                      </span>
                    ) : (
                      <span className="text-slate-500">#{idx + 1}</span>
                    )}
                  </td>

                  {/* Model Name */}
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <span className={`font-semibold ${isWinner ? 'text-amber-300' : 'text-slate-200'}`}>
                        {model.model_name}
                      </span>
                      {isWinner && (
                        <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40">
                          Selected Winner
                        </span>
                      )}
                      {isNaive && (
                        <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                          Naive Baseline
                        </span>
                      )}
                    </div>
                  </td>

                  {/* Engine */}
                  <td className="py-3 px-4">
                    <span className={`text-xs px-2.5 py-0.5 rounded-full border font-mono uppercase font-medium ${getEngineBadgeClass(model.engine)}`}>
                      {model.engine}
                    </span>
                  </td>

                  {/* Validation Primary Score */}
                  <td className="py-3 px-4 text-right font-mono font-bold text-slate-100">
                    {model.status === 'success' ? formatScore(model.validation_primary_score) : '—'}
                  </td>

                  {/* Test Primary Score */}
                  <td className="py-3 px-4 text-right font-mono font-semibold text-slate-300">
                    {model.test_primary_score !== undefined && model.test_primary_score !== null
                      ? formatScore(model.test_primary_score)
                      : '—'}
                  </td>

                  {/* Generalization Gap */}
                  <td className="py-3 px-4 text-right font-mono text-xs">
                    {model.generalization_gap !== undefined && model.generalization_gap !== null ? (
                      <span className={`px-2 py-0.5 rounded border text-xs font-mono ${getGapBadgeClass(model.generalization_gap)}`}>
                        {model.generalization_gap > 0 ? `+${formatScore(model.generalization_gap)}` : formatScore(model.generalization_gap)}
                      </span>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>

                  {/* Training Time */}
                  <td className="py-3 px-4 text-right font-mono text-xs text-slate-400">
                    {model.training_time_seconds.toFixed(1)}s
                  </td>

                  {/* Status */}
                  <td className="py-3 px-4 text-center">
                    {model.status === 'success' ? (
                      <span className="text-xs text-emerald-400 flex items-center justify-center gap-1 font-medium">
                        ✓ OK
                      </span>
                    ) : (
                      <span
                        className="text-xs text-rose-400 font-medium cursor-help"
                        title={model.error_message || 'Training failed'}
                      >
                        ✕ Failed
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
  );
};
