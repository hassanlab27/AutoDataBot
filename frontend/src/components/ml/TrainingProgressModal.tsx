import React from 'react';
import { RunStatus } from '../../types/ml';

interface TrainingProgressModalProps {
  status: RunStatus;
  onCancel: () => void;
  isCancelling: boolean;
}

export const TrainingProgressModal: React.FC<TrainingProgressModalProps> = ({
  status,
  onCancel,
  isCancelling
}) => {
  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainder = Math.floor(secs % 60);
    return `${mins}m ${remainder < 10 ? '0' : ''}${remainder}s`;
  };

  const getEngineColor = (engine?: string) => {
    switch (engine?.toLowerCase()) {
      case 'autogluon':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'flaml':
        return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40';
      case 'lightgbm':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'xgboost':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/40';
      case 'catboost':
        return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40';
      case 'sklearn':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/40';
      default:
        return 'bg-slate-700/40 text-slate-300 border-slate-600/40';
    }
  };

  return (
    <div className="bg-slate-900 border border-indigo-500/30 rounded-xl p-6 shadow-2xl relative overflow-hidden backdrop-blur">
      {/* Background Pulse Glow */}
      <div className="absolute -right-16 -top-16 w-64 h-64 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-indigo-500/20 border border-indigo-500/30 text-indigo-400">
            <div className="w-5 h-5 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              AutoML Optimization in Progress
              {status.current_engine && (
                <span className={`text-xs px-2.5 py-0.5 rounded-full border font-mono uppercase font-semibold ${getEngineColor(status.current_engine)}`}>
                  {status.current_engine}
                </span>
              )}
            </h3>
            <p className="text-sm text-slate-400 mt-0.5">
              {status.current_stage || 'Training tabular models...'}
            </p>
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs text-slate-400 font-medium">Elapsed Time</div>
          <div className="text-sm font-mono font-bold text-slate-200">
            {formatTime(status.elapsed_seconds || 0)}
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mt-4">
        <div className="flex justify-between text-xs text-slate-400 mb-1.5 font-medium">
          <span>Overall Progress</span>
          <span className="text-indigo-400 font-mono font-semibold">{status.progress_pct}%</span>
        </div>
        <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${Math.max(5, status.progress_pct)}%` }}
          />
        </div>
      </div>

      {/* Cancel Action */}
      <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between">
        <div className="text-xs text-slate-400 flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          Job ID: <span className="font-mono text-slate-300">{status.run_id}</span>
        </div>
        <button
          type="button"
          onClick={onCancel}
          disabled={isCancelling || status.cancellation_requested}
          className="px-4 py-1.5 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 rounded-lg text-xs font-semibold transition-all disabled:opacity-50 flex items-center gap-1.5"
        >
          {isCancelling || status.cancellation_requested ? (
            <>
              <div className="w-3 h-3 border border-rose-300/40 border-t-rose-300 rounded-full animate-spin" />
              <span>Cancelling...</span>
            </>
          ) : (
            <>
              <span>⏹</span>
              <span>Stop Training</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
