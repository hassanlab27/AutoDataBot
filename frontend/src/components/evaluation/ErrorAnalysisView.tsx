import React, { useState } from 'react';
import {
  ClassificationErrorAnalysis,
  RegressionErrorAnalysis,
  WorstPrediction
} from '../../types/evaluation';
import { AlertOctagon, Sparkles, ChevronRight } from 'lucide-react';

interface ErrorAnalysisViewProps {
  errorAnalysis: ClassificationErrorAnalysis | RegressionErrorAnalysis;
  problemType: string;
  onExplainPrediction?: (predictionId: number) => void;
}

export const ErrorAnalysisView: React.FC<ErrorAnalysisViewProps> = ({
  errorAnalysis,
  problemType,
  onExplainPrediction
}) => {
  const isClassification = problemType.includes('classification');
  const clsErrors = isClassification ? (errorAnalysis as ClassificationErrorAnalysis) : null;
  const regErrors = !isClassification ? (errorAnalysis as RegressionErrorAnalysis) : null;

  const worstPredictions = errorAnalysis?.worst_predictions || [];
  const totalSamples = errorAnalysis?.total_samples ?? 0;

  const [selectedRow, setSelectedRow] = useState<WorstPrediction | null>(
    worstPredictions.length > 0 ? worstPredictions[0] : null
  );

  const formatScore = (val?: number | null) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(4);
  };

  const largestErrors = regErrors?.largest_absolute_errors || [];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Error Analysis Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 uppercase font-mono">Test Samples Evaluated</div>
          <div className="text-2xl font-black font-mono text-white mt-1">
            {totalSamples}
          </div>
        </div>

        {isClassification && clsErrors && (
          <>
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 uppercase font-mono">Misclassifications</div>
              <div className="text-2xl font-black font-mono text-rose-400 mt-1">
                {clsErrors.misclassification_count ?? 0}
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 uppercase font-mono">Test Error Rate</div>
              <div className="text-2xl font-black font-mono text-rose-300 mt-1">
                {clsErrors.error_rate !== undefined && clsErrors.error_rate !== null
                  ? `${(clsErrors.error_rate * 100).toFixed(2)}%`
                  : '—'}
              </div>
            </div>

            {clsErrors.false_positives !== null && clsErrors.false_positives !== undefined && (
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
                <div className="text-xs text-slate-400 uppercase font-mono">FP / FN Breakdown</div>
                <div className="text-xl font-black font-mono text-slate-200 mt-1">
                  FP: <span className="text-amber-400">{clsErrors.false_positives}</span> | FN: <span className="text-rose-400">{clsErrors.false_negatives}</span>
                </div>
              </div>
            )}
          </>
        )}

        {!isClassification && regErrors && (
          <>
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 uppercase font-mono">Max Absolute Error</div>
              <div className="text-2xl font-black font-mono text-rose-400 mt-1">
                {largestErrors.length > 0 ? formatScore(largestErrors[0]) : '—'}
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 uppercase font-mono">Top 5 Mean Error</div>
              <div className="text-2xl font-black font-mono text-rose-300 mt-1">
                {largestErrors.length > 0
                  ? formatScore(
                      largestErrors.slice(0, 5).reduce((a, b) => a + b, 0) /
                        Math.min(5, largestErrors.length)
                    )
                  : '—'}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Most Confused Pairs (Classification Only) */}
      {isClassification && clsErrors && (clsErrors.most_confused_pairs || []).length > 0 && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
            <AlertOctagon size={16} className="text-amber-400" />
            Most Confused Class Pairs
          </h3>
          <div className="flex flex-wrap gap-3">
            {(clsErrors.most_confused_pairs || []).map((pair, idx) => (
              <div key={idx} className="bg-slate-950 px-4 py-2.5 rounded-xl border border-slate-800 flex items-center gap-3 text-xs">
                <div>
                  <span className="text-slate-400">Actual:</span> <span className="font-bold text-rose-400">{pair.actual}</span>
                </div>
                <ChevronRight size={14} className="text-slate-600" />
                <div>
                  <span className="text-slate-400">Predicted:</span> <span className="font-bold text-indigo-400">{pair.predicted}</span>
                </div>
                <span className="ml-2 px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-mono font-bold">
                  {pair.count} times
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top 20 Worst Predictions Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800 mb-4">
          <div>
            <h3 className="text-base font-bold text-white">Top 20 Worst Test Predictions</h3>
            <p className="text-xs text-slate-400">
              {isClassification
                ? 'Ranked by highest model confidence in incorrect predictions'
                : 'Ranked by largest absolute difference between actual and predicted target'}
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            Showing top {worstPredictions.length} cases (privacy preserved)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase text-[11px]">
                <th className="py-2.5 px-3">Test Row #</th>
                <th className="py-2.5 px-3">Actual Value</th>
                <th className="py-2.5 px-3">Predicted Value</th>
                {isClassification && <th className="py-2.5 px-3">Confidence / Prob</th>}
                {!isClassification && <th className="py-2.5 px-3">Residual</th>}
                {!isClassification && <th className="py-2.5 px-3">Absolute Error</th>}
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {worstPredictions.map((pred) => {
                const isSelected = selectedRow?.id === pred.id;
                return (
                  <tr
                    key={pred.id}
                    onClick={() => setSelectedRow(pred)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-indigo-950/40 text-white' : 'hover:bg-slate-800/40 text-slate-300'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-bold text-slate-400">#{pred.id}</td>
                    <td className="py-2.5 px-3 font-semibold text-emerald-400">{String(pred.actual)}</td>
                    <td className="py-2.5 px-3 font-semibold text-rose-400">{String(pred.predicted)}</td>
                    {isClassification && (
                      <td className="py-2.5 px-3 text-slate-300">
                        {pred.probability !== null && pred.probability !== undefined
                          ? `${(pred.probability * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                    )}
                    {!isClassification && (
                      <td className="py-2.5 px-3 text-indigo-300">
                        {pred.residual !== undefined ? (pred.residual > 0 ? `+${formatScore(pred.residual)}` : formatScore(pred.residual)) : '—'}
                      </td>
                    )}
                    {!isClassification && (
                      <td className="py-2.5 px-3 text-rose-400 font-bold">
                        {formatScore(pred.absolute_error)}
                      </td>
                    )}
                    <td className="py-2.5 px-3 text-right">
                      {onExplainPrediction && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onExplainPrediction(pred.id);
                          }}
                          className="px-2.5 py-1 rounded bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 border border-indigo-500/30 text-[11px] font-sans font-medium flex items-center gap-1.5 ml-auto"
                        >
                          <Sparkles size={12} />
                          Explain with SHAP
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Selected Prediction Feature Inspector */}
        {selectedRow && selectedRow.features && (
          <div className="mt-6 pt-4 border-t border-slate-800 bg-slate-950/60 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs font-bold uppercase tracking-wider text-indigo-300 flex items-center gap-2">
                <span>🔍</span> Features for Prediction #{selectedRow.id}
              </div>
              <span className="text-[11px] text-slate-400">
                Actual: <b className="text-white">{String(selectedRow.actual)}</b> | Predicted: <b className="text-rose-400">{String(selectedRow.predicted)}</b>
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 text-xs font-mono">
              {Object.entries(selectedRow.features).map(([feat, val]) => (
                <div key={feat} className="bg-slate-900 px-3 py-2 rounded-lg border border-slate-800 truncate">
                  <div className="text-[10px] text-slate-400 truncate">{feat}</div>
                  <div className="text-slate-200 font-bold mt-0.5 truncate">{String(val)}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
