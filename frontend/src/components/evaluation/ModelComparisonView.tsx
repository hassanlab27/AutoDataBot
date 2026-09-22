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
    <div className="space-y-8 animate-fadeIn">
      {/* Informational Header Notice */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl flex items-start gap-3">
        <Info size={18} className="text-indigo-400 mt-0.5 shrink-0" />
        <div className="text-xs text-slate-300">
          <span className="font-bold text-white">Generalization Diagnostics:</span> This comparison visualizes the performance stability between validation scores and untouched held-out test scores. Phase 4 selected the final winning model based strictly on validation benchmarks.
        </div>
      </div>

      {/* Validation vs Test Grouped Bar Chart */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <PlotlyChart data={groupedChartData} layout={groupedChartLayout} style={{ width: '100%', height: '400px' }} />
      </div>

      {/* Complete Comparison Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="pb-4 border-b border-slate-800 mb-4">
          <h3 className="text-base font-bold text-white">Candidate Models Comparison Table</h3>
          <p className="text-xs text-slate-400">All trained architectures sorted by validation benchmark performance</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase text-[11px]">
                <th className="py-3 px-4">Model Name</th>
                <th className="py-3 px-4">Engine</th>
                <th className="py-3 px-4">Validation {primary_metric}</th>
                <th className="py-3 px-4">Held-out Test {primary_metric}</th>
                <th className="py-3 px-4">Generalization Gap</th>
                <th className="py-3 px-4">Training Time</th>
                <th className="py-3 px-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {models.map((m) => (
                <tr
                  key={m.model_id}
                  className={`transition-colors ${
                    m.is_winner
                      ? 'bg-indigo-950/40 text-white font-bold'
                      : 'hover:bg-slate-800/30 text-slate-300'
                  }`}
                >
                  <td className="py-3 px-4 flex items-center gap-2">
                    {m.is_winner && <span className="text-amber-400">👑</span>}
                    {m.is_naive_baseline && <span className="text-slate-500">⚓</span>}
                    <span>{m.model_name}</span>
                  </td>
                  <td className="py-3 px-4 uppercase text-[11px] text-slate-400">{m.engine}</td>
                  <td className="py-3 px-4 font-bold text-slate-100">
                    {formatScore(m.validation_score)}
                  </td>
                  <td className="py-3 px-4 font-bold text-emerald-400">
                    {formatScore(m.test_score)}
                  </td>
                  <td className="py-3 px-4 text-indigo-300">
                    {m.generalization_gap !== null
                      ? (m.generalization_gap > 0 ? `+${formatScore(m.generalization_gap)}` : formatScore(m.generalization_gap))
                      : '—'}
                  </td>
                  <td className="py-3 px-4 text-slate-400">
                    {m.training_time_seconds !== undefined && m.training_time_seconds !== null ? `${Number(m.training_time_seconds).toFixed(2)}s` : '—'}
                  </td>
                  <td className="py-3 px-4 text-right">
                    {m.is_winner ? (
                      <span className="px-2.5 py-1 rounded-full text-[10px] uppercase font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                        Selected Winner
                      </span>
                    ) : m.is_naive_baseline ? (
                      <span className="px-2 py-0.5 rounded text-[10px] uppercase font-bold bg-slate-800 text-slate-400">
                        Naive Baseline
                      </span>
                    ) : (
                      <span className="text-slate-500 text-[11px]">Candidate</span>
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
