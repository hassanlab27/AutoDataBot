import React, { useState } from 'react';
import { AutoMLConfig, TrainingMode, EnginePreference } from '../../types/ml';

interface TrainingConfigPanelProps {
  datasetId: string;
  defaultTarget?: string;
  defaultProblemType?: string;
  onStartTraining: (config: AutoMLConfig) => void;
  isLoading: boolean;
}

export const TrainingConfigPanel: React.FC<TrainingConfigPanelProps> = ({
  datasetId,
  defaultTarget = '',
  defaultProblemType = 'binary_classification',
  onStartTraining,
  isLoading
}) => {
  const [target, setTarget] = useState<string>(defaultTarget);
  const [problemType, setProblemType] = useState<string>(defaultProblemType);
  const [trainingMode, setTrainingMode] = useState<TrainingMode>('quick');
  const [customTimeLimit, setCustomTimeLimit] = useState<number>(120);
  const [primaryMetric, setPrimaryMetric] = useState<string>('auto');
  const [enginePreference, setEnginePreference] = useState<EnginePreference>('all');
  const [randomState, setRandomState] = useState<number>(42);
  const [maxCpus, setMaxCpus] = useState<number>(4);
  const [presets, setPresets] = useState<string>('medium_quality');
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!target.trim()) return;

    const config: AutoMLConfig = {
      dataset_id: datasetId,
      target: target.trim(),
      problem_type: problemType,
      training_mode: trainingMode,
      time_limit: trainingMode === 'custom' ? customTimeLimit : undefined,
      primary_metric: primaryMetric,
      random_state: randomState,
      max_cpus: maxCpus > 0 ? maxCpus : undefined,
      presets: presets,
      engine_preference: enginePreference
    };

    onStartTraining(config);
  };

  const isClassification = problemType.includes('classification');

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-xl backdrop-blur">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="p-2 bg-indigo-500/20 text-indigo-400 rounded-lg">⚡</span>
            AutoML Training Configuration
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Automatically train and benchmark multiple tabular ML models without data leakage.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Target and Problem Type */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold text-slate-300 mb-2">
              Target Column <span className="text-rose-400">*</span>
            </label>
            <input
              type="text"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="e.g. churn, price, status"
              required
              disabled={isLoading}
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-300 mb-2">
              Problem Type
            </label>
            <select
              value={problemType}
              onChange={(e) => setProblemType(e.target.value)}
              disabled={isLoading}
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              <option value="binary_classification">Binary Classification</option>
              <option value="multiclass_classification">Multiclass Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
        </div>

        {/* Training Mode Selection */}
        <div>
          <label className="block text-sm font-semibold text-slate-300 mb-2">
            Training Mode & Time Budget
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              { id: 'quick', title: 'Quick (1m)', desc: 'Fast exploration & baselines', budget: '60s' },
              { id: 'standard', title: 'Standard (5m)', desc: 'Balanced search & ensembles', budget: '300s' },
              { id: 'extended', title: 'Extended (15m)', desc: 'Deep tuning & heavy stacks', budget: '900s' },
              { id: 'custom', title: 'Custom', desc: 'Specify exact time budget', budget: 'User defined' }
            ].map((mode) => (
              <button
                type="button"
                key={mode.id}
                onClick={() => setTrainingMode(mode.id as TrainingMode)}
                disabled={isLoading}
                className={`p-3.5 rounded-xl text-left border transition-all ${
                  trainingMode === mode.id
                    ? 'bg-indigo-600/20 border-indigo-500 text-white shadow-lg shadow-indigo-500/10'
                    : 'bg-slate-800/60 border-slate-700/60 text-slate-300 hover:border-slate-600'
                }`}
              >
                <div className="font-semibold text-sm">{mode.title}</div>
                <div className="text-xs text-slate-400 mt-0.5">{mode.desc}</div>
              </button>
            ))}
          </div>

          {trainingMode === 'custom' && (
            <div className="mt-3 flex items-center gap-3">
              <label className="text-xs text-slate-400">Custom Time Limit (seconds):</label>
              <input
                type="number"
                min="10"
                max="3600"
                value={customTimeLimit}
                onChange={(e) => setCustomTimeLimit(Number(e.target.value))}
                disabled={isLoading}
                className="w-32 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          )}
        </div>

        {/* Primary Metric & Engine Preference */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold text-slate-300 mb-2">
              Primary Metric for Model Selection
            </label>
            <select
              value={primaryMetric}
              onChange={(e) => setPrimaryMetric(e.target.value)}
              disabled={isLoading}
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="auto">Auto (Recommended: F1/Accuracy or RMSE)</option>
              {isClassification ? (
                <>
                  <option value="f1">F1 Score (Binary)</option>
                  <option value="f1_macro">F1 Macro</option>
                  <option value="accuracy">Accuracy</option>
                  <option value="balanced_accuracy">Balanced Accuracy</option>
                  <option value="roc_auc">ROC AUC</option>
                  <option value="precision">Precision</option>
                  <option value="recall">Recall</option>
                </>
              ) : (
                <>
                  <option value="rmse">RMSE (Root Mean Squared Error)</option>
                  <option value="mae">MAE (Mean Absolute Error)</option>
                  <option value="r2">R² Score</option>
                  <option value="mape">MAPE</option>
                </>
              )}
            </select>
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-300 mb-2">
              AutoML Engine Preference
            </label>
            <select
              value={enginePreference}
              onChange={(e) => setEnginePreference(e.target.value as EnginePreference)}
              disabled={isLoading}
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="all">All Engines (AutoGluon + FLAML + GBDT + Sklearn)</option>
              <option value="autogluon">AutoGluon Tabular only</option>
              <option value="flaml">FLAML only</option>
              <option value="sklearn_gbdt">Scikit-Learn + GBDT (LightGBM/XGBoost/CatBoost)</option>
            </select>
          </div>
        </div>

        {/* Advanced Settings Toggle */}
        <div className="pt-2 border-t border-slate-800/80">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1.5 focus:outline-none"
          >
            <span>{showAdvanced ? '▼' : '▶'}</span>
            Advanced Settings (Random Seed, CPU Cores, Presets)
          </button>

          {showAdvanced && (
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-4 bg-slate-950/40 p-4 rounded-xl border border-slate-800">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Random Seed</label>
                <input
                  type="number"
                  value={randomState}
                  onChange={(e) => setRandomState(Number(e.target.value))}
                  disabled={isLoading}
                  className="w-full px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Max CPU Cores</label>
                <input
                  type="number"
                  min="1"
                  max="32"
                  value={maxCpus}
                  onChange={(e) => setMaxCpus(Number(e.target.value))}
                  disabled={isLoading}
                  className="w-full px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">AutoGluon Preset</label>
                <select
                  value={presets}
                  onChange={(e) => setPresets(e.target.value)}
                  disabled={isLoading}
                  className="w-full px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="medium_quality">Medium Quality</option>
                  <option value="good_quality">Good Quality</option>
                  <option value="best_quality">Best Quality</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Action Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={isLoading || !target.trim()}
            className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold rounded-xl shadow-lg shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Initializing Training Run...</span>
              </>
            ) : (
              <>
                <span>🚀</span>
                <span>Start AutoML Training</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
