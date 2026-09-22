import React, { useState, useEffect, useRef } from 'react';
import { api } from '../../services/api';
import {
  FeatureImportanceResponse,
  ShapSummaryResponse,
  ShapStatusResponse,
  ShapDependenceResponse,
  LocalExplanationResponse
} from '../../types/explainability';
import { PlotlyChart } from '../eda/PlotlyChart';
import { Sparkles, Brain, AlertCircle, RefreshCw, CheckCircle } from 'lucide-react';

interface ExplainabilityViewProps {
  runId: string;
  problemType?: string;
  initialPredictionId?: number | null;
}

export const ExplainabilityView: React.FC<ExplainabilityViewProps> = ({
  runId,
  initialPredictionId
}) => {
  // Feature Importance State
  const [nativeImp, setNativeImp] = useState<FeatureImportanceResponse | null>(null);
  const [permImp, setPermImp] = useState<FeatureImportanceResponse | null>(null);
  const [loadingImp, setLoadingImp] = useState<boolean>(true);

  // SHAP Global State
  const [shapSummary, setShapSummary] = useState<ShapSummaryResponse | null>(null);
  const [shapStatus, setShapStatus] = useState<ShapStatusResponse | null>(null);
  const [isTriggeringShap, setIsTriggeringShap] = useState<boolean>(false);

  // SHAP Dependence State
  const [selectedDepFeature, setSelectedDepFeature] = useState<string>('');
  const [depData, setDepData] = useState<ShapDependenceResponse | null>(null);
  const [loadingDep, setLoadingDep] = useState<boolean>(false);

  // Local Explanation State
  const [localPredId, setLocalPredId] = useState<number>(initialPredictionId ?? 0);
  const [localExp, setLocalExp] = useState<LocalExplanationResponse | null>(null);
  const [loadingLocal, setLoadingLocal] = useState<boolean>(false);

  const shapPollRef = useRef<any>(null);

  // 1. Fetch feature importance on mount
  useEffect(() => {
    let isMounted = true;
    async function loadImportance() {
      setLoadingImp(true);
      try {
        const [natRes, permRes] = await Promise.allSettled([
          api.getNativeFeatureImportance(runId),
          api.getPermutationImportance(runId, 5)
        ]);

        if (isMounted) {
          if (natRes.status === 'fulfilled' && natRes.value.success) {
            setNativeImp(natRes.value.data);
          }
          if (permRes.status === 'fulfilled' && permRes.value.success) {
            setPermImp(permRes.value.data);
          }
        }
      } catch (err) {
        console.error('Error loading feature importances:', err);
      } finally {
        if (isMounted) setLoadingImp(false);
      }
    }
    loadImportance();
    return () => { isMounted = false; };
  }, [runId]);

  // 2. Fetch SHAP status and summary
  const checkShap = async () => {
    try {
      const statusRes = await api.getShapStatus(runId);
      if (statusRes.success && statusRes.data) {
        setShapStatus(statusRes.data);
        if (statusRes.data.status === 'completed') {
          clearInterval(shapPollRef.current);
          shapPollRef.current = null;
          // Load summary
          const sumRes = await api.getShapSummary(runId, 20);
          if (sumRes.success && sumRes.data) {
            setShapSummary(sumRes.data);
            if (sumRes.data.features && sumRes.data.features.length > 0 && !selectedDepFeature) {
              setSelectedDepFeature(sumRes.data.features[0]);
            }
          }
        }
      }
    } catch (err) {
      console.error('Error checking SHAP status:', err);
    }
  };

  useEffect(() => {
    checkShap();
    return () => {
      if (shapPollRef.current) clearInterval(shapPollRef.current);
    };
  }, [runId]);

  // Trigger SHAP computation
  const handleTriggerShap = async () => {
    setIsTriggeringShap(true);
    try {
      await api.triggerShap(runId, 500);
      setShapStatus({ status: 'running', stage: 'Initializing SHAP calculation...' });
      if (!shapPollRef.current) {
        shapPollRef.current = setInterval(checkShap, 2000);
      }
    } catch (err) {
      console.error('Error triggering SHAP:', err);
    } finally {
      setIsTriggeringShap(false);
    }
  };

  // Fetch SHAP dependence when feature changes
  useEffect(() => {
    if (!selectedDepFeature || shapStatus?.status !== 'completed') return;

    let isMounted = true;
    async function loadDependence() {
      setLoadingDep(true);
      try {
        const res = await api.getShapDependence(runId, selectedDepFeature);
        if (isMounted && res.success && res.data) {
          setDepData(res.data);
        }
      } catch (err) {
        console.error('Error loading SHAP dependence:', err);
      } finally {
        if (isMounted) setLoadingDep(false);
      }
    }
    loadDependence();
    return () => { isMounted = false; };
  }, [runId, selectedDepFeature, shapStatus?.status]);

  // Fetch Local Explanation
  const handleFetchLocalExplanation = async (predId: number) => {
    setLoadingLocal(true);
    try {
      const res = await api.getLocalExplanation(runId, predId);
      if (res.success && res.data) {
        setLocalExp(res.data);
      }
    } catch (err) {
      console.error('Error loading local explanation:', err);
    } finally {
      setLoadingLocal(false);
    }
  };

  useEffect(() => {
    if (initialPredictionId !== undefined && initialPredictionId !== null) {
      setLocalPredId(initialPredictionId);
      handleFetchLocalExplanation(initialPredictionId);
    }
  }, [initialPredictionId]);

  // Charts
  // 1. Native Feature Importance Chart
  const nativeChartData = nativeImp?.features ? [
    {
      x: nativeImp.features.slice(0, 15).map(f => f.importance).reverse(),
      y: nativeImp.features.slice(0, 15).map(f => f.feature).reverse(),
      type: 'bar',
      orientation: 'h',
      marker: { color: '#6366f1' }
    }
  ] : [];

  const nativeChartLayout = {
    title: { text: 'Model Feature Importance (Native)', font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Relative Model Importance', gridcolor: '#1e293b' },
    yaxis: { tickfont: { color: '#cbd5e1', size: 11 }, automargin: true },
    height: 380,
    margin: { l: 140, r: 25, t: 40, b: 50 }
  };

  // 2. Permutation Importance Chart
  const permChartData = permImp?.features ? [
    {
      x: permImp.features.slice(0, 15).map(f => f.importance).reverse(),
      y: permImp.features.slice(0, 15).map(f => f.feature).reverse(),
      type: 'bar',
      orientation: 'h',
      marker: { color: '#10b981' }
    }
  ] : [];

  const permChartLayout = {
    title: { text: 'Permutation Importance (Validation Partition)', font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Mean Score Decrease on Permutation', gridcolor: '#1e293b' },
    yaxis: { tickfont: { color: '#cbd5e1', size: 11 }, automargin: true },
    height: 380,
    margin: { l: 140, r: 25, t: 40, b: 50 }
  };

  // 3. SHAP Global Importance Chart
  const shapGlobalChartData = shapSummary?.mean_abs_shap ? [
    {
      x: shapSummary.mean_abs_shap.slice(0, 15).map(f => f.mean_abs_shap).reverse(),
      y: shapSummary.mean_abs_shap.slice(0, 15).map(f => f.feature).reverse(),
      type: 'bar',
      orientation: 'h',
      marker: { color: '#ec4899' }
    }
  ] : [];

  const shapGlobalChartLayout = {
    title: { text: 'Global Feature Impact (Mean |SHAP Value|)', font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'Mean Absolute SHAP Value (Model Attribution)', gridcolor: '#1e293b' },
    yaxis: { tickfont: { color: '#cbd5e1', size: 11 }, automargin: true },
    height: 380,
    margin: { l: 140, r: 25, t: 40, b: 50 }
  };

  // 4. SHAP Dependence Scatter Plot
  const depChartData = depData?.feature_values ? [
    {
      x: depData.feature_values,
      y: depData.shap_values,
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: '#a855f7',
        size: 7,
        opacity: 0.8,
        line: { color: '#c084fc', width: 0.5 }
      }
    }
  ] : [];

  const depChartLayout = {
    title: { text: `SHAP Dependence: ${selectedDepFeature}`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: `${selectedDepFeature} Value`, gridcolor: '#1e293b' },
    yaxis: { title: 'SHAP Value (Attribution to Output)', gridcolor: '#1e293b' },
    height: 380,
    margin: { l: 60, r: 25, t: 40, b: 50 }
  };

  // 5. Local Explanation Attributions Bar
  const localChartData = localExp?.attributions ? [
    {
      x: localExp.attributions.map(a => a.shap_value),
      y: localExp.attributions.map(a => `${a.feature} (${a.feature_value})`),
      type: 'bar',
      orientation: 'h',
      marker: {
        color: localExp.attributions.map(a => a.shap_value >= 0 ? '#10b981' : '#f43f5e')
      }
    }
  ] : [];

  const localChartLayout = {
    title: { text: `Local Feature Attributions for Sample #${localPredId}`, font: { color: '#f8fafc', size: 14 } },
    xaxis: { title: 'SHAP Contribution to Prediction (+ raises, - lowers)', gridcolor: '#1e293b' },
    yaxis: { tickfont: { color: '#cbd5e1', size: 11 }, automargin: true },
    height: Math.max(340, (localExp?.attributions.length || 5) * 26),
    margin: { l: 170, r: 25, t: 40, b: 50 }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Global Disclaimer Banner */}
      <div className="p-4 bg-indigo-950/30 border border-indigo-500/30 rounded-2xl flex items-start gap-3">
        <Brain size={20} className="text-indigo-400 mt-0.5 shrink-0" />
        <div className="text-xs text-slate-300 space-y-1">
          <div className="font-bold text-white text-sm">Model Attribution & Explainability Principles</div>
          <p>
            The values below describe how the trained model weighs and combines features to produce its outputs.
            Feature attributions and SHAP values describe <b>internal model mechanics</b> and <b>do not establish real-world causality</b>.
          </p>
        </div>
      </div>

      {/* Row 1: Model Feature Importance & Permutation Importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Native Feature Importance */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="pb-3 border-b border-slate-800 mb-4">
            <h3 className="text-base font-bold text-white">Model Feature Importance</h3>
            <p className="text-xs text-slate-400">Native model weights/split frequencies from the winning estimator</p>
          </div>
          {nativeImp?.status === 'available' ? (
            <div className="flex-1">
              <PlotlyChart data={nativeChartData} layout={nativeChartLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500 text-sm p-6 text-center">
              <AlertCircle size={28} className="text-slate-600 mb-2" />
              <span>Native feature importance unavailable for this model architecture.</span>
              <span className="text-xs text-slate-400 mt-1">Check permutation importance or SHAP below for model-agnostic explanations.</span>
            </div>
          )}
        </div>

        {/* Permutation Importance */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
          <div className="pb-3 border-b border-slate-800 mb-4">
            <h3 className="text-base font-bold text-white">Permutation Feature Importance</h3>
            <p className="text-xs text-slate-400">Model-agnostic evaluation computed strictly on validation data (5 shuffles)</p>
          </div>
          {permImp?.status === 'available' ? (
            <div className="flex-1">
              <PlotlyChart data={permChartData} layout={permChartLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
              {loadingImp ? 'Calculating permutation importance...' : 'Permutation importance unavailable.'}
            </div>
          )}
        </div>
      </div>

      {/* Row 2: SHAP Explainability Section */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800 mb-6">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles size={18} className="text-pink-400" />
              <h3 className="text-base font-bold text-white">SHAP (SHapley Additive exPlanations)</h3>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Theoretically grounded Shapley values quantifying feature contributions to model outputs.
            </p>
          </div>

          {/* Trigger or status badge */}
          <div className="flex items-center gap-3">
            {shapStatus?.status === 'completed' && (
              <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5">
                <CheckCircle size={14} /> SHAP Computed ({shapSummary?.explainer_type || 'Explainer'})
              </span>
            )}
            {shapStatus?.status === 'running' && (
              <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 flex items-center gap-1.5 animate-pulse">
                <RefreshCw size={14} className="animate-spin" /> Computing SHAP in background...
              </span>
            )}
            {shapStatus?.status !== 'completed' && shapStatus?.status !== 'running' && (
              <button
                onClick={handleTriggerShap}
                disabled={isTriggeringShap}
                className="px-4 py-2 rounded-xl bg-gradient-to-r from-pink-600 to-indigo-600 hover:from-pink-500 hover:to-indigo-500 text-white text-xs font-bold shadow-lg transition-all flex items-center gap-2"
              >
                <Sparkles size={14} />
                Calculate SHAP Explanations
              </button>
            )}
          </div>
        </div>

        {/* SHAP Status Notice if unavailable or failed */}
        {shapStatus?.status === 'unavailable' && (
          <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-center text-xs text-slate-400">
            SHAP explanation is currently unavailable for this specific model configuration.
            <div className="text-[11px] text-slate-500 mt-1">{shapStatus.reason}</div>
          </div>
        )}

        {/* When SHAP is completed: Global Importance + Dependence */}
        {shapStatus?.status === 'completed' && shapSummary && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Global SHAP Mean Bar Chart */}
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
                <PlotlyChart data={shapGlobalChartData} layout={shapGlobalChartLayout} style={{ width: '100%', height: '380px' }} />
              </div>

              {/* SHAP Dependence Analysis */}
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 flex flex-col">
                <div className="flex items-center justify-between gap-3 mb-3">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-300">Feature Dependence</span>
                  <div className="flex items-center gap-2">
                    <label htmlFor="shap-dep-select" className="text-xs text-slate-400">Feature:</label>
                    <select
                      id="shap-dep-select"
                      value={selectedDepFeature}
                      onChange={(e) => setSelectedDepFeature(e.target.value)}
                      className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1 focus:outline-none focus:border-pink-500"
                    >
                      {shapSummary.features?.map((f) => (
                        <option key={f} value={f}>{f}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="flex-1">
                  {loadingDep ? (
                    <div className="h-[380px] flex items-center justify-center text-xs text-slate-500">
                      Loading dependence plot...
                    </div>
                  ) : depData ? (
                    <PlotlyChart data={depChartData} layout={depChartLayout} style={{ width: '100%', height: '380px' }} />
                  ) : (
                    <div className="h-[380px] flex items-center justify-center text-xs text-slate-500">
                      Select a feature above to inspect its SHAP dependence.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Row 3: Local Prediction-Level Explanation Inspector */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800 mb-6">
          <div>
            <h3 className="text-base font-bold text-white">Local Prediction Explanation</h3>
            <p className="text-xs text-slate-400">
              Inspect feature-level positive and negative contributions for any individual test record
            </p>
          </div>

          <div className="flex items-center gap-2">
            <label htmlFor="pred-id-input" className="text-xs text-slate-400 font-mono">Test Sample ID:</label>
            <input
              id="pred-id-input"
              type="number"
              min={0}
              value={localPredId}
              onChange={(e) => setLocalPredId(parseInt(e.target.value) || 0)}
              className="w-20 bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 font-mono focus:outline-none focus:border-indigo-500"
            />
            <button
              onClick={() => handleFetchLocalExplanation(localPredId)}
              disabled={loadingLocal}
              className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-colors"
            >
              {loadingLocal ? 'Computing...' : 'Explain'}
            </button>
          </div>
        </div>

        {/* Explanation Results */}
        {localExp && localExp.status === 'available' ? (
          <div className="space-y-4">
            {/* Top row: Actual, Predicted, Prob */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Sample #</div>
                <div className="text-base font-mono font-bold text-white mt-0.5">#{localExp.prediction_id}</div>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Actual Target</div>
                <div className="text-base font-mono font-bold text-emerald-400 mt-0.5">{String(localExp.actual)}</div>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Model Predicted</div>
                <div className="text-base font-mono font-bold text-indigo-400 mt-0.5">{String(localExp.predicted)}</div>
              </div>
              {localExp.probability !== undefined && localExp.probability !== null && (
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="text-[10px] text-slate-400 uppercase font-mono">Model Probability</div>
                  <div className="text-base font-mono font-bold text-purple-400 mt-0.5">
                    {(localExp.probability * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>

            {/* Attribution Waterfall / Bar */}
            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <PlotlyChart data={localChartData} layout={localChartLayout} style={{ width: '100%', minHeight: '340px' }} />
            </div>

            {/* Local Disclaimer */}
            <div className="p-3 bg-slate-950 border border-slate-800/80 rounded-xl text-xs text-slate-400 flex items-center gap-2">
              <span className="text-indigo-400 font-bold">ℹ️</span>
              <span>{localExp.disclaimer}</span>
            </div>
          </div>
        ) : (
          <div className="p-8 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-xl">
            Enter a sample ID above (or click "Explain with SHAP" from the Error Analysis table) to inspect its prediction attribution breakdown.
          </div>
        )}
      </div>
    </div>
  );
};
