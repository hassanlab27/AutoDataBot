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
import { Sparkles, Brain, AlertCircle, RefreshCw, CheckCircle, Info } from 'lucide-react';

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
    <div className="eval-container">
      {/* Global Disclaimer Banner */}
      <div className="eval-notice-box">
        <Brain size={22} style={{ color: '#818cf8', flexShrink: 0, marginTop: '2px' }} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <div style={{ fontWeight: 800, color: '#ffffff', fontSize: '0.85rem' }}>Model Attribution & Explainability Principles</div>
          <p style={{ margin: 0, fontSize: '0.8rem', lineHeight: 1.5, color: '#c7d2fe' }}>
            The values below describe how the trained model weighs and combines features to produce its outputs.
            Feature attributions and SHAP values describe <b>internal model mechanics</b> and <b>do not establish real-world causality</b>.
          </p>
        </div>
      </div>

      {/* Row 1: Model Feature Importance & Permutation Importance */}
      <div className="eval-grid-2">
        {/* Native Feature Importance */}
        <div className="eval-card">
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Model Feature Importance</h3>
              <p className="eval-card-desc">Native model weights or split frequencies from the winning estimator</p>
            </div>
            <span className="eval-badge eval-badge-indigo">Native Weights</span>
          </div>
          {nativeImp?.status === 'available' ? (
            <div>
              <PlotlyChart data={nativeChartData} layout={nativeChartLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div style={{ minHeight: '340px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '1.5rem', color: 'var(--eval-text-muted)' }}>
              <AlertCircle size={32} style={{ color: 'var(--eval-text-dim)', marginBottom: '0.75rem' }} />
              <span style={{ fontWeight: 600 }}>Native feature importance is unavailable for this model architecture.</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--eval-text-dim)', marginTop: '0.35rem' }}>
                Check permutation importance or SHAP below for model-agnostic explanations.
              </span>
            </div>
          )}
        </div>

        {/* Permutation Importance */}
        <div className="eval-card">
          <div className="eval-card-header">
            <div>
              <h3 className="eval-card-title">Permutation Feature Importance</h3>
              <p className="eval-card-desc">Model-agnostic evaluation computed strictly on validation data (5 shuffles)</p>
            </div>
            <span className="eval-badge eval-badge-emerald">Validation Shuffles</span>
          </div>
          {permImp?.status === 'available' ? (
            <div>
              <PlotlyChart data={permChartData} layout={permChartLayout} style={{ width: '100%', height: '380px' }} />
            </div>
          ) : (
            <div style={{ minHeight: '340px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--eval-text-dim)', fontSize: '0.85rem' }}>
              {loadingImp ? 'Calculating permutation importance...' : 'Permutation importance unavailable.'}
            </div>
          )}
        </div>
      </div>

      {/* Row 2: SHAP Explainability Section */}
      <div className="eval-card">
        <div className="eval-card-header">
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Sparkles size={18} style={{ color: '#ec4899' }} />
              <h3 className="eval-card-title">SHAP (SHapley Additive exPlanations)</h3>
            </div>
            <p className="eval-card-desc">
              Theoretically grounded Shapley values quantifying feature contributions to model outputs.
            </p>
          </div>

          {/* Trigger or status badge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {shapStatus?.status === 'completed' && (
              <span className="eval-badge eval-badge-emerald">
                <CheckCircle size={14} /> SHAP Computed ({shapSummary?.explainer_type || 'Explainer'})
              </span>
            )}
            {shapStatus?.status === 'running' && (
              <span className="eval-badge eval-badge-indigo">
                <RefreshCw size={14} className="animate-spin" /> Computing SHAP in background...
              </span>
            )}
            {shapStatus?.status !== 'completed' && shapStatus?.status !== 'running' && (
              <button
                onClick={handleTriggerShap}
                disabled={isTriggeringShap}
                className="eval-btn-primary eval-btn-pink"
                aria-label="Calculate SHAP explanations"
              >
                <Sparkles size={14} />
                Calculate SHAP Explanations
              </button>
            )}
          </div>
        </div>

        {/* SHAP Status Notice if unavailable or failed */}
        {shapStatus?.status === 'unavailable' && (
          <div className="eval-notice-box eval-notice-warning" style={{ textAlign: 'center', justifyContent: 'center' }}>
            <div>
              SHAP explanation is currently unavailable for this specific model configuration.
              <div style={{ fontSize: '0.75rem', marginTop: '0.25rem', opacity: 0.8 }}>{shapStatus.reason}</div>
            </div>
          </div>
        )}

        {/* When SHAP is completed: Global Importance + Dependence */}
        {shapStatus?.status === 'completed' && shapSummary && (
          <div className="eval-grid-2">
            {/* Global SHAP Mean Bar Chart */}
            <div className="eval-card" style={{ background: 'rgba(10, 15, 29, 0.65)' }}>
              <div className="eval-card-header">
                <div>
                  <h4 className="eval-card-title" style={{ fontSize: '0.95rem' }}>Global Feature Impact</h4>
                  <p className="eval-card-desc">Mean |SHAP value| across evaluated test partition</p>
                </div>
              </div>
              <PlotlyChart data={shapGlobalChartData} layout={shapGlobalChartLayout} style={{ width: '100%', height: '380px' }} />
            </div>

            {/* SHAP Dependence Analysis */}
            <div className="eval-card" style={{ background: 'rgba(10, 15, 29, 0.65)', display: 'flex', flexDirection: 'column' }}>
              <div className="eval-card-header">
                <div>
                  <h4 className="eval-card-title" style={{ fontSize: '0.95rem' }}>Feature Dependence</h4>
                  <p className="eval-card-desc">Feature values vs corresponding SHAP attribution</p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <label htmlFor="shap-dep-select" style={{ fontSize: '0.75rem', color: 'var(--eval-text-muted)', fontWeight: 600 }}>
                    Feature:
                  </label>
                  <select
                    id="shap-dep-select"
                    value={selectedDepFeature}
                    onChange={(e) => setSelectedDepFeature(e.target.value)}
                    className="eval-select"
                    aria-label="Select feature for SHAP dependence plot"
                  >
                    {shapSummary.features?.map((f) => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div style={{ flex: 1 }}>
                {loadingDep ? (
                  <div style={{ height: '380px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', color: 'var(--eval-text-dim)' }}>
                    Loading dependence plot...
                  </div>
                ) : depData ? (
                  <PlotlyChart data={depChartData} layout={depChartLayout} style={{ width: '100%', height: '380px' }} />
                ) : (
                  <div style={{ height: '380px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', color: 'var(--eval-text-dim)' }}>
                    Select a feature above to inspect its SHAP dependence.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Row 3: Local Prediction-Level Explanation Inspector */}
      <div className="eval-card">
        <div className="eval-card-header">
          <div>
            <h3 className="eval-card-title">Local Prediction Explanation</h3>
            <p className="eval-card-desc">
              Inspect feature-level positive and negative contributions for any individual test record
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label htmlFor="pred-id-input" style={{ fontSize: '0.75rem', color: 'var(--eval-text-muted)', fontFamily: 'monospace' }}>
              Test Row #:
            </label>
            <input
              id="pred-id-input"
              type="number"
              min={0}
              value={localPredId}
              onChange={(e) => setLocalPredId(parseInt(e.target.value) || 0)}
              className="eval-input"
              style={{ width: '80px' }}
              aria-label="Enter test sample row number to explain"
            />
            <button
              onClick={() => handleFetchLocalExplanation(localPredId)}
              disabled={loadingLocal}
              className="eval-btn-primary eval-btn-sm"
              aria-label="Compute local explanation"
            >
              {loadingLocal ? 'Computing...' : 'Explain Sample'}
            </button>
          </div>
        </div>

        {/* Explanation Results */}
        {localExp && localExp.status === 'available' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {/* Top row: Actual, Predicted, Prob */}
            <div className="eval-grid-4">
              <div className="eval-stat-card">
                <span className="eval-stat-label">Sample ID</span>
                <span className="eval-stat-val">#{localExp.prediction_id}</span>
                <span className="eval-stat-subtext">Test Partition Row</span>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Actual Target</span>
                <span className="eval-stat-val eval-stat-val-green">{String(localExp.actual)}</span>
                <span className="eval-stat-subtext">Ground Truth</span>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Model Predicted</span>
                <span className="eval-stat-val eval-stat-val-indigo">{String(localExp.predicted)}</span>
                <span className="eval-stat-subtext">Selected Estimator</span>
              </div>
              {localExp.probability !== undefined && localExp.probability !== null && (
                <div className="eval-stat-card">
                  <span className="eval-stat-label">Confidence</span>
                  <span className="eval-stat-val eval-stat-val-purple" style={{ color: '#d8b4fe' }}>
                    {(localExp.probability * 100).toFixed(1)}%
                  </span>
                  <span className="eval-stat-subtext">Class Probability</span>
                </div>
              )}
            </div>

            {/* Attribution Waterfall / Bar */}
            <div style={{ background: 'rgba(10, 15, 29, 0.6)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '12px', padding: '1rem' }}>
              <PlotlyChart data={localChartData} layout={localChartLayout} style={{ width: '100%', minHeight: '340px' }} />
            </div>

            {/* Local Disclaimer */}
            <div className="eval-notice-box">
              <Info size={16} style={{ color: '#818cf8', flexShrink: 0, marginTop: '2px' }} />
              <span style={{ fontSize: '0.775rem' }}>{localExp.disclaimer}</span>
            </div>
          </div>
        ) : (
          <div style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--eval-text-dim)', fontSize: '0.825rem', border: '1px dashed rgba(255, 255, 255, 0.1)', borderRadius: '14px' }}>
            Enter a sample ID above (or click <b style={{ color: '#a5b4fc' }}>"Explain with SHAP"</b> in the Worst Test Predictions table) to inspect its prediction attribution breakdown.
          </div>
        )}
      </div>
    </div>
  );
};
