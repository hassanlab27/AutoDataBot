import React, { useState, useEffect, useRef } from 'react';
import { PrepareDatasetResponse } from '../../types/preprocessing';
import { AutoMLConfig, RunStatus } from '../../types/ml';
import { api } from '../../services/api';
import { TrainingProgressModal } from '../ml/TrainingProgressModal';
import {
  CheckCircle,
  Layers,
  FileCode,
  RotateCcw,
  Table,
  ChevronDown,
  ChevronUp,
  Zap,
  Sparkles,
  ArrowRight,
  AlertCircle
} from 'lucide-react';

interface PreprocessingResultsProps {
  result: PrepareDatasetResponse;
  onReset: () => void;
  onNavigateToEvaluation?: (runId: string) => void;
}

export const PreprocessingResults: React.FC<PreprocessingResultsProps> = ({
  result,
  onReset,
  onNavigateToEvaluation
}) => {
  const [showAllFeatures, setShowAllFeatures] = useState<boolean>(false);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [latestRunId, setLatestRunId] = useState<string | null>(null);

  const pollingRef = useRef<any>(null);
  const meta = result.metadata;
  const preview = result.preview;

  // Check if any training runs already exist for this dataset
  useEffect(() => {
    let isMounted = true;
    async function checkRuns() {
      try {
        const res = await api.getDatasetRuns(result.dataset_id);
        if (isMounted && res.success && res.runs.length > 0) {
          const completedRun = res.runs.find((r: any) => r.status === 'completed');
          if (completedRun) {
            setLatestRunId(completedRun.run_id);
          }
        }
      } catch {
        // Runs check is best-effort
      }
    }
    checkRuns();
    return () => {
      isMounted = false;
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [result.dataset_id]);

  // Polling loop for active training job
  useEffect(() => {
    if (!activeRunId || !isTraining) return;

    const pollStatus = async () => {
      try {
        const res = await api.getRunStatus(activeRunId);
        if (res.success && res.data) {
          setRunStatus(res.data);
          if (res.data.status === 'completed') {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
            setIsTraining(false);
            setLatestRunId(activeRunId);
            // Seamlessly direct user to Model Evaluation
            if (onNavigateToEvaluation) {
              onNavigateToEvaluation(activeRunId);
            }
          } else if (res.data.status === 'failed' || res.data.status === 'cancelled') {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
            setIsTraining(false);
            if (res.data.error_message) {
              setTrainingError(res.data.error_message);
            }
          }
        }
      } catch (err) {
        console.error('Error polling training status:', err);
      }
    };

    pollingRef.current = setInterval(pollStatus, 1500);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [activeRunId, isTraining, onNavigateToEvaluation]);

  // One-Click Model Training Handler
  const handleStartTraining = async () => {
    setIsTraining(true);
    setTrainingError(null);
    try {
      const config: AutoMLConfig = {
        dataset_id: result.dataset_id,
        target: result.target,
        problem_type: result.problem_type,
        primary_metric: 'auto',
        training_mode: 'quick',
        presets: 'medium_quality_faster_train',
        engine_preference: 'all',
        random_state: 42
      };

      const runRes = await api.startRun(config);
      if (runRes.success && runRes.data?.run_id) {
        const newRunId = runRes.data.run_id;
        setActiveRunId(newRunId);
        setRunStatus({
          run_id: newRunId,
          dataset_id: result.dataset_id,
          status: 'running',
          current_stage: 'Initializing models...',
          current_engine: 'all',
          progress_pct: 5,
          start_time: Date.now() / 1000,
          elapsed_seconds: 0,
          cancellation_requested: false
        });
      } else {
        throw new Error('Could not initiate training job.');
      }
    } catch (err: any) {
      setTrainingError(err?.message || 'Failed to start automated training.');
      setIsTraining(false);
    }
  };

  const handleCancelTraining = async () => {
    if (!activeRunId) return;
    setIsCancelling(true);
    try {
      await api.cancelRun(activeRunId);
    } catch (err) {
      console.error('Error cancelling run:', err);
    } finally {
      setIsCancelling(false);
    }
  };

  return (
    <div>
      {/* Success Notification */}
      <div style={{
        background: 'rgba(16, 185, 129, 0.1)',
        border: '1px solid var(--accent-green)',
        borderRadius: 'var(--radius-md)',
        padding: '1.25rem 1.5rem',
        marginBottom: '1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            background: 'var(--accent-green)',
            color: '#fff',
            padding: '0.5rem',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <CheckCircle size={24} />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.15rem', color: 'var(--accent-green)' }}>
              Dataset Prepared Successfully
            </h3>
            <p style={{ margin: '0.2rem 0 0 0', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Transformations fitted strictly on {meta.split.train_rows.toLocaleString()} training records. Pipeline serialized to disk.
            </p>
          </div>
        </div>

        <button
          className="btn btn-outline"
          onClick={onReset}
          style={{ fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
        >
          <RotateCcw size={15} /> Modify Configuration
        </button>
      </div>

      {/* ONE-CLICK MODEL TRAINING SECTION */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(79, 70, 229, 0.15) 0%, rgba(124, 58, 237, 0.1) 50%, rgba(15, 23, 42, 0.8) 100%)',
        border: '2px solid rgba(99, 102, 241, 0.4)',
        borderRadius: 'var(--radius-lg, 16px)',
        padding: '1.75rem',
        marginBottom: '1.75rem',
        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Background glow accent */}
        <div style={{
          position: 'absolute',
          top: '-50px',
          right: '-50px',
          width: '200px',
          height: '200px',
          background: 'rgba(99, 102, 241, 0.15)',
          borderRadius: '50%',
          filter: 'blur(50px)',
          pointerEvents: 'none'
        }} />

        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1.25rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
              <span style={{ fontSize: '1.25rem' }}>⚡</span>
              <span style={{ textTransform: 'uppercase', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.05em', color: '#818cf8' }}>
                Next Step: Automated Model Training
              </span>
            </div>
            <h3 style={{ margin: 0, fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
              Train Machine Learning Models
            </h3>
            <p style={{ margin: '0.4rem 0 0 0', color: '#cbd5e1', fontSize: '0.875rem', maxWidth: '650px' }}>
              Click the button to automatically train and benchmark multiple candidate models (LightGBM, Random Forest, Logistic/Ridge, HistGradientBoosting, AutoGluon) on your prepared dataset with zero test-set leakage. Once complete, you will be directed straight to the evaluation dashboard.
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.75rem' }}>
              <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem', borderRadius: '9999px', background: 'rgba(99, 102, 241, 0.2)', border: '1px solid rgba(99, 102, 241, 0.4)', color: '#c7d2fe', fontFamily: 'monospace' }}>
                Target: <b>{result.target}</b>
              </span>
              <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem', borderRadius: '9999px', background: 'rgba(168, 85, 247, 0.2)', border: '1px solid rgba(168, 85, 247, 0.4)', color: '#e9d5ff', fontFamily: 'monospace' }}>
                Problem: <b>{result.problem_type.replace('_', ' ')}</b>
              </span>
              <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem', borderRadius: '9999px', background: 'rgba(16, 185, 129, 0.2)', border: '1px solid rgba(16, 185, 129, 0.4)', color: '#a7f3d0', fontFamily: 'monospace' }}>
                Prepared Features: <b>{meta.final_features_count}</b>
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', minWidth: '220px' }}>
            <button
              type="button"
              onClick={handleStartTraining}
              disabled={isTraining}
              style={{
                padding: '0.85rem 1.5rem',
                fontSize: '0.95rem',
                fontWeight: 700,
                color: '#ffffff',
                background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                border: 'none',
                borderRadius: 'var(--radius-md, 8px)',
                cursor: isTraining ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 14px 0 rgba(99, 102, 241, 0.4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                opacity: isTraining ? 0.7 : 1,
                transition: 'all 0.2s ease'
              }}
            >
              <Zap size={18} />
              <span>{isTraining ? 'Training in Progress...' : 'Train Complete Model'}</span>
            </button>

            {latestRunId && !isTraining && onNavigateToEvaluation && (
              <button
                type="button"
                onClick={() => onNavigateToEvaluation(latestRunId)}
                className="btn btn-outline"
                style={{
                  padding: '0.5rem 1rem',
                  fontSize: '0.8rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.4rem',
                  borderColor: 'rgba(148, 163, 184, 0.4)',
                  color: '#cbd5e1'
                }}
              >
                <Sparkles size={14} className="text-pink-400" />
                <span>View Latest Evaluation</span>
                <ArrowRight size={14} />
              </button>
            )}
          </div>
        </div>

        {trainingError && (
          <div style={{
            marginTop: '1rem',
            padding: '0.75rem 1rem',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            color: '#fca5a5',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <AlertCircle size={16} />
            <span>{trainingError}</span>
          </div>
        )}

        {/* Live Training Progress Modal / Overlay */}
        {isTraining && runStatus && (
          <div style={{ marginTop: '1.25rem' }}>
            <TrainingProgressModal
              status={runStatus}
              onCancel={handleCancelTraining}
              isCancelling={isCancelling}
            />
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '1rem',
        marginBottom: '1.5rem'
      }}>
        <div className="card" style={{ padding: '1rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Feature Expansion</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.35rem', color: 'var(--accent-blue)' }}>
            {meta.final_features_count}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            From {meta.selected_features_count} selected inputs
          </div>
        </div>

        <div className="card" style={{ padding: '1rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Partition Rows</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.35rem', color: 'var(--accent-green)' }}>
            {meta.split.train_rows.toLocaleString()} / {meta.split.test_rows.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            {meta.split.train_percentage}% Train / {meta.split.test_percentage}% Test
          </div>
        </div>

        <div className="card" style={{ padding: '1rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>One-Hot Encoded</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.35rem', color: 'var(--accent-purple)' }}>
            {meta.one_hot_generated_features_count}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            Generated binary dummy features
          </div>
        </div>

        <div className="card" style={{ padding: '1rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Datetime Components</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.35rem', color: 'var(--accent-amber)' }}>
            {meta.generated_datetime_features_count}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            Extracted calendar features
          </div>
        </div>
      </div>

      {/* Serialized Artifacts Information */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <FileCode size={20} color="var(--accent-blue)" />
          <h4 style={{ margin: 0, fontSize: '1rem' }}>Saved Preprocessing Pipeline Artifacts</h4>
        </div>
        <p style={{ margin: '0 0 1rem 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          The serialized scikit-learn pipeline and audit configurations are saved in the project output repository:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.75rem' }}>
          <div style={{ background: 'var(--bg-secondary)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--accent-green)' }}>pipeline.joblib</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Full scikit-learn ColumnTransformer & Datetime Extractor
            </div>
          </div>

          <div style={{ background: 'var(--bg-secondary)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--accent-blue)' }}>config.json</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Reproducible parameters, test size, seed, and strategies
            </div>
          </div>

          <div style={{ background: 'var(--bg-secondary)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--accent-purple)' }}>metadata.json</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Feature counts, generated names, and partition profiles
            </div>
          </div>
        </div>
      </div>

      {/* Feature Names List */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={18} color="var(--accent-purple)" />
            <h4 style={{ margin: 0, fontSize: '1rem' }}>
              Transformed Feature Space ({meta.feature_names_out.length} Features)
            </h4>
          </div>
          <button
            className="btn btn-outline"
            onClick={() => setShowAllFeatures(!showAllFeatures)}
            style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
          >
            {showAllFeatures ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {showAllFeatures ? 'Collapse' : 'Expand All'}
          </button>
        </div>

        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.4rem',
          maxHeight: showAllFeatures ? 'none' : '120px',
          overflow: 'hidden',
          position: 'relative'
        }}>
          {meta.feature_names_out.map(f => (
            <span key={f} className="badge" style={{ fontSize: '0.75rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
              {f}
            </span>
          ))}
          {!showAllFeatures && meta.feature_names_out.length > 20 && (
            <div style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: '40px',
              background: 'linear-gradient(transparent, var(--bg-card))',
              pointerEvents: 'none'
            }} />
          )}
        </div>
      </div>

      {/* Processed Data Preview Table */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <Table size={20} color="var(--accent-blue)" />
          <div>
            <h4 style={{ margin: 0, fontSize: '1rem' }}>Processed Training Matrix Sample Preview</h4>
            <p style={{ margin: '0.2rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              First {preview.preview_rows.length} records after missing imputation, categorical encoding, and scaling.
            </p>
          </div>
        </div>

        <div style={{ overflowX: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-color)' }}>
                <th style={{ padding: '0.6rem 0.8rem', width: '40px', color: 'var(--text-muted)' }}>#</th>
                {preview.columns.map(col => (
                  <th key={col} style={{ padding: '0.6rem 0.8rem', whiteSpace: 'nowrap' }}>
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.preview_rows.map((row, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid var(--border-color)' }}>
                  <td style={{ padding: '0.5rem 0.8rem', color: 'var(--text-muted)' }}>{idx + 1}</td>
                  {preview.columns.map(col => (
                    <td key={col} style={{ padding: '0.5rem 0.8rem', whiteSpace: 'nowrap' }}>
                      {row[col] !== null && row[col] !== undefined ? String(row[col]) : '-'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
