import React from 'react';
import { PrepareDatasetRequest } from '../../types/preprocessing';
import { ClipboardCheck, Play, ArrowLeft, ShieldCheck, Loader2 } from 'lucide-react';

interface ConfigurationReviewProps {
  datasetId: string;
  config: PrepareDatasetRequest;
  totalCandidateFeatures: number;
  loading: boolean;
  onExecute: () => void;
  onBack: () => void;
}

export const ConfigurationReview: React.FC<ConfigurationReviewProps> = ({
  datasetId,
  config,
  totalCandidateFeatures,
  loading,
  onExecute,
  onBack
}) => {
  const selectedCount = config.selected_features?.length || 0;
  const excludedCount = totalCandidateFeatures - selectedCount;
  const trainPct = Math.round((1 - config.test_size) * 100);
  const testPct = Math.round(config.test_size * 100);

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
        <div style={{
          background: 'rgba(16, 185, 129, 0.15)',
          padding: '0.5rem',
          borderRadius: 'var(--radius-md)',
          color: 'var(--accent-green)'
        }}>
          <ClipboardCheck size={22} />
        </div>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Step 4: Review Preprocessing Specification</h3>
          <p style={{ margin: '0.2rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Verify pipeline parameters before triggering leakage-resistant preparation.
          </p>
        </div>
      </div>

      <div style={{
        background: 'var(--bg-secondary)',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-color)',
        padding: '1.5rem',
        marginBottom: '1.5rem'
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.25rem' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dataset & Target</span>
            <div style={{ marginTop: '0.35rem', fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)' }}>
              Target: <span style={{ color: 'var(--accent-blue)' }}>{config.target}</span>
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Problem: <strong>{config.problem_type.replace('_', ' ')}</strong>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Dataset ID: {datasetId}
            </div>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Feature Partition</span>
            <div style={{ marginTop: '0.35rem', fontWeight: 600, fontSize: '1rem', color: 'var(--accent-green)' }}>
              {selectedCount} Features Included
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              {excludedCount} Features Excluded
            </div>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Data Split</span>
            <div style={{ marginTop: '0.35rem', fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)' }}>
              {trainPct}% Training / {testPct}% Test
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Random Seed: <strong>{config.random_state}</strong>
            </div>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Transformers & Imputers</span>
            <div style={{ marginTop: '0.35rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <div>Num Impute: <strong>{config.numeric_imputation}</strong></div>
              <div>Cat Impute: <strong>{config.categorical_imputation}</strong></div>
              <div>Scaling: <strong>{config.scaling}</strong></div>
              <div>Encoding: <strong>{config.categorical_encoding}</strong></div>
            </div>
          </div>
        </div>

        {/* Leakage Guarantee Alert */}
        <div style={{
          marginTop: '1.25rem',
          padding: '0.75rem 1rem',
          background: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid var(--accent-green)',
          borderRadius: 'var(--radius-sm)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '0.85rem',
          color: 'var(--accent-green)'
        }}>
          <ShieldCheck size={18} />
          <span>
            <strong>Zero-Leakage Assurance:</strong> Preprocessing pipelines will be fitted strictly on the {trainPct}% training split. The {testPct}% test split is held out untouched.
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <button
          className="btn btn-outline"
          onClick={onBack}
          disabled={loading}
          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}
        >
          <ArrowLeft size={16} /> Back to Configuration
        </button>

        <button
          className="btn btn-primary"
          onClick={onExecute}
          disabled={loading}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.75rem 1.75rem',
            fontSize: '1rem',
            fontWeight: 700
          }}
        >
          {loading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              <span>Preparing Dataset & Fitting Pipeline...</span>
            </>
          ) : (
            <>
              <Play size={18} />
              <span>Prepare Dataset</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
