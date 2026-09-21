import React, { useState } from 'react';
import { PrepareDatasetResponse } from '../../types/preprocessing';
import { CheckCircle, Layers, FileCode, RotateCcw, Table, ChevronDown, ChevronUp } from 'lucide-react';

interface PreprocessingResultsProps {
  result: PrepareDatasetResponse;
  onReset: () => void;
}

export const PreprocessingResults: React.FC<PreprocessingResultsProps> = ({
  result,
  onReset
}) => {
  const [showAllFeatures, setShowAllFeatures] = useState<boolean>(false);
  const meta = result.metadata;
  const preview = result.preview;

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
