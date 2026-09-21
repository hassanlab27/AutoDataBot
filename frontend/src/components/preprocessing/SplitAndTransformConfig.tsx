import React from 'react';
import { PrepareDatasetRequest } from '../../types/preprocessing';
import { Sliders, Calendar, ArrowRight, ArrowLeft, ShieldCheck } from 'lucide-react';

interface SplitAndTransformConfigProps {
  config: PrepareDatasetRequest;
  problemType: string;
  onChangeConfig: (newConfig: PrepareDatasetRequest) => void;
  onProceed: () => void;
  onBack: () => void;
}

export const SplitAndTransformConfig: React.FC<SplitAndTransformConfigProps> = ({
  config,
  problemType,
  onChangeConfig,
  onProceed,
  onBack
}) => {
  const isClassification = problemType.includes('classification');

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
        <div style={{
          background: 'rgba(139, 92, 246, 0.15)',
          padding: '0.5rem',
          borderRadius: 'var(--radius-md)',
          color: 'var(--accent-purple)'
        }}>
          <Sliders size={22} />
        </div>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Step 3: Configure Split & Preprocessing</h3>
          <p style={{ margin: '0.2rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Transformations will be fitted strictly on the training partition to eliminate leakage.
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* Partition Configuration */}
        <div style={{
          background: 'var(--bg-secondary)',
          padding: '1.25rem',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-color)'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.95rem', color: 'var(--text-primary)' }}>
            1. Train / Test Split
          </h4>

          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', fontSize: '0.85rem' }}>
              <label htmlFor="test-size-range" style={{ fontWeight: 600 }}>Test Set Ratio</label>
              <span style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>
                {Math.round(config.test_size * 100)}% Test / {Math.round((1 - config.test_size) * 100)}% Train
              </span>
            </div>
            <input
              id="test-size-range"
              aria-label="Test Set Ratio"
              type="range"
              min="0.10"
              max="0.40"
              step="0.05"
              value={config.test_size}
              onChange={(e) => onChangeConfig({ ...config, test_size: parseFloat(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
            />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="random-seed-input" style={{ display: 'block', marginBottom: '0.4rem', fontSize: '0.85rem', fontWeight: 600 }}>
              Random Seed (Reproducibility)
            </label>
            <input
              id="random-seed-input"
              aria-label="Random Seed"
              type="number"
              value={config.random_state}
              onChange={(e) => onChangeConfig({ ...config, random_state: parseInt(e.target.value) || 42 })}
              className="select-input"
              style={{
                width: '100%',
                padding: '0.5rem 0.75rem',
                background: 'var(--bg-card)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.9rem'
              }}
            />
          </div>

          <div style={{
            background: 'var(--bg-card)',
            padding: '0.75rem',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-color)',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <ShieldCheck size={16} color="var(--accent-green)" />
            <span>
              {isClassification 
                ? 'Stratified shuffle split enabled: Class distributions will be preserved in train and test.'
                : 'Random shuffle split enabled for continuous target regression.'}
            </span>
          </div>
        </div>

        {/* Feature Transformations */}
        <div style={{
          background: 'var(--bg-secondary)',
          padding: '1.25rem',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-color)'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.95rem', color: 'var(--text-primary)' }}>
            2. Missing Values & Feature Encoders
          </h4>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
            <div>
              <label htmlFor="numeric-imputation-select" style={{ display: 'block', marginBottom: '0.3rem', fontSize: '0.8rem', fontWeight: 600 }}>
                Numerical Imputation
              </label>
              <select
                id="numeric-imputation-select"
                aria-label="Numerical Imputation"
                value={config.numeric_imputation}
                onChange={(e) => onChangeConfig({ ...config, numeric_imputation: e.target.value })}
                className="select-input"
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.85rem'
                }}
              >
                <option value="median">Median (Robust to outliers)</option>
                <option value="mean">Mean (Standard)</option>
                <option value="most_frequent">Most Frequent (Mode)</option>
              </select>
            </div>

            <div>
              <label htmlFor="categorical-imputation-select" style={{ display: 'block', marginBottom: '0.3rem', fontSize: '0.8rem', fontWeight: 600 }}>
                Categorical Imputation
              </label>
              <select
                id="categorical-imputation-select"
                aria-label="Categorical Imputation"
                value={config.categorical_imputation}
                onChange={(e) => onChangeConfig({ ...config, categorical_imputation: e.target.value })}
                className="select-input"
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.85rem'
                }}
              >
                <option value="most_frequent">Most Frequent (Mode)</option>
                <option value="constant">Explicit '__MISSING__' category</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
            <div>
              <label htmlFor="categorical-encoding-select" style={{ display: 'block', marginBottom: '0.3rem', fontSize: '0.8rem', fontWeight: 600 }}>
                Categorical Encoding
              </label>
              <select
                id="categorical-encoding-select"
                aria-label="Categorical Encoding"
                value={config.categorical_encoding}
                onChange={(e) => onChangeConfig({ ...config, categorical_encoding: e.target.value })}
                className="select-input"
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.85rem'
                }}
              >
                <option value="one_hot">One-Hot (handle_unknown='ignore')</option>
              </select>
            </div>

            <div>
              <label htmlFor="scaling-select" style={{ display: 'block', marginBottom: '0.3rem', fontSize: '0.8rem', fontWeight: 600 }}>
                Numerical Scaling
              </label>
              <select
                id="scaling-select"
                aria-label="Numerical Scaling"
                value={config.scaling}
                onChange={(e) => onChangeConfig({ ...config, scaling: e.target.value })}
                className="select-input"
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.85rem'
                }}
              >
                <option value="standard">StandardScaler (Zero mean, unit variance)</option>
                <option value="none">None (Raw values)</option>
              </select>
            </div>
          </div>

          {/* Datetime Extraction Toggle */}
          <div style={{
            background: 'var(--bg-card)',
            padding: '0.75rem',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Calendar size={16} color="var(--accent-blue)" />
              <div>
                <label htmlFor="datetime-extract-checkbox" style={{ fontSize: '0.85rem', fontWeight: 600, cursor: 'pointer' }}>
                  Datetime Component Extraction
                </label>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Extract year, month, day, and dayofweek from timestamps
                </div>
              </div>
            </div>
            <label htmlFor="datetime-extract-checkbox" style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <input
                id="datetime-extract-checkbox"
                aria-label="Extract Datetime Components"
                type="checkbox"
                checked={config.extract_datetime}
                onChange={(e) => onChangeConfig({ ...config, extract_datetime: e.target.checked })}
                style={{ width: '18px', height: '18px', accentColor: 'var(--accent-blue)' }}
              />
            </label>
          </div>
        </div>
      </div>

      {/* Navigation Buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <button className="btn btn-outline" onClick={onBack} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <ArrowLeft size={16} /> Back to Feature Selection
        </button>

        <button
          className="btn btn-primary"
          onClick={onProceed}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.25rem' }}
        >
          <span>Review Configuration</span>
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
};
