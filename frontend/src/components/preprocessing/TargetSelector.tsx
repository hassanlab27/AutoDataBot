import React, { useState, useEffect } from 'react';
import { TargetInspectResponse } from '../../types/preprocessing';
import { api } from '../../services/api';
import { Target, CheckCircle2, AlertTriangle, HelpCircle, ArrowRight } from 'lucide-react';

interface TargetSelectorProps {
  datasetId: string;
  columns: Array<{ name: string; inferred_dtype: string }>;
  selectedTarget: string | null;
  onConfirmTarget: (target: string, problemType: string, inspection: TargetInspectResponse) => void;
}

export const TargetSelector: React.FC<TargetSelectorProps> = ({
  datasetId,
  columns,
  selectedTarget,
  onConfirmTarget
}) => {
  const [target, setTarget] = useState<string>(selectedTarget || '');
  const [inspection, setInspection] = useState<TargetInspectResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [userProblemType, setUserProblemType] = useState<string>('');

  useEffect(() => {
    if (target) {
      handleInspectTarget(target);
    }
  }, [target]);

  const handleInspectTarget = async (colName: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.inspectTarget(datasetId, colName);
      setInspection(res);
      setUserProblemType(res.problem_type === 'ambiguous' ? 'binary_classification' : res.problem_type);
    } catch (err: any) {
      setError(err?.message || 'Failed to inspect target column');
      setInspection(null);
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = () => {
    if (!inspection || !target) return;
    const finalProblemType = inspection.problem_type === 'ambiguous' 
      ? userProblemType 
      : inspection.problem_type;
    onConfirmTarget(target, finalProblemType, inspection);
  };

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
        <div style={{
          background: 'rgba(59, 130, 246, 0.15)',
          padding: '0.5rem',
          borderRadius: 'var(--radius-md)',
          color: 'var(--accent-blue)'
        }}>
          <Target size={22} />
        </div>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Step 1: Select Prediction Target</h3>
          <p style={{ margin: '0.2rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Choose the column your machine learning model will predict. Target data is strictly segregated to prevent leakage.
          </p>
        </div>
      </div>

      <div style={{ maxWidth: '500px', marginBottom: '1.5rem' }}>
        <label htmlFor="target-column-select" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600, fontSize: '0.9rem' }}>
          Target Column
        </label>
        <select
          id="target-column-select"
          aria-label="Target Column"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="select-input"
          style={{
            width: '100%',
            padding: '0.75rem 1rem',
            background: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-md)',
            fontSize: '0.95rem'
          }}
        >
          <option value="">-- Select Target Column --</option>
          {columns.map(col => (
            <option key={col.name} value={col.name}>
              {col.name} ({col.inferred_dtype})
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Inspecting target distribution and problem type...
        </div>
      )}

      {error && (
        <div style={{
          padding: '1rem',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid var(--accent-red)',
          color: 'var(--accent-red)',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {inspection && !loading && (
        <div style={{
          background: 'var(--bg-secondary)',
          padding: '1.25rem',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-color)',
          marginTop: '1rem'
        }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Detected Problem Type
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem' }}>
                <span style={{
                  fontSize: '1.1rem',
                  fontWeight: 700,
                  color: inspection.problem_type === 'regression' 
                    ? 'var(--accent-blue)' 
                    : inspection.problem_type === 'ambiguous' 
                    ? 'var(--accent-amber)' 
                    : 'var(--accent-green)'
                }}>
                  {inspection.problem_type === 'binary_classification' && 'Binary Classification'}
                  {inspection.problem_type === 'multiclass_classification' && 'Multiclass Classification'}
                  {inspection.problem_type === 'regression' && 'Regression'}
                  {inspection.problem_type === 'ambiguous' && 'Ambiguous Problem Type'}
                </span>
                <span className="badge" style={{
                  background: inspection.confidence === 'high' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                  color: inspection.confidence === 'high' ? 'var(--accent-green)' : 'var(--accent-amber)',
                  fontSize: '0.75rem'
                }}>
                  {inspection.confidence.toUpperCase()} CONFIDENCE
                </span>
              </div>
              <p style={{ margin: '0.35rem 0 0 0', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                {inspection.detection_reason}
              </p>
            </div>

            {/* Quick Metrics */}
            <div style={{ display: 'flex', gap: '1.5rem', background: 'var(--bg-card)', padding: '0.75rem 1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Valid Observations</div>
                <div style={{ fontSize: '1rem', fontWeight: 600 }}>{inspection.diagnostics.valid_count.toLocaleString()}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Distinct Values</div>
                <div style={{ fontSize: '1rem', fontWeight: 600 }}>{inspection.diagnostics.unique_count}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Missingness</div>
                <div style={{ fontSize: '1rem', fontWeight: 600, color: inspection.diagnostics.missing_count > 0 ? 'var(--accent-amber)' : 'inherit' }}>
                  {inspection.diagnostics.missing_percentage}%
                </div>
              </div>
            </div>
          </div>

          {/* ID Warning */}
          {inspection.diagnostics.id_warning && (
            <div style={{
              padding: '0.75rem 1rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid var(--accent-amber)',
              color: 'var(--accent-amber)',
              marginBottom: '1rem',
              fontSize: '0.85rem',
              display: 'flex',
              gap: '0.5rem',
              alignItems: 'center'
            }}>
              <AlertTriangle size={16} />
              <span>{inspection.diagnostics.id_warning}</span>
            </div>
          )}

          {/* Ambiguous User Resolution */}
          {inspection.problem_type === 'ambiguous' && (
            <div style={{
              background: 'rgba(245, 158, 11, 0.08)',
              border: '1px solid var(--accent-amber)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
              marginBottom: '1rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, color: 'var(--accent-amber)', marginBottom: '0.5rem' }}>
                <HelpCircle size={18} />
                <span>Explicit Problem Type Confirmation Required</span>
              </div>
              <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.85rem' }}>
                This target contains numeric ratings/levels. Please confirm whether AutoDataBot should treat it as Classification or Regression:
              </p>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', fontSize: '0.9rem' }}>
                  <input
                    type="radio"
                    name="problem_type_override"
                    value="multiclass_classification"
                    checked={userProblemType === 'multiclass_classification' || userProblemType === 'binary_classification'}
                    onChange={() => setUserProblemType('multiclass_classification')}
                  />
                  <span>Classification (Distinct categories/levels)</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', fontSize: '0.9rem' }}>
                  <input
                    type="radio"
                    name="problem_type_override"
                    value="regression"
                    checked={userProblemType === 'regression'}
                    onChange={() => setUserProblemType('regression')}
                  />
                  <span>Regression (Continuous numerical scale)</span>
                </label>
              </div>
            </div>
          )}

          {/* Classification Diagnostics */}
          {inspection.diagnostics.classification && (
            <div style={{ marginTop: '1rem' }}>
              <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                Target Class Breakdown ({inspection.diagnostics.classification.classes_count} classes):
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.5rem', marginBottom: '0.75rem' }}>
                {inspection.diagnostics.classification.classes.slice(0, 10).map((cls) => (
                  <div key={cls.class_label} style={{
                    background: 'var(--bg-card)',
                    padding: '0.6rem 0.8rem',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontSize: '0.85rem'
                  }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{cls.class_label}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{cls.count.toLocaleString()} ({cls.percentage}%)</span>
                  </div>
                ))}
              </div>

              {inspection.diagnostics.classification.imbalance_warning && (
                <div style={{
                  padding: '0.6rem 0.8rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(245, 158, 11, 0.1)',
                  border: '1px solid var(--accent-amber)',
                  color: 'var(--accent-amber)',
                  fontSize: '0.825rem',
                  display: 'flex',
                  gap: '0.4rem',
                  alignItems: 'center'
                }}>
                  <AlertTriangle size={15} />
                  <span>{inspection.diagnostics.classification.imbalance_warning}</span>
                </div>
              )}
            </div>
          )}

          {/* Regression Diagnostics */}
          {inspection.diagnostics.regression && (
            <div style={{ marginTop: '1rem' }}>
              <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                Continuous Target Distribution Summary:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <div style={{ background: 'var(--bg-card)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Mean</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>{inspection.diagnostics.regression.mean.toLocaleString()}</div>
                </div>
                <div style={{ background: 'var(--bg-card)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Median</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>{inspection.diagnostics.regression.median.toLocaleString()}</div>
                </div>
                <div style={{ background: 'var(--bg-card)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Std Dev</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>{inspection.diagnostics.regression.std.toLocaleString()}</div>
                </div>
                <div style={{ background: 'var(--bg-card)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Min / Max</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                    {inspection.diagnostics.regression.min} - {inspection.diagnostics.regression.max}
                  </div>
                </div>
                <div style={{ background: 'var(--bg-card)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>IQR (Q1 - Q3)</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                    {inspection.diagnostics.regression.q1} - {inspection.diagnostics.regression.q3}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Confirm Button */}
          <div style={{ marginTop: '1.25rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button
              className="btn btn-primary"
              onClick={handleConfirm}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.25rem' }}
            >
              <CheckCircle2 size={16} />
              Confirm Target ({target}) & Proceed to Feature Selection
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
