import React, { useState, useEffect } from 'react';
import { FeatureAuditResponse, FeatureAuditItem } from '../../types/preprocessing';
import { api } from '../../services/api';
import { Check, ShieldAlert, ArrowRight, ArrowLeft } from 'lucide-react';

interface FeatureSelectorTableProps {
  datasetId: string;
  targetCol: string;
  selectedFeatures: string[];
  onChangeSelectedFeatures: (features: string[]) => void;
  onProceed: () => void;
  onBack: () => void;
}

export const FeatureSelectorTable: React.FC<FeatureSelectorTableProps> = ({
  datasetId,
  targetCol,
  selectedFeatures,
  onChangeSelectedFeatures,
  onProceed,
  onBack
}) => {
  const [audit, setAudit] = useState<FeatureAuditResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAudit = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.auditFeatures(datasetId, targetCol);
        setAudit(res);
        // If no prior selection, pre-select recommended included features
        if (selectedFeatures.length === 0) {
          const recommended = res.features
            .filter(f => f.recommended_action === 'include')
            .map(f => f.name);
          onChangeSelectedFeatures(recommended);
        }
      } catch (err: any) {
        setError(err?.message || 'Failed to audit dataset features');
      } finally {
        setLoading(false);
      }
    };
    fetchAudit();
  }, [datasetId, targetCol]);

  const toggleFeature = (name: string) => {
    if (selectedFeatures.includes(name)) {
      onChangeSelectedFeatures(selectedFeatures.filter(f => f !== name));
    } else {
      onChangeSelectedFeatures([...selectedFeatures, name]);
    }
  };

  const handleSelectAll = () => {
    if (!audit) return;
    onChangeSelectedFeatures(audit.features.map(f => f.name));
  };

  const handleDeselectAll = () => {
    onChangeSelectedFeatures([]);
  };

  const handleApplyRecommended = () => {
    if (!audit) return;
    const recommended = audit.features
      .filter(f => f.recommended_action === 'include')
      .map(f => f.name);
    onChangeSelectedFeatures(recommended);
  };

  if (loading) {
    return (
      <div className="card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Auditing features for potential leakage, identifiers, and high cardinality...
      </div>
    );
  }

  if (error || !audit) {
    return (
      <div className="card" style={{ padding: '1.5rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--accent-red)' }}>
        <p style={{ color: 'var(--accent-red)', margin: 0 }}>{error || 'Unable to audit features.'}</p>
        <button className="btn btn-outline" onClick={onBack} style={{ marginTop: '1rem' }}>
          Back to Target Selection
        </button>
      </div>
    );
  }

  const excludedCount = audit.features.length - selectedFeatures.length;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Step 2: Feature Selection & Leakage Audit</h3>
          <p style={{ margin: '0.2rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Select features to include in training. High-cardinality, ID-like, and constant columns have been flagged with diagnostic explanations.
          </p>
        </div>

        {/* Counter Badges */}
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.15)', color: 'var(--accent-green)', padding: '0.4rem 0.75rem' }}>
            {selectedFeatures.length} Included
          </span>
          <span className="badge" style={{ background: 'rgba(107, 114, 128, 0.2)', color: 'var(--text-muted)', padding: '0.4rem 0.75rem' }}>
            {excludedCount} Excluded
          </span>
        </div>
      </div>

      {/* Target Segregation Notice */}
      <div style={{
        background: 'rgba(59, 130, 246, 0.08)',
        border: '1px solid rgba(59, 130, 246, 0.3)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.6rem 1rem',
        marginBottom: '1rem',
        fontSize: '0.85rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        color: 'var(--accent-blue)'
      }}>
        <ShieldAlert size={16} />
        <span>
          Target column <strong>{targetCol}</strong> has been strictly excluded from the feature candidate matrix to prevent target leakage.
        </span>
      </div>

      {/* Control Actions */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
        <button className="btn btn-outline" onClick={handleApplyRecommended} style={{ fontSize: '0.825rem', padding: '0.4rem 0.8rem' }}>
          <Check size={14} /> Apply Smart Recommendations
        </button>
        <button className="btn btn-outline" onClick={handleSelectAll} style={{ fontSize: '0.825rem', padding: '0.4rem 0.8rem' }}>
          Include All ({audit.features.length})
        </button>
        <button className="btn btn-outline" onClick={handleDeselectAll} style={{ fontSize: '0.825rem', padding: '0.4rem 0.8rem' }}>
          Exclude All
        </button>
      </div>

      {/* Feature Table */}
      <div style={{ overflowX: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-color)' }}>
              <th scope="col" style={{ padding: '0.75rem 1rem', width: '40px' }}>
                <input
                  type="checkbox"
                  aria-label="Select all candidate features"
                  checked={selectedFeatures.length === audit.features.length && audit.features.length > 0}
                  onChange={(e) => e.target.checked ? handleSelectAll() : handleDeselectAll()}
                />
              </th>
              <th scope="col" style={{ padding: '0.75rem 1rem' }}>Feature</th>
              <th scope="col" style={{ padding: '0.75rem 1rem' }}>Inferred Type</th>
              <th scope="col" style={{ padding: '0.75rem 1rem' }}>Missing %</th>
              <th scope="col" style={{ padding: '0.75rem 1rem' }}>Unique Count</th>
              <th scope="col" style={{ padding: '0.75rem 1rem' }}>Audit Status</th>
              <th scope="col" style={{ padding: '0.75rem 1rem' }}>Reason / Details</th>
            </tr>
          </thead>
          <tbody>
            {audit.features.map((feat: FeatureAuditItem) => {
              const isSelected = selectedFeatures.includes(feat.name);
              return (
                <tr
                  key={feat.name}
                  style={{
                    borderBottom: '1px solid var(--border-color)',
                    background: isSelected ? 'transparent' : 'rgba(255, 255, 255, 0.02)',
                    opacity: isSelected ? 1 : 0.65
                  }}
                >
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <input
                      type="checkbox"
                      aria-label={`Select feature ${feat.name}`}
                      checked={isSelected}
                      onChange={() => toggleFeature(feat.name)}
                    />
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600 }}>
                    {feat.name}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span className="badge" style={{ fontSize: '0.75rem' }}>
                      {feat.inferred_dtype}
                    </span>
                  </td>
                  <td style={{
                    padding: '0.75rem 1rem',
                    color: feat.missing_percentage > 20 ? 'var(--accent-amber)' : 'inherit'
                  }}>
                    {feat.missing_percentage}% ({feat.missing_count})
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    {feat.unique_count.toLocaleString()} ({feat.unique_percentage}%)
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    {feat.status === 'included' && (
                      <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.15)', color: 'var(--accent-green)' }}>
                        Included
                      </span>
                    )}
                    {feat.status === 'flagged_exclude' && (
                      <span className="badge" style={{ background: 'rgba(239, 68, 68, 0.15)', color: 'var(--accent-red)' }}>
                        Flagged Exclude
                      </span>
                    )}
                    {feat.status === 'review_recommended' && (
                      <span className="badge" style={{ background: 'rgba(245, 158, 11, 0.15)', color: 'var(--accent-amber)' }}>
                        Review Needed
                      </span>
                    )}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)', fontSize: '0.8rem', maxWidth: '300px' }}>
                    {feat.reason || 'Normal feature characteristics.'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Navigation Buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.5rem' }}>
        <button className="btn btn-outline" onClick={onBack} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <ArrowLeft size={16} /> Back to Target
        </button>

        <button
          className="btn btn-primary"
          onClick={onProceed}
          disabled={selectedFeatures.length === 0}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.25rem' }}
        >
          <span>Continue with {selectedFeatures.length} Features</span>
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
};
