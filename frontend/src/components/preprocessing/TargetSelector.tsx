import React, { useState, useEffect } from 'react';
import { TargetInspectResponse } from '../../types/preprocessing';
import { api } from '../../services/api';
import {
  Target,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  ArrowRight,
  Search,
  X,
  Hash,
  Type,
  Calendar,
  Database,
  Check
} from 'lucide-react';

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
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('all');

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

  // Type category counts
  const numericCount = columns.filter(c => ['integer', 'float', 'numeric', 'number'].some(t => c.inferred_dtype.toLowerCase().includes(t))).length;
  const categoricalCount = columns.filter(c => ['category', 'string', 'text', 'categorical'].some(t => c.inferred_dtype.toLowerCase().includes(t))).length;
  const datetimeCount = columns.filter(c => ['date', 'time'].some(t => c.inferred_dtype.toLowerCase().includes(t))).length;

  const filteredColumns = columns.filter(col => {
    const matchesSearch = col.name.toLowerCase().includes(searchTerm.toLowerCase());
    if (!matchesSearch) return false;
    if (typeFilter === 'all') return true;
    if (typeFilter === 'numeric') {
      return ['integer', 'float', 'numeric', 'number'].some(t => col.inferred_dtype.toLowerCase().includes(t));
    }
    if (typeFilter === 'categorical') {
      return ['category', 'string', 'text', 'categorical'].some(t => col.inferred_dtype.toLowerCase().includes(t));
    }
    if (typeFilter === 'datetime') {
      return ['date', 'time'].some(t => col.inferred_dtype.toLowerCase().includes(t));
    }
    return true;
  });

  const getDtypeBadge = (dtype: string) => {
    const lower = dtype.toLowerCase();
    if (['integer', 'float', 'numeric', 'number'].some(t => lower.includes(t))) {
      return {
        bg: 'rgba(56, 189, 248, 0.15)',
        color: '#38bdf8',
        border: '1px solid rgba(56, 189, 248, 0.35)',
        icon: <Hash size={12} />
      };
    }
    if (['category', 'string', 'text', 'categorical'].some(t => lower.includes(t))) {
      return {
        bg: 'rgba(168, 85, 247, 0.15)',
        color: '#c084fc',
        border: '1px solid rgba(168, 85, 247, 0.35)',
        icon: <Type size={12} />
      };
    }
    if (['date', 'time'].some(t => lower.includes(t))) {
      return {
        bg: 'rgba(245, 158, 11, 0.15)',
        color: '#fbbf24',
        border: '1px solid rgba(245, 158, 11, 0.35)',
        icon: <Calendar size={12} />
      };
    }
    return {
      bg: 'rgba(148, 163, 184, 0.15)',
      color: '#cbd5e1',
      border: '1px solid rgba(148, 163, 184, 0.35)',
      icon: <Database size={12} />
    };
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

      {/* Target Feature Selector Container */}
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: 'var(--radius-md)',
        padding: '1.25rem',
        marginBottom: '1.5rem'
      }}>
        {/* Header / Filter bar */}
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1rem'
        }}>
          <div>
            <label htmlFor="target-column-select" style={{ display: 'block', fontWeight: 600, fontSize: '0.95rem', color: '#ffffff' }}>
              Select Target Column
            </label>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Click a feature tile below or choose directly from the dropdown
            </span>
          </div>

          {/* Quick Dropdown Fallback */}
          <div style={{ minWidth: '240px' }}>
            <select
              id="target-column-select"
              aria-label="Target Column Dropdown"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="select-input"
              style={{
                width: '100%',
                backgroundColor: '#0f172a',
                color: '#ffffff',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.5rem 0.75rem',
                fontSize: '0.85rem'
              }}
            >
              <option value="" style={{ backgroundColor: '#0f172a', color: '#94a3b8' }}>
                -- Choose target from list --
              </option>
              {columns.map(col => (
                <option key={col.name} value={col.name} style={{ backgroundColor: '#0f172a', color: '#ffffff' }}>
                  {col.name} ({col.inferred_dtype})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Search & Filter Chips Row */}
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem',
          alignItems: 'center',
          marginBottom: '1rem',
          paddingBottom: '0.75rem',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)'
        }}>
          {/* Search Box */}
          <div style={{ position: 'relative', flex: '1 1 200px', maxWidth: '300px' }}>
            <Search size={16} style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search features..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              aria-label="Search target features"
              style={{
                width: '100%',
                padding: '0.45rem 2rem 0.45rem 2.2rem',
                backgroundColor: '#1e293b',
                color: '#ffffff',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.85rem'
              }}
            />
            {searchTerm && (
              <button
                type="button"
                onClick={() => setSearchTerm('')}
                aria-label="Clear search"
                style={{
                  position: 'absolute',
                  right: '0.5rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  padding: '2px',
                  display: 'flex'
                }}
              >
                <X size={14} />
              </button>
            )}
          </div>

          {/* Filter Chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', alignItems: 'center' }}>
            <button
              type="button"
              onClick={() => setTypeFilter('all')}
              style={{
                padding: '0.35rem 0.7rem',
                fontSize: '0.75rem',
                fontWeight: 600,
                borderRadius: 'var(--radius-full)',
                border: typeFilter === 'all' ? '1px solid var(--primary)' : '1px solid rgba(255, 255, 255, 0.12)',
                background: typeFilter === 'all' ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                color: typeFilter === 'all' ? '#a5b4fc' : 'var(--text-secondary)',
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              All ({columns.length})
            </button>
            {numericCount > 0 && (
              <button
                type="button"
                onClick={() => setTypeFilter('numeric')}
                style={{
                  padding: '0.35rem 0.7rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  borderRadius: 'var(--radius-full)',
                  border: typeFilter === 'numeric' ? '1px solid var(--accent-blue)' : '1px solid rgba(255, 255, 255, 0.12)',
                  background: typeFilter === 'numeric' ? 'rgba(56, 189, 248, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                  color: typeFilter === 'numeric' ? '#7dd3fc' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                Numeric ({numericCount})
              </button>
            )}
            {categoricalCount > 0 && (
              <button
                type="button"
                onClick={() => setTypeFilter('categorical')}
                style={{
                  padding: '0.35rem 0.7rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  borderRadius: 'var(--radius-full)',
                  border: typeFilter === 'categorical' ? '1px solid var(--accent-purple)' : '1px solid rgba(255, 255, 255, 0.12)',
                  background: typeFilter === 'categorical' ? 'rgba(168, 85, 247, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                  color: typeFilter === 'categorical' ? '#d8b4fe' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                Categorical ({categoricalCount})
              </button>
            )}
            {datetimeCount > 0 && (
              <button
                type="button"
                onClick={() => setTypeFilter('datetime')}
                style={{
                  padding: '0.35rem 0.7rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  borderRadius: 'var(--radius-full)',
                  border: typeFilter === 'datetime' ? '1px solid var(--accent-amber)' : '1px solid rgba(255, 255, 255, 0.12)',
                  background: typeFilter === 'datetime' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                  color: typeFilter === 'datetime' ? '#fde68a' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                Datetime ({datetimeCount})
              </button>
            )}
          </div>
        </div>

        {/* Feature Cards Scrollable Container */}
        <div
          role="listbox"
          aria-label="Target feature options"
          className="custom-scrollbar"
          style={{
            maxHeight: '260px',
            overflowY: 'auto',
            paddingRight: '0.35rem'
          }}
        >
          {filteredColumns.length === 0 ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
              No features match your search criteria.
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))',
              gap: '0.65rem'
            }}>
              {filteredColumns.map(col => {
                const isSelected = target === col.name;
                const badge = getDtypeBadge(col.inferred_dtype);
                return (
                  <div
                    key={col.name}
                    role="option"
                    aria-selected={isSelected}
                    tabIndex={0}
                    onClick={() => setTarget(col.name)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setTarget(col.name);
                      }
                    }}
                    style={{
                      background: isSelected 
                        ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.22) 0%, rgba(59, 130, 246, 0.15) 100%)' 
                        : '#1e293b',
                      border: isSelected 
                        ? '2px solid #6366f1' 
                        : '1px solid rgba(255, 255, 255, 0.12)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '0.75rem 0.85rem',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                      gap: '0.5rem',
                      boxShadow: isSelected 
                        ? '0 4px 14px rgba(99, 102, 241, 0.3)' 
                        : '0 2px 4px rgba(0, 0, 0, 0.2)',
                      transition: 'all 0.18s ease-in-out',
                      outline: 'none'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.5rem' }}>
                      <span
                        title={col.name}
                        style={{
                          fontWeight: 700,
                          fontSize: '0.9rem',
                          color: '#ffffff',
                          wordBreak: 'break-word',
                          lineHeight: 1.3
                        }}
                      >
                        {col.name}
                      </span>
                      {isSelected && (
                        <div style={{
                          color: '#10b981',
                          background: 'rgba(16, 185, 129, 0.2)',
                          borderRadius: 'var(--radius-full)',
                          padding: '2px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0
                        }}>
                          <Check size={14} strokeWidth={3} />
                        </div>
                      )}
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          background: badge.bg,
                          color: badge.color,
                          border: badge.border,
                          fontSize: '0.72rem',
                          fontWeight: 600,
                          padding: '0.15rem 0.45rem',
                          borderRadius: 'var(--radius-full)',
                          fontFamily: 'var(--font-mono)'
                        }}
                      >
                        {badge.icon}
                        {col.inferred_dtype}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
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
