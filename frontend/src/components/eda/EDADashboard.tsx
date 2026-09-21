import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import {
  EDAOverview,
  MissingAnalysis,
  CorrelationAnalysis,
  OutlierStat,
  TargetAnalysisResponse
} from '../../types/eda';
import { MissingValuesChart } from './MissingValuesChart';
import { DistributionViewer } from './DistributionViewer';
import { CorrelationHeatmap } from './CorrelationHeatmap';
import { OutlierTable } from './OutlierTable';
import { RelationshipExplorer } from './RelationshipExplorer';
import {
  Target,
  Database,
  Hash,
  AlertCircle,
  Copy,
  Layers,
  Sparkles,
  Loader2,
  AlertTriangle
} from 'lucide-react';

interface EDADashboardProps {
  datasetId: string;
  columns: Array<{ name: string; inferred_dtype: string }>;
}

export const EDADashboard: React.FC<EDADashboardProps> = ({ datasetId, columns }) => {
  const [targetCol, setTargetCol] = useState<string>('');
  const [overview, setOverview] = useState<EDAOverview | null>(null);
  const [missingData, setMissingData] = useState<MissingAnalysis | null>(null);
  const [correlationData, setCorrelationData] = useState<CorrelationAnalysis | null>(null);
  const [outliers, setOutliers] = useState<OutlierStat[]>([]);
  const [targetAnalysis, setTargetAnalysis] = useState<TargetAnalysisResponse | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [targetLoading, setTargetLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch initial target-independent EDA data
  useEffect(() => {
    let isMounted = true;
    const fetchBaseEDA = async () => {
      setLoading(true);
      setError(null);
      try {
        const [ovRes, missRes, corrRes, outRes] = await Promise.all([
          api.getEDAOverview(datasetId),
          api.getEDAMissing(datasetId),
          api.getEDACorrelation(datasetId),
          api.getEDAOutliers(datasetId)
        ]);

        if (isMounted) {
          setOverview(ovRes);
          setMissingData(missRes);
          setCorrelationData(corrRes);
          setOutliers(outRes);
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Failed to load exploratory data analysis.');
        }
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    fetchBaseEDA();
    return () => { isMounted = false; };
  }, [datasetId]);

  // Fetch target-aware analysis when target selected
  useEffect(() => {
    if (!targetCol) {
      setTargetAnalysis(null);
      return;
    }

    let isMounted = true;
    const fetchTargetEDA = async () => {
      setTargetLoading(true);
      try {
        const res = await api.getEDATargetAnalysis(datasetId, targetCol);
        if (isMounted) setTargetAnalysis(res);
      } catch (err) {
        console.error('Target analysis failed:', err);
      } finally {
        if (isMounted) setTargetLoading(false);
      }
    };

    fetchTargetEDA();
    return () => { isMounted = false; };
  }, [datasetId, targetCol]);

  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '4rem 1rem' }}>
        <Loader2 size={36} className="spin" style={{ color: 'var(--accent-primary)', margin: '0 auto 1rem' }} />
        <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)', margin: '0 0 0.5rem 0' }}>
          Generating Exploratory Data Analysis...
        </h3>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', margin: 0 }}>
          Computing distributions, robust statistics, correlation matrices, and IQR outlier boundaries.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card" style={{ border: '1px solid rgba(239, 68, 68, 0.4)', background: 'rgba(239, 68, 68, 0.05)', textAlign: 'center', padding: '3rem 1rem' }}>
        <AlertTriangle size={36} color="#ef4444" style={{ margin: '0 auto 0.75rem' }} />
        <h3 style={{ color: '#ef4444', margin: '0 0 0.5rem 0' }}>Error Loading EDA</h3>
        <p style={{ color: '#fca5a5', fontSize: '0.9rem', margin: 0 }}>{error}</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Target Column Selector Bar */}
      <div className="card" style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem',
        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9))',
        border: '1px solid rgba(99, 102, 241, 0.3)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            background: 'rgba(99, 102, 241, 0.2)',
            padding: '0.5rem',
            borderRadius: 'var(--radius-sm)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Target size={20} color="var(--accent-primary)" />
          </div>
          <div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Target Column Mode
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {targetCol ? `Target-aware EDA enabled for '${targetCol}'` : 'Currently in target-independent EDA mode'}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <label htmlFor="eda-target-dropdown" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
            Target column:
          </label>
          <select
            id="eda-target-dropdown"
            value={targetCol}
            onChange={(e) => setTargetCol(e.target.value)}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              fontSize: '0.85rem',
              cursor: 'pointer',
              minWidth: '180px'
            }}
          >
            <option value="">(None - Target Independent)</option>
            {columns.map(c => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.inferred_dtype})
              </option>
            ))}
          </select>
          {targetCol && (
            <button
              className="btn btn-outline"
              style={{ padding: '0.45rem 0.75rem', fontSize: '0.8rem' }}
              onClick={() => setTargetCol('')}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Target-Aware Profile Section if selected */}
      {targetCol && targetAnalysis && (
        <div className="card" style={{ border: '1px solid rgba(16, 185, 129, 0.3)', background: 'rgba(16, 185, 129, 0.03)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
              <Sparkles size={18} /> Target-Aware Preliminary Analysis: {targetAnalysis.target_column}
            </h3>
            <span className="badge badge-success">
              Logical Type: {targetAnalysis.target_dtype}
            </span>
          </div>

          <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', margin: '0 0 1rem 0' }}>
            {targetAnalysis.disclaimer}
          </p>

          {targetLoading ? (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <Loader2 size={24} className="spin" style={{ color: '#10b981', margin: '0 auto 0.5rem' }} />
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Analyzing target relationships...</p>
            </div>
          ) : (
            <div>
              {targetAnalysis.profile.is_imbalanced && (
                <div style={{
                  background: 'rgba(245, 158, 11, 0.1)',
                  border: '1px solid rgba(245, 158, 11, 0.3)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.75rem 1rem',
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  <AlertTriangle size={16} color="#f59e0b" />
                  <span style={{ fontSize: '0.85rem', color: '#fde68a' }}>
                    {targetAnalysis.profile.imbalance_warning}
                  </span>
                </div>
              )}

              {/* Numeric Target: Top correlated features */}
              {targetAnalysis.profile.type === 'numeric' && targetAnalysis.profile.top_associated_features && (
                <div>
                  <h4 style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                    Top Numerically Associated Features with Target ({targetCol}):
                  </h4>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {targetAnalysis.profile.top_associated_features.map((feat: any, idx: number) => (
                      <div key={idx} style={{
                        background: 'rgba(30, 41, 59, 0.6)',
                        padding: '0.5rem 0.85rem',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-color)',
                        fontSize: '0.825rem'
                      }}>
                        <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{feat.feature}</span>:{' '}
                        <strong style={{ color: feat.correlation > 0 ? '#38bdf8' : '#f43f5e' }}>
                          {feat.correlation > 0 ? `+${feat.correlation}` : feat.correlation}
                        </strong>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Categorical Target: Top categories */}
              {targetAnalysis.profile.type === 'categorical' && targetAnalysis.profile.stats && (
                <div>
                  <h4 style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                    Target Class Frequencies:
                  </h4>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {targetAnalysis.profile.stats.top_categories.map((cat: any, idx: number) => (
                      <div key={idx} style={{
                        background: 'rgba(30, 41, 59, 0.6)',
                        padding: '0.5rem 0.85rem',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-color)',
                        fontSize: '0.825rem'
                      }}>
                        <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{cat.category}</span>:{' '}
                        <span>{cat.count.toLocaleString()} ({cat.percentage}%)</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Dataset Overview Cards */}
      {overview && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: '1rem'
        }}>
          <div className="card" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              <Database size={15} /> Total Rows
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.35rem' }}>
              {overview.rows.toLocaleString()}
            </div>
          </div>

          <div className="card" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              <Layers size={15} /> Columns
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.35rem' }}>
              {overview.columns}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              {overview.numeric_columns} num | {overview.categorical_columns} cat
            </div>
          </div>

          <div className="card" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              <AlertCircle size={15} /> Missing Cells
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: overview.missing_percentage > 10 ? '#ef4444' : 'var(--text-primary)', marginTop: '0.35rem' }}>
              {overview.missing_percentage}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              {overview.missing_cells.toLocaleString()} cells
            </div>
          </div>

          <div className="card" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              <Copy size={15} /> Duplicate Rows
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: overview.duplicate_rows > 0 ? '#f59e0b' : 'var(--text-primary)', marginTop: '0.35rem' }}>
              {overview.duplicate_rows.toLocaleString()}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              {overview.duplicate_percentage}%
            </div>
          </div>

          <div className="card" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              <Hash size={15} /> Memory Footprint
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.35rem' }}>
              {overview.memory_usage_human}
            </div>
          </div>
        </div>
      )}

      {/* Missing Values Analysis Section */}
      {missingData && <MissingValuesChart missingData={missingData} />}

      {/* Distributions & Statistical Summary */}
      <DistributionViewer datasetId={datasetId} columns={columns} />

      {/* Correlation Heatmap & Strong Associations */}
      {correlationData && <CorrelationHeatmap correlationData={correlationData} />}

      {/* Pairwise Relationship Explorer */}
      <RelationshipExplorer datasetId={datasetId} columns={columns} />

      {/* Statistical Outliers (IQR Method) */}
      <OutlierTable outliers={outliers} />
    </div>
  );
};
