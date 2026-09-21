import React, { useState } from 'react';
import { OutlierStat } from '../../types/eda';
import { AlertCircle, Box, Eye } from 'lucide-react';

interface OutlierTableProps {
  outliers: OutlierStat[];
  onSelectColumn?: (col: string) => void;
}

export const OutlierTable: React.FC<OutlierTableProps> = ({ outliers, onSelectColumn }) => {
  const [activeBoxCol, setActiveBoxCol] = useState<string | null>(outliers[0]?.column || null);

  const selectedStat = outliers.find(o => o.column === activeBoxCol);

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
            Potential Outliers (IQR Method)
          </h2>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Transparent statistical outlier boundary calculations: Q1 - 1.5×IQR and Q3 + 1.5×IQR.
          </p>
        </div>

        <span className="badge badge-info">
          Analyzed Columns: {outliers.length}
        </span>
      </div>

      {/* Epistemological disclaimer notice */}
      <div style={{
        background: 'rgba(245, 158, 11, 0.08)',
        border: '1px solid rgba(245, 158, 11, 0.25)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.75rem 1rem',
        marginBottom: '1rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem'
      }}>
        <AlertCircle size={18} color="#f59e0b" style={{ flexShrink: 0 }} />
        <div style={{ fontSize: '0.825rem', color: '#fde68a' }}>
          <strong>Important Note:</strong> Statistical outliers are mathematical flags, not proven errors or corrupted data. AutoDataBot never modifies or removes records during EDA.
        </div>
      </div>

      {outliers.length === 0 ? (
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', padding: '1rem 0' }}>
          No numerical columns available for outlier detection.
        </p>
      ) : (
        <div>
          <div style={{ overflowX: 'auto', marginBottom: '1.5rem' }}>
            <table className="table" style={{ fontSize: '0.825rem' }}>
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Q1 (25%)</th>
                  <th>Q3 (75%)</th>
                  <th>IQR</th>
                  <th>Lower Bound</th>
                  <th>Upper Bound</th>
                  <th>Potential Outliers</th>
                  <th>% of Data</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {outliers.map((row) => (
                  <tr
                    key={row.column}
                    style={{
                      background: activeBoxCol === row.column ? 'rgba(56, 189, 248, 0.05)' : undefined
                    }}
                  >
                    <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{row.column}</td>
                    <td>{row.q1 ?? 'N/A'}</td>
                    <td>{row.q3 ?? 'N/A'}</td>
                    <td>{row.iqr ?? 'N/A'}</td>
                    <td>{row.lower_bound ?? 'N/A'}</td>
                    <td>{row.upper_bound ?? 'N/A'}</td>
                    <td>
                      <span className={`badge ${row.potential_outliers_count > 0 ? 'badge-warning' : 'badge-success'}`}>
                        {row.potential_outliers_count.toLocaleString()}
                      </span>
                    </td>
                    <td>{row.potential_outliers_percentage}%</td>
                    <td>
                      <button
                        className="btn btn-outline"
                        style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
                        onClick={() => {
                          setActiveBoxCol(row.column);
                          if (onSelectColumn) onSelectColumn(row.column);
                        }}
                      >
                        <Eye size={12} /> Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detailed Box Inspection if column selected */}
          {selectedStat && (
            <div style={{
              background: 'rgba(30, 41, 59, 0.4)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <h4 style={{ margin: 0, fontSize: '0.95rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Box size={16} color="var(--accent-primary)" /> Outlier Distribution for: <strong>{selectedStat.column}</strong>
                </h4>

                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Lower Outliers: <strong>{selectedStat.lower_outliers_count}</strong> | Upper Outliers: <strong>{selectedStat.upper_outliers_count}</strong>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.5rem 0.75rem', borderRadius: '4px' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Observed Minimum</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>{selectedStat.min_val ?? 'N/A'}</div>
                </div>
                <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.5rem 0.75rem', borderRadius: '4px' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Lower IQR Fence</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#38bdf8' }}>{selectedStat.lower_bound ?? 'N/A'}</div>
                </div>
                <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.5rem 0.75rem', borderRadius: '4px' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Upper IQR Fence</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#f43f5e' }}>{selectedStat.upper_bound ?? 'N/A'}</div>
                </div>
                <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.5rem 0.75rem', borderRadius: '4px' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Observed Maximum</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>{selectedStat.max_val ?? 'N/A'}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
