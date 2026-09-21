import React from 'react';
import { ColumnSummary } from '../types/dataset';

interface ColumnTableProps {
  columns: ColumnSummary[];
}

export const ColumnTable: React.FC<ColumnTableProps> = ({ columns }) => {
  const getTypeBadgeClass = (dtype: string) => {
    switch (dtype.toLowerCase()) {
      case 'integer':
      case 'float':
        return 'badge-primary';
      case 'boolean':
        return 'badge-success';
      case 'datetime-like':
        return 'badge-info';
      case 'categorical':
        return 'badge-warning';
      case 'text':
        return 'badge-muted';
      default:
        return 'badge-muted';
    }
  };

  return (
    <div className="card" style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <div>
          <h3 style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Column Schema & Cardinality Audit
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Inferred semantic types, missingness, and cardinality metrics across {columns.length} columns.
          </p>
        </div>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Inferred Type</th>
              <th>Raw Dtype</th>
              <th style={{ textAlign: 'right' }}>Missing</th>
              <th style={{ textAlign: 'right' }}>Unique</th>
              <th style={{ textAlign: 'right' }}>Cardinality</th>
              <th>Flags</th>
              <th>Sample Values</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col) => (
              <tr key={col.name}>
                <td style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                  {col.name}
                </td>
                <td>
                  <span className={`badge ${getTypeBadgeClass(col.inferred_dtype)}`}>
                    {col.inferred_dtype}
                  </span>
                </td>
                <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                  {col.raw_dtype}
                </td>
                <td style={{ textAlign: 'right', color: col.missing_count > 0 ? 'var(--warning)' : 'inherit' }}>
                  {col.missing_count > 0 ? `${col.missing_count} (${col.missing_percentage}%)` : '0 (0%)'}
                </td>
                <td style={{ textAlign: 'right' }}>
                  {col.unique_count.toLocaleString()}
                </td>
                <td style={{ textAlign: 'right', color: col.unique_percentage > 50 ? 'var(--warning)' : 'inherit' }}>
                  {col.unique_percentage}%
                </td>
                <td>
                  <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                    {col.is_constant && (
                      <span className="badge badge-danger" title="Column has only 1 distinct value">
                        Constant
                      </span>
                    )}
                    {col.is_high_cardinality && (
                      <span className="badge badge-warning" title="Categorical feature with > 50% unique values">
                        High Card
                      </span>
                    )}
                    {!col.is_constant && !col.is_high_cardinality && (
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>—</span>
                    )}
                  </div>
                </td>
                <td style={{ color: 'var(--text-muted)', fontSize: '0.75rem', maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {col.sample_values.length > 0 ? col.sample_values.join(', ') : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
