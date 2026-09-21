import React, { useState, useEffect } from 'react';
import { Table, RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import { PreviewResponse } from '../types/dataset';

interface DatasetPreviewProps {
  datasetId: string;
}

export const DatasetPreview: React.FC<DatasetPreviewProps> = ({ datasetId }) => {
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [limit, setLimit] = useState<number>(20);

  useEffect(() => {
    let isMounted = true;
    const fetchPreview = async () => {
      setLoading(true);
      try {
        const data = await api.getDatasetPreview(datasetId, limit);
        if (isMounted) setPreview(data);
      } catch (err) {
        console.error('Failed to load dataset preview:', err);
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    fetchPreview();
    return () => { isMounted = false; };
  }, [datasetId, limit]);

  return (
    <div className="card" style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', gap: '1rem' }}>
        <div>
          <h3 style={{ fontSize: '1.125rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Table size={18} color="var(--primary)" />
            Dataset Sample Preview
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Showing first {preview ? preview.returned_rows : limit} of {preview ? preview.total_rows.toLocaleString() : '...'} rows.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Rows to display:</span>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            style={{
              padding: '0.35rem 0.65rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              fontSize: '0.8rem',
              outline: 'none'
            }}
          >
            <option value={10}>10 rows</option>
            <option value={20}>20 rows</option>
            <option value={50}>50 rows</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
          <RefreshCw className="animate-spin" size={20} />
          Loading preview rows...
        </div>
      ) : preview && preview.rows.length > 0 ? (
        <div className="table-container" style={{ maxHeight: '480px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '48px', color: 'var(--text-muted)' }}>#</th>
                {preview.columns.map((col) => (
                  <th key={col}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{col}</span>
                      <span style={{ fontSize: '0.65rem', fontWeight: 400, color: 'var(--text-muted)' }}>
                        {preview.inferred_dtypes[col] || 'unknown'}
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                    {rowIdx + 1}
                  </td>
                  {preview.columns.map((col) => {
                    const val = row[col];
                    const isNull = val === null || val === undefined;
                    return (
                      <td key={col} style={{ color: isNull ? 'var(--text-muted)' : 'inherit' }}>
                        {isNull ? <em style={{ fontSize: '0.75rem', color: '#64748b' }}>null</em> : String(val)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          No preview rows available for this dataset.
        </div>
      )}
    </div>
  );
};
