import React from 'react';
import { MissingAnalysis } from '../../types/eda';
import { PlotlyChart } from './PlotlyChart';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';

interface MissingValuesChartProps {
  missingData: MissingAnalysis;
}

export const MissingValuesChart: React.FC<MissingValuesChartProps> = ({ missingData }) => {
  const hasMissing = missingData.total_missing_cells > 0;
  const hasSevere = missingData.severe_missing_columns && missingData.severe_missing_columns.length > 0;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
            Missing Values Analysis
          </h2>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Column-by-column missing data distribution and severity assessment.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span className="badge badge-info">
            Total Missing: {missingData.total_missing_cells.toLocaleString()} ({missingData.missing_percentage}%)
          </span>
          <span className={`badge ${hasMissing ? 'badge-warning' : 'badge-success'}`}>
            Columns with Missing: {missingData.columns_with_missing_count}
          </span>
        </div>
      </div>

      {hasSevere && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 'var(--radius-sm)',
          padding: '0.75rem 1rem',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <AlertTriangle size={18} color="#ef4444" style={{ flexShrink: 0 }} />
          <div style={{ fontSize: '0.85rem', color: '#fca5a5' }}>
            <strong>Severe Missingness Warning:</strong> The following columns have ≥ 50% missing values:{' '}
            <strong>{missingData.severe_missing_columns.join(', ')}</strong>.
          </div>
        </div>
      )}

      {!hasMissing ? (
        <div style={{
          textAlign: 'center',
          padding: '2.5rem 1rem',
          background: 'rgba(16, 185, 129, 0.05)',
          border: '1px dashed rgba(16, 185, 129, 0.3)',
          borderRadius: 'var(--radius-md)'
        }}>
          <CheckCircle2 size={36} color="#10b981" style={{ margin: '0 auto 0.75rem' }} />
          <h3 style={{ fontSize: '1.1rem', color: '#10b981', margin: '0 0 0.25rem 0' }}>Clean Dataset: No Missing Values</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
            Every cell in this dataset contains valid recorded data.
          </p>
        </div>
      ) : (
        <div>
          {missingData.chart && (
            <PlotlyChart
              data={missingData.chart.data}
              layout={{
                ...missingData.chart.layout,
                height: Math.max(280, missingData.columns.length * 28)
              }}
              style={{ minHeight: '300px' }}
              chartId="missing-values-plotly"
            />
          )}

          {/* Quick Summary Table of Missing Columns */}
          <div style={{ marginTop: '1rem', overflowX: 'auto' }}>
            <table className="table" style={{ fontSize: '0.825rem' }}>
              <thead>
                <tr>
                  <th>Column Name</th>
                  <th>Missing Count</th>
                  <th>Missing %</th>
                  <th>Present Count</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {missingData.columns.filter(c => c.missing_count > 0).map(col => (
                  <tr key={col.column}>
                    <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{col.column}</td>
                    <td>{col.missing_count.toLocaleString()}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{
                          width: '60px',
                          height: '6px',
                          background: 'rgba(255, 255, 255, 0.1)',
                          borderRadius: '3px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${Math.min(col.missing_percentage, 100)}%`,
                            height: '100%',
                            background: col.missing_percentage >= 50 ? '#ef4444' : (col.missing_percentage > 10 ? '#f59e0b' : '#38bdf8')
                          }} />
                        </div>
                        <span>{col.missing_percentage}%</span>
                      </div>
                    </td>
                    <td>{col.present_count.toLocaleString()} ({col.present_percentage}%)</td>
                    <td>
                      {col.missing_percentage >= 50 ? (
                        <span className="badge badge-danger">Severe</span>
                      ) : col.missing_percentage > 10 ? (
                        <span className="badge badge-warning">Moderate</span>
                      ) : (
                        <span className="badge badge-info">Low</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
