import React from 'react';
import { CorrelationAnalysis } from '../../types/eda';
import { PlotlyChart } from './PlotlyChart';
import { ArrowUpRight, ArrowDownRight, ShieldAlert, Sparkles } from 'lucide-react';

interface CorrelationHeatmapProps {
  correlationData: CorrelationAnalysis;
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ correlationData }) => {
  const hasEnoughCols = correlationData.columns && correlationData.columns.length >= 2;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
            Pearson Correlation Analysis
          </h2>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Pairwise linear association across numerical features.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span className="badge badge-info">
            Features: {correlationData.columns_included} of {correlationData.total_numeric_columns}
          </span>
          {correlationData.truncated && (
            <span className="badge badge-warning" title={correlationData.selection_strategy}>
              Variance Filtered
            </span>
          )}
        </div>
      </div>

      {/* Mandatory Correlation != Causation Disclaimer Alert */}
      <div style={{
        background: 'rgba(56, 189, 248, 0.08)',
        border: '1px solid rgba(56, 189, 248, 0.25)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.75rem 1rem',
        marginBottom: '1rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem'
      }}>
        <ShieldAlert size={18} color="#38bdf8" style={{ flexShrink: 0 }} />
        <div style={{ fontSize: '0.825rem', color: '#bae6fd' }}>
          <strong>Statistical Notice:</strong> {correlationData.disclaimer} High correlation reflects co-movement, not a causal mechanism or feature importance.
        </div>
      </div>

      {!hasEnoughCols ? (
        <div style={{
          textAlign: 'center',
          padding: '2.5rem 1rem',
          background: 'rgba(30, 41, 59, 0.4)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--text-secondary)',
          fontSize: '0.9rem'
        }}>
          A minimum of 2 numerical columns is required to compute a correlation matrix.
        </div>
      ) : (
        <div>
          {/* Truncation explanation if applicable */}
          {correlationData.truncated && (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem', fontStyle: 'italic' }}>
              ℹ️ Displaying {correlationData.selection_strategy} to maintain heatmap readability.
            </p>
          )}

          {/* Plotly Heatmap */}
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', borderRadius: 'var(--radius-md)', padding: '0.5rem', marginBottom: '1.25rem' }}>
            <PlotlyChart
              data={correlationData.heatmap.data}
              layout={{
                ...correlationData.heatmap.layout,
                height: Math.min(650, Math.max(380, correlationData.columns.length * 32))
              }}
              style={{ minHeight: '380px' }}
              chartId="correlation-heatmap-plotly"
            />
          </div>

          {/* Strong Correlations Ranking Table */}
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Sparkles size={16} color="var(--accent-primary)" /> Strong Correlation Pairs (|r| ≥ 0.50)
            </h3>

            {correlationData.strong_correlations.length === 0 ? (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.5rem 0' }}>
                No pairs exceed the absolute correlation threshold of 0.50 in this dataset.
              </p>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table className="table" style={{ fontSize: '0.825rem' }}>
                  <thead>
                    <tr>
                      <th>Feature A</th>
                      <th>Feature B</th>
                      <th>Pearson (r)</th>
                      <th>Strength</th>
                      <th>Direction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {correlationData.strong_correlations.map((pair, idx) => (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{pair.feature_a}</td>
                        <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{pair.feature_b}</td>
                        <td>
                          <span style={{
                            fontWeight: 700,
                            color: pair.correlation > 0 ? '#38bdf8' : '#f43f5e'
                          }}>
                            {pair.correlation > 0 ? `+${pair.correlation}` : pair.correlation}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${pair.relationship_strength === 'strong' ? 'badge-warning' : 'badge-info'}`}>
                            {pair.relationship_strength}
                          </span>
                        </td>
                        <td>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                            {pair.direction === 'positive' ? (
                              <><ArrowUpRight size={14} color="#38bdf8" /> Positive</>
                            ) : (
                              <><ArrowDownRight size={14} color="#f43f5e" /> Negative</>
                            )}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
