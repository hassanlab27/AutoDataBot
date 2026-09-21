import React, { useState } from 'react';
import { api } from '../../services/api';
import { PlotlyFigurePayload } from '../../types/eda';
import { PlotlyChart } from './PlotlyChart';
import { Network, Wand2, Loader2, Sparkles } from 'lucide-react';

interface ColumnMeta {
  name: string;
  inferred_dtype: string;
}

interface RelationshipExplorerProps {
  datasetId: string;
  columns: ColumnMeta[];
}

export const RelationshipExplorer: React.FC<RelationshipExplorerProps> = ({ datasetId, columns }) => {
  const [colX, setColX] = useState<string>(columns[0]?.name || '');
  const [colY, setColY] = useState<string>(columns[1]?.name || columns[0]?.name || '');
  const [chartData, setChartData] = useState<PlotlyFigurePayload | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const metaX = columns.find(c => c.name === colX);
  const metaY = columns.find(c => c.name === colY);

  const getAnticipatedRule = () => {
    if (!metaX || !metaY) return 'Auto Recommender';
    const tx = metaX.inferred_dtype;
    const ty = metaY.inferred_dtype;
    if (['integer', 'float'].includes(tx) && ['integer', 'float'].includes(ty)) {
      return 'Numeric + Numeric → Scatter Plot + Trendline';
    }
    if ((['integer', 'float'].includes(tx) && ['categorical', 'boolean', 'text'].includes(ty)) ||
        (['categorical', 'boolean', 'text'].includes(tx) && ['integer', 'float'].includes(ty))) {
      return 'Categorical + Numeric → Grouped Box Plot';
    }
    if (['categorical', 'boolean', 'text'].includes(tx) && ['categorical', 'boolean', 'text'].includes(ty)) {
      return 'Categorical + Categorical → Contingency Heatmap';
    }
    if ((tx === 'datetime-like' && ['integer', 'float'].includes(ty)) ||
        (ty === 'datetime-like' && ['integer', 'float'].includes(tx))) {
      return 'Datetime + Numeric → Chronological Time Series';
    }
    return 'Deterministic Rule-Based Selection';
  };

  const handleGenerate = async () => {
    if (!colX || !colY) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.createEDAChart(datasetId, colX, colY);
      setChartData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to generate relationship chart');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
            <Network size={20} color="var(--accent-primary)" /> Pairwise Relationship Explorer
          </h2>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Select any two variables to generate deterministic bivariate visualizations.
          </p>
        </div>

        <span className="badge badge-info" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <Sparkles size={13} /> {getAnticipatedRule()}
        </span>
      </div>

      {/* Selector Controls */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'flex-end',
        gap: '1rem',
        background: 'rgba(30, 41, 59, 0.4)',
        padding: '1rem',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-color)',
        marginBottom: '1.25rem'
      }}>
        {/* X Column */}
        <div style={{ flex: '1 1 200px' }}>
          <label htmlFor="select-col-x" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
            Feature X:
          </label>
          <select
            id="select-col-x"
            value={colX}
            onChange={(e) => setColX(e.target.value)}
            style={{
              width: '100%',
              padding: '0.5rem 0.75rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              fontSize: '0.85rem'
            }}
          >
            {columns.map(c => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.inferred_dtype})
              </option>
            ))}
          </select>
        </div>

        {/* Y Column */}
        <div style={{ flex: '1 1 200px' }}>
          <label htmlFor="select-col-y" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
            Feature Y:
          </label>
          <select
            id="select-col-y"
            value={colY}
            onChange={(e) => setColY(e.target.value)}
            style={{
              width: '100%',
              padding: '0.5rem 0.75rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              fontSize: '0.85rem'
            }}
          >
            {columns.map(c => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.inferred_dtype})
              </option>
            ))}
          </select>
        </div>

        {/* Generate Button */}
        <div>
          <button
            className="btn btn-primary"
            onClick={handleGenerate}
            disabled={loading || !colX || !colY}
            style={{ padding: '0.55rem 1.25rem', fontSize: '0.85rem' }}
          >
            {loading ? (
              <>
                <Loader2 size={15} className="spin" /> Generating...
              </>
            ) : (
              <>
                <Wand2 size={15} /> Generate Chart
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 'var(--radius-sm)',
          padding: '1rem',
          color: '#fca5a5',
          fontSize: '0.85rem',
          marginBottom: '1rem'
        }}>
          {error}
        </div>
      )}

      {/* Chart Canvas */}
      {chartData ? (
        <div style={{
          background: 'rgba(15, 23, 42, 0.6)',
          borderRadius: 'var(--radius-md)',
          padding: '0.75rem',
          border: '1px solid var(--border-color)'
        }}>
          {chartData.sampled && (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textAlign: 'right' }}>
              ⚡ Sampled to 2,000 points with fixed seed for high-performance rendering.
            </div>
          )}

          <PlotlyChart
            data={chartData.data}
            layout={{
              ...chartData.layout,
              height: 440
            }}
            style={{ minHeight: '440px' }}
            chartId={`bivariate-${colX}-${colY}`}
          />
        </div>
      ) : (
        <div style={{
          textAlign: 'center',
          padding: '3rem 1rem',
          background: 'rgba(15, 23, 42, 0.3)',
          border: '1px dashed var(--border-color)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--text-secondary)'
        }}>
          <Network size={32} style={{ margin: '0 auto 0.75rem', opacity: 0.5 }} />
          <h4 style={{ margin: '0 0 0.25rem 0', fontSize: '1rem', color: 'var(--text-primary)' }}>
            No Relationship Chart Generated Yet
          </h4>
          <p style={{ margin: 0, fontSize: '0.85rem' }}>
            Choose two columns above and click <strong>Generate Chart</strong> to inspect their relationship.
          </p>
        </div>
      )}
    </div>
  );
};
