import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import { DistributionResponse, NumericalStat, CategoricalStat } from '../../types/eda';
import { PlotlyChart } from './PlotlyChart';
import { BarChart2, Box, Info, Loader2 } from 'lucide-react';

interface DistributionViewerProps {
  datasetId: string;
  columns: Array<{ name: string; inferred_dtype: string }>;
}

export const DistributionViewer: React.FC<DistributionViewerProps> = ({ datasetId, columns }) => {
  const [selectedCol, setSelectedCol] = useState<string>(columns[0]?.name || '');
  const [activeView, setActiveView] = useState<'histogram' | 'box_plot'>('histogram');
  const [data, setData] = useState<DistributionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedCol) return;
    let isMounted = true;
    const fetchDist = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.getEDADistribution(datasetId, selectedCol);
        if (isMounted) setData(res);
      } catch (err: any) {
        if (isMounted) setError(err.message || 'Failed to load distribution');
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchDist();
    return () => { isMounted = false; };
  }, [datasetId, selectedCol]);

  const isNumeric = data?.dtype === 'integer' || data?.dtype === 'float';
  const isCategorical = data?.dtype === 'categorical' || data?.dtype === 'boolean' || data?.dtype === 'text';
  const isDatetime = data?.dtype === 'datetime-like';

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
            Distributions & Summary Statistics
          </h2>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Inspect histograms, box plots, frequencies, and descriptive statistics for each feature.
          </p>
        </div>

        {/* Column Dropdown Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <label htmlFor="distribution-col-select" style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
            Column:
          </label>
          <select
            id="distribution-col-select"
            value={selectedCol}
            onChange={(e) => setSelectedCol(e.target.value)}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(30, 41, 59, 0.8)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              fontSize: '0.85rem',
              cursor: 'pointer'
            }}
          >
            {columns.map(c => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.inferred_dtype})
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '3rem 1rem' }}>
          <Loader2 size={28} className="spin" style={{ color: 'var(--accent-primary)', margin: '0 auto 0.5rem' }} />
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>Computing distribution metrics...</p>
        </div>
      )}

      {error && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 'var(--radius-sm)',
          padding: '1rem',
          color: '#fca5a5',
          fontSize: '0.875rem'
        }}>
          {error}
        </div>
      )}

      {!loading && !error && data && (
        <div>
          {/* Numeric View Toggle (Histogram vs Box Plot) */}
          {isNumeric && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  className={`btn ${activeView === 'histogram' ? 'btn-primary' : 'btn-outline'}`}
                  style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}
                  onClick={() => setActiveView('histogram')}
                >
                  <BarChart2 size={14} /> Histogram
                </button>
                <button
                  className={`btn ${activeView === 'box_plot' ? 'btn-primary' : 'btn-outline'}`}
                  style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}
                  onClick={() => setActiveView('box_plot')}
                >
                  <Box size={14} /> Box Plot & Outliers
                </button>
              </div>

              <span className="badge badge-info" style={{ textTransform: 'capitalize' }}>
                Type: {data.dtype}
              </span>
            </div>
          )}

          {/* Visualization Area */}
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', borderRadius: 'var(--radius-md)', padding: '0.5rem', marginBottom: '1.25rem' }}>
            {isNumeric && activeView === 'histogram' && data.histogram && (
              <PlotlyChart
                data={data.histogram.data}
                layout={data.histogram.layout}
                chartId={`dist-hist-${data.column}`}
              />
            )}
            {isNumeric && activeView === 'box_plot' && data.box_plot && (
              <PlotlyChart
                data={data.box_plot.data}
                layout={data.box_plot.layout}
                chartId={`dist-box-${data.column}`}
              />
            )}
            {(isCategorical || isDatetime) && data.chart && (
              <PlotlyChart
                data={data.chart.data}
                layout={data.chart.layout}
                chartId={`dist-cat-${data.column}`}
              />
            )}
          </div>

          {/* Descriptive Statistics Cards */}
          {isNumeric && data.stats && (
            <div>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Info size={15} /> Descriptive Statistics: {data.column}
              </h3>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                gap: '0.75rem'
              }}>
                {(() => {
                  const numStats = data.stats as NumericalStat;
                  return [
                    { label: 'Mean', value: numStats.mean ?? 'N/A' },
                    { label: 'Median', value: numStats.median ?? 'N/A' },
                    { label: 'Std Dev', value: numStats.std ?? 'N/A' },
                    { label: 'Min', value: numStats.min ?? 'N/A' },
                    { label: 'Max', value: numStats.max ?? 'N/A' },
                    { label: 'Q1 (25%)', value: numStats.q1 ?? 'N/A' },
                    { label: 'Q3 (75%)', value: numStats.q3 ?? 'N/A' },
                    { label: 'IQR', value: numStats.iqr ?? 'N/A' },
                    { label: 'Skewness', value: numStats.skewness ?? 'N/A' },
                    { label: 'Kurtosis', value: numStats.kurtosis ?? 'N/A' }
                  ].map((stat, idx) => (
                    <div key={idx} style={{
                      background: 'rgba(30, 41, 59, 0.5)',
                      padding: '0.75rem',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-color)'
                    }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{stat.label}</div>
                      <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                        {stat.value}
                      </div>
                    </div>
                  ));
                })()}
              </div>
            </div>
          )}

          {isCategorical && data.stats && (
            <div>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Info size={15} /> Cardinality & Frequency Summary
              </h3>
              {(() => {
                const catStats = data.stats as CategoricalStat;
                return (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
                    <div style={{ background: 'rgba(30, 41, 59, 0.5)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Unique Categories</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                        {catStats.unique_values.toLocaleString()}
                      </div>
                    </div>
                    <div style={{ background: 'rgba(30, 41, 59, 0.5)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Most Frequent (Mode)</div>
                      <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                        {catStats.mode || 'N/A'} ({catStats.mode_percentage}%)
                      </div>
                    </div>
                    <div style={{ background: 'rgba(30, 41, 59, 0.5)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Valid / Missing</div>
                      <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                        {catStats.valid_count} / {catStats.missing_count} ({catStats.missing_percentage}%)
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {isDatetime && data.temporal && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
              <div style={{ background: 'rgba(30, 41, 59, 0.5)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Earliest Date</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                  {data.temporal.min_date}
                </div>
              </div>
              <div style={{ background: 'rgba(30, 41, 59, 0.5)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Latest Date</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                  {data.temporal.max_date}
                </div>
              </div>
              <div style={{ background: 'rgba(30, 41, 59, 0.5)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Date Range (Days)</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                  {data.temporal.date_range_days}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
