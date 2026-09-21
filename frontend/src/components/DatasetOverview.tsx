import React from 'react';
import { Layers, Database, Copy, AlertTriangle, FileCheck, HardDrive } from 'lucide-react';
import { DatasetStructureSummary, IngestionReport } from '../types/dataset';

interface DatasetOverviewProps {
  summary: DatasetStructureSummary;
  ingestion: IngestionReport;
}

export const DatasetOverview: React.FC<DatasetOverviewProps> = ({ summary, ingestion }) => {
  const totalMissing = summary.columns.reduce((acc, col) => acc + col.missing_count, 0);
  const totalCells = summary.row_count * summary.column_count;
  const missingPercentage = totalCells > 0 ? ((totalMissing / totalCells) * 100).toFixed(1) : '0.0';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginBottom: '2rem' }}>
      {/* Top Banner: Ingestion Repairs Info */}
      <div className="card" style={{ padding: '1rem 1.5rem', background: 'rgba(30, 41, 69, 0.4)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <FileCheck size={20} color="var(--success)" />
            <div>
              <span style={{ fontWeight: 600, fontSize: '0.925rem' }}>{ingestion.original_filename}</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginLeft: '0.5rem' }}>
                ({(ingestion.file_size_bytes / 1024).toFixed(1)} KB)
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span className="badge badge-muted">Encoding: {ingestion.encoding_used}</span>
            <span className="badge badge-muted">Delimiter: "{ingestion.delimiter_used}"</span>
            {ingestion.actions.length > 0 && (
              <span className="badge badge-info">
                {ingestion.actions.length} Ingestion {ingestion.actions.length === 1 ? 'Repair' : 'Repairs'} Logged
              </span>
            )}
          </div>
        </div>

        {ingestion.actions.length > 0 && (
          <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              {ingestion.actions.map((act, idx) => (
                <li key={idx}><strong>{act.action_type}:</strong> {act.description}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Metric Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '1rem'
      }}>
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
            <Layers size={16} /> Total Rows
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
            {summary.row_count.toLocaleString()}
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
            <Database size={16} /> Columns
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
            {summary.column_count}
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
            <HardDrive size={16} /> Memory Size
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
            {summary.memory_usage_human}
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
            <AlertTriangle size={16} /> Missing Values
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em', color: totalMissing > 0 ? 'var(--warning)' : 'var(--success)' }}>
            {totalMissing.toLocaleString()}
            <span style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-muted)', marginLeft: '0.4rem' }}>
              ({missingPercentage}%)
            </span>
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
            <Copy size={16} /> Duplicate Rows
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em', color: summary.duplicate_rows_count > 0 ? 'var(--warning)' : 'var(--success)' }}>
            {summary.duplicate_rows_count.toLocaleString()}
            <span style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-muted)', marginLeft: '0.4rem' }}>
              ({summary.duplicate_rows_percentage}%)
            </span>
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
            <AlertTriangle size={16} /> Constant Columns
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em', color: summary.constant_columns_count > 0 ? 'var(--warning)' : 'var(--success)' }}>
            {summary.constant_columns_count}
          </div>
        </div>
      </div>
    </div>
  );
};
