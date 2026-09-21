import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, Info, ShieldAlert, Sparkles } from 'lucide-react';
import { DatasetQualityReport } from '../types/dataset';

interface QualitySectionProps {
  quality: DatasetQualityReport;
}

export const QualitySection: React.FC<QualitySectionProps> = ({ quality }) => {
  const getRatingColor = (rating: string) => {
    switch (rating.toLowerCase()) {
      case 'excellent':
        return 'var(--success)';
      case 'good':
        return '#3b82f6';
      case 'fair':
        return 'var(--warning)';
      default:
        return 'var(--danger)';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
        return <CheckCircle2 size={18} color="var(--success)" />;
      case 'warning':
        return <AlertTriangle size={18} color="var(--warning)" />;
      case 'fail':
        return <XCircle size={18} color="var(--danger)" />;
      default:
        return <Info size={18} color="var(--text-muted)" />;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginBottom: '2rem' }}>
      {/* Header & Quality Score Card */}
      <div className="card" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1.5rem' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <Sparkles size={22} color="var(--primary)" />
            Dataset Quality Audit
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            Transparent, deterministic data hygiene score based on missingness, duplicates, constant columns, and identifiers.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Quality Rating
            </span>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: getRatingColor(quality.rating) }}>
              {quality.rating}
            </div>
          </div>

          <div style={{
            width: '80px',
            height: '80px',
            borderRadius: '50%',
            border: `4px solid ${getRatingColor(quality.rating)}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            backgroundColor: 'rgba(0, 0, 0, 0.25)'
          }}>
            <span style={{ fontSize: '1.5rem', fontWeight: 800, lineHeight: 1 }}>
              {quality.quality_score}
            </span>
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>/ 100</span>
          </div>
        </div>
      </div>

      {/* Checklist Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '1rem'
      }}>
        {quality.checks.map((check) => (
          <div
            key={check.name}
            className="card"
            style={{
              padding: '1.25rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
              borderLeft: `4px solid ${check.status === 'pass' ? 'var(--success)' : check.status === 'warning' ? 'var(--warning)' : 'var(--danger)'}`
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: 600, fontSize: '0.925rem' }}>{check.name}</span>
              {getStatusIcon(check.status)}
            </div>
            <p style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)' }}>
              {check.summary}
            </p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {check.description}
            </p>
            {check.affected_columns.length > 0 && (
              <div style={{ marginTop: '0.25rem', display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                {check.affected_columns.slice(0, 6).map((c) => (
                  <span key={c} className="badge badge-muted" style={{ fontSize: '0.7rem' }}>
                    {c}
                  </span>
                ))}
                {check.affected_columns.length > 6 && (
                  <span className="badge badge-muted" style={{ fontSize: '0.7rem' }}>
                    +{check.affected_columns.length - 6} more
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Suspicious Columns Warning Box */}
      {quality.suspicious_columns.length > 0 && (
        <div className="card" style={{ border: '1px solid rgba(245, 158, 11, 0.3)', backgroundColor: 'rgba(245, 158, 11, 0.04)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '1rem' }}>
            <ShieldAlert size={20} color="var(--warning)" />
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#fcd34d' }}>
              Suspicious Columns Advisory ({quality.suspicious_columns.length} Flagged)
            </h3>
          </div>
          <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            AutoDataBot flags these features for transparency. <strong>They are not automatically deleted.</strong> Review each advisory below:
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {quality.suspicious_columns.map((susp, idx) => (
              <div
                key={idx}
                style={{
                  padding: '0.85rem 1rem',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'rgba(15, 23, 42, 0.6)',
                  border: '1px solid var(--border-color)',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem'
                }}
              >
                <span className="badge badge-warning" style={{ marginTop: '0.15rem' }}>
                  {susp.issue_type}
                </span>
                <div>
                  <strong style={{ fontSize: '0.875rem', color: 'var(--text-primary)' }}>{susp.column_name}:</strong>{' '}
                  <span style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>{susp.description}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
