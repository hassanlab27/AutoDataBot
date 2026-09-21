import React from 'react';
import { Database, ShieldCheck, Activity } from 'lucide-react';

interface NavbarProps {
  backendConnected: boolean;
  onReset?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ backendConnected, onReset }) => {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-color)',
      background: 'rgba(11, 15, 29, 0.8)',
      backdropFilter: 'blur(16px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '1rem 0',
      marginBottom: '2rem'
    }}>
      <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 0 }}>
        <div 
          onClick={onReset}
          style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: onReset ? 'pointer' : 'default' }}
        >
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(99, 102, 241, 0.35)'
          }}>
            <Database size={20} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.025em', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              AutoDataBot
              <span className="badge badge-primary">Phase 1</span>
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Local Automated Data-Science Foundation</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className={`badge ${backendConnected ? 'badge-success' : 'badge-danger'}`} style={{ padding: '0.35rem 0.75rem' }}>
            <Activity size={13} />
            {backendConnected ? 'API Connected (FastAPI)' : 'API Offline'}
          </div>
          <div className="badge badge-muted" style={{ padding: '0.35rem 0.75rem' }}>
            <ShieldCheck size={13} />
            100% Local & Free
          </div>
        </div>
      </div>
    </header>
  );
};
