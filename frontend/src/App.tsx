import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { UploadSection } from './components/UploadSection';
import { DatasetOverview } from './components/DatasetOverview';
import { ColumnTable } from './components/ColumnTable';
import { QualitySection } from './components/QualitySection';
import { DatasetPreview } from './components/DatasetPreview';
import { EDADashboard } from './components/eda/EDADashboard';
import { UploadResponse } from './types/dataset';
import { api } from './services/api';
import { FileUp, BarChart3, Columns, ShieldCheck, Table, LineChart } from 'lucide-react';

export const App: React.FC = () => {
  const [dataset, setDataset] = useState<UploadResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'eda' | 'columns' | 'quality' | 'preview'>('overview');
  const [backendConnected, setBackendConnected] = useState<boolean>(false);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const res = await api.checkHealth();
        if (res.status === 'ok') {
          setBackendConnected(true);
        }
      } catch {
        setBackendConnected(false);
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <Navbar
        backendConnected={backendConnected}
        onReset={dataset ? () => setDataset(null) : undefined}
      />

      <main className="container">
        {!dataset ? (
          <UploadSection onUploadSuccess={(data) => {
            setDataset(data);
            setActiveTab('overview');
          }} />
        ) : (
          <div>
            {/* Action Bar */}
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1.5rem',
              gap: '1rem'
            }}>
              {/* Tab Navigation */}
              <div style={{
                display: 'flex',
                gap: '0.5rem',
                background: 'rgba(17, 24, 39, 0.7)',
                padding: '0.35rem',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                flexWrap: 'wrap'
              }}>
                <button
                  className={`btn ${activeTab === 'overview' ? 'btn-primary' : 'btn-outline'}`}
                  style={{ padding: '0.5rem 1rem', fontSize: '0.825rem' }}
                  onClick={() => setActiveTab('overview')}
                >
                  <BarChart3 size={15} /> Overview
                </button>
                <button
                  className={`btn ${activeTab === 'eda' ? 'btn-primary' : 'btn-outline'}`}
                  style={{ padding: '0.5rem 1rem', fontSize: '0.825rem' }}
                  onClick={() => setActiveTab('eda')}
                >
                  <LineChart size={15} /> Exploratory Analysis (EDA)
                </button>
                <button
                  className={`btn ${activeTab === 'columns' ? 'btn-primary' : 'btn-outline'}`}
                  style={{ padding: '0.5rem 1rem', fontSize: '0.825rem' }}
                  onClick={() => setActiveTab('columns')}
                >
                  <Columns size={15} /> Column Schema ({dataset.summary.column_count})
                </button>
                <button
                  className={`btn ${activeTab === 'quality' ? 'btn-primary' : 'btn-outline'}`}
                  style={{ padding: '0.5rem 1rem', fontSize: '0.825rem' }}
                  onClick={() => setActiveTab('quality')}
                >
                  <ShieldCheck size={15} /> Data Quality ({dataset.quality.quality_score}/100)
                </button>
                <button
                  className={`btn ${activeTab === 'preview' ? 'btn-primary' : 'btn-outline'}`}
                  style={{ padding: '0.5rem 1rem', fontSize: '0.825rem' }}
                  onClick={() => setActiveTab('preview')}
                >
                  <Table size={15} /> Sample Preview
                </button>
              </div>

              {/* Upload another dataset */}
              <button
                className="btn btn-outline"
                onClick={() => setDataset(null)}
                style={{ fontSize: '0.825rem' }}
              >
                <FileUp size={15} />
                Upload Another CSV
              </button>
            </div>

            {/* Tab Views */}
            {activeTab === 'overview' && (
              <div>
                <DatasetOverview summary={dataset.summary} ingestion={dataset.ingestion} />
                <QualitySection quality={dataset.quality} />
              </div>
            )}

            {activeTab === 'eda' && (
              <EDADashboard
                datasetId={dataset.dataset_id}
                columns={dataset.summary.columns.map(c => ({
                  name: c.name,
                  inferred_dtype: c.inferred_dtype
                }))}
              />
            )}

            {activeTab === 'columns' && (
              <ColumnTable columns={dataset.summary.columns} />
            )}

            {activeTab === 'quality' && (
              <QualitySection quality={dataset.quality} />
            )}

            {activeTab === 'preview' && (
              <DatasetPreview datasetId={dataset.dataset_id} />
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
