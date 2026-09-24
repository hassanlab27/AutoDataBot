import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import { RunSummaryCard } from './RunSummaryCard';
import { ReportViewerModal } from './ReportViewerModal';
import { RunSummaryData } from '../../types/report';
import { 
  FileText, 
  Archive, 
  RefreshCw, 
  AlertCircle, 
  History,
  ExternalLink
} from 'lucide-react';
import './reporting.css';

interface ReportingDashboardProps {
  datasetId: string;
  initialRunId?: string | null;
  rowsCount?: number | string;
  columnsCount?: number | string;
  onNavigateToEvaluation?: (runId: string) => void;
  onNavigateToExplainability?: (runId: string) => void;
}

export const ReportingDashboard: React.FC<ReportingDashboardProps> = ({
  datasetId,
  initialRunId,
  rowsCount,
  columnsCount,
  onNavigateToEvaluation,
  onNavigateToExplainability
}) => {
  const [runs, setRuns] = useState<any[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(initialRunId || null);
  const [runSummary, setRunSummary] = useState<RunSummaryData | null>(null);
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [isReportModalOpen, setIsReportModalOpen] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isReportLoading, setIsReportLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // 1. Fetch available runs for dataset
  useEffect(() => {
    let isMounted = true;
    async function loadRuns() {
      try {
        const res = await api.getDatasetRuns(datasetId);
        if (isMounted && res.success && res.runs.length > 0) {
          setRuns(res.runs);
          if (!activeRunId) {
            const completedRun = res.runs.find((r: any) => r.status === 'completed') || res.runs[0];
            setActiveRunId(completedRun.run_id);
          }
        } else if (isMounted) {
          setIsLoading(false);
        }
      } catch (err) {
        console.error('Error fetching runs:', err);
        if (isMounted) setIsLoading(false);
      }
    }
    loadRuns();
    return () => { isMounted = false; };
  }, [datasetId]);

  // 2. Fetch run summary & report HTML when activeRunId changes
  useEffect(() => {
    if (!activeRunId) return;

    let isMounted = true;
    async function loadDetails() {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        // Fetch run details
        const detailsRes = await api.getRun(activeRunId!);
        if (isMounted && detailsRes.success && detailsRes.data) {
          const d = detailsRes.data;
          const sumData: RunSummaryData = {
            run_id: d.run_id,
            dataset_id: d.status?.dataset_id || d.summary?.dataset_id || datasetId,
            problem_type: d.summary?.problem_type || d.config?.problem_type,
            target: d.summary?.target || d.config?.target,
            selected_model: d.summary?.winning_model,
            winning_engine: d.summary?.winning_engine,
            validation_score: d.summary?.validation_score,
            test_score: d.summary?.test_score,
            train_score: d.summary?.train_score,
            primary_metric: d.summary?.primary_metric || d.config?.primary_metric,
            duration_seconds: d.status?.elapsed_seconds,
            status: d.status?.status || 'completed',
            created_at: d.summary?.created_at
          };
          setRunSummary(sumData);

          // If completed, fetch report HTML
          if (sumData.status === 'completed') {
            setIsReportLoading(true);
            try {
              const html = await api.getReportHtml(activeRunId!);
              if (isMounted) setReportHtml(html);
            } catch (rErr: any) {
              console.warn('Could not prefetch report HTML:', rErr);
            } finally {
              if (isMounted) setIsReportLoading(false);
            }
          } else {
            setReportHtml(null);
          }
        }
      } catch (err: any) {
        if (isMounted) setErrorMsg(err?.message || 'Failed to load run details.');
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    loadDetails();
    return () => { isMounted = false; };
  }, [activeRunId]);

  if (isLoading && !runSummary) {
    return (
      <div className="reporting-card" style={{ padding: '3.5rem 2rem', textAlign: 'center', maxWidth: '600px', margin: '2rem auto' }}>
        <RefreshCw size={36} className="text-indigo-400 spin" style={{ margin: '0 auto 1rem auto' }} />
        <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#fff', marginBottom: '0.5rem' }}>
          Loading Run Report & Exports...
        </h3>
        <p style={{ fontSize: '0.85rem', color: '#9ca3af' }}>
          Compiling executive metrics, evaluation outputs, and artifact bundles.
        </p>
      </div>
    );
  }

  if (errorMsg || !runSummary) {
    return (
      <div className="reporting-card" style={{ padding: '2.5rem', textAlign: 'center', maxWidth: '580px', margin: '2rem auto' }}>
        <AlertCircle size={44} className="text-amber-400" style={{ margin: '0 auto 1rem auto' }} />
        <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#fff', marginBottom: '0.5rem' }}>
          No Completed Runs Available
        </h3>
        <p style={{ fontSize: '0.85rem', color: '#9ca3af', marginBottom: '1.5rem', lineHeight: 1.5 }}>
          {errorMsg || 'Train models in the Prepare Dataset section to generate final reports and export bundles.'}
        </p>
      </div>
    );
  }

  return (
    <div className="reporting-container">
      {/* Run Selector Bar */}
      {runs.length > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.75rem', fontSize: '0.825rem', color: '#9ca3af' }}>
          <History size={16} />
          <span>Select Run History:</span>
          <select
            value={activeRunId || ''}
            onChange={(e) => setActiveRunId(e.target.value)}
            className="run-select-box"
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.status})
              </option>
            ))}
          </select>
        </div>
      )}

      {/* 1. Executive Summary Card */}
      <RunSummaryCard
        summary={runSummary}
        rowsCount={rowsCount}
        columnsCount={columnsCount}
        onViewEvaluation={onNavigateToEvaluation ? () => onNavigateToEvaluation(runSummary.run_id) : undefined}
        onViewExplainability={onNavigateToExplainability ? () => onNavigateToExplainability(runSummary.run_id) : undefined}
        onOpenReportModal={() => setIsReportModalOpen(true)}
      />

      {/* 2. Embedded Report Preview Card */}
      {runSummary.status === 'completed' && (
        <div className="reporting-card">
          <div className="reporting-header-row">
            <div>
              <h3 className="reporting-title">
                <FileText size={20} className="text-indigo-400" />
                <span>Deterministic Technical Report Preview</span>
              </h3>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#9ca3af' }}>
                Complete multi-section report rendered from structured artifacts.
              </p>
            </div>

            <div className="reporting-actions-bar">
              <button
                className="btn btn-primary"
                onClick={() => setIsReportModalOpen(true)}
                style={{ fontSize: '0.825rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
              >
                <ExternalLink size={15} /> Expand Full Screen
              </button>
              <a
                href={api.getExportZipUrl(runSummary.run_id)}
                download={`autodatabot_${runSummary.run_id}.zip`}
                className="btn btn-outline"
                style={{ fontSize: '0.825rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#fff' }}
              >
                <Archive size={15} /> Download Export ZIP
              </a>
            </div>
          </div>

          <div style={{ height: '700px', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: '#fff' }}>
            {isReportLoading ? (
              <div style={{ padding: '3rem', textAlign: 'center', color: '#64748b' }}>
                <RefreshCw size={24} className="spin" style={{ margin: '0 auto 0.75rem auto' }} />
                <p>Generating report preview...</p>
              </div>
            ) : reportHtml ? (
              <iframe
                srcDoc={reportHtml}
                title="Report Embedded Preview"
                style={{ width: '100%', height: '100%', border: 'none' }}
              />
            ) : (
              <div style={{ padding: '3rem', textAlign: 'center', color: '#dc2626' }}>
                Report preview could not be loaded. Click "View Final Report" above to try again.
              </div>
            )}
          </div>
        </div>
      )}

      {/* 3. Modal Report Viewer */}
      <ReportViewerModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        runId={runSummary.run_id}
        htmlContent={reportHtml}
        isLoading={isReportLoading}
      />
    </div>
  );
};
