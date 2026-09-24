import React from 'react';
import { X, Printer, ExternalLink, Download, FileText } from 'lucide-react';
import { api } from '../../services/api';

interface ReportViewerModalProps {
  isOpen: boolean;
  onClose: () => void;
  runId: string;
  htmlContent: string | null;
  isLoading?: boolean;
}

export const ReportViewerModal: React.FC<ReportViewerModalProps> = ({
  isOpen,
  onClose,
  runId,
  htmlContent,
  isLoading = false
}) => {
  if (!isOpen) return null;

  const handlePrint = () => {
    const iframe = document.getElementById('report-frame') as HTMLIFrameElement;
    if (iframe && iframe.contentWindow) {
      iframe.contentWindow.focus();
      iframe.contentWindow.print();
    }
  };

  const handleOpenNewTab = () => {
    const blob = new Blob([htmlContent || ''], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
  };

  const handleDownloadHtml = () => {
    const blob = new Blob([htmlContent || ''], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `autodatabot_${runId}_report.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="report-modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="report-modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="report-modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <FileText size={20} className="text-indigo-400" />
            <div>
              <div style={{ fontWeight: 700, fontSize: '1rem', color: '#fff' }}>
                AutoDataBot Technical Report
              </div>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                Run: <code>{runId}</code>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              className="btn btn-outline"
              onClick={handlePrint}
              style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem', color: '#fff' }}
              title="Print Report"
            >
              <Printer size={15} /> Print
            </button>
            <button
              className="btn btn-outline"
              onClick={handleDownloadHtml}
              style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem', color: '#fff' }}
              title="Download HTML file"
            >
              <Download size={15} /> Download HTML
            </button>
            <a
              href={api.getReportPdfUrl(runId)}
              download={`autodatabot_${runId}_report.pdf`}
              className="btn btn-outline"
              style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem', color: '#fff' }}
              title="Download PDF file"
            >
              <Download size={15} /> PDF
            </a>
            <button
              className="btn btn-outline"
              onClick={handleOpenNewTab}
              style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem', color: '#fff' }}
              title="Open in new window"
            >
              <ExternalLink size={15} /> Open Tab
            </button>
            <button
              onClick={onClose}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#94a3b8',
                cursor: 'pointer',
                padding: '0.4rem',
                marginLeft: '0.5rem'
              }}
              aria-label="Close report modal"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="report-modal-body">
          {isLoading ? (
            <div style={{ padding: '4rem', textAlign: 'center', color: '#64748b' }}>
              <div className="spin" style={{ display: 'inline-block', marginBottom: '1rem' }}>
                ⌛
              </div>
              <p>Generating deterministic report...</p>
            </div>
          ) : htmlContent ? (
            <iframe
              id="report-frame"
              srcDoc={htmlContent}
              title={`Report for run ${runId}`}
              className="report-iframe"
            />
          ) : (
            <div style={{ padding: '4rem', textAlign: 'center', color: '#dc2626' }}>
              Failed to load report content.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
