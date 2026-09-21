import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, AlertCircle, Loader2 } from 'lucide-react';
import { api } from '../services/api';
import { UploadResponse, ApiError } from '../types/dataset';

interface UploadSectionProps {
  onUploadSuccess: (data: UploadResponse) => void;
}

export const UploadSection: React.FC<UploadSectionProps> = ({ onUploadSuccess }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setErrorMessage(null);

    // 1. Client-side extension validation
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMessage('Unsupported file type. Please upload a standard tabular .csv file.');
      return;
    }

    if (file.size === 0) {
      setErrorMessage('The selected file is empty (0 bytes).');
      return;
    }

    setIsUploading(true);
    try {
      const response = await api.uploadDataset(file);
      onUploadSuccess(response);
    } catch (err: any) {
      const apiErr = err as ApiError;
      setErrorMessage(apiErr.message || 'Failed to upload and parse CSV dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div style={{ maxWidth: '720px', margin: '3rem auto', textAlign: 'center' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>
          Upload Your CSV Dataset
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1rem' }}>
          Inspect, validate, and understand dataset quality locally with zero cloud dependencies.
        </p>
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragging ? 'var(--primary)' : 'rgba(255, 255, 255, 0.15)'}`,
          backgroundColor: isDragging ? 'rgba(99, 102, 241, 0.08)' : 'rgba(17, 24, 39, 0.6)',
          borderRadius: 'var(--radius-lg)',
          padding: '3.5rem 2rem',
          cursor: isUploading ? 'not-allowed' : 'pointer',
          transition: 'all 0.25s ease',
          boxShadow: isDragging ? '0 0 25px rgba(99, 102, 241, 0.25)' : 'none'
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            backgroundColor: 'var(--primary-glow)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--primary)'
          }}>
            {isUploading ? (
              <Loader2 className="animate-spin" size={32} />
            ) : (
              <UploadCloud size={32} />
            )}
          </div>

          <div>
            <p style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
              {isUploading ? 'Validating and Ingesting CSV...' : 'Click to choose or drag & drop your CSV here'}
            </p>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              Supports tabular comma, semicolon, tab, and pipe delimited CSV files up to 100 MB.
            </p>
          </div>

          <button
            type="button"
            className="btn btn-primary"
            disabled={isUploading}
            style={{ marginTop: '0.5rem', pointerEvents: 'none' }}
          >
            <FileText size={16} />
            Choose CSV File
          </button>
        </div>
      </div>

      {errorMessage && (
        <div style={{
          marginTop: '1.5rem',
          padding: '1rem',
          borderRadius: 'var(--radius-md)',
          backgroundColor: 'var(--danger-bg)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          textAlign: 'left'
        }}>
          <AlertCircle size={20} color="var(--danger)" style={{ flexShrink: 0 }} />
          <div style={{ fontSize: '0.875rem', color: '#fca5a5' }}>
            <strong>Upload Error:</strong> {errorMessage}
          </div>
        </div>
      )}
    </div>
  );
};
