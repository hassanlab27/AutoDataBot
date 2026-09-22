import React, { useState, useEffect } from 'react';
import { TargetSelector } from './TargetSelector';
import { FeatureSelectorTable } from './FeatureSelectorTable';
import { SplitAndTransformConfig } from './SplitAndTransformConfig';
import { ConfigurationReview } from './ConfigurationReview';
import { PreprocessingResults } from './PreprocessingResults';
import {
  TargetInspectResponse,
  PrepareDatasetRequest,
  PrepareDatasetResponse
} from '../../types/preprocessing';
import { api } from '../../services/api';
import { Target, CheckSquare, Sliders, ClipboardCheck, CheckCircle2, AlertCircle } from 'lucide-react';

interface PreprocessingDashboardProps {
  datasetId: string;
  columns: Array<{ name: string; inferred_dtype: string }>;
  onNavigateToEvaluation?: (runId: string) => void;
}

export const PreprocessingDashboard: React.FC<PreprocessingDashboardProps> = ({
  datasetId,
  columns,
  onNavigateToEvaluation
}) => {
  const [step, setStep] = useState<number>(1);
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  const [confirmedProblemType, setConfirmedProblemType] = useState<string | null>(null);

  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [config, setConfig] = useState<PrepareDatasetRequest>({
    target: '',
    problem_type: '',
    selected_features: [],
    test_size: 0.20,
    random_state: 42,
    numeric_imputation: 'median',
    categorical_imputation: 'most_frequent',
    categorical_encoding: 'one_hot',
    scaling: 'standard',
    extract_datetime: true
  });

  const [executionLoading, setExecutionLoading] = useState<boolean>(false);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [preparedResult, setPreparedResult] = useState<PrepareDatasetResponse | null>(null);
  const [initialLoading, setInitialLoading] = useState<boolean>(true);

  // Check if dataset was already preprocessed
  useEffect(() => {
    const checkExistingPreparation = async () => {
      setInitialLoading(true);
      try {
        const summary = await api.getPreprocessingSummary(datasetId);
        if (summary && summary.status === 'prepared') {
          const preview = await api.getPreprocessingPreview(datasetId, 20);
          setPreparedResult({
            status: 'prepared',
            dataset_id: datasetId,
            target: summary.target || '',
            problem_type: summary.problem_type || '',
            config: summary.config || {},
            metadata: summary,
            preview: {
              total_rows: preview.total_training_rows,
              preview_rows: preview.rows,
              columns: preview.columns
            }
          });
          setStep(5);
        }
      } catch {
        // No existing preparation; start at Step 1
      } finally {
        setInitialLoading(false);
      }
    };
    checkExistingPreparation();
  }, [datasetId]);

  const handleConfirmTarget = (target: string, problemType: string, _inspection: TargetInspectResponse) => {
    setSelectedTarget(target);
    setConfirmedProblemType(problemType);
    setConfig(prev => ({
      ...prev,
      target,
      problem_type: problemType
    }));
    setStep(2);
  };

  const handleProceedFromFeatures = () => {
    setConfig(prev => ({
      ...prev,
      selected_features: selectedFeatures
    }));
    setStep(3);
  };

  const handleProceedFromConfig = () => {
    setStep(4);
  };

  const handleExecutePreparation = async () => {
    setExecutionLoading(true);
    setExecutionError(null);
    try {
      const payload: PrepareDatasetRequest = {
        ...config,
        selected_features: selectedFeatures
      };
      const res = await api.prepareDataset(datasetId, payload);
      setPreparedResult(res);
      setStep(5);
    } catch (err: any) {
      setExecutionError(err?.message || 'Failed to prepare dataset and serialize pipeline.');
    } finally {
      setExecutionLoading(false);
    }
  };

  const handleReset = () => {
    setPreparedResult(null);
    setStep(1);
  };

  if (initialLoading) {
    return (
      <div className="card" style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading preprocessing specifications...
      </div>
    );
  }

  const stepsList = [
    { num: 1, label: 'Select Target', icon: Target },
    { num: 2, label: 'Select Features', icon: CheckSquare },
    { num: 3, label: 'Configuration', icon: Sliders },
    { num: 4, label: 'Review', icon: ClipboardCheck },
    { num: 5, label: 'Prepared Data', icon: CheckCircle2 }
  ];

  return (
    <div>
      {/* Step Indicator Bar */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'var(--bg-secondary)',
        padding: '0.75rem 1.25rem',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-color)',
        marginBottom: '1.5rem',
        overflowX: 'auto',
        gap: '1rem'
      }}>
        {stepsList.map((s, idx) => {
          const Icon = s.icon;
          const isActive = step === s.num;
          const isCompleted = step > s.num;
          return (
            <div
              key={s.num}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                opacity: isActive || isCompleted ? 1 : 0.5,
                color: isActive ? 'var(--accent-blue)' : isCompleted ? 'var(--accent-green)' : 'var(--text-muted)',
                fontWeight: isActive ? 700 : 500,
                fontSize: '0.85rem',
                whiteSpace: 'nowrap'
              }}
            >
              <div style={{
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                background: isActive ? 'var(--accent-blue)' : isCompleted ? 'var(--accent-green)' : 'var(--bg-card)',
                color: isActive || isCompleted ? '#fff' : 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.75rem',
                fontWeight: 700
              }}>
                {isCompleted ? '✓' : s.num}
              </div>
              <span>
                <Icon size={14} style={{ marginRight: '0.35rem', verticalAlign: 'middle' }} />
                {s.label}
              </span>
              {idx < stepsList.length - 1 && (
                <div style={{ width: '20px', height: '1px', background: 'var(--border-color)', marginLeft: '0.5rem' }} />
              )}
            </div>
          );
        })}
      </div>

      {executionError && (
        <div style={{
          padding: '1rem',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid var(--accent-red)',
          color: 'var(--accent-red)',
          marginBottom: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <AlertCircle size={18} />
          <span>{executionError}</span>
        </div>
      )}

      {/* Step Content */}
      {step === 1 && (
        <TargetSelector
          datasetId={datasetId}
          columns={columns}
          selectedTarget={selectedTarget}
          onConfirmTarget={handleConfirmTarget}
        />
      )}

      {step === 2 && selectedTarget && (
        <FeatureSelectorTable
          datasetId={datasetId}
          targetCol={selectedTarget}
          selectedFeatures={selectedFeatures}
          onChangeSelectedFeatures={setSelectedFeatures}
          onProceed={handleProceedFromFeatures}
          onBack={() => setStep(1)}
        />
      )}

      {step === 3 && (
        <SplitAndTransformConfig
          config={config}
          problemType={confirmedProblemType || 'binary_classification'}
          onChangeConfig={setConfig}
          onProceed={handleProceedFromConfig}
          onBack={() => setStep(2)}
        />
      )}

      {step === 4 && selectedTarget && (
        <ConfigurationReview
          datasetId={datasetId}
          config={config}
          totalCandidateFeatures={columns.length - 1}
          loading={executionLoading}
          onExecute={handleExecutePreparation}
          onBack={() => setStep(3)}
        />
      )}

      {step === 5 && preparedResult && (
        <PreprocessingResults
          result={preparedResult}
          onReset={handleReset}
          onNavigateToEvaluation={onNavigateToEvaluation}
        />
      )}
    </div>
  );
};
