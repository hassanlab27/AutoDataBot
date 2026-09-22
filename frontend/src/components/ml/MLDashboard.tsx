import React, { useState, useEffect, useRef } from 'react';
import { api } from '../../services/api';
import { AutoMLConfig, RunStatus, RunDetails } from '../../types/ml';
import { TrainingConfigPanel } from './TrainingConfigPanel';
import { TrainingProgressModal } from './TrainingProgressModal';
import { LeaderboardTable } from './LeaderboardTable';
import { FinalModelCard } from './FinalModelCard';

interface MLDashboardProps {
  datasetId: string;
  onViewEvaluation?: (runId: string) => void;
}

export const MLDashboard: React.FC<MLDashboardProps> = ({ datasetId, onViewEvaluation }) => {
  const [preprocessingSummary, setPreprocessingSummary] = useState<any>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [runDetails, setRunDetails] = useState<RunDetails | null>(null);
  const [pastRuns, setPastRuns] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const pollingTimerRef = useRef<any>(null);

  // Load preprocessing metadata and past runs for dataset on mount
  useEffect(() => {
    let isMounted = true;

    async function loadInitialData() {
      try {
        const prep = await api.getPreprocessingSummary(datasetId);
        if (isMounted && prep.status === 'prepared') {
          setPreprocessingSummary(prep);
        }
      } catch {
        // Preprocessing may not have been run yet
      }

      try {
        const runsRes = await api.getDatasetRuns(datasetId);
        if (isMounted && runsRes.success && runsRes.runs.length > 0) {
          setPastRuns(runsRes.runs);
          // If there's a recent completed or active run, load it
          const latest = runsRes.runs[0];
          setActiveRunId(latest.run_id);
          loadRunDetails(latest.run_id);
        }
      } catch {
        // No runs found yet
      }
    }

    loadInitialData();

    return () => {
      isMounted = false;
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
      }
    };
  }, [datasetId]);

  // Polling loop for active training run
  useEffect(() => {
    if (!activeRunId) return;

    const checkStatus = async () => {
      try {
        const res = await api.getRunStatus(activeRunId);
        if (res.success && res.data) {
          setRunStatus(res.data);
          if (res.data.status === 'completed' || res.data.status === 'failed' || res.data.status === 'cancelled') {
            clearInterval(pollingTimerRef.current);
            pollingTimerRef.current = null;
            setIsLoading(false);
            loadRunDetails(activeRunId);
            // Refresh past runs list
            api.getDatasetRuns(datasetId).then((r) => {
              if (r.success) setPastRuns(r.runs);
            });
          }
        }
      } catch (err) {
        console.error('Error polling run status:', err);
      }
    };

    if (runStatus?.status === 'running' || runStatus?.status === 'queued' || runStatus?.status === 'created' || runStatus?.status === 'evaluating') {
      if (!pollingTimerRef.current) {
        pollingTimerRef.current = setInterval(checkStatus, 2000);
      }
    }

    return () => {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
        pollingTimerRef.current = null;
      }
    };
  }, [activeRunId, runStatus?.status, datasetId]);

  const loadRunDetails = async (runId: string) => {
    try {
      const res = await api.getRun(runId);
      if (res.success && res.data) {
        setRunDetails(res.data);
        setRunStatus(res.data.status);
      }
    } catch (err: any) {
      console.error('Failed to load run details:', err);
    }
  };

  const handleStartTraining = async (config: AutoMLConfig) => {
    setErrorMsg(null);
    setIsLoading(true);
    try {
      const res = await api.startRun(config);
      if (res.success && res.data) {
        const newRunId = res.data.run_id;
        setActiveRunId(newRunId);
        setRunStatus({
          run_id: newRunId,
          dataset_id: datasetId,
          status: 'running',
          current_stage: 'Training initialized',
          progress_pct: 5,
          start_time: Date.now() / 1000,
          elapsed_seconds: 0,
          cancellation_requested: false
        });
        setRunDetails(null);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to start AutoML run');
      setIsLoading(false);
    }
  };

  const handleCancelTraining = async () => {
    if (!activeRunId) return;
    setIsCancelling(true);
    try {
      await api.cancelRun(activeRunId);
    } catch (err: any) {
      console.error('Failed to cancel run:', err);
    } finally {
      setIsCancelling(false);
    }
  };

  const defaultTarget = preprocessingSummary?.config?.target || '';
  const defaultProblemType = preprocessingSummary?.config?.problem_type || 'binary_classification';

  const isTrainingActive =
    runStatus &&
    ['created', 'queued', 'running', 'evaluating'].includes(runStatus.status);

  // Find winner model from leaderboard or details
  const winnerModel = runDetails?.leaderboard?.find((m) => m.is_winner);

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 py-6">
      {/* Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white flex items-center gap-2.5">
            <span className="p-2 bg-indigo-500/20 text-indigo-400 rounded-xl">🤖</span>
            AutoML & Tabular Model Training
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Zero-leakage training engine featuring AutoGluon, FLAML, Scikit-learn, and Gradient Boosted Trees.
          </p>
        </div>

        {/* Past Runs Selector */}
        {pastRuns.length > 0 && (
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400 font-semibold">Previous Runs:</label>
            <select
              value={activeRunId || ''}
              onChange={(e) => {
                const selected = e.target.value;
                setActiveRunId(selected);
                loadRunDetails(selected);
              }}
              className="px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-xs font-mono text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {pastRuns.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} ({r.target || 'AutoML'} — {r.winning_model || r.status})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Error Message Banner */}
      {errorMsg && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-sm flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span>⚠️</span>
            <span>{errorMsg}</span>
          </div>
          <button
            onClick={() => setErrorMsg(null)}
            className="text-rose-400 hover:text-white text-xs font-bold"
          >
            ✕
          </button>
        </div>
      )}

      {/* Active Training Progress Indicator */}
      {isTrainingActive && runStatus && (
        <TrainingProgressModal
          status={runStatus}
          onCancel={handleCancelTraining}
          isCancelling={isCancelling}
        />
      )}

      {/* Training Configuration Panel (Visible when not actively training) */}
      {!isTrainingActive && (
        <TrainingConfigPanel
          datasetId={datasetId}
          defaultTarget={defaultTarget}
          defaultProblemType={defaultProblemType}
          onStartTraining={handleStartTraining}
          isLoading={isLoading}
        />
      )}

      {/* Completed Run Results */}
      {runDetails && runDetails.status.status === 'completed' && (
        <div className="space-y-8 animate-fadeIn">
          {/* Winner Spotlight Card */}
          {winnerModel && (
            <FinalModelCard
              winner={winnerModel}
              primaryMetric={runDetails.metrics?.primary_metric || 'score'}
              naiveBaselineScore={runDetails.metrics?.naive_baseline_score}
              onViewEvaluation={onViewEvaluation && activeRunId ? () => onViewEvaluation(activeRunId) : undefined}
            />
          )}

          {/* Model Leaderboard */}
          {runDetails.leaderboard && runDetails.leaderboard.length > 0 && (
            <LeaderboardTable
              leaderboard={runDetails.leaderboard}
              primaryMetric={runDetails.metrics?.primary_metric || 'score'}
            />
          )}
        </div>
      )}

      {/* Failed / Cancelled Run Notice */}
      {runDetails && runDetails.status.status === 'failed' && (
        <div className="p-6 bg-rose-950/20 border border-rose-500/40 rounded-xl text-center">
          <div className="text-3xl mb-2">✕</div>
          <h3 className="text-lg font-bold text-rose-300">AutoML Training Run Failed</h3>
          <p className="text-sm text-slate-400 mt-1">
            {runDetails.status.error_message || 'An unexpected error occurred during model fitting.'}
          </p>
        </div>
      )}

      {runDetails && runDetails.status.status === 'cancelled' && (
        <div className="p-6 bg-amber-950/20 border border-amber-500/40 rounded-xl text-center">
          <div className="text-3xl mb-2">⏹</div>
          <h3 className="text-lg font-bold text-amber-300">AutoML Training Run Cancelled</h3>
          <p className="text-sm text-slate-400 mt-1">
            The run was stopped early by user request. Partial progress was safely cleared.
          </p>
        </div>
      )}
    </div>
  );
};
