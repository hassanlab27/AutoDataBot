# AutoDataBot — Phase 5: Model Evaluation & Explainability Walkthrough

## Summary of Completed Work
In Phase 5, we implemented an advanced, leakage-safe model evaluation, error analysis, diagnostics, and explainability engine for **AutoDataBot**, along with a responsive, dark-mode React evaluation dashboard.

Phase 5 strictly adheres to the core principles:
1. **Zero Retraining / Tuning:** Evaluates existing Phase 4 models on held-out test data without retraining or altering models or thresholds.
2. **Leakage Protection:** Evaluation calculations strictly preserve dataset partitioning; permutation importance defaults to validation data.
3. **Statistical Caution:** All diagnostics and attribution views display clear neutral disclaimers that importance and SHAP values describe model attribution rather than real-world causality.
4. **Resilient Architecture:** SHAP computation runs deterministically on configurable samples (`sample_size=500` or `20` in tests) and fails gracefully for unsupported models without crashing the evaluation dashboard.
5. **Persistent Caching:** Evaluation metrics, confusion matrices, residuals, error tables, and SHAP outputs are cached under `outputs/runs/<run_id>/evaluation/` and `outputs/runs/<run_id>/explainability/`.

---

## Key Components Implemented

### 1. Backend Evaluation Engine (`backend/app/evaluation/`)
- `loader.py`: Re-partitions datasets deterministically using Phase 3 split configs, extracts transformed-to-raw feature mappings, and loads model estimators.
- `classification.py`: Computes Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, per-class metrics, confusion matrices (counts & normalized), downsampled ROC/PR curves, threshold analysis, and probability calibration with Brier score.
- `regression.py`: Computes MAE, RMSE, R², Median Absolute Error, Explained Variance, MAPE, Actual vs Predicted scatter, residual statistics & histograms, and neutral pattern diagnostics.
- `errors.py`: Error rate, false positives/negatives, most confused class pairs, and top 20 worst predictions with feature subsets.
- `diagnostics.py`: Evaluates validation vs test performance with neutral labels (`No obvious gap`, `Moderate generalization gap`, etc.).
- `comparison.py`: Cross-model comparison and validation vs test score bar charts.
- `service.py`: Orchestrates evaluation computations and disk caching.

### 2. Explainability Engine (`backend/app/explainability/`)
- `feature_importance.py`: Native tree/linear model feature importance with original-feature mapping.
- `permutation.py`: Model-agnostic permutation importance on the validation partition.
- `shap_service.py`: TreeExplainer / LinearExplainer / Kernel fallback with deterministic sampling, global attributions, feature dependence, local sample attributions, and non-blocking background execution.

### 3. API Endpoints (`backend/app/api/routes/`)
- `GET /api/runs/{run_id}/evaluation`
- `GET /api/runs/{run_id}/evaluation/classification`
- `GET /api/runs/{run_id}/evaluation/regression`
- `GET /api/runs/{run_id}/evaluation/errors`
- `GET /api/runs/{run_id}/evaluation/calibration`
- `GET /api/runs/{run_id}/evaluation/diagnostics`
- `GET /api/runs/{run_id}/evaluation/comparison`
- `GET /api/runs/{run_id}/evaluation/importance`
- `GET /api/runs/{run_id}/evaluation/permutation`
- `POST /api/runs/{run_id}/explainability/shap`
- `GET /api/runs/{run_id}/explainability/shap/status`
- `GET /api/runs/{run_id}/explainability/shap`
- `GET /api/runs/{run_id}/explainability/dependence`
- `GET /api/runs/{run_id}/explainability/local/{prediction_id}`

### 4. Frontend Dashboard (`frontend/src/components/evaluation/`)
- `EvaluationDashboard.tsx`: Top-level evaluation container with run selector and sub-tabs.
- `EvaluationHeader.tsx`: Winner model card, generalization gap, training/prediction time, diagnostic badge.
- `ClassificationPerformanceView.tsx`: Metric cards, Plotly confusion matrix (counts/percentages toggle), ROC curve, PR curve, calibration curve, threshold exploration.
- `RegressionPerformanceView.tsx`: Metric cards, Actual vs Predicted scatter with diagonal reference, residual distribution histogram, residuals vs predicted scatter, diagnostics notes.
- `ErrorAnalysisView.tsx`: Top 20 worst predictions table, most confused pairs, and local feature inspector with direct "Explain with SHAP" jump.
- `ExplainabilityView.tsx`: Native feature importance, Permutation importance, SHAP global summary, SHAP feature dependence, and Local sample attribution waterfall.
- `ModelComparisonView.tsx`: Generalization comparison bar chart (Validation vs Test) and complete candidate leaderboard comparison table.

---

## Verification Results

### Test Suite Execution
```
======================= 92 passed, 11 warnings in 13.22s =======================
```
- **Phase 1 Tests:** PASS (`test_health.py`, `test_ingestion.py`, `test_quality.py`, `test_validation.py`, `test_api_endpoints.py`)
- **Phase 2 Tests:** PASS (`test_eda.py`, `test_phase2_verification_datasets.py`)
- **Phase 3 Tests:** PASS (`test_preprocessing.py`, `test_phase3_verification_datasets.py`)
- **Phase 4 Tests:** PASS (`test_automl.py`, `test_phase4_verification_datasets.py`)
- **Phase 5 Tests:** PASS (`test_evaluation.py` [11 tests], `test_evaluation_api.py` [1 e2e test])

### Frontend Build
```
✓ 1623 modules transformed.
dist/index.html                     0.98 kB │ gzip:     0.55 kB
dist/assets/index-DpbkPFbD.css      5.60 kB │ gzip:     1.66 kB
dist/assets/index-B3P_ex_c.js     357.45 kB │ gzip:    87.33 kB
dist/assets/plotly-BRHOZM6Y.js  4,778.10 kB │ gzip: 1,472.39 kB
✓ built in 19.06s
```
Zero TypeScript errors, clean bundle compilation.
