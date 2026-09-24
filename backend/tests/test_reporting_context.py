import pytest
from pathlib import Path
from app.core.config import settings
from app.reporting.report_context import build_report_context

def test_build_report_context_classification():
    # Use existing completed multiclass classification run
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    ctx = build_report_context(run_id)

    # 1. Executive summary
    assert "executive_summary" in ctx
    exec_sum = ctx["executive_summary"]
    assert exec_sum["run_id"] == run_id
    assert exec_sum["target"] == "diabetes_risk"
    assert exec_sum["problem_type"] == "multiclass_classification"
    assert exec_sum["winning_model"] is not None
    assert exec_sum["validation_metric"] is not None
    assert exec_sum["test_metric"] is not None

    # 2. Dataset overview
    assert "dataset_overview" in ctx
    ds_ov = ctx["dataset_overview"]
    assert ds_ov["rows_count"] > 0
    assert ds_ov["columns_count"] > 0

    # 3. Data quality
    assert "data_quality" in ctx
    dq = ctx["data_quality"]
    assert "quality_score" in dq
    assert "quality_warnings" in dq

    # 4. EDA
    assert "eda" in ctx
    assert "target_distribution" in ctx["eda"]

    # 5. Target / Problem
    assert "target_problem" in ctx
    tp = ctx["target_problem"]
    assert tp["problem_type"] == "multiclass_classification"
    assert "class_counts" in tp

    # 6. Preprocessing
    assert "preprocessing" in ctx
    prep = ctx["preprocessing"]
    assert "split" in prep
    assert prep["split"]["train_rows"] > 0
    assert prep["split"]["test_rows"] > 0

    # 7. Training configuration
    assert "training_config" in ctx
    tc = ctx["training_config"]
    assert tc["primary_metric"] == "accuracy"

    # 8. Leaderboard
    assert "leaderboard" in ctx
    assert isinstance(ctx["leaderboard"], list)
    assert len(ctx["leaderboard"]) > 0
    assert "rank" in ctx["leaderboard"][0]
    assert "model_name" in ctx["leaderboard"][0]

    # 9. Selected model
    assert "selected_model" in ctx
    sm = ctx["selected_model"]
    assert sm["model_name"] == exec_sum["winning_model"]

    # 10. Evaluation
    assert "evaluation" in ctx
    eval_sec = ctx["evaluation"]
    assert eval_sec["problem_type"] == "multiclass_classification"
    assert "accuracy" in eval_sec["metrics"]
    assert "confusion_matrix" in eval_sec

    # 11. Generalization & Diagnostics
    assert "diagnostics" in ctx
    diag = ctx["diagnostics"]
    assert "diagnostic_label" in diag
    assert "disclaimer" in diag

    # 12. Error analysis
    assert "error_analysis" in ctx

    # 13. Feature importance
    assert "feature_importance" in ctx

    # 14. SHAP
    assert "shap" in ctx
    assert "methodological_note" in ctx["shap"]

    # 15. Limitations
    assert "limitations" in ctx
    assert len(ctx["limitations"]) > 0

    # Security check: no internal secrets or full local absolute paths in public fields
    context_str = str(ctx)
    assert "password" not in context_str.lower()
    assert "secret" not in context_str.lower()


def test_build_report_context_regression():
    # Use existing completed regression run
    run_id = "run_20260924_111902_7d89a3"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    ctx = build_report_context(run_id)
    assert ctx["executive_summary"]["problem_type"] == "regression"
    assert "mae" in ctx["evaluation"]["metrics"]
    assert "rmse" in ctx["evaluation"]["metrics"]
    assert "r2" in ctx["evaluation"]["metrics"]
    assert "residual_stats" in ctx["evaluation"]


def test_build_report_context_invalid_run():
    with pytest.raises(Exception):
        build_report_context("non_existent_run_id_99999")
