import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.ml.metrics import (
    compute_metrics,
    is_higher_better,
    compare_scores,
    select_default_primary_metric
)
from app.ml.suitability import assess_training_suitability
from app.ml.evaluator import (
    ModelResult,
    compute_generalization_gap,
    diagnose_model_fit,
    rank_and_select_winner
)
from app.ml.job_runner import job_runner
from app.ml.config import AutoMLConfig
from app.ml.training_service import training_service
from app.storage.dataset_store import dataset_store

client = TestClient(app)

# -------------------------------------------------------------
# 1. Metrics & Directionality Unit Tests
# -------------------------------------------------------------

def test_metrics_binary_classification():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.85, 0.15])

    res = compute_metrics("binary_classification", y_true, y_pred, y_prob)
    assert "accuracy" in res
    assert "f1" in res
    assert "precision" in res
    assert "recall" in res
    assert "roc_auc" in res
    assert res["accuracy"] == 0.875
    assert res["roc_auc"] > 0.8

def test_metrics_multiclass_classification():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])

    res = compute_metrics("multiclass_classification", y_true, y_pred)
    assert "accuracy" in res
    assert "f1_macro" in res
    assert res["accuracy"] == round(5 / 6, 4)

def test_metrics_regression():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 31.0, 39.0])

    res = compute_metrics("regression", y_true, y_pred)
    assert "mae" in res
    assert "rmse" in res
    assert "r2" in res
    assert res["mae"] == 1.5
    assert res["r2"] > 0.9

def test_metric_direction_and_comparison():
    assert is_higher_better("accuracy") is True
    assert is_higher_better("f1") is True
    assert is_higher_better("r2") is True
    assert is_higher_better("rmse") is False
    assert is_higher_better("mae") is False

    # Accuracy: higher is better
    assert compare_scores("accuracy", 0.95, 0.85) == 1
    assert compare_scores("accuracy", 0.80, 0.90) == -1

    # RMSE: lower is better
    assert compare_scores("rmse", 1.5, 3.2) == 1
    assert compare_scores("rmse", 5.0, 2.0) == -1

def test_select_default_primary_metric():
    assert select_default_primary_metric("binary_classification") == "accuracy"
    # Severe imbalance (< 25% minority) selects F1
    diag = {"classification": {"minority_class_percentage": 10.0}}
    assert select_default_primary_metric("binary_classification", diag) == "f1"
    assert select_default_primary_metric("multiclass_classification") == "f1_macro"
    assert select_default_primary_metric("regression") == "rmse"

# -------------------------------------------------------------
# 2. Dataset Suitability & Leakage Guard Tests
# -------------------------------------------------------------

def test_suitability_invalid_dataset():
    suit = assess_training_suitability("non_existent_dataset_id_123", "target")
    assert suit["suitable"] is False
    assert any("does not exist" in e for e in suit["errors"])

import io

def upload_df(df: pd.DataFrame, filename: str = "test.csv") -> str:
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    res = client.post(
        "/api/datasets/upload",
        files={"file": (filename, csv_bytes, "text/csv")}
    )
    assert res.status_code == 201, f"Upload failed: {res.text}"
    return res.json()["dataset_id"]

def test_suitability_validation_synthetic(tmp_path):
    # Create valid synthetic dataset in store
    df = pd.DataFrame({
        "feat1": np.random.randn(50),
        "feat2": np.random.choice(["A", "B"], size=50),
        "target_col": np.random.choice([0, 1], size=50),
        "leak_result_target": np.random.choice([0, 1], size=50)
    })
    ds_id = upload_df(df, filename="suitability_test.csv")

    # 1. Target missing
    suit = assess_training_suitability(ds_id, "unknown_col")
    assert suit["suitable"] is False

    # 2. Valid target with leakage warning
    suit_valid = assess_training_suitability(ds_id, "target_col", "binary_classification")
    assert suit_valid["suitable"] is True
    assert len(suit_valid["warnings"]) > 0  # should flag leak_result_target

# -------------------------------------------------------------
# 3. Evaluator, Diagnostics & Winner Selection
# -------------------------------------------------------------

def test_generalization_gap_computation():
    # Higher is better: 0.90 val vs 0.75 test = 0.15 performance drop
    gap_higher = compute_generalization_gap("accuracy", 0.90, 0.75)
    assert gap_higher == 0.15

    # Lower is better: 5.0 val error vs 8.0 test error = 3.0 performance degradation
    gap_lower = compute_generalization_gap("rmse", 5.0, 8.0)
    assert gap_lower == 3.0

def test_diagnose_model_fit_heuristics():
    # 1. Suspiciously high performance
    label, notes = diagnose_model_fit("accuracy", 0.9995, 0.999, 0.5)
    assert label == "Suspiciously High Performance"

    # 2. Worse than naive baseline
    label, notes = diagnose_model_fit("accuracy", 0.45, 0.42, 0.60)
    assert label == "Worse than Baseline"

    # 3. Severe Overfitting
    label, notes = diagnose_model_fit("accuracy", 0.90, 0.65, 0.50)
    assert label == "Severe Overfitting"

    # 4. Possible Overfitting
    label, notes = diagnose_model_fit("accuracy", 0.85, 0.75, 0.50)
    assert label == "Possible Overfitting"

    # 5. Normal / Well-fit
    label, notes = diagnose_model_fit("accuracy", 0.85, 0.83, 0.50)
    assert label == "Normal / Well-fit"

def test_winner_selection():
    m1 = ModelResult(
        model_id="dummy",
        model_name="DummyClassifier",
        engine="sklearn",
        problem_type="binary_classification",
        validation_primary_score=0.55,
        is_naive_baseline=True
    )
    m2 = ModelResult(
        model_id="rf",
        model_name="RandomForest",
        engine="sklearn",
        problem_type="binary_classification",
        validation_primary_score=0.88,
        is_naive_baseline=False
    )
    m3 = ModelResult(
        model_id="logreg",
        model_name="LogisticRegression",
        engine="sklearn",
        problem_type="binary_classification",
        validation_primary_score=0.79,
        is_naive_baseline=False
    )

    winner, leaderboard = rank_and_select_winner([m1, m2, m3], "accuracy")
    assert winner is not None
    assert winner.model_id == "rf"
    assert winner.is_winner is True
    assert leaderboard[0].model_id == "rf"
    assert leaderboard[1].model_id == "logreg"
    assert leaderboard[2].model_id == "dummy"

# -------------------------------------------------------------
# 4. Job Runner & Cooperative Cancellation
# -------------------------------------------------------------

def test_job_runner_cancellation():
    run_id = "test_cancel_run_001"
    run_dir = Path("/tmp")
    job_runner.register_run(run_id, "ds_1", run_dir)
    assert job_runner.is_cancelled(run_id) is False

    success = job_runner.request_cancellation(run_id)
    assert success is True
    assert job_runner.is_cancelled(run_id) is True
    st = job_runner.get_status(run_id)
    assert st is not None
    assert st.status == "cancelled"

# -------------------------------------------------------------
# 5. End-to-End AutoML Pipeline Execution Test
# -------------------------------------------------------------

def test_e2e_automl_training_fast():
    # Create synthetic dataset with 60 rows
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.uniform(20000, 100000, size=n),
        "department": np.random.choice(["Sales", "Engineering", "Marketing"], size=n),
        "churn": np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    })
    ds_id = upload_df(df, filename="test_automl_fast.csv")

    # Configure AutoML run: sklearn baselines only for ultra-fast unit testing
    config_dict = {
        "dataset_id": ds_id,
        "target": "churn",
        "problem_type": "binary_classification",
        "training_mode": "quick",
        "time_limit": 10,
        "primary_metric": "accuracy",
        "random_state": 42,
        "engine_preference": "sklearn_gbdt"
    }

    # API POST /api/runs
    res = client.post("/api/runs", json=config_dict)
    assert res.status_code == 201
    resp_data = res.json()["data"]
    run_id = resp_data["run_id"]
    assert run_id.startswith("run_")

    # Poll until completed (or max 20s)
    import time
    max_wait = 25
    start = time.time()
    final_status = None
    while time.time() - start < max_wait:
        st_res = client.get(f"/api/runs/{run_id}/status")
        assert st_res.status_code == 200
        st = st_res.json()["data"]
        if st["status"] in ["completed", "failed", "cancelled"]:
            final_status = st["status"]
            break
        time.sleep(0.5)

    assert final_status == "completed", f"Run failed with: {st.get('error_message')}"

    # Verify Leaderboard
    lb_res = client.get(f"/api/runs/{run_id}/leaderboard")
    assert lb_res.status_code == 200
    lb = lb_res.json()["leaderboard"]
    assert len(lb) >= 4  # Dummy, Linear/Logistic, RF, HistGradientBoosting
    
    # Verify Winner
    winners = [m for m in lb if m["is_winner"]]
    assert len(winners) == 1
    winner = winners[0]
    assert winner["test_primary_score"] is not None
    assert winner["generalization_gap"] is not None
    assert winner["diagnostic_label"] is not None

    # Verify Full Run Details
    detail_res = client.get(f"/api/runs/{run_id}")
    assert detail_res.status_code == 200
    details = detail_res.json()["data"]
    assert "metrics" in details
    assert "summary" in details
    assert details["summary"]["winning_model"] == winner["model_name"]
