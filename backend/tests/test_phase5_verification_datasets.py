import io
import time
from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

MODIFIED_DATA_PATH = Path("/home/hassan/Downloads/archive (1)/modified_data.csv")
DIABETES_DATA_PATH = Path("/home/hassan/Downloads/archive (2)/diabetes_risk.csv")

def upload_file_or_df(file_path: Path, fallback_df: pd.DataFrame, filename: str) -> str:
    if file_path.exists():
        with open(file_path, "rb") as f:
            csv_bytes = f.read()
    else:
        csv_buffer = io.StringIO()
        fallback_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

    res = client.post(
        "/api/datasets/upload",
        files={"file": (filename, csv_bytes, "text/csv")}
    )
    assert res.status_code == 201, f"Upload failed: {res.text}"
    return res.json()["dataset_id"]

def wait_for_run(run_id: str, max_timeout: int = 120):
    start = time.time()
    while time.time() - start < max_timeout:
        res = client.get(f"/api/runs/{run_id}/status")
        assert res.status_code == 200
        st = res.json()["data"]
        if st["status"] in ["completed", "failed", "cancelled"]:
            return st
        time.sleep(1.0)
    raise TimeoutError(f"Run {run_id} did not finish within {max_timeout}s")


def test_phase5_regression_modified_data():
    """
    Phase 5 Verification: Regression on modified_data.csv
    Target: price
    Rows: 4600, Train: 3680, Test: 920
    """
    assert MODIFIED_DATA_PATH.exists(), f"Benchmark file {MODIFIED_DATA_PATH} not found"
    ds_id = upload_file_or_df(MODIFIED_DATA_PATH, None, "modified_data.csv")

    # 1. Inspect target
    t_res = client.post(f"/api/datasets/{ds_id}/preprocessing/target-inspect", json={"target": "price"})
    assert t_res.status_code == 200
    assert t_res.json()["problem_type"] == "regression"

    # 2. Prepare dataset
    prep_res = client.post(f"/api/datasets/{ds_id}/preprocessing/prepare", json={
        "target": "price",
        "problem_type": "regression",
        "test_size": 0.20,
        "random_state": 42
    })
    assert prep_res.status_code == 200

    # 3. Start AutoML training
    run_res = client.post("/api/runs", json={
        "dataset_id": ds_id,
        "target": "price",
        "problem_type": "regression",
        "training_mode": "quick",
        "time_limit": 15,
        "primary_metric": "rmse",
        "random_state": 42,
        "engine_preference": "all"
    })
    assert run_res.status_code == 201
    run_id = run_res.json()["data"]["run_id"]

    # 4. Wait for completion
    st = wait_for_run(run_id, max_timeout=240)
    assert st["status"] == "completed", f"Run failed: {st.get('error_message')}"

    # 5. Verify Evaluation Endpoints
    eval_res = client.get(f"/api/runs/{run_id}/evaluation")
    assert eval_res.status_code == 200
    eval_payload = eval_res.json()["data"]
    assert eval_payload["problem_type"] == "regression"
    assert eval_payload["model_name"] != "None"
    assert "metrics" in eval_payload.get("performance", {})
    assert "rmse" in eval_payload["performance"]["metrics"]

    # 6. Verify distinct partition metrics exist
    assert eval_payload.get("test_score") is not None
    assert eval_payload.get("train_score") is not None

    # 7. Verify Explainability Endpoints
    exp_res = client.get(f"/api/runs/{run_id}/explainability/global")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()["data"]
    assert "global_importance" in exp_data or "importance" in exp_data

    # 8. Check SHAP summary
    shap_res = client.get(f"/api/runs/{run_id}/explainability/shap")
    assert shap_res.status_code == 200
    shap_data = shap_res.json()["data"]
    if shap_data.get("available"):
        assert shap_data.get("partition") == "validation"


def test_phase5_classification_diabetes_risk():
    """
    Phase 5 Verification: Classification on diabetes_risk.csv
    Target: diabetes_risk
    Rows: 15000, Train: 12000, Test: 3000
    Internal Train fold: 9600, Internal Val fold: 2400
    CRITICAL: Must complete without the 9600 vs 12000 mismatch!
    """
    assert DIABETES_DATA_PATH.exists(), f"Benchmark file {DIABETES_DATA_PATH} not found"
    ds_id = upload_file_or_df(DIABETES_DATA_PATH, None, "diabetes_risk.csv")

    # 1. Inspect target
    t_res = client.post(f"/api/datasets/{ds_id}/preprocessing/target-inspect", json={"target": "diabetes_risk"})
    assert t_res.status_code == 200
    prob_type = t_res.json()["problem_type"]

    # 2. Prepare dataset
    prep_res = client.post(f"/api/datasets/{ds_id}/preprocessing/prepare", json={
        "target": "diabetes_risk",
        "problem_type": prob_type,
        "test_size": 0.20,
        "random_state": 42
    })
    assert prep_res.status_code == 200

    # 3. Start AutoML training with all engines enabled
    run_res = client.post("/api/runs", json={
        "dataset_id": ds_id,
        "target": "diabetes_risk",
        "problem_type": prob_type,
        "training_mode": "quick",
        "time_limit": 20,
        "primary_metric": "accuracy",
        "random_state": 42,
        "engine_preference": "all"
    })
    assert run_res.status_code == 201
    run_id = run_res.json()["data"]["run_id"]

    # 4. Wait for completion
    st = wait_for_run(run_id, max_timeout=240)
    assert st["status"] == "completed", f"Run failed: {st.get('error_message')}"

    # 5. Verify Evaluation Endpoints
    eval_res = client.get(f"/api/runs/{run_id}/evaluation")
    assert eval_res.status_code == 200
    eval_payload = eval_res.json()["data"]
    assert eval_payload["problem_type"] == prob_type
    assert eval_payload["model_name"] != "None"
    assert "accuracy" in eval_payload["performance"]["metrics"]

    # Invariant check: Test score and Train score are populated without 9600 vs 12000 error
    assert eval_payload.get("test_score") is not None
    assert eval_payload.get("train_score") is not None

    # Check classification endpoints
    cls_res = client.get(f"/api/runs/{run_id}/evaluation/classification")
    assert cls_res.status_code == 200
    cls_data = cls_res.json()["data"]
    assert "confusion_matrix" in cls_data or "metrics" in cls_data

    # 6. Verify Explainability Endpoints
    exp_res = client.get(f"/api/runs/{run_id}/explainability/global")
    assert exp_res.status_code == 200

    shap_res = client.get(f"/api/runs/{run_id}/explainability/shap")
    assert shap_res.status_code == 200
    shap_data = shap_res.json()["data"]
    if shap_data.get("available"):
        assert shap_data.get("partition") == "validation"
