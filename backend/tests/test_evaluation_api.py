import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.storage.dataset_store import dataset_store
from app.preprocessing.preprocessing_service import PreprocessingService
from app.ml.config import AutoMLConfig
from app.ml.training_service import training_service
from app.ml.job_runner import job_runner

client = TestClient(app)

@pytest.fixture
def synthetic_classification_run(tmp_path):
    # Create small synthetic classification dataset
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "income": np.random.uniform(20000, 100000, n),
        "department": np.random.choice(["Sales", "Engineering", "Marketing"], n),
        "churn": np.random.choice([0, 1], n, p=[0.6, 0.4])
    })

    import io
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    up_res = client.post(
        "/api/datasets/upload",
        files={"file": ("test_eval_cls.csv", csv_bytes, "text/csv")}
    )
    assert up_res.status_code == 201
    ds_id = up_res.json()["dataset_id"]

    # Preprocess
    prep_svc = PreprocessingService()
    prep_svc.prepare_dataset(ds_id, target="churn", problem_type="binary_classification", random_state=42)

    # Train fast baseline
    config = AutoMLConfig(
        dataset_id=ds_id,
        target="churn",
        problem_type="binary_classification",
        primary_metric="accuracy",
        training_mode="quick",
        time_limit=10,
        random_state=42,
        engine_preference="sklearn_gbdt",
        presets="medium_quality_faster_train"
    )
    run_info = training_service.create_run(config)
    run_id = run_info["run_id"]

    # Wait for completion
    completed = False
    for _ in range(40):
        st = job_runner.get_status(run_id)
        if st and st.status == "completed":
            completed = True
            break
        elif st and st.status == "failed":
            raise RuntimeError(f"Run failed: {st.error_message}")
        time.sleep(0.5)

    assert completed, f"Run did not complete in time. Last status: {st.status if st else 'None'}"
    return run_id

def test_evaluation_endpoints_e2e(synthetic_classification_run):
    run_id = synthetic_classification_run

    # 1. Full Evaluation Overview
    r1 = client.get(f"/api/runs/{run_id}/evaluation")
    assert r1.status_code == 200
    d1 = r1.json()["data"]
    assert d1["run_id"] == run_id
    assert d1["problem_type"] == "binary_classification"
    assert "performance" in d1
    assert "error_analysis" in d1
    assert "diagnostics" in d1
    assert "comparison" in d1

    # 2. Classification
    r2 = client.get(f"/api/runs/{run_id}/evaluation/classification")
    assert r2.status_code == 200
    d2 = r2.json()["data"]
    assert "confusion_matrix" in d2
    assert "metrics" in d2

    # 3. Errors
    r3 = client.get(f"/api/runs/{run_id}/evaluation/errors")
    assert r3.status_code == 200
    d3 = r3.json()["data"]
    assert "worst_predictions" in d3

    # 4. Calibration
    r4 = client.get(f"/api/runs/{run_id}/evaluation/calibration")
    assert r4.status_code == 200

    # 5. Diagnostics
    r5 = client.get(f"/api/runs/{run_id}/evaluation/diagnostics")
    assert r5.status_code == 200
    d5 = r5.json()["data"]
    assert "diagnostic_label" in d5

    # 6. Comparison
    r6 = client.get(f"/api/runs/{run_id}/evaluation/comparison")
    assert r6.status_code == 200
    d6 = r6.json()["data"]
    assert "models" in d6

    # 7. Importance
    r7 = client.get(f"/api/runs/{run_id}/evaluation/importance")
    assert r7.status_code == 200
    d7 = r7.json()["data"]
    assert "available" in d7

    # 8. Permutation
    r8 = client.get(f"/api/runs/{run_id}/evaluation/permutation?repeats=2")
    assert r8.status_code == 200
    d8 = r8.json()["data"]
    assert "importances" in d8

    # 9. SHAP Summary (direct query with small sample_size)
    r9 = client.get(f"/api/runs/{run_id}/explainability/shap?sample_size=20&max_features=5")
    assert r9.status_code == 200
    d9 = r9.json()["data"]
    assert "available" in d9

    # 10. Local explanation
    r10 = client.get(f"/api/runs/{run_id}/explainability/local/0")
    assert r10.status_code == 200
    d10 = r10.json()["data"]
    assert d10["prediction_id"] == 0
    assert "actual" in d10
    assert "predicted" in d10
