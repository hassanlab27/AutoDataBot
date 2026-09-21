import io
import time
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

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

def wait_for_run(run_id: str, max_timeout: int = 30):
    start = time.time()
    while time.time() - start < max_timeout:
        res = client.get(f"/api/runs/{run_id}/status")
        assert res.status_code == 200
        st = res.json()["data"]
        if st["status"] in ["completed", "failed", "cancelled"]:
            return st
        time.sleep(0.5)
    raise TimeoutError(f"Run {run_id} did not complete within {max_timeout}s")

# -------------------------------------------------------------
# Phase 4 Benchmark 1: Numerical Binary Classification
# -------------------------------------------------------------
def test_benchmark_dataset_a_binary_classification():
    np.random.seed(101)
    n = 80
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    # Signal: y correlates with x1 + x2
    prob = 1 / (1 + np.exp(-(1.5 * x1 + 1.2 * x2)))
    y = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "category_tag": np.random.choice(["type_A", "type_B"], size=n),
        "target_class": y
    })
    ds_id = upload_df(df, filename="bench_a.csv")

    config = {
        "dataset_id": ds_id,
        "target": "target_class",
        "problem_type": "binary_classification",
        "training_mode": "quick",
        "time_limit": 10,
        "primary_metric": "accuracy",
        "random_state": 42,
        "engine_preference": "sklearn_gbdt"
    }

    res = client.post("/api/runs", json=config)
    assert res.status_code == 201
    run_id = res.json()["data"]["run_id"]

    final_status = wait_for_run(run_id)
    assert final_status["status"] == "completed"

    detail_res = client.get(f"/api/runs/{run_id}")
    assert detail_res.status_code == 200
    data = detail_res.json()["data"]
    
    assert data["summary"]["winning_model"] != "None"
    assert data["metrics"]["primary_metric"] == "accuracy"
    assert data["metrics"]["metric_direction"] == "higher"
    assert len(data["leaderboard"]) >= 4

# -------------------------------------------------------------
# Phase 4 Benchmark 2: Continuous Target Regression
# -------------------------------------------------------------
def test_benchmark_dataset_b_regression():
    np.random.seed(202)
    n = 80
    sqft = np.random.uniform(500, 3500, size=n)
    bedrooms = np.random.randint(1, 6, size=n)
    # Linear signal + noise
    price = sqft * 150.0 + bedrooms * 20000.0 + np.random.normal(0, 5000, size=n)

    df = pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "neighborhood": np.random.choice(["North", "South", "East", "West"], size=n),
        "price": price
    })
    ds_id = upload_df(df, filename="bench_b_regression.csv")

    config = {
        "dataset_id": ds_id,
        "target": "price",
        "problem_type": "regression",
        "training_mode": "quick",
        "time_limit": 10,
        "primary_metric": "rmse",
        "random_state": 42,
        "engine_preference": "sklearn_gbdt"
    }

    res = client.post("/api/runs", json=config)
    assert res.status_code == 201
    run_id = res.json()["data"]["run_id"]

    final_status = wait_for_run(run_id)
    assert final_status["status"] == "completed"

    detail_res = client.get(f"/api/runs/{run_id}")
    assert detail_res.status_code == 200
    data = detail_res.json()["data"]

    # In regression, metric direction must be 'lower' for RMSE
    assert data["metrics"]["metric_direction"] == "lower"
    winner = [m for m in data["leaderboard"] if m["is_winner"]][0]
    assert winner["test_primary_score"] is not None
    assert "r2" in winner["test_metrics"]
    assert "mae" in winner["test_metrics"]

# -------------------------------------------------------------
# Phase 4 Benchmark 3: Multiclass Classification
# -------------------------------------------------------------
def test_benchmark_dataset_c_multiclass():
    np.random.seed(303)
    n = 90
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    labels = ["Low", "Medium", "High"]
    target = np.random.choice(labels, size=n, p=[0.33, 0.34, 0.33])

    df = pd.DataFrame({
        "metric_x": f1,
        "metric_y": f2,
        "tier": target
    })
    ds_id = upload_df(df, filename="bench_c_multiclass.csv")

    config = {
        "dataset_id": ds_id,
        "target": "tier",
        "problem_type": "multiclass_classification",
        "training_mode": "quick",
        "time_limit": 10,
        "primary_metric": "f1_macro",
        "random_state": 42,
        "engine_preference": "sklearn_gbdt"
    }

    res = client.post("/api/runs", json=config)
    assert res.status_code == 201
    run_id = res.json()["data"]["run_id"]

    final_status = wait_for_run(run_id)
    assert final_status["status"] == "completed"

    detail_res = client.get(f"/api/runs/{run_id}")
    data = detail_res.json()["data"]
    assert data["metrics"]["primary_metric"] == "f1_macro"
    assert len(data["leaderboard"]) >= 3
