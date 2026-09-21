import io
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def upload_df(df: pd.DataFrame, filename: str = "dataset.csv") -> str:
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    
    res = client.post(
        "/api/datasets/upload",
        files={"file": (filename, csv_bytes, "text/csv")}
    )
    assert res.status_code == 201, f"Upload failed: {res.text}"
    return res.json()["dataset_id"]

# =========================================================================
# Dataset A: Small Numerical Dataset
# =========================================================================
def test_dataset_a_small_numerical():
    np.random.seed(42)
    n = 100
    df_a = pd.DataFrame({
        "feature_1": np.random.normal(50, 10, n),
        "feature_2": np.random.uniform(0, 100, n),
        "feature_3": np.random.exponential(5, n),
        "target_num": np.random.normal(100, 20, n)
    })
    
    ds_id = upload_df(df_a, "dataset_a_numerical.csv")
    
    # 1. Overview
    res = client.get(f"/api/datasets/{ds_id}/eda/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["rows"] == 100
    assert data["columns"] == 4
    assert data["numeric_columns"] == 4
    assert data["missing_cells"] == 0

    # 2. Numerical stats
    res = client.get(f"/api/datasets/{ds_id}/eda/numerical")
    assert res.status_code == 200
    stats = res.json()
    assert len(stats) == 4
    f1 = next(s for s in stats if s["column"] == "feature_1")
    assert 45 < f1["mean"] < 55
    assert f1["missing_count"] == 0

    # 3. Correlation
    res = client.get(f"/api/datasets/{ds_id}/eda/correlation")
    assert res.status_code == 200
    corr = res.json()
    assert len(corr["columns"]) == 4
    assert "Correlation measures statistical linear association" in corr["disclaimer"]

    # 4. Outliers
    res = client.get(f"/api/datasets/{ds_id}/eda/outliers")
    assert res.status_code == 200
    assert len(res.json()) == 4

    # 5. Bivariate Chart (Scatter)
    res = client.post(f"/api/datasets/{ds_id}/eda/chart", json={"x": "feature_1", "y": "feature_2"})
    assert res.status_code == 200
    chart = res.json()
    assert chart["chart_type"] == "scatter"
    assert len(chart["data"]) >= 1

    # 6. Target-aware EDA
    res = client.get(f"/api/datasets/{ds_id}/eda/target-analysis?target=target_num")
    assert res.status_code == 200
    target_data = res.json()
    assert target_data["target_dtype"] == "float"
    assert target_data["profile"]["type"] == "numeric"


# =========================================================================
# Dataset B: Mixed Numerical + Categorical Dataset
# =========================================================================
def test_dataset_b_mixed_types():
    df_b = pd.DataFrame({
        "age": [25, 45, 32, 60, 22, 54, 38, 41, 29, 50],
        "salary": [45000, 95000, 62000, 120000, 38000, 110000, 78000, 82000, 52000, 98000],
        "department": ["Engineering", "HR", "Engineering", "Exec", "Marketing", "Exec", "Engineering", "Marketing", "HR", "Engineering"],
        "is_manager": [False, True, False, True, False, True, False, False, False, True]
    })
    
    ds_id = upload_df(df_b, "dataset_b_mixed.csv")
    
    # 1. Overview
    res = client.get(f"/api/datasets/{ds_id}/eda/overview")
    assert res.status_code == 200
    ov = res.json()
    assert ov["numeric_columns"] == 2
    assert ov["categorical_columns"] >= 1

    # 2. Categorical Stats
    res = client.get(f"/api/datasets/{ds_id}/eda/categorical")
    assert res.status_code == 200
    cats = res.json()
    dept = next(c for c in cats if c["column"] == "department")
    assert dept["unique_values"] == 4
    assert dept["mode"] == "Engineering"

    # 3. Categorical + Numeric Bivariate (Grouped Box Plot)
    res = client.post(f"/api/datasets/{ds_id}/eda/chart", json={"x": "department", "y": "salary"})
    assert res.status_code == 200
    box_chart = res.json()
    assert box_chart["chart_type"] == "box"
    assert len(box_chart["data"]) >= 1

    # 4. Target-aware Categorical Target
    res = client.get(f"/api/datasets/{ds_id}/eda/target-analysis?target=is_manager")
    assert res.status_code == 200
    target_data = res.json()
    assert target_data["profile"]["type"] == "categorical"


# =========================================================================
# Dataset C: Missing Values, Duplicates, Constant, High-Cardinality & Outliers
# =========================================================================
def test_dataset_c_stress_dataset():
    n = 100
    df_c = pd.DataFrame({
        "all_missing": [None] * n,
        "half_missing": [10.0 if i % 2 == 0 else None for i in range(n)],
        "constant_val": ["CONSTANT"] * n,
        "high_card_id": [f"user_{i}" for i in range(n)],
        "outlier_col": [20.0] * (n - 2) + [9999.0, -9999.0], # extreme outliers
        "dup_col": [i % 10 for i in range(n)] # induces duplicates
    })
    
    ds_id = upload_df(df_c, "dataset_c_stress.csv")
    
    # 1. Overview flags
    res = client.get(f"/api/datasets/{ds_id}/eda/overview")
    assert res.status_code == 200
    ov = res.json()
    assert "constant_val" in ov["constant_columns"]
    assert "high_card_id" in ov["high_cardinality_columns"]
    assert ov["missing_percentage"] > 20.0

    # 2. Missing Analysis & Severe Missingness Flag
    res = client.get(f"/api/datasets/{ds_id}/eda/missing")
    assert res.status_code == 200
    miss = res.json()
    assert "all_missing" in miss["severe_missing_columns"]
    assert "half_missing" in miss["severe_missing_columns"]

    # 3. Outlier Detection on extreme values
    res = client.get(f"/api/datasets/{ds_id}/eda/outliers")
    assert res.status_code == 200
    outliers = res.json()
    outlier_stat = next(o for o in outliers if o["column"] == "outlier_col")
    assert outlier_stat["has_outliers"] is True
    assert outlier_stat["potential_outliers_count"] >= 2
    assert outlier_stat["upper_outliers_count"] >= 1
    assert outlier_stat["lower_outliers_count"] >= 1


# =========================================================================
# Dataset D: Dataset with Datetime Column
# =========================================================================
def test_dataset_d_datetime_analysis():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df_d = pd.DataFrame({
        "timestamp": dates.astype(str),
        "temperature": np.sin(np.linspace(0, 3 * np.pi, 60)) * 15 + 20,
        "sensor_id": ["S1"] * 30 + ["S2"] * 30
    })
    
    ds_id = upload_df(df_d, "dataset_d_temporal.csv")
    
    # 1. Overview detects datetime
    res = client.get(f"/api/datasets/{ds_id}/eda/overview")
    assert res.status_code == 200
    ov = res.json()
    assert ov["datetime_columns"] == 1
    assert "timestamp" in ov["column_classification"]["datetime"]

    # 2. Column Distribution for datetime
    res = client.get(f"/api/datasets/{ds_id}/eda/distribution/timestamp")
    assert res.status_code == 200
    dist = res.json()
    assert dist["dtype"] == "datetime-like"
    assert dist["temporal"]["date_range_days"] == 59.0
    assert "2023-01-01" in dist["temporal"]["min_date"]

    # 3. Datetime + Numeric Line Chart
    res = client.post(f"/api/datasets/{ds_id}/eda/chart", json={"x": "timestamp", "y": "temperature"})
    assert res.status_code == 200
    line_chart = res.json()
    assert line_chart["chart_type"] == "line"
    assert len(line_chart["data"]) == 1
    assert line_chart["data"][0]["mode"] == "lines+markers"
