import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.analysis.statistics import (
    compute_dataset_overview,
    compute_numerical_column_stats,
    compute_categorical_column_stats,
    compute_missing_analysis,
    compute_temporal_column_stats
)
from app.analysis.correlations import (
    compute_correlation_matrix,
    extract_strong_correlations
)
from app.analysis.outliers import (
    detect_column_outliers_iqr,
    detect_dataset_outliers
)
from app.analysis.chart_recommender import (
    recommend_bivariate_chart,
    build_missingness_chart,
    build_univariate_numeric_charts,
    build_univariate_categorical_chart
)
from app.storage.dataset_store import dataset_store
from app.ingestion.repair_log import IngestionReport

client = TestClient(app)

# ---------------------------------------------------------
# Unit Tests: Numerical Statistics & Edge Cases
# ---------------------------------------------------------

def test_numerical_stats_standard():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = compute_numerical_column_stats(series, "metric")
    assert stats["count"] == 5
    assert stats["valid_count"] == 5
    assert stats["missing_count"] == 0
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["std"] > 0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["q1"] == 20.0
    assert stats["q3"] == 40.0
    assert stats["iqr"] == 20.0

def test_numerical_stats_with_nan_and_inf():
    # Includes NaN, +inf, -inf
    series = pd.Series([10.0, np.nan, np.inf, -np.inf, 20.0, 30.0])
    stats = compute_numerical_column_stats(series, "messy")
    assert stats["count"] == 6
    assert stats["valid_count"] == 3  # Only 10, 20, 30 valid
    assert stats["missing_count"] == 3
    assert stats["mean"] == 20.0
    assert stats["median"] == 20.0
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0

def test_numerical_stats_all_nan():
    series = pd.Series([np.nan, np.nan, np.nan])
    stats = compute_numerical_column_stats(series, "empty_col")
    assert stats["count"] == 3
    assert stats["valid_count"] == 0
    assert stats["mean"] is None
    assert stats["std"] is None

def test_numerical_stats_constant_column():
    series = pd.Series([42.0, 42.0, 42.0, 42.0])
    stats = compute_numerical_column_stats(series, "constant")
    assert stats["mean"] == 42.0
    assert stats["std"] == 0.0
    assert stats["min"] == 42.0
    assert stats["max"] == 42.0
    assert stats["iqr"] == 0.0

def test_numerical_stats_single_row():
    series = pd.Series([99.0])
    stats = compute_numerical_column_stats(series, "single")
    assert stats["count"] == 1
    assert stats["valid_count"] == 1
    assert stats["mean"] == 99.0
    assert stats["std"] == 0.0

# ---------------------------------------------------------
# Unit Tests: Categorical Analysis
# ---------------------------------------------------------

def test_categorical_stats_standard():
    series = pd.Series(["apple", "banana", "apple", "cherry", "apple", "banana", None])
    stats = compute_categorical_column_stats(series, "fruit", top_n=2)
    assert stats["count"] == 7
    assert stats["valid_count"] == 6
    assert stats["missing_count"] == 1
    assert stats["unique_values"] == 3
    assert stats["mode"] == "apple"
    assert stats["mode_count"] == 3
    assert len(stats["top_categories"]) == 2
    assert stats["top_categories"][0]["category"] == "apple"
    assert stats["other_count"] == 1  # cherry fell into 'other'

def test_categorical_stats_all_missing():
    series = pd.Series([None, None, None])
    stats = compute_categorical_column_stats(series, "empty")
    assert stats["valid_count"] == 0
    assert stats["unique_values"] == 0
    assert stats["mode"] is None

# ---------------------------------------------------------
# Unit Tests: Missing Value Analysis
# ---------------------------------------------------------

def test_missing_analysis():
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": [None, None, None, 1], # 75% missing -> severe
        "c": [10, 20, 30, 40]
    })
    missing = compute_missing_analysis(df)
    assert missing["total_cells"] == 12
    assert missing["total_missing_cells"] == 4
    assert missing["columns_with_missing_count"] == 2
    assert "b" in missing["severe_missing_columns"]
    assert "c" not in missing["columns_with_missing"]

# ---------------------------------------------------------
# Unit Tests: Correlations
# ---------------------------------------------------------

def test_correlation_matrix_and_strong_pairs():
    x = np.linspace(0, 10, 50)
    df = pd.DataFrame({
        "x": x,
        "y_pos": 2 * x + np.random.normal(0, 0.1, 50), # strong pos corr ~ 1.0
        "y_neg": -3 * x + np.random.normal(0, 0.1, 50), # strong neg corr ~ -1.0
        "noise": np.random.normal(0, 5, 50)
    })
    res = compute_correlation_matrix(df)
    assert len(res["columns"]) == 4
    assert len(res["matrix"]) == 4
    assert len(res["strong_correlations"]) >= 2
    
    pairs = [(p["feature_a"], p["feature_b"]) for p in res["strong_correlations"]]
    has_x_pos = ("x", "y_pos") in pairs or ("y_pos", "x") in pairs
    assert has_x_pos

def test_correlation_constant_column():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "const": [5, 5, 5, 5]
    })
    res = compute_correlation_matrix(df)
    assert len(res["matrix"]) == 2
    # Constant correlation should safely be 0.0, not crash
    assert res["matrix"][0][1] == 0.0

# ---------------------------------------------------------
# Unit Tests: Outlier Detection (IQR)
# ---------------------------------------------------------

def test_iqr_outliers_detection():
    # 10 normal values [10..19] + 1 extreme outlier [1000]
    data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1000]
    series = pd.Series(data)
    res = detect_column_outliers_iqr(series, "values")
    assert res["has_outliers"] is True
    assert res["potential_outliers_count"] == 1
    assert res["upper_outliers_count"] == 1
    assert res["lower_outliers_count"] == 0

# ---------------------------------------------------------
# Unit Tests: Rule-Based Chart Recommendation
# ---------------------------------------------------------

def test_chart_recommender_rules():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "num2": [10.0, 20.0, 15.0, 25.0, 30.0],
        "cat1": ["A", "B", "A", "B", "C"],
        "cat2": ["X", "Y", "X", "Y", "Z"],
        "dt": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
    })

    # Numeric + Numeric -> Scatter
    scatter = recommend_bivariate_chart(df, "num1", "num2")
    assert scatter["chart_type"] == "scatter"
    assert len(scatter["data"]) >= 1

    # Categorical + Numeric -> Box
    box = recommend_bivariate_chart(df, "cat1", "num1")
    assert box["chart_type"] == "box"

    # Categorical + Categorical -> Heatmap
    heatmap = recommend_bivariate_chart(df, "cat1", "cat2")
    assert heatmap["chart_type"] == "heatmap"

    # Single Numeric -> Histogram
    hist = recommend_bivariate_chart(df, "num1")
    assert hist["chart_type"] == "histogram"

    # Single Categorical -> Bar
    bar = recommend_bivariate_chart(df, "cat1")
    assert bar["chart_type"] == "bar"

    # Datetime + Numeric -> Line
    line = recommend_bivariate_chart(df, "dt", "num1")
    assert line["chart_type"] == "line"

# ---------------------------------------------------------
# Integration Tests: FastAPI EDA Endpoints
# ---------------------------------------------------------

@pytest.fixture
def sample_dataset_id():
    csv_bytes = b"age,income,gender,signup_date\n25,50000,M,2023-01-01\n30,60000,F,2023-01-02\n35,75000,F,2023-01-03\n40,80000,M,2023-01-04\n120,500000,Other,2023-01-05\n"
    res = client.post(
        "/api/datasets/upload",
        files={"file": ("users.csv", csv_bytes, "text/csv")}
    )
    assert res.status_code == 201
    return res.json()["dataset_id"]

def test_api_eda_overview(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["rows"] == 5
    assert data["columns"] == 4
    assert data["numeric_columns"] >= 2
    assert "income" in data["column_classification"]["numeric"]

def test_api_eda_numerical(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/numerical")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 2
    cols = [c["column"] for c in data]
    assert "age" in cols
    assert "income" in cols

def test_api_eda_categorical(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/categorical")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 1
    assert data[0]["column"] == "gender"

def test_api_eda_missing(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/missing")
    assert res.status_code == 200
    data = res.json()
    assert data["total_cells"] == 20
    assert "chart" in data
    assert data["chart"]["chart_type"] == "bar"

def test_api_eda_correlation(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/correlation")
    assert res.status_code == 200
    data = res.json()
    assert "matrix" in data
    assert "heatmap" in data
    assert "Correlation measures statistical linear association" in data["disclaimer"]

def test_api_eda_outliers(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/outliers")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 2
    # age = 120 and income = 500000 should flag potential outliers
    age_outliers = next(o for o in data if o["column"] == "age")
    assert "Potential outliers are statistical anomalies" in age_outliers["disclaimer"]

def test_api_eda_distribution(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/distribution/age")
    assert res.status_code == 200
    data = res.json()
    assert data["column"] == "age"
    assert "histogram" in data
    assert "box_plot" in data

def test_api_eda_chart_post(sample_dataset_id):
    req_body = {"x": "age", "y": "income"}
    res = client.post(f"/api/datasets/{sample_dataset_id}/eda/chart", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert data["chart_type"] == "scatter"
    assert len(data["data"]) >= 1

def test_api_eda_target_analysis(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/target-analysis?target=income")
    assert res.status_code == 200
    data = res.json()
    assert data["target_column"] == "income"
    assert data["profile"]["type"] == "numeric"
    assert len(data["profile"]["top_associated_features"]) >= 1

def test_api_eda_column_not_found(sample_dataset_id):
    res = client.get(f"/api/datasets/{sample_dataset_id}/eda/distribution/non_existent_column")
    assert res.status_code == 404
    data = res.json()
    assert data["error"] == "COLUMN_NOT_FOUND"

def test_api_eda_dataset_not_found():
    res = client.get("/api/datasets/ds_non_existent_123/eda/overview")
    assert res.status_code == 404
    data = res.json()
    assert data["error"] == "DATASET_NOT_FOUND"
