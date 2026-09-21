import io
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib
from fastapi.testclient import TestClient

from app.main import app
from app.core.errors import InvalidTargetError, TargetLeakageError
from app.preprocessing.target_validator import validate_and_inspect_target
from app.preprocessing.feature_selector import audit_features_for_selection, verify_zero_target_leakage
from app.preprocessing.datetime_transformer import DatetimeComponentExtractor
from app.preprocessing.split_service import perform_train_test_split
from app.preprocessing.pipeline_builder import build_preprocessing_pipeline

client = TestClient(app)

# Helper function to upload DataFrame
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

# ---------------------------------------------------------
# Target Validation & Problem Type Detection Tests
# ---------------------------------------------------------

def test_target_binary_classification():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
    })
    info = validate_and_inspect_target(df, "target")
    assert info["problem_type"] == "binary_classification"
    assert info["confidence"] == "high"
    assert info["diagnostics"]["classification"]["classes_count"] == 2

def test_target_multiclass_classification():
    df = pd.DataFrame({
        "feature": range(15),
        "category": ["A", "B", "C"] * 5
    })
    info = validate_and_inspect_target(df, "category")
    assert info["problem_type"] == "multiclass_classification"
    assert info["diagnostics"]["classification"]["classes_count"] == 3

def test_target_continuous_regression():
    df = pd.DataFrame({
        "feature": range(35),
        "target_val": [i * 2.5 + 1.2 for i in range(35)]
    })
    info = validate_and_inspect_target(df, "target_val")
    assert info["problem_type"] == "regression"
    assert "mean" in info["diagnostics"]["regression"]

def test_target_ambiguous_ratings():
    # 5 discrete ratings: 1, 2, 3, 4, 5
    df = pd.DataFrame({
        "feature": range(20),
        "rating": [1, 2, 3, 4, 5] * 4
    })
    info = validate_and_inspect_target(df, "rating")
    assert info["problem_type"] == "ambiguous"

def test_target_constant_raises_error():
    df = pd.DataFrame({
        "feature": range(15),
        "constant_target": [1] * 15
    })
    with pytest.raises(InvalidTargetError, match="constant"):
        validate_and_inspect_target(df, "constant_target")

def test_target_insufficient_rows_raises_error():
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0]
    })
    with pytest.raises(InvalidTargetError, match="observations"):
        validate_and_inspect_target(df, "target")

# ---------------------------------------------------------
# Feature Selection & Target Leakage Prevention Tests
# ---------------------------------------------------------

def test_feature_audit_flags():
    n = 100
    df = pd.DataFrame({
        "target": [0, 1] * 50,
        "normal_feat": np.random.normal(0, 1, n),
        "const_col": ["CONST"] * n,
        "id_col": [f"user_{i}" for i in range(n)],
        "missing_col": [1.0 if i < 15 else None for i in range(n)], # 85% missing
        "target_copy": [0, 1] * 50 # exact duplicate of target
    })
    audit = audit_features_for_selection(df, "target")
    assert audit["target"] == "target"
    assert "target" not in [f["name"] for f in audit["features"]]

    feat_map = {f["name"]: f for f in audit["features"]}
    assert "constant_column" in feat_map["const_col"]["flags"]
    assert "potential_id" in feat_map["id_col"]["flags"]
    assert "excessive_missingness" in feat_map["missing_col"]["flags"]
    assert "duplicate_feature" in feat_map["target_copy"]["flags"]

def test_leakage_barrier_raises_error():
    with pytest.raises(TargetLeakageError):
        verify_zero_target_leakage(["age", "income", "target"], "target")

# ---------------------------------------------------------
# Train/Test Split & Index Disjointness Tests
# ---------------------------------------------------------

def test_train_test_split_disjoint_and_stratified():
    n = 100
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.uniform(0, 10, n),
        "y": [0] * 80 + [1] * 20 # 80/20 class balance
    })
    X_train, X_test, y_train, y_test, meta = perform_train_test_split(
        df=df,
        target_col="y",
        feature_cols=["x1", "x2"],
        test_size=0.2,
        random_state=42,
        problem_type="binary_classification"
    )
    # Check disjoint indices
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert meta["stratified"] is True
    # Verify stratification preserved 80/20 ratio in test set (4 ones, 16 zeros)
    assert y_test.value_counts()[1] == 4
    assert y_test.value_counts()[0] == 16

# ---------------------------------------------------------
# MANDATORY LEAKAGE TEST (Section 43)
# ---------------------------------------------------------

def test_zero_leakage_in_preprocessing_fitting():
    """
    CRITICAL PROOF: Test data statistics must NOT leak into fitted transformers.
    Training data has mean = 20, median = 20.
    Test data has mean = 2000, median = 2000.
    Fitted scaler mean must equal 20.0, NOT contaminated by the test set!
    """
    # 1. Construct distinct splits
    train_df = pd.DataFrame({
        "numeric": [10.0, 20.0, 30.0],
        "category": ["A", "B", "A"]
    })
    test_df = pd.DataFrame({
        "numeric": [1000.0, 2000.0, 3000.0],
        "category": ["A", "C", "A"] # C is unseen in training
    })

    # 2. Build pipeline
    preprocessor = build_preprocessing_pipeline(
        numeric_features=["numeric"],
        categorical_features=["category"],
        numeric_imputation="median",
        scaling="standard"
    )

    # 3. Fit STRICTLY on training data
    preprocessor.fit(train_df)

    # 4. Extract fitted statistics from scaler and imputer
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]

    fitted_mean = float(scaler.mean_[0])
    fitted_median = float(imputer.statistics_[0])

    # STRICT ASSERTIONS: Must equal training statistics exactly
    assert fitted_mean == 20.0, f"Leakage violation! Expected 20.0, got {fitted_mean}"
    assert fitted_median == 20.0, f"Leakage violation! Expected 20.0, got {fitted_median}"

    # 5. Transform test set: Unseen category 'C' must not crash OneHotEncoder
    test_transformed = preprocessor.transform(test_df)
    assert test_transformed.shape[0] == 3
    # Unseen category 'C' was safely ignored (all 0s in one-hot)

# ---------------------------------------------------------
# Datetime Component Extraction Tests
# ---------------------------------------------------------

def test_datetime_component_extraction():
    df = pd.DataFrame({
        "signup": ["2023-01-15", "2023-06-20", "2023-12-31"],
        "other": [1, 2, 3]
    })
    extractor = DatetimeComponentExtractor(datetime_columns=["signup"])
    extractor.fit(df)
    transformed = extractor.transform(df)

    assert "signup" not in transformed.columns
    assert "signup_year" in transformed.columns
    assert "signup_month" in transformed.columns
    assert "signup_day" in transformed.columns
    assert "signup_dayofweek" in transformed.columns
    assert transformed["signup_year"].tolist() == [2023, 2023, 2023]
    assert transformed["signup_month"].tolist() == [1, 6, 12]

# ---------------------------------------------------------
# Infinite Value Handling & Serialization Tests
# ---------------------------------------------------------

def test_infinite_values_and_joblib_serialization(tmp_path):
    df_train = pd.DataFrame({
        "num": [10.0, np.inf, -np.inf, 40.0],
        "cat": ["X", "Y", "X", "Y"]
    })
    preprocessor = build_preprocessing_pipeline(
        numeric_features=["num"],
        categorical_features=["cat"],
        numeric_imputation="median",
        scaling="standard"
    )
    preprocessor.fit(df_train)

    # Transform without errors despite infinite values
    res = preprocessor.transform(df_train)
    assert res.shape == (4, 3) # 1 scaled num + 2 one-hot cats

    # Test serialization via joblib
    save_file = tmp_path / "pipeline.joblib"
    joblib.dump(preprocessor, save_file)
    assert save_file.is_file()

    loaded_pipeline = joblib.load(save_file)
    res_loaded = loaded_pipeline.transform(df_train)
    np.testing.assert_array_almost_equal(res, res_loaded)

# ---------------------------------------------------------
# Full API Integration Tests
# ---------------------------------------------------------

@pytest.fixture
def uploaded_dataset_id():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        "income": [40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "churn": [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    })
    return upload_df(df, "churn_data.csv")

def test_api_target_inspect(uploaded_dataset_id):
    res = client.post(
        f"/api/datasets/{uploaded_dataset_id}/preprocessing/target-inspect",
        json={"target": "churn"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["target"] == "churn"
    assert data["problem_type"] == "binary_classification"

def test_api_features_inspect(uploaded_dataset_id):
    res = client.post(
        f"/api/datasets/{uploaded_dataset_id}/preprocessing/features-inspect",
        json={"target": "churn"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["target"] == "churn"
    assert len(data["features"]) == 3
    assert "churn" not in [f["name"] for f in data["features"]]

def test_api_prepare_dataset(uploaded_dataset_id):
    req_body = {
        "target": "churn",
        "problem_type": "binary_classification",
        "selected_features": ["age", "income", "gender"],
        "test_size": 0.25,
        "random_state": 42,
        "numeric_imputation": "median",
        "scaling": "standard"
    }
    res = client.post(
        f"/api/datasets/{uploaded_dataset_id}/preprocessing/prepare",
        json=req_body
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "prepared"
    assert data["metadata"]["split"]["train_rows"] == 9
    assert data["metadata"]["split"]["test_rows"] == 3
    assert "preview" in data
    assert len(data["preview"]["preview_rows"]) > 0

def test_api_get_summary_and_preview(uploaded_dataset_id):
    # Ensure prepared
    client.post(
        f"/api/datasets/{uploaded_dataset_id}/preprocessing/prepare",
        json={
            "target": "churn",
            "problem_type": "binary_classification",
            "selected_features": ["age", "income"]
        }
    )
    
    # Test GET summary
    res_sum = client.get(f"/api/datasets/{uploaded_dataset_id}/preprocessing/summary")
    assert res_sum.status_code == 200
    sum_data = res_sum.json()
    assert sum_data["status"] == "prepared"
    assert sum_data["config"]["target"] == "churn"

    # Test GET preview
    res_prev = client.get(f"/api/datasets/{uploaded_dataset_id}/preprocessing/preview?limit=5")
    assert res_prev.status_code == 200
    prev_data = res_prev.json()
    assert prev_data["status"] == "prepared"
    assert len(prev_data["rows"]) <= 5
