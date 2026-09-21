import io
import os
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing.target_validator import validate_and_inspect_target
from app.preprocessing.feature_selector import audit_features_for_selection, verify_zero_target_leakage
from app.preprocessing.datetime_transformer import DatetimeComponentExtractor
from app.preprocessing.split_service import perform_train_test_split
from app.preprocessing.pipeline_builder import build_preprocessing_pipeline
from app.preprocessing.preprocessing_service import preprocessing_service

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


def test_dataset_a_numerical_classification():
    """Dataset A: Pure numerical features for binary classification."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feat1": np.random.normal(10, 2, n),
        "feat2": np.random.uniform(0, 100, n),
        "feat3": np.random.exponential(1.5, n),
        "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })

    # 1. Target detection
    target_info = validate_and_inspect_target(df, "target")
    assert target_info["problem_type"] == "binary_classification"
    assert target_info["diagnostics"]["classification"]["classes_count"] == 2
    assert target_info["diagnostics"]["classification"]["minority_class_percentage"] > 0

    # 2. Feature selection
    audit = audit_features_for_selection(df, "target")
    assert len(audit["features"]) == 3
    selected = [f["name"] for f in audit["features"]]
    verify_zero_target_leakage(selected, "target")

    # 3. Pipeline execution via service
    ds_id = upload_df(df, "dataset_a.csv")
    res = preprocessing_service.prepare_dataset(
        dataset_id=ds_id,
        target="target",
        problem_type="binary_classification",
        selected_features=selected,
        test_size=0.2,
        random_state=42,
        scaling="standard"
    )
    assert res["status"] == "prepared"
    assert res["metadata"]["split"]["train_rows"] == 160
    assert res["metadata"]["split"]["test_rows"] == 40
    assert res["metadata"]["final_features_count"] == 3

    # Check serialized artifact exists
    prep_dir = preprocessing_service.get_preprocessing_dir(ds_id)
    assert (prep_dir / "pipeline.joblib").exists()
    assert (prep_dir / "config.json").exists()
    assert (prep_dir / "metadata.json").exists()


def test_dataset_b_mixed_classification():
    """Dataset B: Mixed categorical and numerical features for classification."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "income": np.random.normal(50000, 15000, n),
        "department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], n),
        "tier": np.random.choice(["Bronze", "Silver", "Gold"], n),
        "churn": np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])
    })

    target_info = validate_and_inspect_target(df, "churn")
    assert target_info["problem_type"] == "binary_classification"

    selected = ["age", "income", "department", "tier"]
    ds_id = upload_df(df, "dataset_b.csv")
    res = preprocessing_service.prepare_dataset(
        dataset_id=ds_id,
        target="churn",
        problem_type="binary_classification",
        selected_features=selected,
        test_size=0.2,
        random_state=42,
        categorical_encoding="one_hot",
        scaling="standard"
    )
    assert res["status"] == "prepared"
    assert res["metadata"]["final_features_count"] >= 7
    assert any("department_" in col for col in res["metadata"]["feature_names_out"])
    assert any("tier_" in col for col in res["metadata"]["feature_names_out"])


def test_dataset_c_regression():
    """Dataset C: Continuous target regression with skewness and outliers."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "sqft": np.random.uniform(500, 4000, n),
        "bedrooms": np.random.choice([1, 2, 3, 4, 5], n),
        "distance_to_center": np.random.exponential(5, n),
        "price": np.random.normal(300000, 75000, n)
    })

    target_info = validate_and_inspect_target(df, "price")
    assert target_info["problem_type"] == "regression"
    assert "regression" in target_info["diagnostics"]
    assert target_info["diagnostics"]["regression"]["mean"] > 0
    assert "q1" in target_info["diagnostics"]["regression"]
    assert "q3" in target_info["diagnostics"]["regression"]

    selected = ["sqft", "bedrooms", "distance_to_center"]
    ds_id = upload_df(df, "dataset_c.csv")
    res = preprocessing_service.prepare_dataset(
        dataset_id=ds_id,
        target="price",
        problem_type="regression",
        selected_features=selected,
        test_size=0.25,
        random_state=42,
        scaling="standard"
    )
    assert res["status"] == "prepared"
    assert res["metadata"]["split"]["train_rows"] == 90
    assert res["metadata"]["split"]["test_rows"] == 30
    assert "Shuffle Split" in res["metadata"]["split"]["split_strategy"]


def test_dataset_d_missing_and_unseen_categories():
    """Dataset D: Missing values, infinite numbers, and unseen categories in test set."""
    train_df = pd.DataFrame({
        "num_feat": [10.0, np.nan, 20.0, np.inf, 30.0, -np.inf, 40.0, 50.0],
        "cat_feat": ["Alpha", "Alpha", "Beta", "Beta", "Alpha", None, "Beta", "Alpha"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })
    test_df = pd.DataFrame({
        "num_feat": [np.nan, 100.0],
        "cat_feat": ["Gamma", "Alpha"],  # 'Gamma' is unseen
        "target": [0, 1]
    })
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    selected = ["num_feat", "cat_feat"]
    ds_id = upload_df(combined_df, "dataset_d.csv")
    res = preprocessing_service.prepare_dataset(
        dataset_id=ds_id,
        target="target",
        problem_type="binary_classification",
        selected_features=selected,
        test_size=0.2,
        random_state=42,
        numeric_imputation="median",
        categorical_imputation="most_frequent",
        categorical_encoding="one_hot",
        scaling="standard"
    )
    assert res["status"] == "prepared"

    # Load saved pipeline and pass new unseen test data
    prep_dir = preprocessing_service.get_preprocessing_dir(ds_id)
    pipeline_artifact = joblib.load(prep_dir / "pipeline.joblib")
    column_transformer = pipeline_artifact["column_transformer"]
    unseen_test = pd.DataFrame({
        "num_feat": [np.inf, -np.inf, 25.0],
        "cat_feat": ["UnknownDelta", "UnseenEpsilon", "Alpha"]
    })
    # Transform must not raise ValueError for unknown categories
    transformed = column_transformer.transform(unseen_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()


def test_dataset_e_datetime_categorical_numerical():
    """Dataset E: Datetime feature extraction combined with categorical and numerical features."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d %H:%M:%S").tolist()
    df = pd.DataFrame({
        "timestamp": dates,
        "region": ["North", "South", "East", "West"] * 25,
        "score": np.random.uniform(50, 100, n),
        "success": [0, 1] * 50
    })

    selected = ["timestamp", "region", "score"]
    ds_id = upload_df(df, "dataset_e.csv")
    res = preprocessing_service.prepare_dataset(
        dataset_id=ds_id,
        target="success",
        problem_type="binary_classification",
        selected_features=selected,
        test_size=0.2,
        random_state=42,
        extract_datetime=True
    )
    assert res["status"] == "prepared"
    metadata = res["metadata"]
    assert metadata["generated_datetime_features_count"] == 4  # year, month, day, dayofweek
    feature_names = metadata["feature_names_out"]
    assert any("timestamp_year" in f for f in feature_names)
    assert any("timestamp_month" in f for f in feature_names)
    assert any("region_" in f for f in feature_names)
    assert any("score" in f for f in feature_names)
