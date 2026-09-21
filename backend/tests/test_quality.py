import pandas as pd
from app.validation.schema_inspector import inspect_dataset_structure
from app.analysis.quality_scorer import detect_suspicious_columns, calculate_quality_score

def test_suspicious_columns_detection():
    # 50 rows to trigger ratio thresholds
    df = pd.DataFrame({
        "user_id": [f"ID_{i}" for i in range(50)],
        "created_at": ["2024-01-01"] * 50,  # constant date
        "normal_val": list(range(50))
    })

    summary = inspect_dataset_structure(df)
    suspicious = detect_suspicious_columns(summary)

    issue_types = [s.issue_type for s in suspicious]
    assert "id_like" in issue_types
    assert "constant" in issue_types

def test_quality_score_perfect_dataset():
    df = pd.DataFrame({
        "feature_1": range(100),
        "feature_2": ["A", "B"] * 50,
        "target": [0, 1] * 50
    })
    summary = inspect_dataset_structure(df)
    suspicious = detect_suspicious_columns(summary)
    quality = calculate_quality_score(summary, suspicious)

    assert quality.quality_score >= 90
    assert quality.rating == "Excellent"

def test_quality_score_degraded_dataset():
    # Many missing values, duplicates, and constant columns
    df = pd.DataFrame({
        "const": [99] * 30,
        "missing_col": [None] * 25 + [1.0] * 5,
        "val": range(30)
    })
    # Add duplicates
    df = pd.concat([df, df.iloc[[0] * 10]], ignore_index=True)

    summary = inspect_dataset_structure(df)
    suspicious = detect_suspicious_columns(summary)
    quality = calculate_quality_score(summary, suspicious)

    assert quality.quality_score < 80
    assert quality.score_breakdown["constant_columns"] > 0
    assert quality.score_breakdown["missing_data"] > 0
