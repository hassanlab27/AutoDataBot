from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from app.core.config import settings
from app.storage.dataset_store import dataset_store

def check_dataset_suitability_for_training(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    selected_features: List[str]
) -> Dict[str, Any]:
    """
    Validates that a dataset and target are suitable for machine learning training.
    Returns:
        is_suitable: bool
        reasons: List[str] (blocking reasons if not suitable)
        warnings: List[str] (non-blocking diagnostic warnings)
    """
    reasons: List[str] = []
    warnings: List[str] = []

    # 1. Target Column Existence
    if target_col not in df.columns:
        reasons.append(f"Target column '{target_col}' not found in dataset.")
        return {"is_suitable": False, "reasons": reasons, "warnings": warnings}

    # 2. Features Availability & Non-Empty Matrix
    if not selected_features:
        reasons.append("No features selected for model training. At least one feature is required.")
    
    missing_feats = [f for f in selected_features if f not in df.columns]
    if missing_feats:
        reasons.append(f"Selected features not found in dataset: {missing_feats}")

    # 3. Direct Target Leakage Barrier
    if target_col in selected_features:
        reasons.append(f"CRITICAL TARGET LEAKAGE: Target '{target_col}' cannot be included inside the feature matrix.")

    # 4. Check for Potential Target Leakage by Feature Name
    for feat in selected_features:
        feat_lower = feat.lower()
        target_lower = target_col.lower()
        if target_lower in feat_lower and feat_lower != target_lower:
            warnings.append(
                f"Feature '{feat}' strongly resembles target '{target_col}'. Potential target leakage risk."
            )

    # 5. Non-Null Observations & Sample Size Checks
    y = df[target_col].dropna()
    valid_count = len(y)
    
    if valid_count < 10:
        reasons.append(f"Dataset has only {valid_count} valid target observations. Minimum required is 10.")
    elif valid_count < 100:
        warnings.append(
            f"Small dataset warning: Contains only {valid_count} rows. Model validation metrics may exhibit high variance."
        )

    # 6. Target Distribution & Cardinality Checks
    unique_count = y.nunique()
    if unique_count <= 1:
        reasons.append(f"Target '{target_col}' is constant ({unique_count} unique value). No predictive signal.")

    p_type = problem_type.lower().strip()
    if "classification" in p_type:
        if unique_count < 2:
            reasons.append(f"Classification problem requires at least 2 distinct classes. Found: {unique_count}.")
        
        # Check class counts
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if min_class_count < 2:
            warnings.append(
                f"Minority class '{class_counts.idxmin()}' has only {min_class_count} sample(s). "
                "Stratification may be limited."
            )
        
        # Check class imbalance
        imbalance_ratio = class_counts.max() / max(min_class_count, 1)
        if imbalance_ratio > 4.0:
            minority_pct = round((min_class_count / valid_count) * 100, 1)
            warnings.append(
                f"Class imbalance detected: minority class represents {minority_pct}% of samples "
                f"(imbalance ratio {imbalance_ratio:.1f}:1)."
            )

    elif p_type == "regression":
        clean_num = pd.to_numeric(y.replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
        if len(clean_num) < 10:
            reasons.append(f"Regression target '{target_col}' does not have enough numeric observations ({len(clean_num)}).")

    is_suitable = len(reasons) == 0
    return {
        "is_suitable": is_suitable,
        "reasons": reasons,
        "warnings": warnings
    }

def assess_training_suitability(
    dataset_id: str,
    target_col: str,
    expected_problem_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level suitability check loading dataset directly from dataset_store.
    Returns:
        suitable: bool
        errors: List[str]
        warnings: List[str]
    """
    try:
        df = dataset_store.load_dataset(dataset_id)
    except Exception as e:
        return {
            "suitable": False,
            "errors": [f"Dataset '{dataset_id}' does not exist or could not be loaded: {str(e)}"],
            "warnings": []
        }

    candidate_features = [c for c in df.columns if c != target_col]
    p_type = expected_problem_type or ("classification" if df[target_col].nunique() < 20 else "regression") if target_col in df.columns else "binary_classification"
    
    res = check_dataset_suitability_for_training(
        df=df,
        target_col=target_col,
        problem_type=p_type,
        selected_features=candidate_features
    )
    return {
        "suitable": res["is_suitable"],
        "errors": res["reasons"],
        "warnings": res["warnings"]
    }

def verify_preprocessing_artifacts_exist(dataset_id: str) -> bool:
    """Verifies that Phase 3 preprocessing artifacts exist for Path B models."""
    clean_id = "".join(c for c in dataset_id if c.isalnum() or c in "_-")
    prep_dir = settings.OUTPUTS_DIR / "datasets" / clean_id / "preprocessing"
    pipeline_file = prep_dir / "pipeline.joblib"
    config_file = prep_dir / "config.json"
    return pipeline_file.exists() and config_file.exists()
