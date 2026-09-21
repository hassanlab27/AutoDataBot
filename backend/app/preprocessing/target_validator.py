import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from app.core.errors import InvalidTargetError, ColumnNotFoundError
from app.validation.schema_inspector import infer_logical_dtype

def validate_and_inspect_target(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Validates the target column and determines the problem type conservatively.
    Checks:
    - Existence in DataFrame
    - Sufficient non-null observations
    - Non-constant check
    - ID-like warning
    - Deterministic, conservative problem-type detection (binary, multiclass, regression, or ambiguous)
    - Diagnostics (class balance for classification, summary stats for regression)
    """
    if target_col not in df.columns:
        raise ColumnNotFoundError(target_col, available_columns=list(df.columns))

    series = df[target_col]
    total_count = len(series)
    non_null_s = series.dropna()
    valid_count = len(non_null_s)
    missing_count = total_count - valid_count
    missing_pct = round((missing_count / max(total_count, 1)) * 100, 2)

    # 1. Target Validation: Empty or insufficient observations
    if valid_count < 10:
        raise InvalidTargetError(
            f"Target column '{target_col}' has only {valid_count} non-null observations. "
            f"At least 10 valid observations are required for meaningful machine learning."
        )

    unique_vals = non_null_s.unique()
    unique_count = len(unique_vals)
    unique_ratio = unique_count / valid_count

    # 2. Constant Target check
    if unique_count <= 1:
        raise InvalidTargetError(
            f"Target column '{target_col}' is constant (contains only 1 unique value: '{unique_vals[0]}'). "
            f"A target variable must have at least 2 distinct outcomes to be predictable."
        )

    # 3. ID-like warning
    id_warning = None
    if unique_ratio > 0.95 and valid_count > 50:
        id_warning = (
            f"Target '{target_col}' contains {unique_ratio*100:.1f}% unique values and may represent an identifier. "
            f"Please confirm that this is genuinely the prediction target."
        )

    # 4. Conservative Problem-Type Detection
    logical_dtype = infer_logical_dtype(series)
    is_numeric = logical_dtype in ("integer", "float")
    
    problem_type = "ambiguous"
    confidence = "high"
    reason = ""

    # Binary Classification
    if unique_count == 2:
        problem_type = "binary_classification"
        confidence = "high"
        reason = f"Target contains exactly 2 distinct outcomes: {list(unique_vals)}."
    
    # Non-numeric string / boolean
    elif logical_dtype in ("categorical", "boolean", "text"):
        if unique_count <= 100:
            problem_type = "multiclass_classification"
            confidence = "high"
            reason = f"Categorical target with {unique_count} discrete classes."
        else:
            problem_type = "ambiguous"
            confidence = "low"
            reason = f"High-cardinality categorical target ({unique_count} unique text values). Check if column is an ID or free text."

    # Numeric target (Float or Integer)
    elif is_numeric:
        # Check if float contains non-integers (continuous)
        clean_num = pd.to_numeric(non_null_s.replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
        has_decimals = not (clean_num == clean_num.round()).all()

        if has_decimals or unique_count > 30:
            problem_type = "regression"
            confidence = "high"
            reason = f"Continuous numerical target with {unique_count} distinct numeric values."
        elif 3 <= unique_count <= 20:
            # Ambiguous integer: Could be discrete rating (1-5) or multiclass (0, 1, 2)
            problem_type = "ambiguous"
            confidence = "medium"
            reason = (
                f"Numerical target with {unique_count} discrete integers. "
                f"It can be modeled either as Multiclass Classification or Regression. User confirmation required."
            )
        else:
            problem_type = "regression"
            confidence = "medium"
            reason = f"Integer numerical target with {unique_count} distinct levels."

    # 5. Diagnostic Profile
    diagnostics: Dict[str, Any] = {
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_percentage": missing_pct,
        "unique_count": unique_count,
        "id_warning": id_warning
    }

    if problem_type in ("binary_classification", "multiclass_classification") or (problem_type == "ambiguous" and unique_count <= 30):
        # Class distribution
        val_counts = non_null_s.value_counts()
        classes_info = []
        for val, cnt in val_counts.items():
            classes_info.append({
                "class_label": str(val),
                "count": int(cnt),
                "percentage": round((int(cnt) / valid_count) * 100, 2)
            })

        minority_pct = min(c["percentage"] for c in classes_info)
        is_imbalanced = minority_pct < 15.0

        diagnostics["classification"] = {
            "classes_count": unique_count,
            "classes": classes_info,
            "minority_class_percentage": minority_pct,
            "is_imbalanced": is_imbalanced,
            "imbalance_warning": (
                f"Severe class imbalance detected: minority class represents only {minority_pct}% of samples."
                if is_imbalanced else None
            )
        }

    if problem_type == "regression" or is_numeric:
        clean_vals = pd.to_numeric(non_null_s.replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
        if len(clean_vals) > 0:
            q1 = float(np.percentile(clean_vals, 25))
            q3 = float(np.percentile(clean_vals, 75))
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            extreme_count = int(((clean_vals < lower_fence) | (clean_vals > upper_fence)).sum())

            diagnostics["regression"] = {
                "mean": round(float(clean_vals.mean()), 4),
                "median": round(float(clean_vals.median()), 4),
                "std": round(float(clean_vals.std()), 4) if len(clean_vals) > 1 else 0.0,
                "min": round(float(clean_vals.min()), 4),
                "max": round(float(clean_vals.max()), 4),
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "potential_extreme_values_count": extreme_count
            }

    return {
        "target": target_col,
        "inferred_dtype": logical_dtype,
        "problem_type": problem_type,
        "confidence": confidence,
        "detection_reason": reason,
        "diagnostics": diagnostics
    }
