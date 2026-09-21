from typing import Dict, Any, List, Set, Optional, Tuple
import pandas as pd
import numpy as np

from app.core.config import settings
from app.core.errors import TargetLeakageError
from app.validation.schema_inspector import infer_logical_dtype

def audit_features_for_selection(
    df: pd.DataFrame,
    target_col: str
) -> Dict[str, Any]:
    """
    Examines all features relative to the target, flags suspicious or problematic columns,
    and enforces zero target leakage.
    Does NOT automatically delete columns—flags them with transparent reasons for user review.
    """
    # 1. Strict Target Leakage Check
    candidate_features = [col for col in df.columns if col != target_col]
    
    if target_col in candidate_features:
        raise TargetLeakageError(
            f"Target column '{target_col}' detected inside candidate feature set."
        )

    row_count = len(df)
    target_series = df[target_col]
    
    features_audit = []
    duplicate_candidates: Dict[str, str] = {}
    
    # Pre-calculate column hashes / series equality for duplicate detection (on small-medium data)
    col_signatures: Dict[str, str] = {}
    for col in candidate_features:
        s = df[col]
        # Check if identical to target (Direct Leakage Candidate)
        if s.equals(target_series):
            duplicate_candidates[col] = f"Exact duplicate of target '{target_col}' (Severe Leakage Risk)"

    # Check pairwise feature duplicates
    checked_pairs: Set[Tuple[str, str]] = set()
    for i in range(len(candidate_features)):
        col_i = candidate_features[i]
        for j in range(i + 1, len(candidate_features)):
            col_j = candidate_features[j]
            if col_i not in duplicate_candidates and df[col_i].equals(df[col_j]):
                duplicate_candidates[col_j] = f"Exact duplicate of feature '{col_i}'"

    for col in candidate_features:
        s = df[col]
        dtype = infer_logical_dtype(s)
        missing_count = int(s.isna().sum())
        missing_pct = round((missing_count / max(row_count, 1)) * 100, 2)
        unique_count = int(s.nunique(dropna=True))
        unique_pct = round((unique_count / max(row_count, 1)) * 100, 2)

        flags: List[str] = []
        reasons: List[str] = []

        # Check Constant Feature
        if unique_count <= settings.CONSTANT_COLUMN_THRESHOLD:
            flags.append("constant_column")
            reasons.append(f"Column has {unique_count} unique value; provides no predictive variance.")

        # Check Duplicate Feature
        if col in duplicate_candidates:
            flags.append("duplicate_feature")
            reasons.append(duplicate_candidates[col])

        # Check Obvious Row ID / High Cardinality
        if (
            unique_count > 50 and 
            (unique_count / max(row_count, 1)) >= settings.SUSPICIOUS_ID_RATIO_THRESHOLD
        ):
            flags.append("potential_id")
            reasons.append(f"{unique_pct}% of values are unique. Likely an identifier or primary key.")

        # Check Excessive Missingness
        if missing_pct >= settings.PREPROCESSING_EXCESSIVE_MISSING_THRESHOLD * 100:
            flags.append("excessive_missingness")
            reasons.append(f"{missing_pct}% of values are missing (exceeds {settings.PREPROCESSING_EXCESSIVE_MISSING_THRESHOLD*100:.0f}% threshold).")

        # Check Free-Form Text
        if dtype == "text":
            flags.append("free_form_text")
            reasons.append("Free-form text detected. Advanced NLP is not enabled in this version.")

        # Check High-Cardinality Categorical (> 50 unique categories)
        if dtype == "categorical" and unique_count > settings.PREPROCESSING_MAX_ONE_HOT_CATEGORIES:
            flags.append("high_cardinality_categorical")
            reasons.append(
                f"Contains {unique_count} distinct categories. "
                f"One-hot encoding may generate excessive dimensional expansion."
            )

        if "constant_column" in flags or "duplicate_feature" in flags:
            status = "flagged_exclude"
            recommended_action = "exclude"
        elif flags:
            status = "review_recommended"
            recommended_action = "exclude"
        else:
            status = "included"
            recommended_action = "include"

        reason = " | ".join(reasons) if reasons else None

        features_audit.append({
            "name": str(col),
            "inferred_dtype": dtype,
            "raw_dtype": str(s.dtype),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "unique_count": unique_count,
            "unique_percentage": unique_pct,
            "flags": flags,
            "status": status,
            "recommended_action": recommended_action,
            "reason": reason
        })

    return {
        "target": target_col,
        "total_candidate_features": len(candidate_features),
        "recommended_included_count": len([f for f in features_audit if f["recommended_action"] == "include"]),
        "recommended_excluded_count": len([f for f in features_audit if f["recommended_action"] == "exclude"]),
        "review_recommended_count": len([f for f in features_audit if f["recommended_action"] == "review"]),
        "features": features_audit
    }

def verify_zero_target_leakage(selected_features: List[str], target_col: str) -> None:
    """Explicit barrier assertion guaranteeing no target leakage."""
    if target_col in selected_features:
        raise TargetLeakageError(
            f"Fatal Leakage Violation: Target column '{target_col}' cannot be included in feature set X."
        )
