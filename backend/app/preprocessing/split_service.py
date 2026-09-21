from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from app.core.config import settings

def perform_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_size: float = settings.PREPROCESSING_DEFAULT_TEST_SIZE,
    random_state: int = settings.PREPROCESSING_DEFAULT_RANDOM_STATE,
    problem_type: str = "binary_classification"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Executes a leakage-free train/test partition before any preprocessing or fitting.
    Verifies index disjointness: train_idx ∩ test_idx = ∅.
    Applies stratification for classification targets when minimum class counts allow.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is missing (standard ML practice: cannot train or evaluate without ground truth)
    valid_target_mask = y.notna()
    if not valid_target_mask.all():
        X = X[valid_target_mask]
        y = y[valid_target_mask]

    is_classification = problem_type in ("binary_classification", "multiclass_classification")
    stratify_target = None
    stratification_applied = False

    if is_classification:
        # Check if every class has at least 2 samples to allow stratification
        class_counts = y.value_counts()
        if (class_counts >= 2).all() and len(class_counts) > 1:
            stratify_target = y
            stratification_applied = True

    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
        shuffle=True
    )

    # Rigorous isolation validation: Verify index disjointness
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    overlap = train_idx.intersection(test_idx)
    if len(overlap) > 0:
        raise RuntimeError(f"CRITICAL LEAKAGE DETECTED: {len(overlap)} samples overlap between train and test sets!")

    # Calculate class distributions for classification targets
    train_dist = {}
    test_dist = {}
    if is_classification:
        train_dist = {str(k): int(v) for k, v in y_train.value_counts().items()}
        test_dist = {str(k): int(v) for k, v in y_test.value_counts().items()}

    split_metadata = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_rows": len(X),
        "train_percentage": round((len(X_train) / len(X)) * 100, 2),
        "test_percentage": round((len(X_test) / len(X)) * 100, 2),
        "test_size": test_size,
        "random_state": random_state,
        "stratified": stratification_applied,
        "split_strategy": "Stratified Shuffle Split" if stratification_applied else "Random Shuffle Split",
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist
    }

    return X_train, X_test, y_train, y_test, split_metadata
