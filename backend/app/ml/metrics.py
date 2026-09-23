from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Metric direction registry
METRIC_DIRECTIONS = {
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "f1": "higher",
    "f1_macro": "higher",
    "precision": "higher",
    "recall": "higher",
    "roc_auc": "higher",
    "r2": "higher",
    "mae": "lower",
    "rmse": "lower",
    "mape": "lower"
}

def is_higher_better(metric_name: str) -> bool:
    """Returns True if higher score represents better performance."""
    canonical = metric_name.lower().strip()
    return METRIC_DIRECTIONS.get(canonical, "higher") == "higher"

def select_default_primary_metric(
    problem_type: str,
    target_diagnostics: Optional[Dict[str, Any]] = None
) -> str:
    """
    Deterministic rule-based primary metric selection:
    - Binary Classification: 'f1' if class imbalance (< 25% minority), else 'accuracy'
    - Multiclass Classification: 'f1_macro'
    - Regression: 'rmse'
    """
    p_type = problem_type.lower().strip()
    if p_type == "binary_classification":
        if target_diagnostics and "classification" in target_diagnostics:
            minority_pct = target_diagnostics["classification"].get("minority_class_percentage", 50.0)
            if minority_pct < 25.0:
                return "f1"
        return "accuracy"
    elif p_type == "multiclass_classification":
        return "f1_macro"
    elif p_type == "regression":
        return "rmse"
    return "f1" if "classification" in p_type else "rmse"

from app.core.errors import EvaluationError

def compute_metrics(
    problem_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    partition: Optional[str] = None,
    run_id: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate appropriate evaluation metrics for predictions.
    Safely ignores mathematically invalid metrics (e.g., MAPE with 0s).
    Strictly validates that prediction array and target array have identical length.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    # Invariant: y_true and y_pred must have identical length
    if len(y_t) != len(y_p):
        part_str = f" for partition '{partition}'" if partition else ""
        run_str = f" (run_id={run_id})" if run_id else ""
        mod_str = f" (model={model_name})" if model_name else ""
        raise EvaluationError(
            f"Prediction/target length mismatch{part_str}{run_str}{mod_str}: "
            f"expected {len(y_t)} samples, but got {len(y_p)} predictions.",
            details={
                "partition": partition,
                "run_id": run_id,
                "model_name": model_name,
                "expected_length": len(y_t),
                "actual_length": len(y_p)
            }
        )

    res: Dict[str, float] = {}

    p_type = problem_type.lower().strip()

    if p_type == "binary_classification":
        # Ensure scalar types match
        res["accuracy"] = round(float(accuracy_score(y_t, y_p)), 4)
        res["balanced_accuracy"] = round(float(balanced_accuracy_score(y_t, y_p)), 4)
        
        # Determine positive label if possible
        pos_label = 1 if 1 in y_t else (y_t[0] if len(np.unique(y_t)) == 2 else 1)
        try:
            res["f1"] = round(float(f1_score(y_t, y_p, pos_label=pos_label, zero_division=0)), 4)
            res["precision"] = round(float(precision_score(y_t, y_p, pos_label=pos_label, zero_division=0)), 4)
            res["recall"] = round(float(recall_score(y_t, y_p, pos_label=pos_label, zero_division=0)), 4)
        except Exception:
            res["f1"] = round(float(f1_score(y_t, y_p, average="macro", zero_division=0)), 4)
            res["precision"] = round(float(precision_score(y_t, y_p, average="macro", zero_division=0)), 4)
            res["recall"] = round(float(recall_score(y_t, y_p, average="macro", zero_division=0)), 4)

        if y_prob is not None:
            try:
                # Handle 1D or 2D probability outputs
                prob_vec = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
                res["roc_auc"] = round(float(roc_auc_score(y_t, prob_vec)), 4)
            except Exception:
                pass

    elif p_type == "multiclass_classification":
        res["accuracy"] = round(float(accuracy_score(y_t, y_p)), 4)
        res["balanced_accuracy"] = round(float(balanced_accuracy_score(y_t, y_p)), 4)
        res["f1_macro"] = round(float(f1_score(y_t, y_p, average="macro", zero_division=0)), 4)
        res["precision_macro"] = round(float(precision_score(y_t, y_p, average="macro", zero_division=0)), 4)
        res["recall_macro"] = round(float(recall_score(y_t, y_p, average="macro", zero_division=0)), 4)

        if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] > 2:
            try:
                res["roc_auc"] = round(float(roc_auc_score(y_t, y_prob, multi_class="ovr")), 4)
            except Exception:
                pass

    elif p_type == "regression":
        y_t_clean = pd.to_numeric(pd.Series(y_t), errors="coerce").fillna(0).to_numpy()
        y_p_clean = pd.to_numeric(pd.Series(y_p), errors="coerce").fillna(0).to_numpy()

        res["mae"] = round(float(mean_absolute_error(y_t_clean, y_p_clean)), 4)
        res["rmse"] = round(float(root_mean_squared_error(y_t_clean, y_p_clean)), 4)
        res["r2"] = round(float(r2_score(y_t_clean, y_p_clean)), 4)

        # MAPE calculation only if no values are zero to prevent infinite division
        if not np.any(np.isclose(y_t_clean, 0.0)):
            try:
                res["mape"] = round(float(mean_absolute_percentage_error(y_t_clean, y_p_clean)), 4)
            except Exception:
                pass

    return res

def compare_scores(metric_name: str, score_a: float, score_b: float) -> int:
    """
    Compares two model scores taking metric direction into account.
    Returns 1 if score_a is better than score_b, -1 if score_b is better, 0 if equal.
    """
    higher = is_higher_better(metric_name)
    if np.isclose(score_a, score_b, atol=1e-6):
        return 0
    if higher:
        return 1 if score_a > score_b else -1
    else:
        return 1 if score_a < score_b else -1
