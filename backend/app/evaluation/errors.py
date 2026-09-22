from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def analyze_classification_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Identifies classification errors, confusion pairs, and the top N worst misclassifications.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    n = len(y_t)

    is_error = (y_t != y_p)
    misclass_count = int(np.sum(is_error))
    error_rate = round(float(misclass_count / n), 4) if n > 0 else 0.0

    # 1. Most confused class pairs
    error_indices = np.where(is_error)[0]
    confused_pairs: Dict[str, int] = {}
    for idx in error_indices:
        pair_key = f"{y_t[idx]} -> {y_p[idx]}"
        confused_pairs[pair_key] = confused_pairs.get(pair_key, 0) + 1

    sorted_pairs = [
        {
            "pair": k,
            "actual": k.split(" -> ")[0],
            "predicted": k.split(" -> ")[1],
            "count": v
        }
        for k, v in sorted(confused_pairs.items(), key=lambda item: item[1], reverse=True)
    ]

    # 2. Worst predictions (ordered by highest confidence on wrong prediction)
    worst_records: List[Dict[str, Any]] = []
    if misclass_count > 0:
        candidate_records = []
        for idx in error_indices:
            conf = 1.0
            if y_prob is not None:
                try:
                    if y_prob.ndim == 2:
                        # Grab probability assigned to predicted class
                        unique_classes = sorted(list(np.unique(y_t)))
                        pred_val = y_p[idx]
                        if pred_val in unique_classes:
                            col_idx = unique_classes.index(pred_val)
                            conf = float(y_prob[idx, col_idx])
                        else:
                            conf = float(np.max(y_prob[idx]))
                    elif y_prob.ndim == 1:
                        p = float(y_prob[idx])
                        unique_classes = sorted(list(np.unique(y_t)))
                        if len(unique_classes) == 2 and str(y_p[idx]) == str(unique_classes[0]):
                            conf = 1.0 - p
                        else:
                            conf = p
                except Exception:
                    conf = 1.0

            candidate_records.append({
                "prediction_id": int(idx),
                "actual": str(y_t[idx]),
                "predicted": str(y_p[idx]),
                "confidence": round(conf, 4),
                "correct": False
            })

        # Sort by confidence descending (model was most stubbornly wrong)
        candidate_records.sort(key=lambda r: r["confidence"], reverse=True)
        worst_records = candidate_records[:limit]

    return {
        "problem_type": "classification",
        "total_evaluated": n,
        "misclassifications_count": misclass_count,
        "error_rate": error_rate,
        "most_confused_pairs": sorted_pairs[:10],
        "worst_predictions": worst_records
    }

def analyze_regression_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Identifies the top N worst absolute regression prediction errors.
    """
    y_t = pd.to_numeric(pd.Series(np.asarray(y_true)), errors="coerce").fillna(0).to_numpy()
    y_p = pd.to_numeric(pd.Series(np.asarray(y_pred)), errors="coerce").fillna(0).to_numpy()
    n = len(y_t)

    residuals = y_t - y_p
    abs_errors = np.abs(residuals)

    sorted_indices = np.argsort(abs_errors)[::-1]
    top_indices = sorted_indices[:limit]

    worst_records: List[Dict[str, Any]] = []
    for idx in top_indices:
        actual_val = float(y_t[idx])
        pred_val = float(y_p[idx])
        res_val = float(residuals[idx])
        err_val = float(abs_errors[idx])
        rel_pct = round((err_val / abs(actual_val)) * 100, 2) if abs(actual_val) > 1e-5 else None

        worst_records.append({
            "prediction_id": int(idx),
            "actual": round(actual_val, 4),
            "predicted": round(pred_val, 4),
            "residual": round(res_val, 4),
            "abs_error": round(err_val, 4),
            "relative_error_pct": rel_pct
        })

    return {
        "problem_type": "regression",
        "total_evaluated": n,
        "mae": round(float(np.mean(abs_errors)), 4) if n > 0 else 0.0,
        "median_absolute_error": round(float(np.median(abs_errors)), 4) if n > 0 else 0.0,
        "max_absolute_error": round(float(np.max(abs_errors)), 4) if n > 0 else 0.0,
        "worst_predictions": worst_records
    }
