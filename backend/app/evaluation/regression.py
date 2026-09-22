from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error
)

def sample_regression_points(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    max_points: int = 500
) -> List[Dict[str, Any]]:
    """Samples points for responsive Plotly rendering while retaining extremes."""
    n = len(y_true)
    if n <= max_points:
        return [
            {
                "index": int(i),
                "actual": round(float(y_true[i]), 4),
                "predicted": round(float(y_pred[i]), 4),
                "residual": round(float(residuals[i]), 4),
                "abs_error": round(float(abs(residuals[i])), 4)
            }
            for i in range(n)
        ]

    # Always include top 10 largest positive residuals and top 10 largest negative residuals
    sorted_res_idx = np.argsort(residuals)
    extreme_indices = set(sorted_res_idx[:15]).union(set(sorted_res_idx[-15:]))

    # Uniformly sample remaining quota
    remaining_quota = max_points - len(extreme_indices)
    step = max(1, n // remaining_quota)
    sampled_indices = set(range(0, n, step))
    all_indices = sorted(list(extreme_indices.union(sampled_indices))[:max_points])

    return [
        {
            "index": int(i),
            "actual": round(float(y_true[i]), 4),
            "predicted": round(float(y_pred[i]), 4),
            "residual": round(float(residuals[i]), 4),
            "abs_error": round(float(abs(residuals[i])), 4)
        }
        for i in all_indices
    ]

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Computes comprehensive regression metrics, actual vs predicted analysis,
    residual distributions, and residual pattern diagnostics.
    """
    y_t = pd.to_numeric(pd.Series(np.asarray(y_true)), errors="coerce").fillna(0).to_numpy()
    y_p = pd.to_numeric(pd.Series(np.asarray(y_pred)), errors="coerce").fillna(0).to_numpy()

    # 1. Primary Metrics
    mae = round(float(mean_absolute_error(y_t, y_p)), 4)
    rmse = round(float(root_mean_squared_error(y_t, y_p)), 4)
    r2 = round(float(r2_score(y_t, y_p)), 4)
    med_ae = round(float(median_absolute_error(y_t, y_p)), 4)
    exp_var = round(float(explained_variance_score(y_t, y_p)), 4)

    metrics_summary: Dict[str, Any] = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_absolute_error": med_ae,
        "explained_variance": exp_var
    }

    # Safe MAPE calculation (only valid if no target values are near zero)
    if not np.any(np.isclose(y_t, 0.0, atol=1e-5)):
        try:
            mape = round(float(mean_absolute_percentage_error(y_t, y_p)), 4)
            metrics_summary["mape"] = mape
        except Exception:
            pass

    # 2. Residuals
    residuals = y_t - y_p
    mean_res = round(float(np.mean(residuals)), 4)
    median_res = round(float(np.median(residuals)), 4)
    std_res = round(float(np.std(residuals)), 4)
    min_res = round(float(np.min(residuals)), 4)
    max_res = round(float(np.max(residuals)), 4)

    p10 = round(float(np.percentile(residuals, 10)), 4)
    p25 = round(float(np.percentile(residuals, 25)), 4)
    p50 = median_res
    p75 = round(float(np.percentile(residuals, 75)), 4)
    p90 = round(float(np.percentile(residuals, 90)), 4)
    iqr = round(float(p75 - p25), 4)

    # 3. Residual Histogram (20 bins)
    counts, bin_edges = np.histogram(residuals, bins=20)
    histogram_data = {
        "counts": [int(c) for c in counts],
        "bin_edges": [round(float(b), 4) for b in bin_edges],
        "bin_centers": [round(float((bin_edges[i] + bin_edges[i+1])/2), 4) for i in range(len(counts))]
    }

    # 4. Actual vs Predicted Scatter Points & Reference Line
    scatter_points = sample_regression_points(y_t, y_p, residuals, max_points=500)
    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))

    # 5. Residual Pattern Diagnostics
    diagnostics: List[Dict[str, str]] = []
    y_std = float(np.std(y_t)) if float(np.std(y_t)) > 1e-6 else 1.0

    if abs(mean_res) > 0.10 * y_std:
        diagnostics.append({
            "type": "residual_bias",
            "severity": "moderate",
            "message": f"Potential residual bias detected: Mean residual ({mean_res}) deviates notably from zero relative to target variance."
        })

    # Check for heteroscedasticity (differing variance across predictions)
    try:
        med_pred = float(np.median(y_p))
        lower_half_res = residuals[y_p <= med_pred]
        upper_half_res = residuals[y_p > med_pred]
        if len(lower_half_res) > 5 and len(upper_half_res) > 5:
            var_low = float(np.var(lower_half_res))
            var_high = float(np.var(upper_half_res))
            ratio = (var_high / var_low) if var_low > 1e-6 else 1.0
            if ratio > 3.0 or ratio < 0.33:
                diagnostics.append({
                    "type": "heteroscedasticity",
                    "severity": "mild",
                    "message": "Potential residual pattern detected: Residual variance appears to vary across lower vs higher predicted values."
                })
    except Exception:
        pass

    # Extreme residuals count (> 3 sigma)
    extreme_count = int(np.sum(np.abs(residuals - mean_res) > 3 * std_res)) if std_res > 1e-6 else 0
    if extreme_count > 0:
        diagnostics.append({
            "type": "outliers",
            "severity": "info",
            "message": f"{extreme_count} observation(s) exhibit large residuals exceeding 3 standard deviations."
        })

    if not diagnostics:
        diagnostics.append({
            "type": "normal",
            "severity": "info",
            "message": "Residuals appear approximately symmetric around zero without severe systematic anomalies."
        })

    return {
        "problem_type": "regression",
        "metrics": metrics_summary,
        "residual_stats": {
            "mean": mean_res,
            "median": median_res,
            "std": std_res,
            "min": min_res,
            "max": max_res,
            "iqr": iqr,
            "quantiles": {
                "p10": p10,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p90": p90
            }
        },
        "histogram": histogram_data,
        "actual_vs_predicted": {
            "points": scatter_points,
            "diagonal_ref": {
                "min": round(min_val, 4),
                "max": round(max_val, 4)
            }
        },
        "diagnostics": diagnostics,
        "disclaimer": "Residual diagnostic heuristics are exploratory analysis tools and do not establish formal statistical proof."
    }
