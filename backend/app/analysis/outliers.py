import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from app.core.config import settings
from app.validation.schema_inspector import infer_logical_dtype
from app.analysis.statistics import sanitize_float

def detect_column_outliers_iqr(
    series: pd.Series,
    name: str,
    multiplier: float = settings.EDA_OUTLIER_IQR_MULTIPLIER
) -> Dict[str, Any]:
    """
    Detect potential outliers using the transparent Interquartile Range (IQR) method.
    Labels them strictly as 'potential outliers' without modifying the underlying data.
    """
    s_clean = pd.to_numeric(series.replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
    total_valid = len(s_clean)
    
    if total_valid < 4:
        return {
            "column": name,
            "method": "IQR",
            "multiplier": multiplier,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "potential_outliers_count": 0,
            "potential_outliers_percentage": 0.0,
            "lower_outliers_count": 0,
            "upper_outliers_count": 0,
            "min_val": None,
            "max_val": None,
            "has_outliers": False,
            "disclaimer": "Potential outliers are statistical anomalies and do not automatically indicate invalid or erroneous data."
        }
        
    q1 = float(np.percentile(s_clean, 25))
    q3 = float(np.percentile(s_clean, 75))
    iqr = q3 - q1
    
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)
    
    lower_outliers = s_clean[s_clean < lower_bound]
    upper_outliers = s_clean[s_clean > upper_bound]
    
    outlier_count = len(lower_outliers) + len(upper_outliers)
    outlier_pct = round((outlier_count / max(total_valid, 1)) * 100, 2)
    
    return {
        "column": name,
        "method": "IQR",
        "multiplier": multiplier,
        "q1": sanitize_float(q1),
        "q3": sanitize_float(q3),
        "iqr": sanitize_float(iqr),
        "lower_bound": sanitize_float(lower_bound),
        "upper_bound": sanitize_float(upper_bound),
        "potential_outliers_count": outlier_count,
        "potential_outliers_percentage": outlier_pct,
        "lower_outliers_count": len(lower_outliers),
        "upper_outliers_count": len(upper_outliers),
        "min_val": sanitize_float(float(s_clean.min())),
        "max_val": sanitize_float(float(s_clean.max())),
        "has_outliers": outlier_count > 0,
        "disclaimer": "Potential outliers are statistical anomalies and do not automatically indicate invalid or erroneous data."
    }

def detect_dataset_outliers(
    df: pd.DataFrame,
    multiplier: float = settings.EDA_OUTLIER_IQR_MULTIPLIER
) -> List[Dict[str, Any]]:
    """Detect potential outliers across all numerical columns in dataset."""
    results = []
    for col in df.columns:
        s = df[col]
        dtype = infer_logical_dtype(s)
        if dtype in ("integer", "float"):
            results.append(detect_column_outliers_iqr(s, str(col), multiplier=multiplier))
    return results
