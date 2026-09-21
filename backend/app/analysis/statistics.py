import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from app.core.config import settings
from app.validation.schema_inspector import infer_logical_dtype, format_memory_size

def sanitize_float(val: Any, decimals: int = 4) -> Optional[float]:
    """Safely convert numpy/pandas/python float to JSON-serializable float or None."""
    if val is None or pd.isna(val):
        return None
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return None
        return round(f_val, decimals)
    except (ValueError, TypeError):
        return None

def compute_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall machine-readable dataset statistics."""
    rows = len(df)
    columns = len(df.columns)
    
    numeric_cols = []
    categorical_cols = []
    boolean_cols = []
    datetime_cols = []
    text_cols = []
    
    constant_cols = []
    high_cardinality_cols = []
    
    total_missing_cells = 0
    
    for col in df.columns:
        s = df[col]
        dtype = infer_logical_dtype(s)
        missing_count = int(s.isna().sum())
        total_missing_cells += missing_count
        unique_count = int(s.nunique(dropna=True))
        
        if dtype in ("integer", "float"):
            numeric_cols.append(str(col))
        elif dtype == "boolean":
            boolean_cols.append(str(col))
        elif dtype == "datetime-like":
            datetime_cols.append(str(col))
        elif dtype == "text":
            text_cols.append(str(col))
        else:
            categorical_cols.append(str(col))
            
        if unique_count <= settings.CONSTANT_COLUMN_THRESHOLD:
            constant_cols.append(str(col))
            
        if unique_count > 50 and (unique_count / max(rows, 1)) >= settings.HIGH_CARDINALITY_RATIO_THRESHOLD:
            high_cardinality_cols.append(str(col))
            
    total_cells = max(rows * columns, 1)
    missing_pct = round((total_missing_cells / total_cells) * 100, 2)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / max(rows, 1)) * 100, 2)
    
    mem_bytes = int(df.memory_usage(deep=True).sum())
    
    return {
        "rows": rows,
        "columns": columns,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "boolean_columns": len(boolean_cols),
        "datetime_columns": len(datetime_cols),
        "text_columns": len(text_cols),
        "missing_cells": total_missing_cells,
        "missing_percentage": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_pct,
        "memory_usage_bytes": mem_bytes,
        "memory_usage_human": format_memory_size(mem_bytes),
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_cardinality_cols,
        "column_classification": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "boolean": boolean_cols,
            "datetime": datetime_cols,
            "text": text_cols
        }
    }

def compute_numerical_column_stats(series: pd.Series, name: str) -> Dict[str, Any]:
    """Compute robust numerical descriptive statistics handling NaN, +inf, -inf."""
    total_count = len(series)
    # Replace inf with NaN for safe calculations
    s_clean = pd.to_numeric(series.replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
    valid_count = len(s_clean)
    missing_count = total_count - valid_count
    missing_pct = round((missing_count / max(total_count, 1)) * 100, 2)
    
    if valid_count == 0:
        return {
            "column": name,
            "count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None
        }
        
    mean_val = float(s_clean.mean())
    median_val = float(s_clean.median())
    std_val = float(s_clean.std()) if valid_count > 1 else 0.0
    min_val = float(s_clean.min())
    max_val = float(s_clean.max())
    
    q1_val = float(np.percentile(s_clean, 25))
    q3_val = float(np.percentile(s_clean, 75))
    iqr_val = float(q3_val - q1_val)
    
    # Skewness and Kurtosis (require at least 3 values and non-zero variance)
    skew_val = None
    kurt_val = None
    if valid_count >= 3 and std_val > 1e-9:
        try:
            s_skew = float(sp_stats.skew(s_clean, nan_policy="omit", bias=False))
            if not (math.isnan(s_skew) or math.isinf(s_skew)):
                skew_val = round(s_skew, 4)
        except Exception:
            pass
            
        try:
            s_kurt = float(sp_stats.kurtosis(s_clean, nan_policy="omit", bias=False))
            if not (math.isnan(s_kurt) or math.isinf(s_kurt)):
                kurt_val = round(s_kurt, 4)
        except Exception:
            pass

    return {
        "column": name,
        "count": total_count,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_percentage": missing_pct,
        "mean": sanitize_float(mean_val),
        "median": sanitize_float(median_val),
        "std": sanitize_float(std_val),
        "min": sanitize_float(min_val),
        "max": sanitize_float(max_val),
        "q1": sanitize_float(q1_val),
        "q3": sanitize_float(q3_val),
        "iqr": sanitize_float(iqr_val),
        "skewness": skew_val,
        "kurtosis": kurt_val
    }

def compute_all_numerical_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute statistics for all numerical columns in dataset."""
    results = []
    for col in df.columns:
        s = df[col]
        dtype = infer_logical_dtype(s)
        if dtype in ("integer", "float"):
            results.append(compute_numerical_column_stats(s, str(col)))
    return results

def compute_categorical_column_stats(
    series: pd.Series,
    name: str,
    top_n: int = settings.EDA_MAX_CATEGORIES_PER_CHART
) -> Dict[str, Any]:
    """Compute frequency and cardinality metrics for a categorical/boolean column."""
    total_count = len(series)
    s_nonnull = series.dropna().astype(str).str.strip()
    valid_count = len(s_nonnull)
    missing_count = total_count - valid_count
    missing_pct = round((missing_count / max(total_count, 1)) * 100, 2)
    
    unique_count = int(s_nonnull.nunique())
    
    if valid_count == 0:
        return {
            "column": name,
            "count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "unique_values": 0,
            "mode": None,
            "mode_count": 0,
            "mode_percentage": 0.0,
            "top_categories": [],
            "other_count": 0,
            "other_percentage": 0.0
        }
        
    val_counts = s_nonnull.value_counts()
    mode_val = str(val_counts.index[0])
    mode_count = int(val_counts.iloc[0])
    mode_pct = round((mode_count / max(total_count, 1)) * 100, 2)
    
    top_items = val_counts.head(top_n)
    top_categories = []
    top_sum = 0
    
    for cat_name, cnt in top_items.items():
        cnt_int = int(cnt)
        top_sum += cnt_int
        top_categories.append({
            "category": str(cat_name),
            "count": cnt_int,
            "percentage": round((cnt_int / max(total_count, 1)) * 100, 2)
        })
        
    other_count = valid_count - top_sum
    other_pct = round((other_count / max(total_count, 1)) * 100, 2)
    
    return {
        "column": name,
        "count": total_count,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_percentage": missing_pct,
        "unique_values": unique_count,
        "mode": mode_val,
        "mode_count": mode_count,
        "mode_percentage": mode_pct,
        "top_categories": top_categories,
        "other_count": other_count,
        "other_percentage": other_pct
    }

def compute_all_categorical_stats(
    df: pd.DataFrame,
    top_n: int = settings.EDA_MAX_CATEGORIES_PER_CHART
) -> List[Dict[str, Any]]:
    """Compute statistics for all categorical/boolean/text columns."""
    results = []
    for col in df.columns:
        s = df[col]
        dtype = infer_logical_dtype(s)
        if dtype in ("categorical", "boolean", "text"):
            results.append(compute_categorical_column_stats(s, str(col), top_n=top_n))
    return results

def compute_missing_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute granular missing value distribution and highlight severe missingness."""
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = max(total_rows * total_cols, 1)
    
    columns_missing = []
    total_missing_cells = 0
    severe_missing_cols = []
    
    for col in df.columns:
        s = df[col]
        miss_count = int(s.isna().sum())
        total_missing_cells += miss_count
        miss_pct = round((miss_count / max(total_rows, 1)) * 100, 2)
        
        col_info = {
            "column": str(col),
            "missing_count": miss_count,
            "missing_percentage": miss_pct,
            "present_count": total_rows - miss_count,
            "present_percentage": round(100.0 - miss_pct, 2)
        }
        columns_missing.append(col_info)
        
        if miss_pct >= 50.0:
            severe_missing_cols.append(str(col))
            
    # Sort columns by missing percentage descending
    columns_missing.sort(key=lambda x: x["missing_count"], reverse=True)
    
    cols_with_missing = [c["column"] for c in columns_missing if c["missing_count"] > 0]
    
    return {
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "missing_percentage": round((total_missing_cells / total_cells) * 100, 2),
        "columns_with_missing_count": len(cols_with_missing),
        "columns_with_missing": cols_with_missing,
        "severe_missing_columns": severe_missing_cols,
        "columns": columns_missing
    }

def compute_temporal_column_stats(series: pd.Series, name: str) -> Optional[Dict[str, Any]]:
    """Compute temporal statistics for a datetime-like column."""
    s_clean = series.dropna()
    if len(s_clean) == 0:
        return None
        
    try:
        dt_series = pd.to_datetime(s_clean, errors="coerce").dropna()
        if len(dt_series) == 0:
            return None
            
        min_dt = dt_series.min()
        max_dt = dt_series.max()
        duration_days = (max_dt - min_dt).total_seconds() / 86400.0
        
        return {
            "column": name,
            "total_count": len(series),
            "valid_dates_count": len(dt_series),
            "missing_count": len(series) - len(dt_series),
            "min_date": str(min_dt.isoformat()),
            "max_date": str(max_dt.isoformat()),
            "date_range_days": round(duration_days, 2)
        }
    except Exception:
        return None
