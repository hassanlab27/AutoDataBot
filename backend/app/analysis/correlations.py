import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.core.config import settings
from app.validation.schema_inspector import infer_logical_dtype

def compute_correlation_matrix(
    df: pd.DataFrame,
    max_cols: int = settings.EDA_MAX_CORRELATION_COLUMNS,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Calculate Pearson correlation matrix for numerical features.
    If total numeric columns exceed max_cols, select columns with highest normalized variance.
    Handles NaNs safely via pairwise complete observations.
    Explicitly clarifies correlation != causation.
    """
    numeric_cols = [
        str(col) for col in df.columns 
        if infer_logical_dtype(df[col]) in ("integer", "float")
    ]
    
    if len(numeric_cols) < 2:
        return {
            "columns": numeric_cols,
            "matrix": [],
            "total_numeric_columns": len(numeric_cols),
            "columns_included": len(numeric_cols),
            "truncated": False,
            "selection_strategy": "All numerical columns included",
            "strong_correlations": [],
            "disclaimer": "Correlation measures statistical linear association and does not imply causation."
        }
        
    truncated = False
    strategy = "All numerical columns included"
    selected_cols = list(numeric_cols)
    
    if len(numeric_cols) > max_cols:
        truncated = True
        strategy = f"Top {max_cols} columns ranked by normalized variance (std / (mean + eps)) to prioritize informative features"
        
        # Rank by normalized variance
        variances = []
        for col in numeric_cols:
            clean = pd.to_numeric(df[col].replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
            if len(clean) > 1:
                std = float(clean.std())
                mean = abs(float(clean.mean()))
                norm_var = std / (mean + 1e-5)
                variances.append((col, norm_var))
            else:
                variances.append((col, 0.0))
                
        variances.sort(key=lambda x: x[1], reverse=True)
        selected_cols = [col for col, _ in variances[:max_cols]]

    # Compute correlation matrix
    sub_df = df[selected_cols].apply(pd.to_numeric, errors="coerce")
    corr_df = sub_df.corr(method=method)
    
    # Replace NaN correlations (e.g. constant columns) with 0.0
    corr_df = corr_df.fillna(0.0)
    
    matrix = []
    for col_i in selected_cols:
        row = []
        for col_j in selected_cols:
            val = float(corr_df.loc[col_i, col_j])
            if math.isnan(val) or math.isinf(val):
                val = 0.0
            row.append(round(val, 4))
        matrix.append(row)
        
    strong_correlations = extract_strong_correlations(
        corr_df=corr_df,
        threshold=settings.EDA_STRONG_CORRELATION_THRESHOLD,
        top_n=settings.EDA_TOP_CORRELATION_PAIRS
    )
    
    return {
        "columns": selected_cols,
        "matrix": matrix,
        "total_numeric_columns": len(numeric_cols),
        "columns_included": len(selected_cols),
        "truncated": truncated,
        "selection_strategy": strategy,
        "strong_correlations": strong_correlations,
        "disclaimer": "Correlation measures statistical linear association and does not imply causation."
    }

def extract_strong_correlations(
    corr_df: pd.DataFrame,
    threshold: float = settings.EDA_STRONG_CORRELATION_THRESHOLD,
    top_n: int = settings.EDA_TOP_CORRELATION_PAIRS
) -> List[Dict[str, Any]]:
    """
    Extract unique pairs of columns with absolute correlation >= threshold,
    sorted descending by absolute strength.
    """
    pairs = []
    cols = list(corr_df.columns)
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a = cols[i]
            col_b = cols[j]
            r = float(corr_df.loc[col_a, col_b])
            if math.isnan(r) or math.isinf(r):
                continue
            abs_r = abs(r)
            if abs_r >= threshold:
                pairs.append({
                    "feature_a": col_a,
                    "feature_b": col_b,
                    "correlation": round(r, 4),
                    "abs_correlation": round(abs_r, 4),
                    "relationship_strength": "strong" if abs_r >= 0.7 else "moderate",
                    "direction": "positive" if r > 0 else "negative"
                })
                
    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return pairs[:top_n]
