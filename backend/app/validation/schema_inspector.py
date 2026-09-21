from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel
from app.core.config import settings

class ColumnSummary(BaseModel):
    name: str
    raw_dtype: str
    inferred_dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_constant: bool
    is_high_cardinality: bool
    sample_values: List[Any]

class DatasetStructureSummary(BaseModel):
    row_count: int
    column_count: int
    memory_usage_bytes: int
    memory_usage_human: str
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    constant_columns_count: int
    columns: List[ColumnSummary]

def format_memory_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def infer_logical_dtype(series: pd.Series) -> str:
    """
    Infer logical data type without mutating or coercing original values.
    Returns: 'integer', 'float', 'boolean', 'datetime-like', 'categorical', 'text', or 'mixed'.
    """
    # 1. Native Booleans
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # 2. Native Numerics
    if pd.api.types.is_integer_dtype(series):
        # Could be boolean represented as 0/1 with exactly 2 unique values
        non_null = series.dropna()
        if len(non_null) > 0 and set(non_null.unique()).issubset({0, 1}):
            return "boolean"
        return "integer"

    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0 and (non_null == non_null.round()).all() and non_null.max() < 1e12:
            return "integer"
        return "float"

    # 3. Object / String inspection
    non_null_s = series.dropna().astype(str).str.strip()
    if len(non_null_s) == 0:
        return "mixed"

    # Check for string representation of booleans
    lowered = non_null_s.str.lower()
    if set(lowered.unique()).issubset({"true", "false", "yes", "no", "t", "f", "1", "0"}):
        return "boolean"

    # Check for Datetime-like strings (sample up to 50 non-null values)
    sample_values = non_null_s.head(50)
    # Check if strings look like dates (contain separators like '-', '/', ':')
    date_separators = sum(any(sep in val for sep in ["-", "/", ":"]) for val in sample_values)
    if date_separators >= len(sample_values) * 0.8:
        try:
            pd.to_datetime(sample_values, errors="raise", format="mixed")
            return "datetime-like"
        except (ValueError, TypeError):
            pass

    # Check for numeric strings stored as objects
    numeric_count = pd.to_numeric(sample_values, errors="coerce").notna().sum()
    if numeric_count == len(sample_values):
        # Determine int vs float
        numeric_series = pd.to_numeric(sample_values, errors="coerce")
        if (numeric_series == numeric_series.round()).all():
            return "integer"
        return "float"

    # Categorical vs Text based on cardinality and length
    unique_count = series.nunique(dropna=True)
    avg_len = non_null_s.str.len().mean()

    # If average text length is long and mostly unique, it's unstructured text
    if avg_len > 60 and (unique_count / max(len(series), 1)) > 0.70:
        return "text"

    return "categorical"

def inspect_dataset_structure(df: pd.DataFrame) -> DatasetStructureSummary:
    """Perform full schema, dimension, cardinality, and type inspection."""
    row_count = len(df)
    column_count = len(df.columns)
    mem_bytes = int(df.memory_usage(deep=True).sum())

    # Duplicates check
    dup_rows = int(df.duplicated().sum())
    dup_pct = round((dup_rows / max(row_count, 1)) * 100, 2)

    column_summaries = []
    constant_cols_count = 0

    for col in df.columns:
        series = df[col]
        raw_dtype = str(series.dtype)
        inferred_dtype = infer_logical_dtype(series)

        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / max(row_count, 1)) * 100, 2)

        unique_count = int(series.nunique(dropna=True))
        unique_pct = round((unique_count / max(row_count, 1)) * 100, 2)

        is_constant = (unique_count <= settings.CONSTANT_COLUMN_THRESHOLD)
        if is_constant:
            constant_cols_count += 1

        is_high_card = (
            unique_count > 50 and 
            (unique_count / max(row_count, 1)) >= settings.HIGH_CARDINALITY_RATIO_THRESHOLD
        )

        # Sample up to 5 non-null serializable values
        sample_vals = series.dropna().head(5).tolist()
        # Ensure values are JSON serializable
        clean_samples = []
        for val in sample_vals:
            if isinstance(val, (np.integer, int)):
                clean_samples.append(int(val))
            elif isinstance(val, (np.floating, float)):
                clean_samples.append(float(val) if not np.isnan(val) else None)
            elif isinstance(val, (np.bool_, bool)):
                clean_samples.append(bool(val))
            else:
                clean_samples.append(str(val))

        column_summaries.append(
            ColumnSummary(
                name=str(col),
                raw_dtype=raw_dtype,
                inferred_dtype=inferred_dtype,
                missing_count=missing_count,
                missing_percentage=missing_pct,
                unique_count=unique_count,
                unique_percentage=unique_pct,
                is_constant=is_constant,
                is_high_cardinality=is_high_card,
                sample_values=clean_samples
            )
        )

    return DatasetStructureSummary(
        row_count=row_count,
        column_count=column_count,
        memory_usage_bytes=mem_bytes,
        memory_usage_human=format_memory_size(mem_bytes),
        duplicate_rows_count=dup_rows,
        duplicate_rows_percentage=dup_pct,
        constant_columns_count=constant_cols_count,
        columns=column_summaries
    )
