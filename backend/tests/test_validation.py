import pandas as pd
from app.validation.schema_inspector import inspect_dataset_structure, infer_logical_dtype

def test_dtype_inference():
    s_int = pd.Series([1, 2, 3, 4, 5])
    assert infer_logical_dtype(s_int) == "integer"

    s_float = pd.Series([1.5, 2.7, 3.14])
    assert infer_logical_dtype(s_float) == "float"

    s_bool = pd.Series([True, False, True])
    assert infer_logical_dtype(s_bool) == "boolean"

    s_bool_str = pd.Series(["yes", "no", "yes"])
    assert infer_logical_dtype(s_bool_str) == "boolean"

    s_date = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
    assert infer_logical_dtype(s_date) == "datetime-like"

    s_cat = pd.Series(["Red", "Green", "Blue", "Red"])
    assert infer_logical_dtype(s_cat) == "categorical"

def test_dataset_structure_inspection():
    df = pd.DataFrame({
        "num": [10, 20, 30, 10],
        "cat": ["A", "B", None, "A"],
        "const": [1, 1, 1, 1]
    })
    # Add a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    summary = inspect_dataset_structure(df)
    assert summary.row_count == 5
    assert summary.column_count == 3
    assert summary.duplicate_rows_count >= 1
    assert summary.constant_columns_count == 1

    # Check column summary
    cat_col = next(c for c in summary.columns if c.name == "cat")
    assert cat_col.missing_count == 1
    assert cat_col.inferred_dtype == "categorical"

    const_col = next(c for c in summary.columns if c.name == "const")
    assert const_col.is_constant is True
