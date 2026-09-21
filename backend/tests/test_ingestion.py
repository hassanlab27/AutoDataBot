import pytest
from app.ingestion.reader import safe_read_csv, detect_encoding, detect_delimiter
from app.core.errors import EmptyDatasetError, InvalidCSVError

def test_detect_encoding(sample_clean_csv_bytes, sample_semicolon_csv_bytes):
    assert detect_encoding(sample_clean_csv_bytes) in ("utf-8", "utf-8-sig")
    assert detect_encoding(sample_semicolon_csv_bytes) in ("latin-1", "ISO-8859-1", "utf-8")

def test_detect_delimiter():
    assert detect_delimiter("a,b,c\n1,2,3") == ","
    assert detect_delimiter("a;b;c\n1;2;3") == ";"
    assert detect_delimiter("a\tb\tc\n1\t2\t3") == "\t"
    assert detect_delimiter("a|b|c\n1|2|3") == "|"

def test_safe_read_clean_csv(sample_clean_csv_bytes):
    df, report = safe_read_csv(sample_clean_csv_bytes, "clean.csv")
    assert len(df) == 5
    assert len(df.columns) == 4
    assert list(df.columns) == ["age", "gender", "income", "target"]
    assert report.total_raw_rows == 5

def test_safe_read_messy_csv(sample_messy_csv_bytes):
    df, report = safe_read_csv(sample_messy_csv_bytes, "messy.csv")
    # Redundant "Unnamed: 0" was dropped
    assert "Unnamed: 0" not in df.columns
    # Duplicate column "age" was renamed safely to "age_1"
    assert "age" in df.columns
    assert "age_1" in df.columns
    # Verify report logged the actions
    action_types = [a.action_type for a in report.actions]
    assert "unnamed_index" in action_types
    assert "duplicate_or_empty_columns" in action_types

def test_safe_read_semicolon_csv(sample_semicolon_csv_bytes):
    df, report = safe_read_csv(sample_semicolon_csv_bytes, "cities.csv")
    assert len(df) == 3
    assert len(df.columns) == 3
    assert report.delimiter_used == ";"

def test_empty_csv_raises_error():
    with pytest.raises(EmptyDatasetError):
        safe_read_csv(b"", "empty.csv")
