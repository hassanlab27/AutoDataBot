import csv
import io
from typing import Tuple
import charset_normalizer
import pandas as pd
from app.core.errors import EmptyDatasetError, InvalidCSVError
from app.core.logging import logger
from app.ingestion.repair_log import IngestionReport

def detect_encoding(file_bytes: bytes) -> str:
    """Detect text encoding using sample bytes."""
    # Fast path: test utf-8 / utf-8-sig
    sample = file_bytes[:65536]
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        sample.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        pass

    # Use charset_normalizer
    match = charset_normalizer.from_bytes(sample).best()
    if match and match.encoding:
        return match.encoding

    # Safe fallback
    return "latin-1"


def detect_delimiter(sample_text: str) -> str:
    """Sniff delimiter from sample text, falling back to frequency check."""
    common_delimiters = [",", ";", "\t", "|"]
    
    # Try Python's csv.Sniffer
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text[:10000], delimiters=";,|\t,")
        if dialect and dialect.delimiter in common_delimiters:
            return dialect.delimiter
    except Exception:
        pass

    # Frequency-based fallback across lines
    lines = [line for line in sample_text[:10000].splitlines() if line.strip()]
    if lines:
        counts = {d: sum(line.count(d) for line in lines[:5]) for d in common_delimiters}
        best_delimiter = max(counts, key=counts.get)
        if counts[best_delimiter] > 0:
            return best_delimiter

    return ","


def safe_read_csv(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, IngestionReport]:
    """
    Safely ingest and validate CSV file bytes into a Pandas DataFrame.
    Preserves raw data integrity while normalizing columns and logging all repairs.
    """
    file_size = len(file_bytes)
    if file_size == 0:
        raise EmptyDatasetError(filename=filename)

    # 1. Detect Encoding
    encoding = detect_encoding(file_bytes)
    try:
        text_content = file_bytes.decode(encoding)
    except Exception as e:
        logger.warning(f"Failed to decode with {encoding}, falling back to latin-1: {e}")
        encoding = "latin-1"
        text_content = file_bytes.decode(encoding, errors="replace")

    # 2. Detect Delimiter
    delimiter = detect_delimiter(text_content[:20000])

    report = IngestionReport(
        original_filename=filename,
        file_size_bytes=file_size,
        encoding_used=encoding,
        delimiter_used=delimiter,
        total_raw_rows=0,
        total_raw_columns=0
    )

    report.add_action(
        action_type="encoding_detection",
        description=f"Detected file encoding as '{encoding}'.",
        details={"encoding": encoding}
    )

    report.add_action(
        action_type="delimiter_detection",
        description=f"Detected column delimiter as '{delimiter}'.",
        details={"delimiter": delimiter}
    )

    # 3. Read DataFrame using Pandas
    string_buffer = io.StringIO(text_content)
    try:
        # Read with on_bad_lines='skip' to catch corrupted rows gracefully
        df = pd.read_csv(
            string_buffer,
            sep=delimiter,
            encoding=encoding,
            on_bad_lines="skip",
            low_memory=False
        )
    except pd.errors.EmptyDataError:
        raise EmptyDatasetError(filename=filename)
    except Exception as e:
        raise InvalidCSVError(
            message=f"Failed to parse CSV: {str(e)}",
            details={"filename": filename, "error": str(e)}
        )

    if df.empty and len(df.columns) == 0:
        raise EmptyDatasetError(filename=filename)

    report.total_raw_rows = len(df)
    report.total_raw_columns = len(df.columns)

    # 4. Handle Unnamed Index Column (e.g. "Unnamed: 0" from index export)
    cols_to_drop = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.lower() in ("unnamed: 0", "index"):
            series = df[col]
            # If values are sequential numbers (0, 1, 2, ...), it's an exported index
            if pd.api.types.is_numeric_dtype(series):
                sample_s = series.dropna().head(100)
                if len(sample_s) > 0 and (sample_s == sample_s.astype(int)).all():
                    diffs = sample_s.diff().dropna()
                    if (diffs == 1).all() or len(sample_s) == 1:
                        cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        for col in cols_to_drop:
            report.add_action(
                action_type="unnamed_index",
                description=f"Dropped redundant sequential index column '{col}'.",
                details={"column": str(col)}
            )

    # 5. Normalize Column Names & Handle Duplicates Safely
    new_columns = []
    seen_counts = {}
    renamed_map = {}

    import re
    for i, col in enumerate(df.columns):
        col_name = str(col).strip()
        if not col_name:
            col_name = f"column_{i+1}"
            renamed_map[str(col)] = col_name

        # Standardize pandas .1, .2 duplicates to _1, _2
        match = re.match(r"^(.*)\.(\d+)$", col_name)
        if match:
            base_name, suffix = match.groups()
            col_name = f"{base_name}_{suffix}"
            renamed_map[str(col)] = col_name

        # Sanitize whitespace and control chars
        col_clean = " ".join(col_name.split())
        
        # Check duplicate
        if col_clean in seen_counts:
            seen_counts[col_clean] += 1
            unique_name = f"{col_clean}_{seen_counts[col_clean]}"
            new_columns.append(unique_name)
            renamed_map[col_clean] = unique_name
        else:
            seen_counts[col_clean] = 0
            new_columns.append(col_clean)

    if renamed_map:
        report.add_action(
            action_type="duplicate_or_empty_columns",
            description=f"Renamed {len(renamed_map)} duplicate, empty, or unformatted column headers.",
            details={"renamed": renamed_map}
        )

    df.columns = new_columns

    if len(df.columns) == 0:
        raise InvalidCSVError("Dataset has no usable columns after normalization.")

    return df, report
