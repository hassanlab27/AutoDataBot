import math
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, UploadFile, File, Query, status
import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.core.config import settings
from app.core.errors import (
    InvalidFileTypeError,
    FileTooLargeError,
    DatasetNotFoundError,
    InvalidCSVError
)
from app.core.logging import logger
from app.ingestion.reader import safe_read_csv
from app.storage.dataset_store import dataset_store
from app.validation.schema_inspector import inspect_dataset_structure, DatasetStructureSummary
from app.analysis.quality_scorer import detect_suspicious_columns, calculate_quality_score, DatasetQualityReport

router = APIRouter(prefix="/datasets", tags=["Datasets"])

class UploadResponse(BaseModel):
    dataset_id: str
    original_filename: str
    rows_count: int
    columns_count: int
    summary: DatasetStructureSummary
    quality: DatasetQualityReport
    ingestion: Dict[str, Any]

class PreviewResponse(BaseModel):
    dataset_id: str
    total_rows: int
    returned_rows: int
    columns: List[str]
    inferred_dtypes: Dict[str, str]
    rows: List[Dict[str, Any]]

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset, validate, store original bytes, and compute initial audit."""
    filename = file.filename or "uploaded.csv"
    
    # 1. File Type Check
    if not filename.lower().endswith(".csv"):
        raise InvalidFileTypeError(filename=filename, allowed=".csv")

    # 2. Read File Bytes with Size Check
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise FileTooLargeError(max_size_mb=settings.MAX_UPLOAD_SIZE_MB)

    logger.info(f"Received CSV upload: '{filename}' ({file_size_mb:.2f} MB)")

    # 3. Ingestion & CSV Parsing
    df, ingestion_report = safe_read_csv(content, filename)

    # 4. Store Raw File & Metadata
    dataset_id = dataset_store.save_dataset(
        raw_bytes=content,
        original_filename=filename,
        df=df,
        ingestion_report=ingestion_report
    )

    # 5. Validation & Schema Inspection
    summary = inspect_dataset_structure(df)

    # 6. Quality Analysis
    suspicious = detect_suspicious_columns(summary)
    quality = calculate_quality_score(summary, suspicious)

    return UploadResponse(
        dataset_id=dataset_id,
        original_filename=filename,
        rows_count=summary.row_count,
        columns_count=summary.column_count,
        summary=summary,
        quality=quality,
        ingestion=ingestion_report.model_dump()
    )


@router.get("/{dataset_id}/summary", response_model=DatasetStructureSummary)
def get_dataset_summary(dataset_id: str):
    """Retrieve full structural, dimension, and column-level inspection of a dataset."""
    df = dataset_store.load_dataset(dataset_id)
    return inspect_dataset_structure(df)


@router.get("/{dataset_id}/quality", response_model=DatasetQualityReport)
def get_dataset_quality(dataset_id: str):
    """Retrieve data quality score, checklist, and suspicious column detections."""
    df = dataset_store.load_dataset(dataset_id)
    summary = inspect_dataset_structure(df)
    suspicious = detect_suspicious_columns(summary)
    return calculate_quality_score(summary, suspicious)


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
def get_dataset_preview(
    dataset_id: str,
    limit: int = Query(settings.DEFAULT_PREVIEW_ROWS, ge=1, le=settings.MAX_PREVIEW_ROWS)
):
    """Retrieve the first N rows of a dataset safely formatted for UI table display."""
    df = dataset_store.load_dataset(dataset_id)
    summary = inspect_dataset_structure(df)
    
    inferred_dtypes = {col.name: col.inferred_dtype for col in summary.columns}
    
    preview_df = df.head(limit).copy()
    
    # Clean NaN/inf for strict JSON compliance
    clean_rows = []
    for record in preview_df.to_dict(orient="records"):
        clean_record = {}
        for k, v in record.items():
            if pd.isna(v) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                clean_record[k] = None
            elif isinstance(v, (np.integer, int)):
                clean_record[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_record[k] = round(float(v), 6)
            elif isinstance(v, (np.bool_, bool)):
                clean_record[k] = bool(v)
            else:
                clean_record[k] = str(v)
        clean_rows.append(clean_record)

    return PreviewResponse(
        dataset_id=dataset_id,
        total_rows=len(df),
        returned_rows=len(clean_rows),
        columns=list(df.columns),
        inferred_dtypes=inferred_dtypes,
        rows=clean_rows
    )
