import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from app.core.config import settings
from app.core.errors import DatasetNotFoundError
from app.core.logging import logger
from app.ingestion.repair_log import IngestionReport

class DatasetStore:
    """Manages local storage for uploaded datasets and run metadata."""

    def __init__(self, upload_dir: Path = settings.UPLOAD_DIR):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:8]
        return f"ds_{timestamp}_{unique_token}"

    def _sanitize_id(self, dataset_id: str) -> str:
        # Prevent path traversal: only allow alphanumeric, underscores
        clean_id = "".join(c for c in dataset_id if c.isalnum() or c in "_-")
        if not clean_id or clean_id != dataset_id:
            raise DatasetNotFoundError(dataset_id)
        return clean_id

    def save_dataset(
        self,
        raw_bytes: bytes,
        original_filename: str,
        df: pd.DataFrame,
        ingestion_report: IngestionReport
    ) -> str:
        dataset_id = self.generate_dataset_id()
        raw_path = self.upload_dir / f"{dataset_id}.csv"
        meta_path = self.upload_dir / f"{dataset_id}_meta.json"

        # 1. Save original raw bytes untouched
        with open(raw_path, "wb") as f:
            f.write(raw_bytes)

        # 2. Save metadata & ingestion report
        metadata = {
            "dataset_id": dataset_id,
            "original_filename": original_filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "file_size_bytes": len(raw_bytes),
            "rows_count": len(df),
            "columns_count": len(df.columns),
            "columns": list(df.columns),
            "ingestion_report": ingestion_report.model_dump()
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset stored successfully with ID: {dataset_id} ({len(df)} rows, {len(df.columns)} cols)")
        return dataset_id

    def get_dataset_path(self, dataset_id: str) -> Path:
        clean_id = self._sanitize_id(dataset_id)
        path = self.upload_dir / f"{clean_id}.csv"
        if not path.is_file():
            raise DatasetNotFoundError(dataset_id)
        return path

    def get_metadata(self, dataset_id: str) -> Dict[str, Any]:
        clean_id = self._sanitize_id(dataset_id)
        meta_path = self.upload_dir / f"{clean_id}_meta.json"
        if not meta_path.is_file():
            raise DatasetNotFoundError(dataset_id)
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        path = self.get_dataset_path(dataset_id)
        meta = self.get_metadata(dataset_id)
        encoding = meta.get("ingestion_report", {}).get("encoding_used", "utf-8")
        delimiter = meta.get("ingestion_report", {}).get("delimiter_used", ",")

        try:
            return pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)
        except Exception:
            # Fallback to standard read
            return pd.read_csv(path, low_memory=False)

dataset_store = DatasetStore()
