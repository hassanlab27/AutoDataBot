from typing import Any, Dict, Optional
from fastapi import HTTPException, status
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

class AutoDataBotException(Exception):
    def __init__(
        self,
        error: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error = error
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

class DatasetNotFoundError(AutoDataBotException):
    def __init__(self, dataset_id: str):
        super().__init__(
            error="DATASET_NOT_FOUND",
            message=f"Dataset with ID '{dataset_id}' was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"dataset_id": dataset_id}
        )

class InvalidFileTypeError(AutoDataBotException):
    def __init__(self, filename: str, allowed: str = ".csv"):
        super().__init__(
            error="INVALID_FILE_TYPE",
            message=f"File '{filename}' is not supported. Only {allowed} files are allowed.",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            details={"filename": filename, "allowed": allowed}
        )

class FileTooLargeError(AutoDataBotException):
    def __init__(self, max_size_mb: int):
        super().__init__(
            error="FILE_TOO_LARGE",
            message=f"The uploaded file exceeds the maximum limit of {max_size_mb} MB.",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details={"max_size_mb": max_size_mb}
        )

class InvalidCSVError(AutoDataBotException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error="INVALID_CSV",
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )

class EmptyDatasetError(AutoDataBotException):
    def __init__(self, filename: str):
        super().__init__(
            error="EMPTY_DATASET",
            message=f"The file '{filename}' is empty and contains no data rows.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"filename": filename}
        )

class ColumnNotFoundError(AutoDataBotException):
    def __init__(self, column: str, available_columns: Optional[list] = None):
        super().__init__(
            error="COLUMN_NOT_FOUND",
            message=f"Column '{column}' does not exist in this dataset.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"column": column, "available_columns": available_columns or []}
        )

class UnsupportedChartError(AutoDataBotException):
    def __init__(self, message: str = "This combination of columns cannot be visualized using the requested chart.", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error="UNSUPPORTED_CHART",
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )
