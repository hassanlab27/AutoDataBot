from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Body, status
from pydantic import BaseModel, Field

from app.preprocessing.preprocessing_service import preprocessing_service

router = APIRouter(prefix="/datasets/{dataset_id}/preprocessing", tags=["Preprocessing"])

class TargetInspectRequest(BaseModel):
    target: str = Field(..., description="Target column name to validate and inspect")

class FeaturesInspectRequest(BaseModel):
    target: str = Field(..., description="Target column name")

class PrepareDatasetRequest(BaseModel):
    target: str = Field(..., description="Target column name")
    problem_type: str = Field(..., description="Target problem type (binary_classification, multiclass_classification, regression)")
    selected_features: Optional[List[str]] = Field(None, description="List of feature column names to include")
    excluded_features: Optional[List[str]] = Field(None, description="List of feature column names to exclude")
    test_size: float = Field(0.20, ge=0.05, le=0.50, description="Test split proportion (default 0.20)")
    random_state: int = Field(42, description="Random seed for reproducibility")
    numeric_imputation: str = Field("median", description="Strategy for numeric missing values (median, mean)")
    categorical_imputation: str = Field("most_frequent", description="Strategy for categorical missing values (most_frequent)")
    categorical_encoding: str = Field("one_hot", description="Strategy for categorical features (one_hot)")
    scaling: str = Field("standard", description="Numerical feature scaling strategy (standard, none)")
    extract_datetime: bool = Field(True, description="Whether to extract calendar components from datetime features")

@router.post("/target-inspect")
def inspect_target(dataset_id: str, req: TargetInspectRequest = Body(...)):
    """Validate target, determine problem type conservatively, and report diagnostics."""
    return preprocessing_service.inspect_target(dataset_id, req.target)

@router.post("/features-inspect")
def audit_features(dataset_id: str, req: FeaturesInspectRequest = Body(...)):
    """Audit candidate features, flag suspicious columns (ID, constant, text), and check leakage."""
    return preprocessing_service.audit_features(dataset_id, req.target)

@router.post("/prepare")
def prepare_dataset(dataset_id: str, req: PrepareDatasetRequest = Body(...)):
    """Execute leakage-free train/test partition, pipeline fitting, and serialization."""
    return preprocessing_service.prepare_dataset(
        dataset_id=dataset_id,
        target=req.target,
        problem_type=req.problem_type,
        selected_features=req.selected_features,
        excluded_features=req.excluded_features,
        test_size=req.test_size,
        random_state=req.random_state,
        numeric_imputation=req.numeric_imputation,
        categorical_imputation=req.categorical_imputation,
        categorical_encoding=req.categorical_encoding,
        scaling=req.scaling,
        extract_datetime=req.extract_datetime
    )

@router.get("/summary")
def get_preprocessing_summary(dataset_id: str):
    """Retrieve saved preprocessing metadata, train/test counts, and generated feature names."""
    return preprocessing_service.get_summary(dataset_id)

@router.get("/preview")
def get_preprocessing_preview(
    dataset_id: str,
    limit: int = Query(20, ge=1, le=100, description="Number of preview rows")
):
    """Retrieve limited preview of transformed training features."""
    return preprocessing_service.get_preview(dataset_id, limit=limit)
