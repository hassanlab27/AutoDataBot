from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Query, Body, status
from pydantic import BaseModel, Field

from app.analysis.eda_service import eda_service

router = APIRouter(prefix="/datasets/{dataset_id}/eda", tags=["EDA"])

class ChartRequest(BaseModel):
    x: str = Field(..., description="Name of the X-axis feature/column")
    y: Optional[str] = Field(None, description="Optional name of the Y-axis feature/column for bivariate plots")
    chart_type: Optional[str] = Field("auto", description="Optional requested chart type (auto, scatter, box, bar, line, heatmap)")

@router.get("/overview")
def get_eda_overview(dataset_id: str):
    """Retrieve high-level statistical and structural dataset overview."""
    return eda_service.get_overview(dataset_id)

@router.get("/numerical")
def get_numerical_analysis(dataset_id: str):
    """Retrieve descriptive statistics for all numerical columns."""
    return eda_service.get_numerical_analysis(dataset_id)

@router.get("/categorical")
def get_categorical_analysis(dataset_id: str):
    """Retrieve frequency and cardinality metrics for all categorical columns."""
    return eda_service.get_categorical_analysis(dataset_id)

@router.get("/missing")
def get_missing_analysis(dataset_id: str):
    """Retrieve missing value percentages, count, severe missingness flags, and chart payload."""
    return eda_service.get_missing_analysis(dataset_id)

@router.get("/correlation")
def get_correlation_analysis(dataset_id: str):
    """Retrieve Pearson correlation matrix, strong correlation pairs, and heatmap chart."""
    return eda_service.get_correlation_analysis(dataset_id)

@router.get("/outliers")
def get_outliers_summary(dataset_id: str):
    """Retrieve IQR outlier detection summary across all numerical variables."""
    return eda_service.get_outliers_summary(dataset_id)

@router.get("/distribution/{column_name}")
def get_column_distribution(dataset_id: str, column_name: str):
    """Retrieve detailed distribution metrics and histogram/box/bar chart for a specific column."""
    return eda_service.get_column_distribution(dataset_id, column_name)

@router.post("/chart")
def create_chart(dataset_id: str, req: ChartRequest = Body(...)):
    """Deterministically recommend and generate an interactive Plotly chart for X (and optional Y)."""
    return eda_service.generate_chart(
        dataset_id=dataset_id,
        x=req.x,
        y=req.y,
        chart_type=req.chart_type
    )

@router.get("/target-analysis")
def get_target_analysis(
    dataset_id: str,
    target: str = Query(..., description="Target column name")
):
    """Retrieve target-aware preliminary statistical associations and distributions."""
    return eda_service.get_target_analysis(dataset_id, target)
