import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.evaluation.service import evaluation_service
from app.explainability.shap_service import explainability_service
from app.core.errors import AutoDataBotError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs/{run_id}/evaluation", tags=["Model Evaluation"])

@router.get("")
def get_evaluation_overview(run_id: str, refresh: bool = Query(False)) -> Dict[str, Any]:
    """
    Retrieves complete model evaluation, including metrics, confusion matrix/residuals,
    generalization diagnostics, error analysis, and comparison.
    """
    try:
        data = evaluation_service.get_full_evaluation(run_id, force_refresh=refresh)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error fetching evaluation for run {run_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate evaluation: {str(e)}")

@router.get("/classification")
def get_classification_metrics(run_id: str) -> Dict[str, Any]:
    """
    Retrieves classification-specific evaluation metrics, confusion matrix, ROC and PR curves.
    """
    try:
        data = evaluation_service.get_classification_evaluation(run_id)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/regression")
def get_regression_metrics(run_id: str) -> Dict[str, Any]:
    """
    Retrieves regression-specific evaluation metrics, actual vs predicted, and residual distributions.
    """
    try:
        data = evaluation_service.get_regression_evaluation(run_id)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/errors")
def get_error_analysis(run_id: str) -> Dict[str, Any]:
    """
    Retrieves detailed error analysis and top worst prediction records.
    """
    try:
        data = evaluation_service.get_error_analysis(run_id)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/calibration")
def get_calibration_curve(run_id: str) -> Dict[str, Any]:
    """
    Retrieves probability calibration curve points and Brier score.
    """
    try:
        data = evaluation_service.get_calibration(run_id)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/diagnostics")
def get_generalization_diagnostics(run_id: str) -> Dict[str, Any]:
    """
    Retrieves rule-based overfitting/underfitting generalization diagnostics.
    """
    try:
        data = evaluation_service.get_diagnostics(run_id)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/comparison")
def get_models_comparison(run_id: str) -> Dict[str, Any]:
    """
    Retrieves comparison data across all models trained in the run.
    """
    try:
        data = evaluation_service.get_comparison(run_id)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/importance")
def get_native_importance(run_id: str) -> Dict[str, Any]:
    """
    Returns native model feature importance.
    """
    try:
        data = explainability_service.get_native_importance(run_id)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/permutation")
def get_permutation_importance(run_id: str, repeats: int = Query(5, ge=1, le=20)) -> Dict[str, Any]:
    """
    Returns permutation feature importance computed on evaluation data.
    """
    try:
        data = explainability_service.get_permutation_importance(run_id, n_repeats=repeats)
        return {
            "success": True,
            "data": data
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
