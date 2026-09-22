import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.explainability.shap_service import explainability_service
from app.core.errors import AutoDataBotError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs/{run_id}/explainability", tags=["Model Explainability"])

@router.post("/shap")
def start_shap_job(
    run_id: str,
    sample_size: int = Query(500, ge=10, le=2000),
    max_features: int = Query(20, ge=5, le=50)
) -> Dict[str, Any]:
    """
    Initiates asynchronous SHAP calculation in the background.
    """
    try:
        res = explainability_service.start_shap_computation(
            run_id=run_id,
            sample_size=sample_size,
            max_features=max_features
        )
        return {
            "success": True,
            "data": res
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error starting SHAP job for run {run_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/shap/status")
def get_shap_job_status(run_id: str) -> Dict[str, Any]:
    """
    Checks the status of background SHAP computation.
    """
    try:
        res = explainability_service.get_shap_status(run_id)
        return {
            "success": True,
            "data": res
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/shap")
def get_shap_summary(
    run_id: str,
    sample_size: int = Query(500, ge=10, le=2000),
    max_features: int = Query(20, ge=5, le=50)
) -> Dict[str, Any]:
    """
    Retrieves the full SHAP summary, including global importance and summary plot points.
    """
    try:
        res = explainability_service.get_shap_summary(
            run_id=run_id,
            sample_size=sample_size,
            max_features=max_features
        )
        return {
            "success": True,
            "data": res
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting SHAP summary for {run_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/global")
def get_global_shap_importance(run_id: str) -> Dict[str, Any]:
    """
    Retrieves global mean absolute SHAP values per feature.
    """
    try:
        summary = explainability_service.get_shap_summary(run_id)
        if not summary.get("available"):
            return {
                "success": True,
                "data": {
                    "available": False,
                    "reason": summary.get("reason"),
                    "global_importance": []
                }
            }
        return {
            "success": True,
            "data": {
                "available": True,
                "global_importance": summary.get("global_importance", []),
                "disclaimer": summary.get("disclaimer")
            }
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/local/{prediction_id}")
def get_local_prediction_explanation(run_id: str, prediction_id: int) -> Dict[str, Any]:
    """
    Explains an individual prediction from the test set with local feature contributions.
    """
    try:
        res = explainability_service.get_local_prediction_explanation(run_id, prediction_id)
        return {
            "success": True,
            "data": res
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting local explanation for {run_id}, pred {prediction_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/dependence")
def get_feature_dependence(run_id: str, feature: str = Query(..., description="Feature name to inspect")) -> Dict[str, Any]:
    """
    Returns feature value vs SHAP value scatter points for a specific feature.
    """
    try:
        res = explainability_service.get_feature_dependence(run_id, feature)
        return {
            "success": True,
            "data": res
        }
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
