import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.ml.config import AutoMLConfig
from app.ml.training_service import training_service
from app.ml.job_runner import job_runner
from app.core.errors import AutoDataBotError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["AutoML Runs"])

@router.post("", status_code=status.HTTP_201_CREATED)
def start_automl_run(config: AutoMLConfig) -> Dict[str, Any]:
    """
    Initializes and starts an asynchronous AutoML training run.
    """
    try:
        run_info = training_service.create_run(config)
        return {
            "success": True,
            "message": "AutoML training run initiated successfully",
            "data": run_info
        }
    except AutoDataBotError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Unexpected error starting AutoML run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training run: {str(e)}"
        )

@router.get("/{run_id}")
def get_run_details(run_id: str) -> Dict[str, Any]:
    """
    Retrieves full details of an AutoML run including status, config, leaderboard, and metrics.
    """
    run_data = training_service.get_run(run_id)
    if not run_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found"
        )
    return {
        "success": True,
        "data": run_data
    }

@router.get("/{run_id}/status")
def get_run_status(run_id: str) -> Dict[str, Any]:
    """
    Lightweight polling endpoint for live status updates.
    """
    run_status = job_runner.get_status(run_id)
    if not run_status:
        # Check if run exists on disk
        run_data = training_service.get_run(run_id)
        if not run_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )
        return {
            "success": True,
            "data": run_data["status"]
        }
    return {
        "success": True,
        "data": run_status.model_dump()
    }

@router.get("/{run_id}/leaderboard")
def get_run_leaderboard(run_id: str) -> Dict[str, Any]:
    """
    Returns the current leaderboard for an AutoML run.
    """
    run_data = training_service.get_run(run_id)
    if not run_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found"
        )
    return {
        "success": True,
        "run_id": run_id,
        "leaderboard": run_data.get("leaderboard", [])
    }

@router.post("/{run_id}/cancel")
def cancel_automl_run(run_id: str) -> Dict[str, Any]:
    """
    Cooperatively requests cancellation of an active training run.
    """
    success = job_runner.request_cancellation(run_id)
    if not success:
        run_status = job_runner.get_status(run_id)
        current_state = run_status.status if run_status else "unknown"
        return {
            "success": False,
            "message": f"Run '{run_id}' cannot be cancelled (current state: {current_state})"
        }
    return {
        "success": True,
        "message": f"Cancellation requested for run '{run_id}'"
    }

@router.get("/dataset/{dataset_id}")
def list_dataset_runs(dataset_id: str) -> Dict[str, Any]:
    """
    Lists all past and present training runs for a specific dataset.
    """
    runs = training_service.list_runs_for_dataset(dataset_id)
    return {
        "success": True,
        "dataset_id": dataset_id,
        "runs": runs
    }
