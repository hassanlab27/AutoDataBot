import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from app.ml.metrics import compute_metrics
from app.core.logging import logger

def train_flaml_engine(
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    time_limit: int = 60,
    random_state: int = 42,
    eval_metric: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Path B: Executes FLAML AutoML on Phase 3 preprocessed feature matrices.
    Acts as the lightweight secondary AutoML engine and fallback.
    """
    try:
        from flaml import AutoML
    except ImportError as e:
        logger.warning(f"FLAML is not installed: {str(e)}")
        return [{
            "model_name": "FLAML",
            "engine": "FLAML",
            "estimator": None,
            "problem_type": problem_type,
            "training_time_seconds": 0.0,
            "validation_method": "Unavailable",
            "validation_metrics": {},
            "train_metrics": {},
            "status": "failed",
            "error": "FLAML is not installed in the current environment."
        }]

    # Map problem_type and metric to FLAML expected values
    flaml_task = "classification" if "classification" in problem_type.lower() else "regression"
    flaml_metric = "auto"
    if eval_metric:
        if eval_metric in ("f1", "f1_macro", "accuracy", "roc_auc", "rmse", "mae", "r2"):
            flaml_metric = eval_metric

    automl = AutoML()
    settings = {
        "time_limit": max(5, time_limit),
        "metric": flaml_metric,
        "task": flaml_task,
        "seed": random_state,
        "verbose": 0
    }

    t0 = time.perf_counter()
    try:
        automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **settings)
        train_time = round(time.perf_counter() - t0, 3)

        best_estimator_name = automl.best_estimator
        best_model = automl.model.estimator if hasattr(automl, "model") and hasattr(automl.model, "estimator") else automl

        y_val_pred = automl.predict(X_val)
        y_val_prob = None
        if flaml_task == "classification" and hasattr(automl, "predict_proba"):
            try:
                y_val_prob = automl.predict_proba(X_val)
            except Exception:
                pass

        val_metrics = compute_metrics(problem_type, y_val, y_val_pred, y_val_prob)
        train_metrics = compute_metrics(problem_type, y_train, automl.predict(X_train))

        logger.info(f"FLAML finished search in {train_time}s; best model: {best_estimator_name}")
        return [{
            "model_name": f"FLAML_{best_estimator_name}",
            "engine": "FLAML",
            "estimator": best_model,
            "problem_type": problem_type,
            "training_time_seconds": train_time,
            "validation_method": "FLAML validation search",
            "validation_metrics": val_metrics,
            "train_metrics": train_metrics,
            "status": "completed",
            "error": None
        }]

    except Exception as e:
        train_time = round(time.perf_counter() - t0, 3)
        logger.warning(f"FLAML AutoML training failed: {str(e)}")
        return [{
            "model_name": "FLAML",
            "engine": "FLAML",
            "estimator": None,
            "problem_type": problem_type,
            "training_time_seconds": train_time,
            "validation_method": "FLAML validation search",
            "validation_metrics": {},
            "train_metrics": {},
            "status": "failed",
            "error": str(e)
        }]
