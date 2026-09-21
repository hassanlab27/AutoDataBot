import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from app.ml.metrics import compute_metrics
from app.core.logging import logger

def train_gradient_boosting_models(
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    max_cpus: int = 2
) -> List[Dict[str, Any]]:
    """
    Trains LightGBM, XGBoost, and CatBoost standalone models.
    Each engine runs inside its own try/except so one failure does not crash the others.
    """
    results: List[Dict[str, Any]] = []
    is_classification = "classification" in problem_type.lower()
    is_multiclass = problem_type.lower() == "multiclass_classification"

    # 1. LightGBM
    try:
        import lightgbm as lgb
        t0 = time.perf_counter()
        if is_classification:
            model = lgb.LGBMClassifier(
                random_state=random_state,
                n_jobs=max_cpus,
                verbose=-1
            )
        else:
            model = lgb.LGBMRegressor(
                random_state=random_state,
                n_jobs=max_cpus,
                verbose=-1
            )
        model.fit(X_train, y_train)
        train_time = round(time.perf_counter() - t0, 3)

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val) if is_classification else None
        val_metrics = compute_metrics(problem_type, y_val, y_val_pred, y_val_prob)
        train_metrics = compute_metrics(problem_type, y_train, model.predict(X_train))

        results.append({
            "model_name": "LightGBM",
            "engine": "LightGBM",
            "estimator": model,
            "problem_type": problem_type,
            "training_time_seconds": train_time,
            "validation_method": "Holdout Validation",
            "validation_metrics": val_metrics,
            "train_metrics": train_metrics,
            "status": "completed",
            "error": None
        })
        logger.info(f"Trained LightGBM model in {train_time}s")
    except Exception as e:
        logger.warning(f"LightGBM model training failed or skipped: {str(e)}")
        results.append({
            "model_name": "LightGBM",
            "engine": "LightGBM",
            "estimator": None,
            "problem_type": problem_type,
            "training_time_seconds": 0.0,
            "validation_method": "Holdout Validation",
            "validation_metrics": {},
            "train_metrics": {},
            "status": "failed",
            "error": str(e)
        })

    # 2. XGBoost
    try:
        import xgboost as xgb
        t0 = time.perf_counter()
        
        # XGBoost requires class labels to be integers 0..k-1 for multiclass
        y_tr_xgb = y_train
        y_val_xgb = y_val
        class_mapping = None
        if is_classification:
            unique_classes = np.unique(y_train)
            if not np.issubdtype(unique_classes.dtype, np.integer) or unique_classes.min() != 0:
                class_mapping = {val: idx for idx, val in enumerate(unique_classes)}
                y_tr_xgb = np.array([class_mapping[v] for v in y_train])
                y_val_xgb = np.array([class_mapping.get(v, 0) for v in y_val])

        if is_classification:
            model = xgb.XGBClassifier(
                random_state=random_state,
                n_jobs=max_cpus,
                eval_metric="logloss" if not is_multiclass else "mlogloss",
                verbosity=0
            )
        else:
            model = xgb.XGBRegressor(
                random_state=random_state,
                n_jobs=max_cpus,
                eval_metric="rmse",
                verbosity=0
            )
        
        model.fit(X_train, y_tr_xgb)
        train_time = round(time.perf_counter() - t0, 3)

        raw_pred = model.predict(X_val)
        if class_mapping is not None:
            inv_map = {idx: val for val, idx in class_mapping.items()}
            y_val_pred = np.array([inv_map[v] for v in raw_pred])
        else:
            y_val_pred = raw_pred

        y_val_prob = model.predict_proba(X_val) if is_classification else None
        val_metrics = compute_metrics(problem_type, y_val, y_val_pred, y_val_prob)
        train_metrics = compute_metrics(problem_type, y_tr_xgb, model.predict(X_train))

        results.append({
            "model_name": "XGBoost",
            "engine": "XGBoost",
            "estimator": model,
            "problem_type": problem_type,
            "training_time_seconds": train_time,
            "validation_method": "Holdout Validation",
            "validation_metrics": val_metrics,
            "train_metrics": train_metrics,
            "status": "completed",
            "error": None
        })
        logger.info(f"Trained XGBoost model in {train_time}s")
    except Exception as e:
        logger.warning(f"XGBoost model training failed or skipped: {str(e)}")
        results.append({
            "model_name": "XGBoost",
            "engine": "XGBoost",
            "estimator": None,
            "problem_type": problem_type,
            "training_time_seconds": 0.0,
            "validation_method": "Holdout Validation",
            "validation_metrics": {},
            "train_metrics": {},
            "status": "failed",
            "error": str(e)
        })

    # 3. CatBoost
    try:
        import catboost as cb
        t0 = time.perf_counter()
        if is_classification:
            model = cb.CatBoostClassifier(
                random_seed=random_state,
                thread_count=max_cpus,
                verbose=False
            )
        else:
            model = cb.CatBoostRegressor(
                random_seed=random_state,
                thread_count=max_cpus,
                verbose=False
            )
        model.fit(X_train, y_train)
        train_time = round(time.perf_counter() - t0, 3)

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val) if is_classification else None
        val_metrics = compute_metrics(problem_type, y_val, y_val_pred, y_val_prob)
        train_metrics = compute_metrics(problem_type, y_train, model.predict(X_train))

        results.append({
            "model_name": "CatBoost",
            "engine": "CatBoost",
            "estimator": model,
            "problem_type": problem_type,
            "training_time_seconds": train_time,
            "validation_method": "Holdout Validation",
            "validation_metrics": val_metrics,
            "train_metrics": train_metrics,
            "status": "completed",
            "error": None
        })
        logger.info(f"Trained CatBoost model in {train_time}s")
    except Exception as e:
        logger.warning(f"CatBoost model training failed or skipped: {str(e)}")
        results.append({
            "model_name": "CatBoost",
            "engine": "CatBoost",
            "estimator": None,
            "problem_type": problem_type,
            "training_time_seconds": 0.0,
            "validation_method": "Holdout Validation",
            "validation_metrics": {},
            "train_metrics": {},
            "status": "failed",
            "error": str(e)
        })

    return results
