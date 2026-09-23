import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from app.ml.metrics import compute_metrics
from app.core.logging import logger

class XGBModelWrapper:
    """Wraps XGBoost estimator to automatically map integer classes back to original labels."""
    def __init__(self, raw_model, class_mapping=None):
        self.raw_model = raw_model
        self.class_mapping = class_mapping
        self.inv_mapping = {v: k for k, v in class_mapping.items()} if class_mapping else None

    def predict(self, X):
        preds = np.asarray(self.raw_model.predict(X)).ravel()
        if self.inv_mapping is not None:
            return np.array([self.inv_mapping.get(v, v) for v in preds])
        return preds

    def predict_proba(self, X):
        return self.raw_model.predict_proba(X)

    def __getattr__(self, name):
        if name.startswith("__") or "raw_model" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.raw_model, name)

class CatBoostModelWrapper:
    """Wraps CatBoost estimator to guarantee 1D prediction output."""
    def __init__(self, raw_model):
        self.raw_model = raw_model

    def predict(self, X):
        return np.asarray(self.raw_model.predict(X)).ravel()

    def predict_proba(self, X):
        return self.raw_model.predict_proba(X)

    def __getattr__(self, name):
        if name.startswith("__") or "raw_model" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.raw_model, name)

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
        val_metrics = compute_metrics(
            problem_type,
            y_val,
            y_val_pred,
            y_val_prob,
            partition="validation",
            model_name="LightGBM"
        )
        train_metrics = compute_metrics(
            problem_type,
            y_train,
            model.predict(X_train),
            partition="train",
            model_name="LightGBM"
        )

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

        estimator_wrapped = XGBModelWrapper(model, class_mapping) if class_mapping else model
        y_val_pred = estimator_wrapped.predict(X_val)
        y_val_prob = model.predict_proba(X_val) if is_classification else None
        val_metrics = compute_metrics(
            problem_type,
            y_val,
            y_val_pred,
            y_val_prob,
            partition="validation",
            model_name="XGBoost"
        )
        train_metrics = compute_metrics(
            problem_type,
            y_train,
            estimator_wrapped.predict(X_train),
            partition="train",
            model_name="XGBoost"
        )

        results.append({
            "model_name": "XGBoost",
            "engine": "XGBoost",
            "estimator": estimator_wrapped,
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

        estimator_wrapped = CatBoostModelWrapper(model)
        y_val_pred = estimator_wrapped.predict(X_val)
        y_val_prob = model.predict_proba(X_val) if is_classification else None
        val_metrics = compute_metrics(
            problem_type,
            y_val,
            y_val_pred,
            y_val_prob,
            partition="validation",
            model_name="CatBoost"
        )
        train_metrics = compute_metrics(
            problem_type,
            y_train,
            estimator_wrapped.predict(X_train),
            partition="train",
            model_name="CatBoost"
        )

        results.append({
            "model_name": "CatBoost",
            "engine": "CatBoost",
            "estimator": estimator_wrapped,
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

