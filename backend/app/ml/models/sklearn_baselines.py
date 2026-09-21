import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

from app.ml.metrics import compute_metrics
from app.core.logging import logger

def train_sklearn_baselines(
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    max_cpus: int = 2
) -> List[Dict[str, Any]]:
    """
    Trains standard scikit-learn baselines including naive predictors.
    Returns list of standardized model evaluation results.
    """
    results: List[Dict[str, Any]] = []
    is_classification = "classification" in problem_type.lower()

    if is_classification:
        candidates = [
            ("NaiveBaseline (Majority Class)", DummyClassifier(strategy="most_frequent")),
            ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=max_cpus)),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=max_cpus)),
            ("HistGradientBoostingClassifier", HistGradientBoostingClassifier(random_state=random_state))
        ]
    else:
        candidates = [
            ("NaiveBaseline (Mean Predictor)", DummyRegressor(strategy="mean")),
            ("LinearRegression", LinearRegression(n_jobs=max_cpus)),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=max_cpus)),
            ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(random_state=random_state))
        ]

    for model_name, estimator in candidates:
        t0 = time.perf_counter()
        try:
            estimator.fit(X_train, y_train)
            train_time = round(time.perf_counter() - t0, 3)

            # Predictions on Validation Set
            y_val_pred = estimator.predict(X_val)
            y_val_prob = None
            if is_classification and hasattr(estimator, "predict_proba"):
                try:
                    y_val_prob = estimator.predict_proba(X_val)
                except Exception:
                    pass

            val_metrics = compute_metrics(problem_type, y_val, y_val_pred, y_val_prob)

            # Predictions on Training Set (for overfitting diagnostics)
            y_train_pred = estimator.predict(X_train)
            train_metrics = compute_metrics(problem_type, y_train, y_train_pred)

            results.append({
                "model_name": model_name,
                "engine": "sklearn",
                "estimator": estimator,
                "problem_type": problem_type,
                "training_time_seconds": train_time,
                "validation_method": "Holdout Validation",
                "validation_metrics": val_metrics,
                "train_metrics": train_metrics,
                "status": "completed",
                "error": None
            })
            logger.info(f"Trained sklearn baseline '{model_name}' in {train_time}s")
        except Exception as e:
            train_time = round(time.perf_counter() - t0, 3)
            logger.warning(f"Failed to train sklearn baseline '{model_name}': {str(e)}")
            results.append({
                "model_name": model_name,
                "engine": "sklearn",
                "estimator": None,
                "problem_type": problem_type,
                "training_time_seconds": train_time,
                "validation_method": "Holdout Validation",
                "validation_metrics": {},
                "train_metrics": {},
                "status": "failed",
                "error": str(e)
            })

    return results
