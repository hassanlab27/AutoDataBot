import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from app.ml.metrics import compute_metrics, is_higher_better
from app.core.logging import logger

def train_autogluon_engine(
    train_data_raw: pd.DataFrame,
    target_col: str,
    problem_type: str,
    val_data_raw: Optional[pd.DataFrame],
    run_models_dir: Path,
    time_limit: int = 60,
    presets: str = "medium_quality",
    random_state: int = 42,
    eval_metric: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Path A: Executes AutoGluon TabularPredictor on validated RAW tabular data.
    AutoGluon applies its own native feature transformations, ensembling, and bagging.
    Returns list of candidate model results extracted from AutoGluon leaderboard.
    """
    results: List[Dict[str, Any]] = []
    ag_path = run_models_dir / "autogluon"
    ag_path.mkdir(parents=True, exist_ok=True)

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as e:
        logger.warning(f"AutoGluon is not installed: {str(e)}")
        return [{
            "model_name": "AutoGluon (Ensemble)",
            "engine": "AutoGluon",
            "estimator": None,
            "problem_type": problem_type,
            "training_time_seconds": 0.0,
            "validation_method": "Unavailable",
            "validation_metrics": {},
            "train_metrics": {},
            "status": "failed",
            "error": "AutoGluon is not installed in the current environment."
        }]

    # Map problem_type to AutoGluon expected format
    ag_problem_type = None
    if problem_type == "binary_classification":
        ag_problem_type = "binary"
    elif problem_type == "multiclass_classification":
        ag_problem_type = "multiclass"
    elif problem_type == "regression":
        ag_problem_type = "regression"

    # Map metric to AutoGluon metric string if needed
    ag_metric = eval_metric
    if eval_metric == "f1_macro":
        ag_metric = "f1_macro"
    elif eval_metric == "rmse":
        ag_metric = "root_mean_squared_error"
    elif eval_metric == "mae":
        ag_metric = "mean_absolute_error"

    t0 = time.perf_counter()
    try:
        predictor = TabularPredictor(
            label=target_col,
            problem_type=ag_problem_type,
            eval_metric=ag_metric,
            path=str(ag_path),
            verbosity=1
        )

        fit_kwargs = {
            "train_data": train_data_raw,
            "time_limit": time_limit,
            "presets": presets
        }
        if val_data_raw is not None:
            fit_kwargs["tuning_data"] = val_data_raw

        predictor.fit(**fit_kwargs)
        total_time = round(time.perf_counter() - t0, 3)

        # Retrieve Leaderboard
        if val_data_raw is not None:
            lb = predictor.leaderboard(data=val_data_raw, silent=True)
        else:
            lb = predictor.leaderboard(silent=True)
        logger.info(f"AutoGluon completed training {len(lb)} candidate models in {total_time}s")

        # Extract models from leaderboard
        for _, row in lb.iterrows():
            m_name = str(row["model"])
            score_val = float(row["score_val"]) if "score_val" in row and pd.notna(row["score_val"]) else 0.0
            fit_time = float(row["fit_time"]) if "fit_time" in row and pd.notna(row["fit_time"]) else 0.0

            # Compute detailed validation metrics if validation data is available
            val_metrics: Dict[str, float] = {}
            if val_data_raw is not None:
                try:
                    X_val_no_target = val_data_raw.drop(columns=[target_col])
                    y_val_true = val_data_raw[target_col].to_numpy()
                    y_val_pred = predictor.predict(X_val_no_target, model=m_name).to_numpy()
                    y_val_prob = None
                    if "classification" in problem_type:
                        try:
                            prob_df = predictor.predict_proba(X_val_no_target, model=m_name)
                            y_val_prob = prob_df.to_numpy()
                        except Exception:
                            pass
                    val_metrics = compute_metrics(problem_type, y_val_true, y_val_pred, y_val_prob)
                except Exception as eval_err:
                    logger.debug(f"Could not calculate detailed metrics for AutoGluon model {m_name}: {eval_err}")

            if not val_metrics:
                # Fallback to primary score reported by AutoGluon
                primary_key = eval_metric or ("f1" if "classification" in problem_type else "rmse")
                val_metrics[primary_key] = score_val

            results.append({
                "model_name": f"AutoGluon_{m_name}",
                "engine": "AutoGluon",
                "estimator": predictor,
                "submodel_name": m_name,
                "problem_type": problem_type,
                "training_time_seconds": round(fit_time, 3),
                "validation_method": "AutoGluon internal validation/bagging",
                "validation_metrics": val_metrics,
                "train_metrics": {},
                "status": "completed",
                "artifact_path": str(ag_path),
                "error": None
            })

    except Exception as e:
        total_time = round(time.perf_counter() - t0, 3)
        logger.warning(f"AutoGluon training encountered an issue: {str(e)}")
        results.append({
            "model_name": "AutoGluon",
            "engine": "AutoGluon",
            "estimator": None,
            "problem_type": problem_type,
            "training_time_seconds": total_time,
            "validation_method": "AutoGluon internal validation",
            "validation_metrics": {},
            "train_metrics": {},
            "status": "failed",
            "error": str(e)
        })

    return results
