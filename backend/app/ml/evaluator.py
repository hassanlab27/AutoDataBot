from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import numpy as np

from app.ml.metrics import is_higher_better, compare_scores

class ModelResult(BaseModel):
    model_id: str
    model_name: str
    engine: str  # "sklearn", "lightgbm", "xgboost", "catboost", "flaml", "autogluon"
    problem_type: str
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    validation_primary_score: float = 0.0
    test_metrics: Optional[Dict[str, float]] = None
    test_primary_score: Optional[float] = None
    generalization_gap: Optional[float] = None
    training_time_seconds: float = 0.0
    status: str = "success"  # "success", "failed", "timeout"
    error_message: Optional[str] = None
    diagnostic_label: Optional[str] = None
    diagnostic_notes: List[str] = Field(default_factory=list)
    is_naive_baseline: bool = False
    is_winner: bool = False
    model_path: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None

def compute_generalization_gap(
    primary_metric: str,
    val_score: float,
    test_score: float
) -> float:
    """
    Computes performance drop from validation to test set.
    Positive value always represents performance degradation on the test set.
    """
    if is_higher_better(primary_metric):
        return round(float(val_score - test_score), 4)
    else:
        return round(float(test_score - val_score), 4)

def diagnose_model_fit(
    primary_metric: str,
    val_score: float,
    test_score: Optional[float],
    naive_baseline_score: Optional[float],
    train_score: Optional[float] = None
) -> tuple[str, List[str]]:
    """
    Rule-based diagnostic heuristic evaluating overfitting, underfitting,
    and comparison against naive baseline.
    """
    notes: List[str] = []
    higher_better = is_higher_better(primary_metric)

    # 1. Suspiciously high performance (possible leakage)
    if higher_better:
        if primary_metric in ["accuracy", "balanced_accuracy", "f1", "f1_macro", "r2", "roc_auc"]:
            if val_score >= 0.999:
                notes.append("Validation score is near-perfect (>= 0.999), which strongly indicates potential target leakage.")
                return "Suspiciously High Performance", notes
    else:
        if val_score <= 1e-5:
            notes.append("Validation error is near zero, which strongly indicates potential target leakage.")
            return "Suspiciously High Performance", notes

    # 2. Comparison against naive baseline
    if naive_baseline_score is not None:
        val_cmp = compare_scores(primary_metric, val_score, naive_baseline_score)
        if val_cmp < 0:
            notes.append("Model performs worse than a naive dummy baseline.")
            return "Worse than Baseline", notes
        elif val_cmp == 0:
            notes.append("Model performs identically to a naive dummy baseline.")
            return "Underfitting", notes

    # 3. Test-based overfitting / underfitting analysis
    if test_score is not None:
        gap = compute_generalization_gap(primary_metric, val_score, test_score)
        
        # Underfitting check
        if higher_better and primary_metric in ["accuracy", "balanced_accuracy", "f1", "f1_macro", "r2"]:
            if primary_metric == "r2" and val_score <= 0.05 and test_score <= 0.05:
                notes.append("Both validation and test R2 are extremely low (<= 0.05). Model lacks predictive power.")
                return "Underfitting", notes
            if naive_baseline_score is not None and abs(val_score - naive_baseline_score) < 0.02 and abs(test_score - naive_baseline_score) < 0.02:
                notes.append("Validation and test scores show minimal improvement over naive baseline (< 2%).")
                return "Underfitting", notes

        # Overfitting check based on generalization gap
        if gap >= 0.20:
            notes.append(f"Significant performance drop ({gap:.4f}) on test data indicates severe overfitting.")
            return "Severe Overfitting", notes
        elif gap >= 0.08:
            notes.append(f"Moderate performance drop ({gap:.4f}) on test data indicates possible overfitting.")
            return "Possible Overfitting", notes
        elif gap < -0.15:
            notes.append(f"Test score is unexpectedly much higher than validation score ({abs(gap):.4f}). Check split distribution.")
            return "Normal / Well-fit", notes

    notes.append("Model generalizes well between validation and held-out test data.")
    return "Normal / Well-fit", notes

def rank_and_select_winner(
    results: List[ModelResult],
    primary_metric: str
) -> tuple[Optional[ModelResult], List[ModelResult]]:
    """
    Ranks successful models by validation_primary_score using metric direction.
    Excludes naive baselines from winning unless no other model succeeded.
    Marks the highest-ranking model with is_winner=True.
    """
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]

    if not successful:
        return None, results

    higher_better = is_higher_better(primary_metric)

    # Sort successful models by validation_primary_score
    sorted_successful = sorted(
        successful,
        key=lambda r: r.validation_primary_score if higher_better else -r.validation_primary_score,
        reverse=higher_better
    )

    # Determine winner candidate (prefer non-naive baseline)
    candidates = [r for r in sorted_successful if not r.is_naive_baseline]
    winner = candidates[0] if candidates else sorted_successful[0]

    for r in sorted_successful:
        r.is_winner = (r.model_id == winner.model_id)

    # Combined leaderboard: successful models first, then failed models
    leaderboard = sorted_successful + failed
    return winner, leaderboard
