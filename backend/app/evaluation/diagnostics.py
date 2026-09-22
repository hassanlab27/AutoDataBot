from typing import Dict, Any, List, Optional
from app.ml.metrics import is_higher_better, compare_scores

def compute_detailed_diagnostics(
    primary_metric: str,
    val_score: float,
    test_score: Optional[float],
    naive_baseline_score: Optional[float] = None,
    train_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes neutral generalization and fitting diagnostics.
    """
    higher_better = is_higher_better(primary_metric)
    notes: List[str] = []
    label = "Normal / Well-fit"

    # Generalization gap calculation
    gap: Optional[float] = None
    if test_score is not None:
        if higher_better:
            gap = round(float(val_score - test_score), 4)
        else:
            gap = round(float(test_score - val_score), 4)

    # Train to validation gap calculation
    train_val_gap: Optional[float] = None
    if train_score is not None:
        if higher_better:
            train_val_gap = round(float(train_score - val_score), 4)
        else:
            train_val_gap = round(float(val_score - train_score), 4)

    # 1. Check for suspiciously high performance (leakage risk)
    if higher_better:
        if primary_metric in ["accuracy", "balanced_accuracy", "f1", "f1_macro", "r2", "roc_auc"] and val_score >= 0.999:
            label = "Suspiciously High Performance"
            notes.append("Validation score is near-perfect (>= 0.999), which may indicate target leakage or trivial proxy features.")
    else:
        if val_score <= 1e-5:
            label = "Suspiciously High Performance"
            notes.append("Validation error is near zero (<= 1e-5), which may indicate target leakage or trivial proxy features.")

    # 2. Check against naive baseline
    if label == "Normal / Well-fit" and naive_baseline_score is not None:
        cmp_res = compare_scores(primary_metric, val_score, naive_baseline_score)
        if cmp_res < 0:
            label = "Worse than Baseline"
            notes.append(f"Model validation score ({val_score}) is lower than a naive baseline ({naive_baseline_score}).")
        elif cmp_res == 0:
            label = "Potential underfitting indicator"
            notes.append("Model performs identically to a naive baseline, suggesting insufficient predictive signal.")

    # 3. Overfitting / Underfitting heuristic
    if label == "Normal / Well-fit" and gap is not None:
        if gap >= 0.20:
            label = "Large generalization gap"
            notes.append(f"Performance drops by {gap:.4f} between validation and test sets (Potential overfitting indicator).")
        elif gap >= 0.08:
            label = "Moderate generalization gap"
            notes.append(f"Moderate performance decrease ({gap:.4f}) observed between validation and test sets.")
        elif gap < -0.15:
            notes.append(f"Test performance exceeds validation performance by {abs(gap):.4f}. Possible slight distribution difference between partitions.")

        # Underfitting check
        if higher_better and primary_metric in ["accuracy", "balanced_accuracy", "f1", "r2"]:
            if primary_metric == "r2" and val_score <= 0.05 and (test_score is None or test_score <= 0.05):
                label = "Potential underfitting indicator"
                notes.append("R2 score is very low (<= 0.05). Model appears to struggle capturing underlying patterns.")

    if not notes:
        notes.append("Model exhibits consistent generalization across validation and test partitions without large performance divergence.")

    return {
        "primary_metric": primary_metric,
        "metric_direction": "higher" if higher_better else "lower",
        "validation_score": val_score,
        "test_score": test_score,
        "generalization_gap": gap,
        "train_validation_gap": train_val_gap,
        "diagnostic_label": label,
        "diagnostic_notes": notes,
        "disclaimer": "Diagnostic heuristics describe observed partition differences and do not constitute absolute statistical proof."
    }
