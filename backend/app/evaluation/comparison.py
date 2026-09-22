import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.core.config import settings

def compare_run_models(run_id: str) -> Dict[str, Any]:
    """
    Extracts and standardizes model comparison data across all models trained in a run.
    """
    clean_id = "".join(c for c in run_id if c.isalnum() or c in "_-")
    run_dir = settings.OUTPUTS_DIR / "runs" / clean_id
    lb_path = run_dir / "leaderboard.json"
    metrics_path = run_dir / "metrics.json"

    if not lb_path.exists():
        return {
            "run_id": run_id,
            "models": [],
            "chart_data": {"models": [], "validation_scores": [], "test_scores": []}
        }

    with open(lb_path, "r", encoding="utf-8") as f:
        leaderboard = json.load(f)

    primary_metric = "metric"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                primary_metric = metrics_data.get("primary_metric", "metric")
        except Exception:
            pass

    models_summary: List[Dict[str, Any]] = []
    chart_models: List[str] = []
    chart_val_scores: List[float] = []
    chart_test_scores: List[Optional[float]] = []

    for m in leaderboard:
        m_name = m.get("model_name", "Unknown")
        val_score = m.get("validation_primary_score", 0.0)
        test_score = m.get("test_primary_score")
        gap = m.get("generalization_gap")
        is_winner = m.get("is_winner", False)

        models_summary.append({
            "model_id": m.get("model_id"),
            "model_name": m_name,
            "engine": m.get("engine", "sklearn"),
            "status": m.get("status", "success"),
            "is_winner": is_winner,
            "is_naive_baseline": m.get("is_naive_baseline", False),
            "validation_score": val_score,
            "test_score": test_score,
            "generalization_gap": gap,
            "training_time_seconds": m.get("training_time_seconds", 0.0),
            "diagnostic_label": m.get("diagnostic_label")
        })

        if m.get("status") == "success":
            chart_models.append(m_name)
            chart_val_scores.append(round(float(val_score), 4) if val_score is not None else 0.0)
            chart_test_scores.append(round(float(test_score), 4) if test_score is not None else None)

    return {
        "run_id": run_id,
        "primary_metric": primary_metric,
        "models": models_summary,
        "chart_data": {
            "models": chart_models,
            "validation_scores": chart_val_scores,
            "test_scores": chart_test_scores
        }
    }
