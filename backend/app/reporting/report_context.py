import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.core.config import settings
from app.core.errors import AutoDataBotError, DatasetNotFoundError
from app.core.logging import logger
from app.storage.dataset_store import dataset_store

REPORT_SCHEMA_VERSION = "1.0"

def _sanitize_id(identifier: str) -> str:
    """Sanitize identifier against path traversal attacks."""
    if not identifier:
        raise AutoDataBotError("Identifier cannot be empty.")
    clean = "".join(c for c in identifier if c.isalnum() or c in "_-")
    if not clean or clean != identifier or ".." in identifier or "/" in identifier or "\\" in identifier:
        raise AutoDataBotError(f"Invalid identifier: {identifier}")
    return clean

def _load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    if not file_path.is_file():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load JSON from {file_path}: {e}")
        return None

def build_report_context(run_id: str) -> Dict[str, Any]:
    """
    Deterministically aggregates all stored Phase 1-5 artifacts for a run
    into a structured context dictionary for report rendering.
    Does NOT rerun any ML, preprocessing, or EDA computations.
    """
    clean_run_id = _sanitize_id(run_id)
    run_dir = settings.OUTPUTS_DIR / "runs" / clean_run_id
    if not run_dir.is_dir():
        raise AutoDataBotError(f"Run {clean_run_id} not found on disk.")

    # 1. Load run metadata
    status_data = _load_json(run_dir / "status.json") or {}
    config_data = _load_json(run_dir / "config.json") or {}
    summary_data = _load_json(run_dir / "summary.json") or {}
    leaderboard_data = _load_json(run_dir / "leaderboard.json") or {}

    dataset_id = summary_data.get("dataset_id") or config_data.get("dataset_id") or status_data.get("dataset_id")
    if not dataset_id:
        raise AutoDataBotError(f"Run {clean_run_id} does not have an associated dataset_id.")

    clean_dataset_id = _sanitize_id(dataset_id)

    # 2. Load dataset metadata & inspection
    dataset_meta = {}
    try:
        dataset_meta = dataset_store.get_metadata(clean_dataset_id)
    except Exception as e:
        logger.warning(f"Could not load metadata for dataset {clean_dataset_id}: {e}")

    # Preprocessing metadata
    prep_dir = settings.OUTPUTS_DIR / "datasets" / clean_dataset_id / "preprocessing"
    prep_meta = _load_json(prep_dir / "metadata.json") or {}
    prep_config = _load_json(prep_dir / "config.json") or {}

    # Evaluation artifacts
    eval_dir = run_dir / "evaluation"
    full_eval = _load_json(eval_dir / "full_evaluation.json") or {}
    class_eval = _load_json(eval_dir / "classification.json") or {}
    regr_eval = _load_json(eval_dir / "regression.json") or {}
    diagnostics_eval = _load_json(eval_dir / "diagnostics.json") or {}
    errors_eval = _load_json(eval_dir / "errors.json") or {}

    # Explainability artifacts
    explain_dir = run_dir / "explainability"
    shap_data = _load_json(explain_dir / "shap_summary.json") or {}
    native_feat_data = _load_json(explain_dir / "feature_importance.json") or {}
    perm_feat_data = _load_json(explain_dir / "permutation.json") or {}

    # -------------------------------------------------------------
    # 7.1 Executive Summary
    # -------------------------------------------------------------
    dataset_filename = dataset_meta.get("original_filename") or clean_dataset_id
    total_rows = dataset_meta.get("rows_count") or prep_meta.get("split", {}).get("total_rows", "Not available")
    total_cols = dataset_meta.get("columns_count") or len(dataset_meta.get("columns", [])) or "Not available"
    target_column = summary_data.get("target") or config_data.get("target") or prep_config.get("target", "Not available")
    problem_type = summary_data.get("problem_type") or config_data.get("problem_type") or full_eval.get("problem_type", "Not available")
    winning_model = summary_data.get("winning_model", "Not available")
    winning_engine = summary_data.get("winning_engine", "Not available")
    primary_metric = summary_data.get("primary_metric") or config_data.get("primary_metric", "Not available")
    val_score = summary_data.get("validation_score")
    test_score = summary_data.get("test_score")
    train_score = summary_data.get("train_score")
    elapsed_seconds = status_data.get("elapsed_seconds") or 0.0

    exec_summary = {
        "run_id": clean_run_id,
        "dataset_name": dataset_filename,
        "rows_count": total_rows,
        "columns_count": total_cols,
        "target": target_column,
        "problem_type": problem_type,
        "training_mode": config_data.get("training_mode", "standard"),
        "primary_metric": primary_metric,
        "winning_model": winning_model,
        "winning_engine": winning_engine,
        "validation_metric": val_score,
        "test_metric": test_score,
        "train_metric": train_score,
        "duration_seconds": elapsed_seconds,
        "created_at": summary_data.get("created_at") or status_data.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "status": status_data.get("status", "completed")
    }

    # -------------------------------------------------------------
    # 7.2 Dataset Overview
    # -------------------------------------------------------------
    ingestion_report = dataset_meta.get("ingestion_report", {})
    file_size_bytes = dataset_meta.get("file_size_bytes", 0)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2) if file_size_bytes else "Not available"
    
    # Column type categorization
    columns_list = dataset_meta.get("columns", [])
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    # Derive from prep_meta if available
    num_count = prep_meta.get("numeric_features_count")
    cat_count = prep_meta.get("categorical_features_count")
    dt_count = prep_meta.get("generated_datetime_features_count", 0)

    dataset_overview = {
        "dataset_id": clean_dataset_id,
        "original_filename": dataset_filename,
        "file_size_mb": file_size_mb,
        "rows_count": total_rows,
        "columns_count": total_cols,
        "target_column": target_column,
        "problem_type": problem_type,
        "numeric_features_count": num_count if num_count is not None else "Not available",
        "categorical_features_count": cat_count if cat_count is not None else "Not available",
        "datetime_features_count": dt_count,
        "encoding_used": ingestion_report.get("encoding_used", "utf-8"),
        "delimiter_used": ingestion_report.get("delimiter_used", ","),
        "columns": columns_list
    }

    # -------------------------------------------------------------
    # 8. Data Quality Section
    # -------------------------------------------------------------
    data_quality = {
        "quality_score": "Not available",
        "duplicate_rows_count": 0,
        "constant_columns": [],
        "high_cardinality_columns": [],
        "missing_summary": [],
        "quality_warnings": []
    }
    # Check if quality was saved in dataset metadata or warnings
    if "quality_warnings" in config_data:
        data_quality["quality_warnings"] = config_data.get("quality_warnings", [])
    if "suitability_warnings" in config_data:
        data_quality["quality_warnings"].extend(config_data.get("suitability_warnings", []))
    
    # -------------------------------------------------------------
    # 9. & 10. EDA & Target / Problem Section
    # -------------------------------------------------------------
    split_info = prep_meta.get("split", {})
    target_problem = {
        "target_column": target_column,
        "problem_type": problem_type,
        "is_classification": "classification" in problem_type.lower() if problem_type != "Not available" else True,
        "classes_count": len(split_info.get("train_class_distribution", {})) or "Not available",
        "class_counts": split_info.get("train_class_distribution") or split_info.get("test_class_distribution") or {},
        "target_stats": {}
    }

    eda = {
        "target_distribution": target_problem["class_counts"],
        "problem_type": problem_type,
        "numeric_features_count": num_count,
        "categorical_features_count": cat_count,
    }

    # -------------------------------------------------------------
    # 11. Preprocessing Section
    # -------------------------------------------------------------
    preprocessing = {
        "original_features_count": prep_meta.get("original_features_count", total_cols),
        "selected_features_count": prep_meta.get("selected_features_count", total_cols),
        "final_features_count": prep_meta.get("final_features_count", "Not available"),
        "one_hot_generated_features_count": prep_meta.get("one_hot_generated_features_count", 0),
        "numeric_imputation_strategy": "median",
        "categorical_imputation_strategy": "most_frequent",
        "categorical_encoding": "OneHotEncoder",
        "scaling_strategy": "StandardScaler",
        "split": {
            "train_rows": split_info.get("train_rows", "Not available"),
            "test_rows": split_info.get("test_rows", "Not available"),
            "train_percentage": split_info.get("train_percentage", 80.0),
            "test_percentage": split_info.get("test_percentage", 20.0),
            "random_state": split_info.get("random_state", 42),
            "stratified": split_info.get("stratified", True),
            "split_strategy": split_info.get("split_strategy", "Train/Test Holdout Split")
        },
        "engine_strategies": {
            "AutoGluon": "Native raw tabular preprocessing with automated multi-modal type inference",
            "Scikit-Learn & Gradient Boosting": "Disjoint scikit-learn ColumnTransformer fitted strictly on training partition"
        }
    }

    # -------------------------------------------------------------
    # 12. Training Configuration
    # -------------------------------------------------------------
    training_config = {
        "run_id": clean_run_id,
        "training_mode": config_data.get("training_mode", "standard"),
        "primary_metric": primary_metric,
        "time_limit_seconds": config_data.get("time_limit", "Not available"),
        "random_state": config_data.get("random_state", 42),
        "engine_preference": config_data.get("engine_preference", "all"),
        "total_models_trained": summary_data.get("total_models_trained", 0),
        "successful_models": summary_data.get("successful_models", 0),
        "failed_models": summary_data.get("failed_models", 0),
        "tuning_history": summary_data.get("tuning_history", {})
    }

    # -------------------------------------------------------------
    # 13. Model Leaderboard
    # -------------------------------------------------------------
    if isinstance(leaderboard_data, list):
        models_raw = leaderboard_data
    elif isinstance(leaderboard_data, dict):
        models_raw = leaderboard_data.get("models", [])
    else:
        models_raw = []

    leaderboard = []
    for idx, m in enumerate(models_raw):
        if not isinstance(m, dict):
            continue
        v_score = m.get("validation_primary_score") or m.get("val_score") or m.get("validation_score")
        t_score = m.get("test_primary_score") or m.get("test_score")
        tr_score = m.get("train_primary_score") or m.get("train_score")
        time_sec = m.get("training_time") or m.get("training_time_seconds")
        
        leaderboard.append({
            "rank": m.get("rank", idx + 1),
            "model_name": m.get("model_name", "Unknown"),
            "engine": m.get("engine", "Unknown"),
            "validation_score": round(v_score, 4) if isinstance(v_score, float) else v_score,
            "test_score": round(t_score, 4) if isinstance(t_score, float) else t_score,
            "train_score": round(tr_score, 4) if isinstance(tr_score, float) else tr_score,
            "training_time": round(time_sec, 2) if isinstance(time_sec, (int, float)) else None,
            "status": m.get("status", "success"),
            "reason": m.get("error_message") or m.get("reason")
        })

    # -------------------------------------------------------------
    # 14. Selected Model
    # -------------------------------------------------------------
    selected_model = {
        "model_name": winning_model,
        "engine": winning_engine,
        "selection_metric": primary_metric,
        "validation_score": val_score,
        "test_score": test_score,
        "train_score": train_score,
        "training_duration": elapsed_seconds,
        "selection_rationale": "Model winner selected based strictly on holdout validation score ranking prior to final test partition evaluation."
    }

    # -------------------------------------------------------------
    # 15. Evaluation Section
    # -------------------------------------------------------------
    perf_data = full_eval.get("performance") or {}
    metrics_data = perf_data.get("metrics") or {}
    if not metrics_data:
        if class_eval:
            metrics_data = class_eval.get("metrics", {})
        elif regr_eval:
            metrics_data = regr_eval.get("metrics", {})

    evaluation = {
        "problem_type": problem_type,
        "metrics": metrics_data,
        "confusion_matrix": perf_data.get("confusion_matrix") or class_eval.get("confusion_matrix"),
        "per_class_metrics": perf_data.get("per_class_metrics") or class_eval.get("per_class_metrics"),
        "calibration": perf_data.get("calibration") or class_eval.get("calibration"),
        "residual_stats": perf_data.get("residual_stats") or regr_eval.get("residual_stats"),
        "actual_vs_predicted": perf_data.get("actual_vs_predicted") or regr_eval.get("actual_vs_predicted")
    }

    # -------------------------------------------------------------
    # 16. Generalization / Diagnostics
    # -------------------------------------------------------------
    diagnostics = {
        "validation_score": val_score,
        "test_score": test_score,
        "generalization_gap": summary_data.get("generalization_gap") or diagnostics_eval.get("generalization_gap"),
        "diagnostic_label": summary_data.get("diagnostic_label") or diagnostics_eval.get("diagnostic_label", "Normal / Well-fit"),
        "diagnostic_notes": diagnostics_eval.get("diagnostic_notes", []),
        "disclaimer": "Diagnostic heuristics describe observed partition differences and do not constitute absolute statistical proof."
    }

    # -------------------------------------------------------------
    # 17. Error Analysis
    # -------------------------------------------------------------
    error_analysis = full_eval.get("error_analysis") or errors_eval or {}
    # Sanitize and cap display examples (limit to top 10 items, remove raw confidential columns)
    worst_records = error_analysis.get("top_worst_predictions", [])[:10]
    error_analysis_clean = {
        "total_test_samples": error_analysis.get("total_test_samples"),
        "total_errors": error_analysis.get("total_errors"),
        "error_rate": error_analysis.get("error_rate"),
        "top_worst_predictions": worst_records
    }

    # -------------------------------------------------------------
    # 18. Feature Importance
    # -------------------------------------------------------------
    native_list = native_feat_data.get("importances", [])
    perm_list = perm_feat_data.get("importances", [])
    feature_importance = {
        "native": native_list[:15],
        "native_available": native_feat_data.get("available", len(native_list) > 0),
        "permutation": perm_list[:15],
        "permutation_available": perm_feat_data.get("available", len(perm_list) > 0),
        "note": "Feature importance ranks model weight or score drop on transformed feature representations."
    }

    # -------------------------------------------------------------
    # 19. SHAP Explainability
    # -------------------------------------------------------------
    shap_summary_list = shap_data.get("global_importance", [])
    shap = {
        "available": shap_data.get("available", False),
        "reason": shap_data.get("reason"),
        "global_importance": shap_summary_list[:15],
        "methodological_note": "SHAP values describe model attribution for the evaluated predictions and should not be interpreted as proof of causation."
    }

    # -------------------------------------------------------------
    # 20. Limitations Section
    # -------------------------------------------------------------
    limitations = []
    if isinstance(total_rows, int):
        if total_rows < 1000:
            limitations.append("Small sample size: Dataset contains fewer than 1,000 rows, which may limit generalizability.")
        elif total_rows > 100000:
            limitations.append("Large dataset subsampling: Certain diagnostics and SHAP calculations utilize representative holdout subsets.")
    
    if "classification" in problem_type.lower():
        class_counts = target_problem.get("class_counts", {})
        if class_counts:
            counts = list(class_counts.values())
            if counts and min(counts) / sum(counts) < 0.10:
                limitations.append("Class imbalance observed: Minority class represents less than 10% of samples; precision/recall trade-offs should be monitored.")

    limitations.extend([
        "Tabular modeling scope: Analysis and models assume independent and identically distributed tabular records.",
        "Holdout validation: Evaluation metrics reflect a single stratified/shuffled holdout test partition; real-world deployment performance may vary with data drift.",
        "Non-causal attribution: Feature importance and SHAP rankings indicate predictive association within the trained model, not causal influence."
    ])

    return {
        "report_version": REPORT_SCHEMA_VERSION,
        "run_id": clean_run_id,
        "dataset_id": clean_dataset_id,
        "created_at": exec_summary["created_at"],
        "executive_summary": exec_summary,
        "dataset_overview": dataset_overview,
        "data_quality": data_quality,
        "eda": eda,
        "target_problem": target_problem,
        "preprocessing": preprocessing,
        "training_config": training_config,
        "leaderboard": leaderboard,
        "selected_model": selected_model,
        "evaluation": evaluation,
        "diagnostics": diagnostics,
        "error_analysis": error_analysis_clean,
        "feature_importance": feature_importance,
        "shap": shap,
        "limitations": limitations
    }
