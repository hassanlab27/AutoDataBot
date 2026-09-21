import time
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.errors import AutoDataBotException, AutoDataBotError
from app.storage.dataset_store import dataset_store
from app.preprocessing.preprocessing_service import PreprocessingService
from app.preprocessing.split_service import perform_train_test_split
from app.ml.config import AutoMLConfig
from app.ml.metrics import (
    select_default_primary_metric,
    compute_metrics,
    is_higher_better,
    compare_scores
)
from app.ml.suitability import assess_training_suitability
from app.ml.evaluator import (
    ModelResult,
    compute_generalization_gap,
    diagnose_model_fit,
    rank_and_select_winner
)
from app.ml.models.sklearn_baselines import train_sklearn_baselines
from app.ml.models.gradient_boosting import train_gradient_boosting_models
from app.ml.engines.flaml_engine import train_flaml_engine
from app.ml.engines.autogluon_engine import train_autogluon_engine
from app.ml.job_runner import job_runner, RunStatus

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.preprocessing_service = PreprocessingService()

    def get_run_dir(self, run_id: str) -> Path:
        clean_id = "".join(c for c in run_id if c.isalnum() or c in "_-")
        run_dir = settings.OUTPUTS_DIR / "runs" / clean_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "models").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        return run_dir

    def create_run(self, config: AutoMLConfig) -> Dict[str, Any]:
        """
        Validates suitability, allocates run directory, persists config,
        and initializes run state.
        """
        # 1. Dataset existence & suitability checks
        suitability = assess_training_suitability(
            dataset_id=config.dataset_id,
            target_col=config.target,
            expected_problem_type=config.problem_type
        )
        if not suitability["suitable"]:
            errors = "; ".join(suitability["errors"])
            raise AutoDataBotError(f"Dataset is not suitable for AutoML training: {errors}")

        # 2. Allocate Run ID and Run Directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}_{uuid.uuid4().hex[:6]}"
        run_dir = self.get_run_dir(run_id)

        # 3. Determine primary metric if 'auto'
        primary_metric = config.primary_metric
        if primary_metric == "auto":
            try:
                target_diag = self.preprocessing_service.inspect_target(config.dataset_id, config.target)
            except Exception:
                target_diag = None
            primary_metric = select_default_primary_metric(config.problem_type, target_diag)

        # 4. Save config.json
        config_data = config.model_dump()
        config_data["effective_primary_metric"] = primary_metric
        config_data["effective_time_limit"] = config.get_effective_time_limit()
        config_data["effective_cpus"] = config.get_effective_cpus()
        config_data["suitability_warnings"] = suitability.get("warnings", [])

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # 5. Register with JobRunner
        status = job_runner.register_run(run_id, config.dataset_id, run_dir)

        # 6. Launch worker thread
        job_runner.start_job(run_id, self._execute_run, (run_id, config, primary_metric))

        return {
            "run_id": run_id,
            "dataset_id": config.dataset_id,
            "target": config.target,
            "problem_type": config.problem_type,
            "primary_metric": primary_metric,
            "training_mode": config.training_mode,
            "time_limit": config_data["effective_time_limit"],
            "status": status.status,
            "warnings": suitability.get("warnings", [])
        }

    def _execute_run(self, run_id: str, config: AutoMLConfig, primary_metric: str) -> None:
        """Background execution worker for the entire AutoML pipeline."""
        run_dir = self.get_run_dir(run_id)
        results: List[ModelResult] = []
        trained_model_objs: Dict[str, Any] = {}

        try:
            logger.info(f"Starting AutoML run {run_id} for dataset {config.dataset_id}")
            job_runner.update_status(
                run_id,
                status="running",
                stage="Preparing training and validation datasets",
                progress_pct=10
            )

            # 1. Prepare / Load Preprocessed Data
            (
                X_train_raw, X_test_raw,
                X_train_prep, X_test_prep,
                y_train, y_test
            ) = self._prepare_data(config)

            if job_runner.is_cancelled(run_id):
                return

            # Internal validation split from training partition for fair ranking
            # Held-out test set (X_test_prep, y_test) is strictly untouched!
            stratify = y_train if config.problem_type.endswith("classification") else None
            try:
                (
                    X_tr_prep, X_val_prep,
                    y_tr, y_val
                ) = train_test_split(
                    X_train_prep,
                    y_train,
                    test_size=0.20,
                    random_state=config.random_state,
                    stratify=stratify
                )
            except Exception:
                (
                    X_tr_prep, X_val_prep,
                    y_tr, y_val
                ) = train_test_split(
                    X_train_prep,
                    y_train,
                    test_size=0.20,
                    random_state=config.random_state
                )

            # 2. Stage: Scikit-learn Baselines
            job_runner.update_status(
                run_id,
                stage="Training Scikit-learn models & baselines",
                engine="sklearn",
                progress_pct=25
            )
            sk_raw_results = train_sklearn_baselines(
                problem_type=config.problem_type,
                X_train=X_tr_prep,
                y_train=y_tr,
                X_val=X_val_prep,
                y_val=y_val,
                random_state=config.random_state,
                max_cpus=config.get_effective_cpus()
            )

            for item in sk_raw_results:
                m_id = f"model_{len(results)+1}_{item['engine'].lower()}"
                is_naive = "naive" in item["model_name"].lower()
                val_score = item["validation_metrics"].get(primary_metric, 0.0) if item["status"] == "completed" else 0.0
                
                m_res = ModelResult(
                    model_id=m_id,
                    model_name=item["model_name"],
                    engine=item["engine"],
                    problem_type=config.problem_type,
                    validation_metrics=item["validation_metrics"],
                    validation_primary_score=val_score,
                    training_time_seconds=item["training_time_seconds"],
                    status="success" if item["status"] == "completed" else "failed",
                    error_message=item.get("error"),
                    is_naive_baseline=is_naive
                )
                results.append(m_res)
                if item.get("estimator") is not None:
                    trained_model_objs[m_id] = item["estimator"]

            # Save naive baseline score for benchmark comparison
            naive_baseline_score = None
            naive_models = [m for m in results if m.is_naive_baseline and m.status == "success"]
            if naive_models:
                naive_baseline_score = naive_models[0].validation_primary_score

            if job_runner.is_cancelled(run_id):
                return

            # 3. Stage: Standalone Gradient Boosting Models (LightGBM, XGBoost, CatBoost)
            if config.engine_preference in ["all", "sklearn_gbdt"]:
                job_runner.update_status(
                    run_id,
                    stage="Training Gradient Boosted Trees (LightGBM, XGBoost, CatBoost)",
                    engine="gbdt",
                    progress_pct=50
                )
                gb_raw_results = train_gradient_boosting_models(
                    problem_type=config.problem_type,
                    X_train=X_tr_prep,
                    y_train=y_tr,
                    X_val=X_val_prep,
                    y_val=y_val,
                    random_state=config.random_state,
                    max_cpus=config.get_effective_cpus()
                )
                for item in gb_raw_results:
                    m_id = f"model_{len(results)+1}_{item['engine'].lower()}"
                    val_score = item["validation_metrics"].get(primary_metric, 0.0) if item["status"] == "completed" else 0.0
                    m_res = ModelResult(
                        model_id=m_id,
                        model_name=item["model_name"],
                        engine=item["engine"],
                        problem_type=config.problem_type,
                        validation_metrics=item["validation_metrics"],
                        validation_primary_score=val_score,
                        training_time_seconds=item["training_time_seconds"],
                        status="success" if item["status"] == "completed" else "failed",
                        error_message=item.get("error"),
                        is_naive_baseline=False
                    )
                    results.append(m_res)
                    if item.get("estimator") is not None:
                        trained_model_objs[m_id] = item["estimator"]

            if job_runner.is_cancelled(run_id):
                return

            # 4. Stage: FLAML AutoML Engine
            if config.engine_preference in ["all", "flaml"]:
                flaml_time = max(15, min(config.get_effective_time_limit() // 3, 120))
                job_runner.update_status(
                    run_id,
                    stage="Running FLAML optimization",
                    engine="flaml",
                    progress_pct=70
                )
                flaml_raw_results = train_flaml_engine(
                    problem_type=config.problem_type,
                    X_train=X_tr_prep,
                    y_train=y_tr,
                    X_val=X_val_prep,
                    y_val=y_val,
                    time_limit=flaml_time,
                    random_state=config.random_state,
                    eval_metric=primary_metric
                )
                for item in flaml_raw_results:
                    m_id = f"model_{len(results)+1}_{item['engine'].lower()}"
                    val_score = item["validation_metrics"].get(primary_metric, 0.0) if item["status"] == "completed" else 0.0
                    m_res = ModelResult(
                        model_id=m_id,
                        model_name=item["model_name"],
                        engine=item["engine"],
                        problem_type=config.problem_type,
                        validation_metrics=item["validation_metrics"],
                        validation_primary_score=val_score,
                        training_time_seconds=item["training_time_seconds"],
                        status="success" if item["status"] == "completed" else "failed",
                        error_message=item.get("error"),
                        is_naive_baseline=False
                    )
                    results.append(m_res)
                    if item.get("estimator") is not None:
                        trained_model_objs[m_id] = item["estimator"]

            if job_runner.is_cancelled(run_id):
                return

            # 5. Stage: AutoGluon AutoML Engine (Path A - validated raw tabular data)
            if config.engine_preference in ["all", "autogluon"]:
                ag_time = max(20, min(config.get_effective_time_limit() // 2, 240))
                job_runner.update_status(
                    run_id,
                    stage="Running AutoGluon ensembling",
                    engine="autogluon",
                    progress_pct=85
                )
                # Combine raw features and target for AutoGluon
                train_data_raw = X_train_raw.copy()
                train_data_raw[config.target] = y_train

                ag_raw_results = train_autogluon_engine(
                    train_data_raw=train_data_raw,
                    target_col=config.target,
                    problem_type=config.problem_type,
                    val_data_raw=None,
                    run_models_dir=run_dir / "models",
                    time_limit=ag_time,
                    presets=config.presets,
                    random_state=config.random_state,
                    eval_metric=primary_metric
                )
                for item in ag_raw_results:
                    m_id = f"model_{len(results)+1}_{item['engine'].lower()}"
                    val_score = item["validation_metrics"].get(primary_metric, 0.0) if item["status"] == "completed" else 0.0
                    m_res = ModelResult(
                        model_id=m_id,
                        model_name=item["model_name"],
                        engine=item["engine"],
                        problem_type=config.problem_type,
                        validation_metrics=item["validation_metrics"],
                        validation_primary_score=val_score,
                        training_time_seconds=item["training_time_seconds"],
                        status="success" if item["status"] == "completed" else "failed",
                        error_message=item.get("error"),
                        is_naive_baseline=False
                    )
                    results.append(m_res)
                    if item.get("estimator") is not None:
                        trained_model_objs[m_id] = item["estimator"]

            if job_runner.is_cancelled(run_id):
                return

            # 6. Model Ranking & Winner Selection (Based STRICTLY on validation_primary_score)
            job_runner.update_status(
                run_id,
                stage="Ranking models & evaluating winner on held-out test data",
                engine="evaluator",
                progress_pct=92
            )

            winner, leaderboard = rank_and_select_winner(results, primary_metric)

            # 7. Test Set Evaluation on Winner & Baselines (Held-out test partition untouched till now!)
            if winner is not None:
                self._evaluate_on_test_data(
                    winner=winner,
                    trained_model_objs=trained_model_objs,
                    config=config,
                    primary_metric=primary_metric,
                    X_test_prep=X_test_prep,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    naive_baseline_score=naive_baseline_score
                )

                # Persist winner model bundle to disk
                self._save_winner_artifact(winner, trained_model_objs, run_dir)

            # Direct test evaluation for naive baseline to show benchmark in leaderboard
            for naive_res in [m for m in leaderboard if m.is_naive_baseline and m.status == "success"]:
                dummy_model = trained_model_objs.get(naive_res.model_id)
                if dummy_model:
                    try:
                        dummy_test_preds = dummy_model.predict(X_test_prep)
                        naive_res.test_metrics = compute_metrics(
                            config.problem_type,
                            y_test,
                            dummy_test_preds
                        )
                        naive_res.test_primary_score = naive_res.test_metrics.get(primary_metric)
                        naive_res.generalization_gap = compute_generalization_gap(
                            primary_metric,
                            naive_res.validation_primary_score,
                            naive_res.test_primary_score
                        )
                    except Exception:
                        pass

            # 8. Persist Outputs (leaderboard.json, metrics.json, summary.json)
            self._save_run_artifacts(run_dir, config, primary_metric, leaderboard, winner, naive_baseline_score)

            job_runner.update_status(
                run_id,
                status="completed",
                stage="Completed",
                progress_pct=100
            )
            logger.info(f"AutoML run {run_id} successfully completed. Winner: {winner.model_name if winner else 'None'}")

        except Exception as e:
            logger.exception(f"Fatal error during AutoML run {run_id}: {e}")
            job_runner.update_status(
                run_id,
                status="failed",
                stage="Failed",
                error=str(e)
            )

    def _prepare_data(
        self,
        config: AutoMLConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads dataset and prepares deterministic splits:
        Path A: Raw train/test splits.
        Path B: Preprocessed train/test splits via Phase 3 pipeline.
        """
        df = dataset_store.load_dataset(config.dataset_id)
        prep_dir = self.preprocessing_service.get_preprocessing_dir(config.dataset_id)
        pipeline_path = prep_dir / "pipeline.joblib"
        config_path = prep_dir / "config.json"

        # If preprocessing hasn't been run yet, run it with default settings
        if not pipeline_path.exists() or not config_path.exists():
            logger.info(f"No prior preprocessing found for dataset {config.dataset_id}. Preparing dataset now.")
            prep_res = self.preprocessing_service.prepare_dataset(
                dataset_id=config.dataset_id,
                target=config.target,
                problem_type=config.problem_type,
                random_state=config.random_state
            )
            prep_config = prep_res["config"]
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                prep_config = json.load(f)

        # Re-partition raw data using saved configuration
        feature_cols = prep_config.get("selected_features")
        test_size = prep_config.get("test_size", settings.PREPROCESSING_DEFAULT_TEST_SIZE)
        random_state = prep_config.get("random_state", config.random_state)

        X_train_raw, X_test_raw, y_train, y_test, _ = perform_train_test_split(
            df=df,
            target_col=config.target,
            feature_cols=feature_cols,
            test_size=test_size,
            random_state=random_state,
            problem_type=config.problem_type
        )

        # Load fitted ColumnTransformer pipeline
        bundle = joblib.load(pipeline_path)
        dt_extractor = bundle.get("datetime_extractor")
        col_transformer = bundle.get("column_transformer")

        # Transform features for Path B
        if dt_extractor:
            X_tr_dt = dt_extractor.transform(X_train_raw)
            X_te_dt = dt_extractor.transform(X_test_raw)
        else:
            X_tr_dt = X_train_raw.copy()
            X_te_dt = X_test_raw.copy()

        X_train_prep = col_transformer.transform(X_tr_dt)
        X_test_prep = col_transformer.transform(X_te_dt)

        if hasattr(X_train_prep, "toarray"):
            X_train_prep = X_train_prep.toarray()
        if hasattr(X_test_prep, "toarray"):
            X_test_prep = X_test_prep.toarray()

        return (
            X_train_raw, X_test_raw,
            np.asarray(X_train_prep, dtype=np.float32),
            np.asarray(X_test_prep, dtype=np.float32),
            np.asarray(y_train),
            np.asarray(y_test)
        )

    def _evaluate_on_test_data(
        self,
        winner: ModelResult,
        trained_model_objs: Dict[str, Any],
        config: AutoMLConfig,
        primary_metric: str,
        X_test_prep: np.ndarray,
        X_test_raw: pd.DataFrame,
        y_test: np.ndarray,
        naive_baseline_score: Optional[float]
    ) -> None:
        """Strictly evaluates winning model on untouched test partition."""
        model_obj = trained_model_objs.get(winner.model_id)
        if model_obj is None:
            return

        try:
            if winner.engine.lower() == "autogluon":
                # Path A evaluation on raw test features
                test_df = X_test_raw.copy()
                test_preds = model_obj.predict(test_df)
                test_probs = None
                try:
                    test_probs = model_obj.predict_proba(test_df).to_numpy()
                except Exception:
                    pass
            else:
                # Path B evaluation on preprocessed test features
                test_preds = model_obj.predict(X_test_prep)
                test_probs = None
                if hasattr(model_obj, "predict_proba"):
                    try:
                        test_probs = model_obj.predict_proba(X_test_prep)
                    except Exception:
                        pass

            # Calculate test metrics
            winner.test_metrics = compute_metrics(
                config.problem_type,
                y_test,
                test_preds,
                test_probs
            )
            winner.test_primary_score = winner.test_metrics.get(primary_metric)

            # Compute Generalization Gap
            if winner.test_primary_score is not None:
                winner.generalization_gap = compute_generalization_gap(
                    primary_metric,
                    winner.validation_primary_score,
                    winner.test_primary_score
                )

            # Heuristic Overfitting / Underfitting Diagnostics
            diag_label, diag_notes = diagnose_model_fit(
                primary_metric=primary_metric,
                val_score=winner.validation_primary_score,
                test_score=winner.test_primary_score,
                naive_baseline_score=naive_baseline_score
            )
            winner.diagnostic_label = diag_label
            winner.diagnostic_notes = diag_notes

        except Exception as e:
            logger.error(f"Error evaluating winner on test partition: {e}", exc_info=True)
            winner.diagnostic_label = "Evaluation Incomplete"
            winner.diagnostic_notes = [f"Failed during test set evaluation: {str(e)}"]

    def _save_winner_artifact(
        self,
        winner: ModelResult,
        trained_model_objs: Dict[str, Any],
        run_dir: Path
    ) -> None:
        """Saves winning model and metadata bundle."""
        model_obj = trained_model_objs.get(winner.model_id)
        if model_obj is None or winner.engine.lower() == "autogluon":
            return
        
        bundle_path = run_dir / "models" / "final_model.joblib"
        try:
            joblib.dump({
                "model_id": winner.model_id,
                "model_name": winner.model_name,
                "engine": winner.engine,
                "model_object": model_obj,
                "hyperparameters": winner.hyperparameters,
                "validation_metrics": winner.validation_metrics,
                "test_metrics": winner.test_metrics
            }, bundle_path)
            winner.model_path = str(bundle_path)
        except Exception as e:
            logger.warning(f"Could not dump winner model to disk: {e}")

    def _save_run_artifacts(
        self,
        run_dir: Path,
        config: AutoMLConfig,
        primary_metric: str,
        leaderboard: List[ModelResult],
        winner: Optional[ModelResult],
        naive_baseline_score: Optional[float]
    ) -> None:
        """Persists leaderboard.json, metrics.json, and summary.json."""
        lb_dicts = [m.model_dump() for m in leaderboard]
        with open(run_dir / "leaderboard.json", "w", encoding="utf-8") as f:
            json.dump(lb_dicts, f, indent=2)

        metrics_payload = {
            "run_id": run_dir.name,
            "primary_metric": primary_metric,
            "metric_direction": "higher" if is_higher_better(primary_metric) else "lower",
            "naive_baseline_score": naive_baseline_score,
            "winner_id": winner.model_id if winner else None,
            "winner_validation_metrics": winner.validation_metrics if winner else {},
            "winner_test_metrics": winner.test_metrics if winner else {},
            "generalization_gap": winner.generalization_gap if winner else None,
            "diagnostic_label": winner.diagnostic_label if winner else None,
            "diagnostic_notes": winner.diagnostic_notes if winner else []
        }
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        summary_payload = {
            "run_id": run_dir.name,
            "dataset_id": config.dataset_id,
            "target": config.target,
            "problem_type": config.problem_type,
            "primary_metric": primary_metric,
            "total_models_trained": len(leaderboard),
            "successful_models": len([m for m in leaderboard if m.status == "success"]),
            "failed_models": len([m for m in leaderboard if m.status != "success"]),
            "winning_model": winner.model_name if winner else "None",
            "winning_engine": winner.engine if winner else "None",
            "validation_score": winner.validation_primary_score if winner else None,
            "test_score": winner.test_primary_score if winner else None,
            "generalization_gap": winner.generalization_gap if winner else None,
            "diagnostic_label": winner.diagnostic_label if winner else None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        run_dir = settings.OUTPUTS_DIR / "runs" / run_id
        if not run_dir.exists():
            return None

        status = job_runner.get_status(run_id)
        config_data = {}
        if (run_dir / "config.json").exists():
            with open(run_dir / "config.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)

        leaderboard_data = []
        if (run_dir / "leaderboard.json").exists():
            with open(run_dir / "leaderboard.json", "r", encoding="utf-8") as f:
                leaderboard_data = json.load(f)

        metrics_data = {}
        if (run_dir / "metrics.json").exists():
            with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

        summary_data = {}
        if (run_dir / "summary.json").exists():
            with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
                summary_data = json.load(f)

        return {
            "run_id": run_id,
            "status": status.model_dump() if status else {"status": "unknown"},
            "config": config_data,
            "leaderboard": leaderboard_data,
            "metrics": metrics_data,
            "summary": summary_data
        }

    def list_runs_for_dataset(self, dataset_id: str) -> List[Dict[str, Any]]:
        runs_dir = settings.OUTPUTS_DIR / "runs"
        if not runs_dir.exists():
            return []

        matched_runs = []
        for d in sorted(runs_dir.iterdir(), reverse=True):
            if d.is_dir():
                summary_file = d / "summary.json"
                config_file = d / "config.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            s = json.load(f)
                            if s.get("dataset_id") == dataset_id:
                                matched_runs.append(s)
                    except Exception:
                        pass
                elif config_file.exists():
                    try:
                        with open(config_file, "r", encoding="utf-8") as f:
                            c = json.load(f)
                            if c.get("dataset_id") == dataset_id:
                                status = job_runner.get_status(d.name)
                                matched_runs.append({
                                    "run_id": d.name,
                                    "dataset_id": dataset_id,
                                    "target": c.get("target"),
                                    "status": status.status if status else "unknown"
                                })
                    except Exception:
                        pass
        return matched_runs

training_service = TrainingService()
