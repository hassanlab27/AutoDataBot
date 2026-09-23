import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from app.core.config import settings
from app.core.errors import AutoDataBotError
from app.evaluation.loader import data_loader, EvaluationData
from app.evaluation.classification import evaluate_classification
from app.evaluation.regression import evaluate_regression
from app.evaluation.errors import analyze_classification_errors, analyze_regression_errors
from app.evaluation.diagnostics import compute_detailed_diagnostics
from app.evaluation.comparison import compare_run_models

logger = logging.getLogger(__name__)

class EvaluationService:
    def get_eval_dir(self, run_id: str) -> Path:
        clean_id = "".join(c for c in run_id if c.isalnum() or c in "_-")
        eval_dir = settings.OUTPUTS_DIR / "runs" / clean_id / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir

    def get_full_evaluation(self, run_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieves or generates full evaluation results for a run, with persistent caching.
        """
        eval_dir = self.get_eval_dir(run_id)
        cache_file = eval_dir / "full_evaluation.json"

        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    self._normalize_evaluation_payload(cached)
                    return cached
            except Exception as e:
                logger.warning(f"Could not load cached evaluation for {run_id}: {e}")

        # Compute evaluation
        data = data_loader.load_evaluation_data(run_id)
        model = data.model_obj
        is_regression = (data.problem_type == "regression")

        # 1. Predictions on test data
        t0 = time.time()
        y_prob: Optional[np.ndarray] = None
        if data.engine.lower() == "autogluon":
            test_preds = model.predict(data.X_test_raw)
            try:
                y_prob = model.predict_proba(data.X_test_raw).to_numpy()
            except Exception:
                pass
        else:
            test_preds = model.predict(data.X_test_prep)
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(data.X_test_prep)
                except Exception:
                    pass
        pred_time = time.time() - t0

        y_test = data.y_test
        if len(y_test) != len(test_preds):
            raise AutoDataBotError(
                f"Test partition mismatch for run '{run_id}', model '{data.model_name}': "
                f"expected {len(y_test)} samples, got {len(test_preds)} predictions"
            )

        # 2. Performance on Training Data (for Authenticity & Overfitting Verification)
        train_score: Optional[float] = None
        train_metrics: Dict[str, Any] = {}
        eval_train_y = data.y_tr if data.y_tr is not None else data.y_train
        eval_train_raw = data.X_tr_raw if data.X_tr_raw is not None else data.X_train_raw
        eval_train_prep = data.X_tr_prep if data.X_tr_prep is not None else data.X_train_prep

        if eval_train_y is not None and (eval_train_prep is not None or eval_train_raw is not None):
            try:
                y_tr_prob = None
                if data.engine.lower() == "autogluon" and eval_train_raw is not None:
                    tr_preds = model.predict(eval_train_raw)
                    try:
                        y_tr_prob = model.predict_proba(eval_train_raw).to_numpy()
                    except Exception:
                        pass
                elif eval_train_prep is not None:
                    tr_preds = model.predict(eval_train_prep)
                    if hasattr(model, "predict_proba"):
                        try:
                            y_tr_prob = model.predict_proba(eval_train_prep)
                        except Exception:
                            pass
                else:
                    tr_preds = None

                if tr_preds is not None:
                    if len(eval_train_y) != len(tr_preds):
                        raise AutoDataBotError(
                            f"Train partition mismatch for run '{run_id}', model '{data.model_name}': "
                            f"expected {len(eval_train_y)} samples, got {len(tr_preds)} predictions"
                        )
                    if is_regression:
                        tr_perf = evaluate_regression(eval_train_y, tr_preds)
                        train_metrics = tr_perf.get("metrics", {})
                    else:
                        tr_perf = evaluate_classification(eval_train_y, tr_preds, y_prob=y_tr_prob, problem_type=data.problem_type)
                        train_metrics = tr_perf.get("metrics", {})
                    train_score = train_metrics.get(data.primary_metric)
            except Exception as e:
                logger.warning(f"Could not compute train metrics in evaluation service: {e}")

        # 3. Performance & Error Analysis on Test Data
        if is_regression:
            perf = evaluate_regression(y_test, test_preds)
            errors = analyze_regression_errors(y_test, test_preds, limit=20)
        else:
            perf = evaluate_classification(y_test, test_preds, y_prob=y_prob, problem_type=data.problem_type)
            errors = analyze_classification_errors(y_test, test_preds, y_prob=y_prob, limit=20)

        # 4. Generalization Diagnostics
        metrics_dict = perf.get("metrics", {})
        test_score = metrics_dict.get(data.primary_metric)
        val_score = data.config.get("validation_score", 0.0)

        # Try to pull validation score & tuning history from leaderboard/metrics
        metrics_json_path = settings.OUTPUTS_DIR / "runs" / data.run_id / "metrics.json"
        naive_score = None
        tuning_history = {}
        if metrics_json_path.exists():
            try:
                with open(metrics_json_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                    naive_score = m_data.get("naive_baseline_score")
                    val_metrics = m_data.get("winner_validation_metrics", {})
                    if data.primary_metric in val_metrics:
                        val_score = val_metrics[data.primary_metric]
                    if train_score is None and m_data.get("winner_train_score") is not None:
                        train_score = m_data.get("winner_train_score")
                    if not train_metrics and m_data.get("winner_train_metrics"):
                        train_metrics = m_data.get("winner_train_metrics", {})
                    tuning_history = m_data.get("tuning_history", {})
            except Exception:
                pass

        diag = compute_detailed_diagnostics(
            primary_metric=data.primary_metric,
            val_score=val_score,
            test_score=test_score,
            naive_baseline_score=naive_score
        )

        # 5. Multi-model Comparison
        comp = compare_run_models(run_id)
        training_time = 0.0
        lb_models = comp.get("models", [])
        winner_lb = next((m for m in lb_models if m.get("is_winner")), None)
        if winner_lb:
            training_time = winner_lb.get("training_time_seconds", 0.0)

        # Build combined payload
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "dataset_id": data.dataset_id,
            "target": data.target_col,
            "problem_type": data.problem_type,
            "primary_metric": data.primary_metric,
            "model_name": data.model_name,
            "engine": data.engine,
            "validation_score": val_score,
            "train_score": train_score,
            "test_score": test_score,
            "train_metrics": train_metrics,
            "tuning_history": tuning_history,
            "generalization_gap": diag.get("generalization_gap"),
            "training_time_seconds": round(float(training_time), 3),
            "prediction_time_seconds": round(float(pred_time), 4),
            "winning_model": {
                "model_name": data.model_name,
                "engine": data.engine,
                "validation_score": val_score,
                "train_score": train_score,
                "test_score": test_score,
                "train_metrics": train_metrics,
                "tuning_history": tuning_history,
                "generalization_gap": diag.get("generalization_gap"),
                "diagnostic_label": diag.get("diagnostic_label"),
                "training_time_seconds": round(float(training_time), 3),
                "prediction_time_seconds": round(float(pred_time), 4)
            },
            "performance": perf,
            "error_analysis": errors,
            "diagnostics": diag,
            "comparison": comp
        }

        # Cache artifacts to disk
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(perf.get("metrics", {}), f, indent=2)
            with open(eval_dir / "errors.json", "w", encoding="utf-8") as f:
                json.dump(errors, f, indent=2)
            with open(eval_dir / "diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2)
            if not is_regression:
                with open(eval_dir / "classification.json", "w", encoding="utf-8") as f:
                    json.dump(perf, f, indent=2)
                if perf.get("calibration"):
                    with open(eval_dir / "calibration.json", "w", encoding="utf-8") as f:
                        json.dump(perf.get("calibration"), f, indent=2)
            else:
                with open(eval_dir / "regression.json", "w", encoding="utf-8") as f:
                    json.dump(perf, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist evaluation cache for {run_id}: {e}")

        self._normalize_evaluation_payload(payload)
        return payload

    def _normalize_evaluation_payload(self, payload: Dict[str, Any]) -> None:
        """Enriches cached or newly generated payloads with backward-compatible aliases and fields."""
        if not isinstance(payload, dict):
            return

        winning = payload.get("winning_model") or {}
        if not isinstance(winning, dict):
            winning = {}
            payload["winning_model"] = winning

        # Ensure top-level fields
        if not payload.get("model_name") and winning.get("model_name"):
            payload["model_name"] = winning["model_name"]
        if not payload.get("engine") and winning.get("engine"):
            payload["engine"] = winning["engine"]
        if payload.get("validation_score") is None and winning.get("validation_score") is not None:
            payload["validation_score"] = winning["validation_score"]
        if payload.get("train_score") is None and winning.get("train_score") is not None:
            payload["train_score"] = winning["train_score"]
        if "train_score" not in winning and payload.get("train_score") is not None:
            winning["train_score"] = payload.get("train_score")
        if not payload.get("train_metrics") and winning.get("train_metrics"):
            payload["train_metrics"] = winning["train_metrics"]
        if "train_metrics" not in winning and payload.get("train_metrics"):
            winning["train_metrics"] = payload.get("train_metrics")
        if not payload.get("tuning_history") and winning.get("tuning_history"):
            payload["tuning_history"] = winning["tuning_history"]
        if "tuning_history" not in winning and payload.get("tuning_history"):
            winning["tuning_history"] = payload.get("tuning_history")
        if payload.get("test_score") is None and winning.get("test_score") is not None:
            payload["test_score"] = winning["test_score"]
        if payload.get("generalization_gap") is None and winning.get("generalization_gap") is not None:
            payload["generalization_gap"] = winning.get("generalization_gap")

        # Times
        train_time = payload.get("training_time_seconds")
        if train_time is None:
            train_time = winning.get("training_time_seconds", 0.0)
            payload["training_time_seconds"] = train_time
        if "training_time_seconds" not in winning:
            winning["training_time_seconds"] = train_time

        pred_time = payload.get("prediction_time_seconds")
        if pred_time is None:
            pred_time = winning.get("prediction_time_seconds", 0.0)
            payload["prediction_time_seconds"] = pred_time
        if "prediction_time_seconds" not in winning:
            winning["prediction_time_seconds"] = pred_time

        # Performance normalization
        perf = payload.get("performance") or {}
        if isinstance(perf, dict):
            # Per class
            if "per_class_metrics" not in perf and "per_class" in perf:
                pcm = {}
                for item in perf.get("per_class", []):
                    lbl = str(item.get("class_label", item.get("class", "")))
                    pcm[lbl] = item
                perf["per_class_metrics"] = pcm
            elif "per_class" not in perf and "per_class_metrics" in perf:
                perf["per_class"] = list(perf["per_class_metrics"].values())

            # Confusion matrix
            cm = perf.get("confusion_matrix")
            if isinstance(cm, dict):
                if "matrix_normalized" not in cm and "normalized_matrix" in cm:
                    cm["matrix_normalized"] = cm["normalized_matrix"]
                elif "normalized_matrix" not in cm and "matrix_normalized" in cm:
                    cm["normalized_matrix"] = cm["matrix_normalized"]

            # ROC curve
            roc = perf.get("roc_curve")
            if isinstance(roc, dict):
                if "x" not in roc and "fpr" in roc:
                    roc["x"] = roc["fpr"]
                if "y" not in roc and "tpr" in roc:
                    roc["y"] = roc["tpr"]

            # PR curve
            pr = perf.get("pr_curve")
            if isinstance(pr, dict):
                if "x" not in pr and "recall" in pr:
                    pr["x"] = pr["recall"]
                if "y" not in pr and "precision" in pr:
                    pr["y"] = pr["precision"]

            # Calibration
            cal = perf.get("calibration")
            if isinstance(cal, dict):
                if "prob_pred" not in cal and "prob_predicted" in cal:
                    cal["prob_pred"] = cal["prob_predicted"]

        # Comparison chart_data
        comp = payload.get("comparison") or {}
        if isinstance(comp, dict):
            cd = comp.get("chart_data") or {}
            if isinstance(cd, dict):
                if "model_names" not in cd and "models" in cd:
                    cd["model_names"] = cd["models"]
                elif "models" not in cd and "model_names" in cd:
                    cd["models"] = cd["model_names"]

    def get_classification_evaluation(self, run_id: str) -> Dict[str, Any]:
        full = self.get_full_evaluation(run_id)
        if full.get("problem_type") == "regression":
            raise AutoDataBotError("Classification evaluation is not applicable to regression tasks")
        return full.get("performance", {})

    def get_regression_evaluation(self, run_id: str) -> Dict[str, Any]:
        full = self.get_full_evaluation(run_id)
        if full.get("problem_type") != "regression":
            raise AutoDataBotError("Regression evaluation is not applicable to classification tasks")
        return full.get("performance", {})

    def get_error_analysis(self, run_id: str) -> Dict[str, Any]:
        full = self.get_full_evaluation(run_id)
        return full.get("error_analysis", {})

    def get_calibration(self, run_id: str) -> Dict[str, Any]:
        full = self.get_full_evaluation(run_id)
        cal = full.get("performance", {}).get("calibration")
        if not cal:
            return {
                "available": False,
                "reason": "Calibration is only available for classification models providing probability estimates."
            }
        return {"available": True, **cal}

    def get_diagnostics(self, run_id: str) -> Dict[str, Any]:
        full = self.get_full_evaluation(run_id)
        return full.get("diagnostics", {})

    def get_comparison(self, run_id: str) -> Dict[str, Any]:
        return compare_run_models(run_id)

evaluation_service = EvaluationService()
