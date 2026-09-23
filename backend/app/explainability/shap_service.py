import os
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.errors import AutoDataBotError
from app.evaluation.loader import data_loader, EvaluationData
from app.explainability.feature_importance import extract_native_feature_importance
from app.explainability.permutation import compute_permutation_importance

logger = logging.getLogger(__name__)

# Execution status registry for background SHAP computations
_shap_jobs_lock = threading.Lock()
_shap_jobs_status: Dict[str, Dict[str, Any]] = {}

class ExplainabilityService:
    DEFAULT_SAMPLE_SIZE = 500
    DEFAULT_MAX_FEATURES = 20

    def get_explain_dir(self, run_id: str) -> Path:
        clean_id = "".join(c for c in run_id if c.isalnum() or c in "_-")
        exp_dir = settings.OUTPUTS_DIR / "runs" / clean_id / "explainability"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "local").mkdir(parents=True, exist_ok=True)
        return exp_dir

    def get_native_importance(self, run_id: str) -> Dict[str, Any]:
        """Returns model native feature importance."""
        exp_dir = self.get_explain_dir(run_id)
        cache_path = exp_dir / "feature_importance.json"

        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        data = data_loader.load_evaluation_data(run_id)
        res = extract_native_feature_importance(
            model=data.model_obj,
            feature_names=data.feature_names,
            feature_map=data.feature_map
        )

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        except Exception:
            pass

        return res

    def get_permutation_importance(self, run_id: str, n_repeats: int = 5) -> Dict[str, Any]:
        """Returns permutation feature importance computed on validation partition."""
        exp_dir = self.get_explain_dir(run_id)
        cache_path = exp_dir / "permutation.json"

        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        data = data_loader.load_evaluation_data(run_id)
        scoring = "r2" if data.problem_type == "regression" else "accuracy"
        res = compute_permutation_importance(
            model=data.model_obj,
            X=data.X_val_prep,
            y=data.y_val,
            feature_names=data.feature_names,
            feature_map=data.feature_map,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=42,
            data_partition_label="validation"
        )

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        except Exception:
            pass

        return res

    def get_shap_status(self, run_id: str) -> Dict[str, Any]:
        """Returns the execution state of SHAP calculation."""
        exp_dir = self.get_explain_dir(run_id)
        cache_path = exp_dir / "shap_summary.json"
        if cache_path.exists():
            return {"run_id": run_id, "status": "completed", "progress_pct": 100}

        with _shap_jobs_lock:
            job = _shap_jobs_status.get(run_id)
            if job:
                return job

        return {"run_id": run_id, "status": "not_started", "progress_pct": 0}

    def start_shap_computation(
        self,
        run_id: str,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        max_features: int = DEFAULT_MAX_FEATURES
    ) -> Dict[str, Any]:
        """Starts asynchronous SHAP computation in the background if not already cached."""
        exp_dir = self.get_explain_dir(run_id)
        cache_path = exp_dir / "shap_summary.json"
        if cache_path.exists():
            return {"run_id": run_id, "status": "completed", "message": "SHAP results already cached."}

        with _shap_jobs_lock:
            current = _shap_jobs_status.get(run_id)
            if current and current.get("status") == "running":
                return current

            _shap_jobs_status[run_id] = {
                "run_id": run_id,
                "status": "running",
                "progress_pct": 10,
                "message": "Initializing SHAP explainer..."
            }

        thread = threading.Thread(
            target=self._run_shap_worker,
            args=(run_id, sample_size, max_features),
            daemon=True
        )
        thread.start()

        return {"run_id": run_id, "status": "running", "message": "SHAP calculation started in background."}

    def _run_shap_worker(self, run_id: str, sample_size: int, max_features: int) -> None:
        try:
            logger.info(f"Starting background SHAP calculation for run {run_id}")
            data = data_loader.load_evaluation_data(run_id)
            res = self._compute_shap(data, sample_size=sample_size, max_features=max_features)

            exp_dir = self.get_explain_dir(run_id)
            with open(exp_dir / "shap_summary.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

            with _shap_jobs_lock:
                _shap_jobs_status[run_id] = {
                    "run_id": run_id,
                    "status": "completed" if res.get("available") else "failed",
                    "progress_pct": 100,
                    "reason": res.get("reason"),
                    "message": "SHAP calculation completed."
                }
            logger.info(f"SHAP calculation completed for run {run_id}")
        except Exception as e:
            logger.error(f"Error in SHAP calculation for run {run_id}: {e}", exc_info=True)
            with _shap_jobs_lock:
                _shap_jobs_status[run_id] = {
                    "run_id": run_id,
                    "status": "failed",
                    "progress_pct": 100,
                    "error": str(e),
                    "message": f"SHAP failed: {str(e)}"
                }

    def get_shap_summary(
        self,
        run_id: str,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        max_features: int = DEFAULT_MAX_FEATURES
    ) -> Dict[str, Any]:
        """Synchronously retrieves or computes SHAP summary."""
        exp_dir = self.get_explain_dir(run_id)
        cache_path = exp_dir / "shap_summary.json"

        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        data = data_loader.load_evaluation_data(run_id)
        res = self._compute_shap(data, sample_size=sample_size, max_features=max_features)

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        except Exception:
            pass

        return res

    def _compute_shap(
        self,
        data: EvaluationData,
        sample_size: int,
        max_features: int
    ) -> Dict[str, Any]:
        """Core SHAP execution logic with model-type dispatch and safe failure handling."""
        try:
            import shap
        except ImportError:
            return {
                "available": False,
                "reason": "SHAP library is not installed in the environment.",
                "global_importance": [],
                "summary_points": []
            }

        model = data.model_obj
        if data.engine.lower() == "autogluon":
            return {
                "available": False,
                "reason": "SHAP explainer not supported for this AutoGluon ensemble model structure.",
                "global_importance": [],
                "summary_points": []
            }

        # Compute SHAP on validation partition to keep external test set untouched
        X = data.X_val_prep if (data.X_val_prep is not None and len(data.X_val_prep) > 0) else data.X_tr_prep
        if X is None or len(X) == 0:
            X = data.X_train_prep
        n_samples = len(X) if X is not None else 0
        if n_samples == 0:
            return {"available": False, "reason": "Validation evaluation dataset is empty."}

        # Deterministic sampling for speed
        rng = np.random.RandomState(42)
        if n_samples > sample_size:
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[sample_idx]
        else:
            sample_idx = np.arange(n_samples)
            X_sample = X

        try:
            explainer = None
            shap_values = None

            # 1. Try TreeExplainer first for tree ensembles
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except Exception:
                pass

            # 2. Try LinearExplainer for linear models
            if shap_values is None:
                try:
                    # Use a small background summary
                    bg = shap.sample(X, min(50, len(X)), random_state=42)
                    explainer = shap.LinearExplainer(model, bg)
                    shap_values = explainer.shap_values(X_sample)
                except Exception:
                    pass

            # 3. Model-agnostic Explainer fallback on background subset
            if shap_values is None:
                try:
                    bg = shap.sample(X, min(30, len(X)), random_state=42)
                    if hasattr(model, "predict_proba") and not (data.problem_type == "regression"):
                        explainer = shap.Explainer(model.predict_proba, bg)
                    else:
                        explainer = shap.Explainer(model.predict, bg)
                    res_obj = explainer(X_sample)
                    shap_values = res_obj.values
                except Exception as e:
                    logger.warning(f"Fallback Explainer failed: {e}")

            if shap_values is None:
                return {
                    "available": False,
                    "reason": "Model architecture is not supported by available SHAP explainers.",
                    "global_importance": [],
                    "summary_points": []
                }

            # Standardize shap_values shape:
            # - For binary classification, TreeExplainer may return a list of 2 arrays [neg, pos] or 2D array
            # - For multiclass, list of arrays or 3D array (samples, features, classes)
            # - For regression, 2D array (samples, features)
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    # Binary classification: positive class values
                    shap_matrix = np.asarray(shap_values[1], dtype=float)
                else:
                    # Multiclass: average absolute SHAP values across classes
                    shap_matrix = np.mean(np.abs(np.asarray(shap_values, dtype=float)), axis=0)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Shape (samples, features, classes) -> average absolute over classes
                shap_matrix = np.mean(np.abs(shap_values), axis=-1)
            else:
                shap_matrix = np.asarray(shap_values, dtype=float)

            # Check dimensions match feature_names
            n_feats = shap_matrix.shape[1] if shap_matrix.ndim >= 2 else 1
            feature_names = data.feature_names[:n_feats]
            while len(feature_names) < n_feats:
                feature_names.append(f"feature_{len(feature_names)}")

            # Global Mean Absolute SHAP values
            mean_abs_vals = np.mean(np.abs(shap_matrix), axis=0)
            global_records: List[Dict[str, Any]] = []
            for i in range(n_feats):
                feat_name = feature_names[i]
                orig_name = data.feature_map.get(feat_name, feat_name)
                global_records.append({
                    "feature": feat_name,
                    "original_feature": orig_name,
                    "mean_abs_shap": round(float(mean_abs_vals[i]), 4)
                })

            global_records.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
            top_features_list = [r["feature"] for r in global_records[:max_features]]

            # Summary points for top features (downsampled for Plotly)
            summary_points: List[Dict[str, Any]] = []
            n_plot_points = min(len(X_sample), 100)
            plot_indices = rng.choice(len(X_sample), size=n_plot_points, replace=False) if len(X_sample) > 100 else np.arange(len(X_sample))

            for feat in top_features_list:
                f_idx = feature_names.index(feat)
                f_vals = X_sample[plot_indices, f_idx]
                s_vals = shap_matrix[plot_indices, f_idx]

                min_f = float(np.min(f_vals))
                max_f = float(np.max(f_vals))
                span = max_f - min_f if max_f - min_f > 1e-6 else 1.0

                for p_i, val in enumerate(f_vals):
                    norm_val = round((float(val) - min_f) / span, 3)
                    summary_points.append({
                        "feature": feat,
                        "original_feature": data.feature_map.get(feat, feat),
                        "shap_value": round(float(s_vals[p_i]), 4),
                        "feature_value": round(float(val), 4),
                        "normalized_value": norm_val
                    })

            return {
                "available": True,
                "partition": "validation",
                "sample_size": len(X_sample),
                "total_features": n_feats,
                "global_importance": global_records,
                "top_features": top_features_list,
                "summary_points": summary_points,
                "disclaimer": "SHAP values describe how the model attributes output variations; they do not establish real-world causality."
            }

        except Exception as e:
            logger.error(f"SHAP calculation encountered an exception: {e}", exc_info=True)
            return {
                "available": False,
                "reason": f"SHAP calculation failed: {str(e)}",
                "global_importance": [],
                "summary_points": []
            }

    def get_local_prediction_explanation(
        self,
        run_id: str,
        prediction_id: int
    ) -> Dict[str, Any]:
        """Explains a single test prediction using local SHAP attribution."""
        exp_dir = self.get_explain_dir(run_id)
        local_cache = exp_dir / "local" / f"pred_{prediction_id}.json"

        if local_cache.exists():
            try:
                with open(local_cache, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        data = data_loader.load_evaluation_data(run_id)
        model = data.model_obj
        X = data.X_test_prep
        y = data.y_test

        if prediction_id < 0 or prediction_id >= len(X):
            raise AutoDataBotError(f"Prediction ID {prediction_id} out of bounds (test set has {len(X)} records).")

        row = X[prediction_id:prediction_id+1]
        actual_val = str(y[prediction_id])
        prob_val = None

        if data.engine.lower() == "autogluon":
            raw_row = data.X_test_raw.iloc[prediction_id:prediction_id+1]
            ag_preds = model.predict(raw_row)
            pred_val = ag_preds.iloc[0] if hasattr(ag_preds, "iloc") else ag_preds[0]
            try:
                probs = model.predict_proba(raw_row)
                prob_val = round(float(np.max(probs.to_numpy())), 4)
            except Exception:
                pass
        else:
            pred_val = model.predict(row)[0]
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(row)[0]
                    prob_val = round(float(np.max(probs)), 4)
                except Exception:
                    pass

        # Calculate SHAP for this single row
        shap_vals_row = None
        base_value = 0.0

        try:
            import shap
            explainer = None
            try:
                explainer = shap.TreeExplainer(model)
                shap_obj = explainer.shap_values(row)
            except Exception:
                bg_pool = data.X_val_prep if (data.X_val_prep is not None and len(data.X_val_prep) > 0) else X
                bg = shap.sample(bg_pool, min(30, len(bg_pool)), random_state=42)
                explainer = shap.Explainer(model.predict, bg)
                res_obj = explainer(row)
                shap_obj = res_obj.values

            if isinstance(shap_obj, list):
                # Binary or multiclass
                shap_vals_row = shap_obj[1][0] if len(shap_obj) == 2 else np.mean(shap_obj, axis=0)[0]
            elif isinstance(shap_obj, np.ndarray) and shap_obj.ndim == 3:
                shap_vals_row = np.mean(shap_obj, axis=-1)[0]
            else:
                shap_vals_row = shap_obj[0]

            if hasattr(explainer, "expected_value"):
                ev = explainer.expected_value
                base_value = round(float(ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) == 2 else (ev[0] if isinstance(ev, (list, np.ndarray)) else ev)), 4)
        except Exception as e:
            logger.warning(f"Could not compute local SHAP for prediction {prediction_id}: {e}")

        top_contributions: List[Dict[str, Any]] = []
        if shap_vals_row is not None:
            for i, val in enumerate(shap_vals_row):
                feat_name = data.feature_names[i] if i < len(data.feature_names) else f"feature_{i}"
                orig_name = data.feature_map.get(feat_name, feat_name)
                f_val = round(float(row[0, i]), 4)
                top_contributions.append({
                    "feature": feat_name,
                    "original_feature": orig_name,
                    "feature_value": f_val,
                    "shap_value": round(float(val), 4),
                    "abs_shap": round(float(abs(val)), 4),
                    "direction": "positive" if val >= 0 else "negative"
                })

            top_contributions.sort(key=lambda r: r["abs_shap"], reverse=True)

        res = {
            "prediction_id": prediction_id,
            "actual": actual_val,
            "predicted": str(pred_val),
            "probability": prob_val,
            "base_value": base_value,
            "available": (shap_vals_row is not None),
            "top_contributions": top_contributions[:15],
            "disclaimer": "These values describe how the model attributed this prediction; they do not establish causation."
        }

        try:
            with open(local_cache, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        except Exception:
            pass

        return res

    def get_feature_dependence(
        self,
        run_id: str,
        feature_name: str
    ) -> Dict[str, Any]:
        """Returns feature values vs SHAP values for a selected feature."""
        summary = self.get_shap_summary(run_id)
        if not summary.get("available"):
            return {"available": False, "reason": summary.get("reason")}

        points = [p for p in summary.get("summary_points", []) if p.get("feature") == feature_name or p.get("original_feature") == feature_name]
        return {
            "available": len(points) > 0,
            "feature": feature_name,
            "points": points
        }

explainability_service = ExplainabilityService()
