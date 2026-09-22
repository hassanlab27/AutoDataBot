import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.errors import AutoDataBotError
from app.storage.dataset_store import dataset_store
from app.preprocessing.preprocessing_service import PreprocessingService
from app.preprocessing.split_service import perform_train_test_split

logger = logging.getLogger(__name__)

@dataclass
class EvaluationData:
    run_id: str
    dataset_id: str
    target_col: str
    problem_type: str
    primary_metric: str
    model_obj: Any
    model_name: str
    engine: str
    model_bundle: Dict[str, Any]
    config: Dict[str, Any]
    X_test_prep: np.ndarray
    X_test_raw: pd.DataFrame
    y_test: np.ndarray
    X_val_prep: np.ndarray
    y_val: np.ndarray
    feature_names: List[str]
    feature_map: Dict[str, str]  # transformed_feature -> original_feature

class EvaluationDataLoader:
    def __init__(self):
        self.prep_service = PreprocessingService()

    def get_run_dir(self, run_id: str) -> Path:
        clean_id = "".join(c for c in run_id if c.isalnum() or c in "_-")
        return settings.OUTPUTS_DIR / "runs" / clean_id

    def load_evaluation_data(self, run_id: str) -> EvaluationData:
        run_dir = self.get_run_dir(run_id)
        if not run_dir.exists():
            raise AutoDataBotError(f"Run directory for '{run_id}' not found")

        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise AutoDataBotError(f"Run '{run_id}' is missing configuration file (config.json)")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        dataset_id = config.get("dataset_id")
        target_col = config.get("target")
        problem_type = config.get("problem_type")
        primary_metric = config.get("effective_primary_metric", config.get("primary_metric", "accuracy"))

        # 1. Load trained winning model
        model_bundle_path = run_dir / "models" / "final_model.joblib"
        ag_models_dir = run_dir / "models" / "autogluon"
        if not ag_models_dir.exists():
            ag_models_dir = run_dir / "models" / "ag_models"

        model_obj = None
        model_bundle: Dict[str, Any] = {}
        model_name = "Model"
        engine = "sklearn"

        if model_bundle_path.exists():
            model_bundle = joblib.load(model_bundle_path)
            model_obj = model_bundle.get("model_object")
            model_name = model_bundle.get("model_name", "Model")
            engine = model_bundle.get("engine", "sklearn")
        elif ag_models_dir.exists():
            try:
                from autogluon.tabular import TabularPredictor
                model_obj = TabularPredictor.load(str(ag_models_dir))
                model_name = "AutoGluon Ensemble"
                engine = "autogluon"
                model_bundle = {
                    "model_id": "autogluon_predictor",
                    "model_name": model_name,
                    "engine": engine
                }
            except Exception as e:
                logger.warning(f"Could not load AutoGluon predictor: {e}")
        else:
            # Check leaderboard for details
            lb_path = run_dir / "leaderboard.json"
            if lb_path.exists():
                with open(lb_path, "r", encoding="utf-8") as f:
                    lb = json.load(f)
                    winners = [m for m in lb if m.get("is_winner")]
                    if winners:
                        model_name = winners[0].get("model_name", "Model")
                        engine = winners[0].get("engine", "sklearn")

        if model_obj is None:
            raise AutoDataBotError(f"No trained model object found for run '{run_id}'. Run may have failed or was cancelled.")

        # 2. Re-partition dataset using saved deterministic split configuration
        df = dataset_store.load_dataset(dataset_id)
        prep_dir = self.prep_service.get_preprocessing_dir(dataset_id)
        prep_cfg_path = prep_dir / "config.json"
        prep_config: Dict[str, Any] = {}
        if prep_cfg_path.exists():
            with open(prep_cfg_path, "r", encoding="utf-8") as f:
                prep_config = json.load(f)

        feature_cols = prep_config.get("selected_features", config.get("selected_features"))
        test_size = prep_config.get("test_size", settings.PREPROCESSING_DEFAULT_TEST_SIZE)
        random_state = prep_config.get("random_state", config.get("random_state", 42))

        X_train_raw, X_test_raw, y_train, y_test, _ = perform_train_test_split(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            test_size=test_size,
            random_state=random_state,
            problem_type=problem_type
        )

        # 3. Transform via fitted preprocessing pipeline
        pipeline_path = prep_dir / "pipeline.joblib"
        if not pipeline_path.exists():
            raise AutoDataBotError(f"Preprocessing pipeline not found for dataset '{dataset_id}'")

        bundle = joblib.load(pipeline_path)
        dt_extractor = bundle.get("datetime_extractor")
        col_transformer = bundle.get("column_transformer")

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

        X_train_prep = np.asarray(X_train_prep, dtype=np.float32)
        X_test_prep = np.asarray(X_test_prep, dtype=np.float32)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # Internal validation split matching Phase 4 training
        stratify = y_train if problem_type.endswith("classification") else None
        try:
            X_tr_prep, X_val_prep, y_tr, y_val = train_test_split(
                X_train_prep,
                y_train,
                test_size=0.20,
                random_state=random_state,
                stratify=stratify
            )
        except Exception:
            X_tr_prep, X_val_prep, y_tr, y_val = train_test_split(
                X_train_prep,
                y_train,
                test_size=0.20,
                random_state=random_state
            )

        # 4. Extract feature names and build transformed -> raw feature mapping
        feature_names: List[str] = []
        feature_map: Dict[str, str] = {}

        try:
            raw_feature_names = list(col_transformer.get_feature_names_out())
            for name in raw_feature_names:
                clean_name = str(name)
                feature_names.append(clean_name)
                # Map e.g. "cat__Sex_female" -> "Sex", "num__Age" -> "Age", "remainder__Col" -> "Col"
                # Strip prefix before double underscore if present
                if "__" in clean_name:
                    prefix_removed = clean_name.split("__", 1)[1]
                else:
                    prefix_removed = clean_name
                
                # Check which original column it originates from
                matched_orig = prefix_removed
                if feature_cols:
                    for orig in feature_cols:
                        if prefix_removed == orig or prefix_removed.startswith(f"{orig}_"):
                            matched_orig = orig
                            break
                feature_map[clean_name] = matched_orig
        except Exception as e:
            logger.warning(f"Could not extract feature names out from column transformer: {e}")
            feature_names = [f"feature_{i}" for i in range(X_test_prep.shape[1])]
            feature_map = {f: f for f in feature_names}

        return EvaluationData(
            run_id=run_id,
            dataset_id=dataset_id,
            target_col=target_col,
            problem_type=problem_type,
            primary_metric=primary_metric,
            model_obj=model_obj,
            model_name=model_name,
            engine=engine,
            model_bundle=model_bundle,
            config=config,
            X_test_prep=X_test_prep,
            X_test_raw=X_test_raw,
            y_test=y_test,
            X_val_prep=X_val_prep,
            y_val=y_val,
            feature_names=feature_names,
            feature_map=feature_map
        )

data_loader = EvaluationDataLoader()
