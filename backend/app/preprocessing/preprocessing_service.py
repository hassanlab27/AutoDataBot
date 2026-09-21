import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from app.core.config import settings
from app.core.logging import logger
from app.core.errors import (
    DatasetNotFoundError,
    InvalidTargetError,
    TargetLeakageError,
    PreprocessingError
)
from app.storage.dataset_store import dataset_store
from app.validation.schema_inspector import infer_logical_dtype
from app.preprocessing.target_validator import validate_and_inspect_target
from app.preprocessing.feature_selector import audit_features_for_selection, verify_zero_target_leakage
from app.preprocessing.datetime_transformer import DatetimeComponentExtractor
from app.preprocessing.split_service import perform_train_test_split
from app.preprocessing.pipeline_builder import build_preprocessing_pipeline

class PreprocessingService:
    """
    Coordinates end-to-end leakage-free data preparation for AutoML:
    1. Validates target and confirms problem type
    2. Validates candidate feature list and enforces zero target leakage
    3. Partitions train/test sets before any fitting occurs
    4. Extracts datetime components if enabled
    5. Fits sklearn ColumnTransformer strictly on training data
    6. Transforms train and test splits
    7. Serializes pipeline.joblib and records config/metadata artifacts
    """

    def get_preprocessing_dir(self, dataset_id: str) -> Path:
        clean_id = "".join(c for c in dataset_id if c.isalnum() or c in "_-")
        target_dir = settings.OUTPUTS_DIR / "datasets" / clean_id / "preprocessing"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def inspect_target(self, dataset_id: str, target_col: str) -> Dict[str, Any]:
        """Validate target column, determine problem type, and return diagnostics."""
        df = dataset_store.load_dataset(dataset_id)
        return validate_and_inspect_target(df, target_col)

    def audit_features(self, dataset_id: str, target_col: str) -> Dict[str, Any]:
        """Audit candidate features, flag potential issues, and check leakage risks."""
        df = dataset_store.load_dataset(dataset_id)
        return audit_features_for_selection(df, target_col)

    def prepare_dataset(
        self,
        dataset_id: str,
        target: str,
        problem_type: str,
        selected_features: Optional[List[str]] = None,
        excluded_features: Optional[List[str]] = None,
        test_size: float = settings.PREPROCESSING_DEFAULT_TEST_SIZE,
        random_state: int = settings.PREPROCESSING_DEFAULT_RANDOM_STATE,
        numeric_imputation: str = "median",
        categorical_imputation: str = "most_frequent",
        categorical_encoding: str = "one_hot",
        scaling: str = "standard",
        extract_datetime: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full leakage-free preparation and pipeline serialization.
        """
        df = dataset_store.load_dataset(dataset_id)
        logger.info(f"Beginning preprocessing preparation for dataset: {dataset_id}, target: {target}")

        # 1. Target Validation
        target_info = validate_and_inspect_target(df, target)
        
        # 2. Determine Feature Set & Guard Against Leakage
        all_candidate_cols = [c for c in df.columns if c != target]
        
        if selected_features is not None:
            active_features = [f for f in selected_features if f in df.columns and f != target]
        elif excluded_features is not None:
            active_features = [f for f in all_candidate_cols if f not in excluded_features]
        else:
            # Default: audit features and exclude constant/direct-duplicate columns
            feature_audit = audit_features_for_selection(df, target)
            active_features = [
                f["name"] for f in feature_audit["features"] 
                if f["status"] != "flagged_exclude"
            ]

        # Explicit Zero Leakage Barrier
        verify_zero_target_leakage(active_features, target)

        if not active_features:
            raise PreprocessingError("No features selected for preprocessing. At least 1 feature is required.")

        # 3. Leakage-Free Train/Test Split (Executed BEFORE any transformation fitting)
        X_train_raw, X_test_raw, y_train, y_test, split_meta = perform_train_test_split(
            df=df,
            target_col=target,
            feature_cols=active_features,
            test_size=test_size,
            random_state=random_state,
            problem_type=problem_type
        )

        logger.info(
            f"Dataset split successfully: Train={split_meta['train_rows']} rows, "
            f"Test={split_meta['test_rows']} rows ({split_meta['split_strategy']})"
        )

        # 4. Datetime Feature Component Extraction (if enabled)
        datetime_cols = [
            col for col in active_features 
            if infer_logical_dtype(X_train_raw[col]) == "datetime-like"
        ]
        
        dt_extractor = None
        if extract_datetime and datetime_cols:
            dt_extractor = DatetimeComponentExtractor(datetime_columns=datetime_cols)
            dt_extractor.fit(X_train_raw)
            X_train_dt = dt_extractor.transform(X_train_raw)
            X_test_dt = dt_extractor.transform(X_test_raw)
            generated_dt_count = len(dt_extractor.feature_names_out_)
        else:
            X_train_dt = X_train_raw.copy()
            X_test_dt = X_test_raw.copy()
            generated_dt_count = 0

        # 5. Classify Remaining Features into Numeric vs Categorical
        numeric_features = []
        categorical_features = []
        
        for col in X_train_dt.columns:
            l_dtype = infer_logical_dtype(X_train_dt[col])
            if l_dtype in ("integer", "float"):
                numeric_features.append(str(col))
            else:
                categorical_features.append(str(col))

        # 6. Build & Fit ColumnTransformer Pipeline STRICTLY on Training Data
        preprocessor = build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_imputation=numeric_imputation,
            categorical_imputation=categorical_imputation,
            categorical_encoding=categorical_encoding,
            scaling=scaling
        )

        try:
            # FIT strictly on X_train_dt
            preprocessor.fit(X_train_dt)
            # Transform train and test
            X_train_transformed = preprocessor.transform(X_train_dt)
            X_test_transformed = preprocessor.transform(X_test_dt)
        except Exception as e:
            logger.error(f"Error fitting ColumnTransformer pipeline: {str(e)}", exc_info=True)
            raise PreprocessingError(f"Failed to fit preprocessing pipeline: {str(e)}")

        # Retrieve output feature names
        try:
            raw_names = list(preprocessor.get_feature_names_out())
            # Clean transformer prefixes (e.g., 'num__age' -> 'age', 'cat__dept_HR' -> 'dept_HR')
            output_feature_names = [
                name.split("__", 1)[-1] if "__" in name else name 
                for name in raw_names
            ]
        except Exception:
            output_feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

        final_feature_count = len(output_feature_names)
        one_hot_count = max(final_feature_count - len(numeric_features), 0)

        # 7. Persist Artifacts (pipeline.joblib, config.json, metadata.json)
        prep_dir = self.get_preprocessing_dir(dataset_id)
        pipeline_path = prep_dir / "pipeline.joblib"
        config_path = prep_dir / "config.json"
        meta_path = prep_dir / "metadata.json"

        # Bundle full pipeline (including datetime extractor if used)
        serializable_bundle = {
            "datetime_extractor": dt_extractor,
            "column_transformer": preprocessor,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "problem_type": problem_type,
            "feature_names_in": active_features,
            "feature_names_out": output_feature_names
        }
        joblib.dump(serializable_bundle, pipeline_path)

        config_data = {
            "dataset_id": dataset_id,
            "target": target,
            "problem_type": problem_type,
            "selected_features": active_features,
            "excluded_features": [c for c in all_candidate_cols if c not in active_features],
            "test_size": test_size,
            "random_state": random_state,
            "numeric_imputation": numeric_imputation,
            "categorical_imputation": categorical_imputation,
            "categorical_encoding": categorical_encoding,
            "scaling": scaling,
            "datetime_features": extract_datetime and bool(datetime_cols)
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        meta_data = {
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_features_count": len(all_candidate_cols),
            "selected_features_count": len(active_features),
            "excluded_features_count": len(all_candidate_cols) - len(active_features),
            "final_features_count": final_feature_count,
            "numeric_features_count": len(numeric_features),
            "categorical_features_count": len(categorical_features),
            "generated_datetime_features_count": generated_dt_count,
            "one_hot_generated_features_count": one_hot_count,
            "feature_names_out": output_feature_names,
            "split": split_meta
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        # Save small cache of preview records in memory for quick retrieval
        preview_rows = []
        preview_limit = min(20, X_train_transformed.shape[0])
        preview_arr = X_train_transformed[:preview_limit]
        if hasattr(preview_arr, "toarray"):
            preview_arr = preview_arr.toarray()

        for i in range(preview_limit):
            row_dict = {}
            for col_idx, col_name in enumerate(output_feature_names):
                val = preview_arr[i, col_idx]
                row_dict[col_name] = round(float(val), 4) if not (np.isnan(val) or np.isinf(val)) else None
            preview_rows.append(row_dict)

        logger.info(
            f"Preprocessing completed successfully for {dataset_id}: "
            f"{len(active_features)} features -> {final_feature_count} processed features"
        )

        return {
            "status": "prepared",
            "dataset_id": dataset_id,
            "target": target,
            "problem_type": problem_type,
            "config": config_data,
            "metadata": meta_data,
            "preview": {
                "total_rows": X_train_transformed.shape[0],
                "preview_rows": preview_rows,
                "columns": output_feature_names[:50]  # First 50 columns for UI preview
            }
        }

    def get_summary(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve stored metadata of prepared dataset."""
        prep_dir = self.get_preprocessing_dir(dataset_id)
        meta_path = prep_dir / "metadata.json"
        config_path = prep_dir / "config.json"

        if not meta_path.is_file() or not config_path.is_file():
            return {
                "status": "not_prepared",
                "dataset_id": dataset_id,
                "message": "Dataset has not been preprocessed yet."
            }

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        return {
            "status": "prepared",
            "dataset_id": dataset_id,
            "config": config_data,
            "metadata": meta_data
        }

    def get_preview(self, dataset_id: str, limit: int = 20) -> Dict[str, Any]:
        """Retrieve small preview of processed training features."""
        prep_dir = self.get_preprocessing_dir(dataset_id)
        pipeline_path = prep_dir / "pipeline.joblib"
        meta_path = prep_dir / "metadata.json"
        config_path = prep_dir / "config.json"

        if not pipeline_path.is_file() or not meta_path.is_file():
            return {
                "status": "not_prepared",
                "dataset_id": dataset_id,
                "rows": [],
                "columns": []
            }

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        bundle = joblib.load(pipeline_path)
        df = dataset_store.load_dataset(dataset_id)
        target = config_data["target"]
        active_features = config_data["selected_features"]
        
        # Reproduce train split using stored seed
        X_train_raw, _, _, _, _ = perform_train_test_split(
            df=df,
            target_col=target,
            feature_cols=active_features,
            test_size=config_data["test_size"],
            random_state=config_data["random_state"],
            problem_type=config_data["problem_type"]
        )

        dt_extractor = bundle.get("datetime_extractor")
        if dt_extractor:
            X_train_dt = dt_extractor.transform(X_train_raw)
        else:
            X_train_dt = X_train_raw.copy()

        column_transformer = bundle["column_transformer"]
        X_transformed = column_transformer.transform(X_train_dt.head(limit))
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        output_names = meta_data["feature_names_out"]
        rows = []
        for i in range(min(limit, X_transformed.shape[0])):
            row_dict = {}
            for col_idx, col_name in enumerate(output_names):
                val = X_transformed[i, col_idx]
                row_dict[col_name] = round(float(val), 4) if not (np.isnan(val) or np.isinf(val)) else None
            rows.append(row_dict)

        return {
            "status": "prepared",
            "dataset_id": dataset_id,
            "total_training_rows": meta_data["split"]["train_rows"],
            "returned_rows": len(rows),
            "columns": output_names,
            "rows": rows
        }

preprocessing_service = PreprocessingService()
