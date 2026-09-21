from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class InfiniteValueSanitizer(BaseEstimator, TransformerMixin):
    """
    Sanitizes infinite numerical values (+inf, -inf) into np.nan
    so that downstream SimpleImputer handles them transparently without throwing exceptions.
    """
    def fit(self, X, y=None):
        if hasattr(X, "shape") and len(X.shape) > 1:
            self.n_features_in_ = X.shape[1]
        else:
            self.n_features_in_ = 1
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=float, copy=True)
        X_arr[np.isinf(X_arr)] = np.nan
        return X_arr

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(getattr(self, "n_features_in_", 0))], dtype=object)
        return np.asarray(input_features, dtype=object)

def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    numeric_imputation: str = "median",
    categorical_imputation: str = "most_frequent",
    categorical_encoding: str = "one_hot",
    scaling: str = "standard"
) -> ColumnTransformer:
    """
    Builds a serializable scikit-learn ColumnTransformer configured with
    leakage-resistant imputers, encoders, and scalers.
    """
    transformers = []

    # 1. Numerical Pipeline
    if numeric_features:
        num_steps = [
            ("inf_sanitizer", InfiniteValueSanitizer()),
            ("imputer", SimpleImputer(strategy=numeric_imputation))
        ]
        if scaling == "standard":
            num_steps.append(("scaler", StandardScaler()))
            
        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("numeric", num_pipeline, numeric_features))

    # 2. Categorical Pipeline
    if categorical_features:
        cat_steps = [
            ("imputer", SimpleImputer(strategy=categorical_imputation, fill_value="__MISSING__"))
        ]
        if categorical_encoding == "one_hot":
            cat_steps.append((
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ))
            
        cat_pipeline = Pipeline(steps=cat_steps)
        transformers.append(("categorical", cat_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No valid numerical or categorical features provided to build preprocessing pipeline.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor
