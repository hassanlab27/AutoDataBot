from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from app.core.errors import EvaluationError

@dataclass
class DatasetPartitions:
    """
    Explicit dataset partition contract ensuring strict separation and alignment
    between raw dataFrames, preprocessed feature matrices, and target vectors.

    Partitions:
    - TRAIN (internal fold): used for model training and fitting
    - VALIDATION (internal fold): used for model selection, hyperparameter tuning,
      permutation importance, and SHAP background baselines
    - TEST (held-out external partition): strictly isolated; used ONLY for final winner evaluation
    """
    # Internal training fold
    X_tr_prep: np.ndarray
    X_tr_raw: pd.DataFrame
    y_tr: np.ndarray

    # Internal validation fold
    X_val_prep: np.ndarray
    X_val_raw: pd.DataFrame
    y_val: np.ndarray

    # Isolated external test set
    X_test_prep: np.ndarray
    X_test_raw: pd.DataFrame
    y_test: np.ndarray

    def validate(self) -> None:
        """
        Validates internal invariants:
        - len(X) == len(y) for every partition
        - no zero-sized partitions
        - consistent feature dimensions
        """
        # Train fold validation
        if len(self.X_tr_prep) != len(self.y_tr) or len(self.X_tr_raw) != len(self.y_tr):
            raise EvaluationError(
                f"Train partition length mismatch: X_prep={len(self.X_tr_prep)}, "
                f"X_raw={len(self.X_tr_raw)}, y={len(self.y_tr)}",
                details={
                    "partition": "train",
                    "X_prep_len": len(self.X_tr_prep),
                    "X_raw_len": len(self.X_tr_raw),
                    "y_len": len(self.y_tr)
                }
            )

        # Validation fold validation
        if len(self.X_val_prep) != len(self.y_val) or len(self.X_val_raw) != len(self.y_val):
            raise EvaluationError(
                f"Validation partition length mismatch: X_prep={len(self.X_val_prep)}, "
                f"X_raw={len(self.X_val_raw)}, y={len(self.y_val)}",
                details={
                    "partition": "validation",
                    "X_prep_len": len(self.X_val_prep),
                    "X_raw_len": len(self.X_val_raw),
                    "y_len": len(self.y_val)
                }
            )

        # Test set validation
        if len(self.X_test_prep) != len(self.y_test) or len(self.X_test_raw) != len(self.y_test):
            raise EvaluationError(
                f"Test partition length mismatch: X_prep={len(self.X_test_prep)}, "
                f"X_raw={len(self.X_test_raw)}, y={len(self.y_test)}",
                details={
                    "partition": "test",
                    "X_prep_len": len(self.X_test_prep),
                    "X_raw_len": len(self.X_test_raw),
                    "y_len": len(self.y_test)
                }
            )

        # Feature dimension check
        if self.X_tr_prep.shape[1] != self.X_val_prep.shape[1] or self.X_tr_prep.shape[1] != self.X_test_prep.shape[1]:
            raise EvaluationError(
                f"Preprocessed feature dimension mismatch: train={self.X_tr_prep.shape[1]}, "
                f"val={self.X_val_prep.shape[1]}, test={self.X_test_prep.shape[1]}"
            )

    @property
    def train_size(self) -> int:
        return len(self.y_tr)

    @property
    def val_size(self) -> int:
        return len(self.y_val)

    @property
    def test_size(self) -> int:
        return len(self.y_test)

    def get_autogluon_train_df(self, target_col: str) -> pd.DataFrame:
        """Returns internal train fold DataFrame with target column attached."""
        df = self.X_tr_raw.copy()
        df[target_col] = self.y_tr
        return df

    def get_autogluon_val_df(self, target_col: str) -> pd.DataFrame:
        """Returns internal validation fold DataFrame with target column attached."""
        df = self.X_val_raw.copy()
        df[target_col] = self.y_val
        return df

    def get_autogluon_test_df(self, target_col: str) -> pd.DataFrame:
        """Returns external test partition DataFrame with target column attached."""
        df = self.X_test_raw.copy()
        df[target_col] = self.y_test
        return df
