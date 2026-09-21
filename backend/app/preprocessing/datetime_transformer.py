from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeComponentExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numeric calendar components from datetime-like columns:
    - year
    - month
    - day
    - dayofweek
    - hour (if variance exists)
    Drops the original raw string/datetime column.
    Handles unparseable dates safely by coercing to NaT.
    """
    def __init__(self, datetime_columns: Optional[List[str]] = None):
        self.datetime_columns = datetime_columns or []
        self.feature_names_out_: List[str] = []

    def fit(self, X, y=None):
        # Determine output feature names
        names = []
        for col in self.datetime_columns:
            names.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek"
            ])
        self.feature_names_out_ = names
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        extracted_dfs = []

        for col in self.datetime_columns:
            if col in X_df.columns:
                s_dt = pd.to_datetime(X_df[col], errors="coerce")
                
                # Check for median fallback for unparseable dates
                valid_years = s_dt.dt.year.dropna()
                med_year = int(valid_years.median()) if len(valid_years) > 0 else 2020
                med_month = int(s_dt.dt.month.dropna().median()) if len(valid_years) > 0 else 1
                med_day = int(s_dt.dt.day.dropna().median()) if len(valid_years) > 0 else 1
                med_dow = int(s_dt.dt.dayofweek.dropna().median()) if len(valid_years) > 0 else 0

                comp_df = pd.DataFrame({
                    f"{col}_year": s_dt.dt.year.fillna(med_year).astype(int),
                    f"{col}_month": s_dt.dt.month.fillna(med_month).astype(int),
                    f"{col}_day": s_dt.dt.day.fillna(med_day).astype(int),
                    f"{col}_dayofweek": s_dt.dt.dayofweek.fillna(med_dow).astype(int),
                }, index=X_df.index)
                
                extracted_dfs.append(comp_df)
                X_df = X_df.drop(columns=[col])

        if extracted_dfs:
            X_df = pd.concat([X_df] + extracted_dfs, axis=1)

        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)
