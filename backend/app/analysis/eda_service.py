from typing import Dict, Any, List, Optional
from collections import OrderedDict
import pandas as pd
import numpy as np

from app.core.config import settings
from app.core.errors import DatasetNotFoundError, ColumnNotFoundError
from app.storage.dataset_store import dataset_store
from app.validation.schema_inspector import infer_logical_dtype
from app.analysis.statistics import (
    compute_dataset_overview,
    compute_all_numerical_stats,
    compute_all_categorical_stats,
    compute_missing_analysis,
    compute_numerical_column_stats,
    compute_categorical_column_stats,
    compute_temporal_column_stats,
    sanitize_float
)
from app.analysis.correlations import (
    compute_correlation_matrix,
    extract_strong_correlations
)
from app.analysis.outliers import (
    detect_dataset_outliers,
    detect_column_outliers_iqr
)
from app.analysis.chart_recommender import (
    build_missingness_chart,
    build_univariate_numeric_charts,
    build_univariate_categorical_chart,
    build_correlation_heatmap,
    recommend_bivariate_chart
)

class EDAService:
    """
    Coordinates all exploratory data analysis operations.
    Maintains a simple bounded in-memory cache to avoid repeated disk reads.
    """
    def __init__(self, cache_size: int = 8):
        self.cache_size = cache_size
        self._df_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """Retrieve dataset DataFrame, using LRU cache when possible."""
        if dataset_id in self._df_cache:
            self._df_cache.move_to_end(dataset_id)
            return self._df_cache[dataset_id]

        df = dataset_store.load_dataset(dataset_id)
        if len(self._df_cache) >= self.cache_size:
            self._df_cache.popitem(last=False)
        self._df_cache[dataset_id] = df
        return df

    def get_overview(self, dataset_id: str) -> Dict[str, Any]:
        """Dataset overview statistics."""
        df = self.get_dataframe(dataset_id)
        return compute_dataset_overview(df)

    def get_numerical_analysis(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Numerical column descriptive metrics."""
        df = self.get_dataframe(dataset_id)
        return compute_all_numerical_stats(df)

    def get_categorical_analysis(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Categorical column frequencies and top categories."""
        df = self.get_dataframe(dataset_id)
        return compute_all_categorical_stats(df)

    def get_missing_analysis(self, dataset_id: str) -> Dict[str, Any]:
        """Missing values summary and chart data."""
        df = self.get_dataframe(dataset_id)
        missing_data = compute_missing_analysis(df)
        chart = build_missingness_chart(df)
        return {
            **missing_data,
            "chart": chart
        }

    def get_correlation_analysis(self, dataset_id: str) -> Dict[str, Any]:
        """Pearson correlation matrix and heatmap visualization payload."""
        df = self.get_dataframe(dataset_id)
        corr_data = compute_correlation_matrix(df)
        heatmap = build_correlation_heatmap(corr_data)
        return {
            **corr_data,
            "heatmap": heatmap
        }

    def get_outliers_summary(self, dataset_id: str) -> List[Dict[str, Any]]:
        """IQR outlier detection summary for all numeric columns."""
        df = self.get_dataframe(dataset_id)
        return detect_dataset_outliers(df)

    def get_column_distribution(self, dataset_id: str, column_name: str) -> Dict[str, Any]:
        """Distribution metrics and chart for a single column."""
        df = self.get_dataframe(dataset_id)
        if column_name not in df.columns:
            raise ColumnNotFoundError(column_name, available_columns=list(df.columns))

        s = df[column_name]
        dtype = infer_logical_dtype(s)

        if dtype in ("integer", "float"):
            stats = compute_numerical_column_stats(s, column_name)
            outliers = detect_column_outliers_iqr(s, column_name)
            charts = build_univariate_numeric_charts(s, column_name)
            return {
                "column": column_name,
                "dtype": dtype,
                "stats": stats,
                "outliers": outliers,
                "histogram": charts["histogram"],
                "box_plot": charts["box_plot"]
            }
        elif dtype == "datetime-like":
            temporal = compute_temporal_column_stats(s, column_name)
            chart = recommend_bivariate_chart(df, column_name)
            return {
                "column": column_name,
                "dtype": dtype,
                "temporal": temporal,
                "chart": chart
            }
        else:
            cat_stats = compute_categorical_column_stats(s, column_name)
            bar_chart = build_univariate_categorical_chart(s, column_name)
            return {
                "column": column_name,
                "dtype": dtype,
                "stats": cat_stats,
                "chart": bar_chart
            }

    def generate_chart(
        self,
        dataset_id: str,
        x: str,
        y: Optional[str] = None,
        chart_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Dynamic chart generator using rule-based recommendation."""
        df = self.get_dataframe(dataset_id)
        return recommend_bivariate_chart(df, col_x=x, col_y=y, requested_chart_type=chart_type)

    def get_target_analysis(self, dataset_id: str, target_col: str) -> Dict[str, Any]:
        """
        Optional target-aware EDA when the user selects a target variable.
        Computes associations, correlations, class balance, and feature relationships with target.
        """
        df = self.get_dataframe(dataset_id)
        if target_col not in df.columns:
            raise ColumnNotFoundError(target_col, available_columns=list(df.columns))

        target_s = df[target_col]
        target_dtype = infer_logical_dtype(target_s)
        
        # 1. Target column summary
        if target_dtype in ("integer", "float"):
            target_profile = {
                "type": "numeric",
                "stats": compute_numerical_column_stats(target_s, target_col),
                "outliers": detect_column_outliers_iqr(target_s, target_col),
                "charts": build_univariate_numeric_charts(target_s, target_col)
            }
            # Correlation with other numeric variables
            correlations_with_target = []
            for col in df.columns:
                if col == target_col:
                    continue
                if infer_logical_dtype(df[col]) in ("integer", "float"):
                    clean = df[[col, target_col]].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                    if len(clean) > 2:
                        s_x = pd.to_numeric(clean[col], errors="coerce")
                        s_y = pd.to_numeric(clean[target_col], errors="coerce")
                        valid_mask = s_x.notna() & s_y.notna()
                        if valid_mask.sum() > 2 and s_x.std() > 1e-9 and s_y.std() > 1e-9:
                            corr = float(np.corrcoef(s_x[valid_mask], s_y[valid_mask])[0, 1])
                            if not (np.isnan(corr) or np.isinf(corr)):
                                correlations_with_target.append({
                                    "feature": str(col),
                                    "correlation": round(corr, 4),
                                    "abs_correlation": round(abs(corr), 4)
                                })
            correlations_with_target.sort(key=lambda x: x["abs_correlation"], reverse=True)
            top_features = correlations_with_target[:5]
            target_profile["top_associated_features"] = top_features
        else:
            cat_stats = compute_categorical_column_stats(target_s, target_col)
            bar_chart = build_univariate_categorical_chart(target_s, target_col)
            
            # Check class balance
            is_imbalanced = False
            if cat_stats["top_categories"]:
                first_pct = cat_stats["top_categories"][0]["percentage"]
                if first_pct > 80.0:
                    is_imbalanced = True

            target_profile = {
                "type": "categorical",
                "stats": cat_stats,
                "chart": bar_chart,
                "is_imbalanced": is_imbalanced,
                "imbalance_warning": "Target classes are severely imbalanced (>80% in top category)." if is_imbalanced else None
            }

        return {
            "target_column": target_col,
            "target_dtype": target_dtype,
            "profile": target_profile,
            "disclaimer": "Target-aware analysis indicates preliminary statistical association only. ML feature importance and modeling will be determined in subsequent phases."
        }

eda_service = EDAService()
