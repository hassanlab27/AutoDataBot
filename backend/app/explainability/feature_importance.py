from typing import Dict, Any, List, Optional
import numpy as np

def extract_native_feature_importance(
    model: Any,
    feature_names: List[str],
    feature_map: Dict[str, str]
) -> Dict[str, Any]:
    """
    Extracts model-native feature importances (e.g., Gini/split importance for trees,
    absolute coefficients for linear models) and pairs them with transformed and original feature names.
    """
    raw_importances: Optional[np.ndarray] = None
    importance_type = "feature_importances"

    # 1. Tree-based models (RandomForest, HistGradientBoosting, LightGBM, XGBoost, CatBoost, DecisionTree)
    if hasattr(model, "feature_importances_"):
        try:
            raw_importances = np.asarray(model.feature_importances_, dtype=float)
            importance_type = "Gini / Split Importance"
        except Exception:
            pass

    # 2. Linear models (LogisticRegression, Ridge, Lasso)
    elif hasattr(model, "coef_"):
        try:
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim == 2:
                # Average absolute coefficient across classes
                raw_importances = np.mean(np.abs(coef), axis=0)
            else:
                raw_importances = np.abs(coef)
            importance_type = "Mean Absolute Coefficient"
        except Exception:
            pass

    # 3. AutoGluon TabularPredictor
    elif hasattr(model, "feature_importance"):
        try:
            # We skip heavy live re-computation here; AutoGluon provides its own feature_importance table
            importance_type = "AutoGluon Permutation Importance"
        except Exception:
            pass

    if raw_importances is None or len(raw_importances) == 0:
        return {
            "available": False,
            "reason": "Model architecture does not provide native feature importance.",
            "importances": []
        }

    # Normalize if sum > 0
    total = np.sum(raw_importances)
    normalized = (raw_importances / total) if total > 1e-7 else raw_importances

    records: List[Dict[str, Any]] = []
    for i, imp in enumerate(normalized):
        feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        orig_name = feature_map.get(feat_name, feat_name)
        records.append({
            "feature": feat_name,
            "original_feature": orig_name,
            "importance": round(float(imp), 4),
            "raw_value": round(float(raw_importances[i]), 4)
        })

    # Sort descending
    records.sort(key=lambda r: r["importance"], reverse=True)

    return {
        "available": True,
        "importance_type": importance_type,
        "label": "Model Feature Importance",
        "disclaimer": "Feature importance represents relative contribution to model predictions; it does not establish real-world causality.",
        "importances": records
    }
