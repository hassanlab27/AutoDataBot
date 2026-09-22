from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.inspection import permutation_importance

def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    feature_map: Dict[str, str],
    scoring: Optional[str] = None,
    n_repeats: int = 5,
    random_state: int = 42,
    data_partition_label: str = "validation"
) -> Dict[str, Any]:
    """
    Computes model-agnostic permutation feature importance.
    """
    try:
        # Downsample evaluation data if large to keep calculation fast and responsive
        n_samples = len(X)
        if n_samples > 1000:
            rng = np.random.RandomState(random_state)
            sample_idx = rng.choice(n_samples, size=1000, replace=False)
            X_eval = X[sample_idx]
            y_eval = y[sample_idx]
        else:
            X_eval = X
            y_eval = y

        result = permutation_importance(
            model,
            X_eval,
            y_eval,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1  # safe threading
        )

        records: List[Dict[str, Any]] = []
        for i, mean_imp in enumerate(result.importances_mean):
            feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            orig_name = feature_map.get(feat_name, feat_name)
            std_imp = float(result.importances_std[i])
            records.append({
                "feature": feat_name,
                "original_feature": orig_name,
                "importance_mean": round(float(mean_imp), 4),
                "importance_std": round(float(std_imp), 4)
            })

        # Sort descending by importance_mean
        records.sort(key=lambda r: r["importance_mean"], reverse=True)

        return {
            "available": True,
            "data_partition": data_partition_label,
            "n_repeats": n_repeats,
            "scoring": scoring or "default",
            "label": "Permutation Feature Importance",
            "disclaimer": "Permutation importance measures drop in model performance when feature values are shuffled; it does not indicate real-world causal impact.",
            "importances": records
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"Permutation importance calculation failed: {str(e)}",
            "importances": []
        }
