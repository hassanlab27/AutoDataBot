import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from fastapi.testclient import TestClient

from app.main import app
from app.evaluation.classification import evaluate_classification, downsample_curve
from app.evaluation.regression import evaluate_regression
from app.evaluation.errors import analyze_classification_errors, analyze_regression_errors
from app.evaluation.diagnostics import compute_detailed_diagnostics
from app.evaluation.comparison import compare_run_models
from app.explainability.feature_importance import extract_native_feature_importance
from app.explainability.permutation import compute_permutation_importance
from app.explainability.shap_service import explainability_service
from app.evaluation.loader import EvaluationData

client = TestClient(app)

# -------------------------------------------------------------
# 1. Classification Evaluation Tests
# -------------------------------------------------------------

def test_classification_evaluation_binary():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
    y_prob = np.array([
        [0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9],
        [0.85, 0.15], [0.55, 0.45], [0.95, 0.05], [0.05, 0.95], [0.9, 0.1]
    ])

    res = evaluate_classification(y_true, y_pred, y_prob, problem_type="binary_classification")
    assert res["problem_type"] == "binary_classification"
    metrics = res["metrics"]
    assert metrics["accuracy"] == 0.9
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "specificity" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics

    cm = res["confusion_matrix"]
    assert cm["is_binary"] is True
    assert cm["matrix"] == [[5, 0], [1, 4]]
    assert "binary_breakdown" in cm
    assert cm["binary_breakdown"]["true_positives"] == 4
    assert cm["binary_breakdown"]["false_negatives"] == 1

    assert res["roc_curve"] is not None
    assert len(res["roc_curve"]["fpr"]) > 0
    assert res["roc_curve"]["auc"] > 0.8

    assert res["pr_curve"] is not None
    assert res["threshold_analysis"] is not None
    assert len(res["threshold_analysis"]) == 19  # 0.05 to 0.95
    assert res["calibration"] is not None
    assert len(res["calibration"]["prob_predicted"]) > 0

def test_classification_evaluation_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])

    res = evaluate_classification(y_true, y_pred, problem_type="multiclass_classification")
    assert res["problem_type"] == "multiclass_classification"
    assert res["metrics"]["accuracy"] == round(7 / 9, 4)
    assert "f1" in res["metrics"]
    assert "weighted_f1" in res["metrics"]

    assert len(res["per_class"]) == 3
    for pc in res["per_class"]:
        assert "class_label" in pc
        assert "precision" in pc
        assert "recall" in pc
        assert "f1" in pc

    cm = res["confusion_matrix"]
    assert cm["is_binary"] is False
    assert len(cm["labels"]) == 3
    assert len(cm["matrix"]) == 3

def test_downsample_curve():
    x = np.linspace(0, 1, 500)
    y = np.linspace(0, 1, 500)
    thresh = np.linspace(1, 0, 500)
    sub_x, sub_y, sub_thresh = downsample_curve(x, y, thresh, max_points=50)
    assert len(sub_x) <= 50
    assert sub_x[0] == 0.0
    assert sub_x[-1] == 1.0
    assert len(sub_thresh) == len(sub_x)

# -------------------------------------------------------------
# 2. Regression Evaluation Tests
# -------------------------------------------------------------

def test_regression_evaluation_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 38.0, 52.0])

    res = evaluate_regression(y_true, y_pred)
    assert res["problem_type"] == "regression"
    metrics = res["metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "median_absolute_error" in metrics
    assert "explained_variance" in metrics
    assert "mape" in metrics  # Valid because no zeros

    r_stats = res["residual_stats"]
    assert r_stats["mean"] == round(float(np.mean(y_true - y_pred)), 4)
    assert r_stats["std"] == round(float(np.std(y_true - y_pred)), 4)

    assert "histogram" in res
    assert len(res["histogram"]["counts"]) == 20
    assert "actual_vs_predicted" in res
    assert len(res["actual_vs_predicted"]["points"]) == 5
    assert len(res["diagnostics"]) > 0

# -------------------------------------------------------------
# 3. Error Analysis Tests
# -------------------------------------------------------------

def test_error_analysis_classification():
    y_true = np.array(["cat", "dog", "bird", "cat", "dog"])
    y_pred = np.array(["cat", "bird", "bird", "dog", "dog"])
    y_prob = np.array([0.95, 0.88, 0.92, 0.76, 0.99])

    res = analyze_classification_errors(y_true, y_pred, y_prob, limit=2)
    assert res["total_evaluated"] == 5
    assert res["misclassifications_count"] == 2
    assert res["error_rate"] == 0.4
    assert len(res["worst_predictions"]) == 2
    # Highest confidence wrong prediction should be first
    assert res["worst_predictions"][0]["predicted"] == "bird"
    assert res["worst_predictions"][0]["actual"] == "dog"

def test_error_analysis_regression():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([105.0, 250.0, 290.0, 401.0])

    res = analyze_regression_errors(y_true, y_pred, limit=2)
    assert res["total_evaluated"] == 4
    assert len(res["worst_predictions"]) == 2
    # Worst is 200 vs 250 (error = 50.0)
    assert res["worst_predictions"][0]["actual"] == 200.0
    assert res["worst_predictions"][0]["abs_error"] == 50.0
    assert res["worst_predictions"][0]["relative_error_pct"] == 25.0

# -------------------------------------------------------------
# 4. Diagnostics & Comparison Tests
# -------------------------------------------------------------

def test_generalization_diagnostics_rules():
    # Large generalization gap
    diag1 = compute_detailed_diagnostics("accuracy", val_score=0.85, test_score=0.60)
    assert diag1["diagnostic_label"] == "Large generalization gap"
    assert diag1["generalization_gap"] == 0.25

    # Suspicious leakage
    diag2 = compute_detailed_diagnostics("f1", val_score=1.0, test_score=0.99)
    assert diag2["diagnostic_label"] == "Suspiciously High Performance"

    # Worse than baseline
    diag3 = compute_detailed_diagnostics("accuracy", val_score=0.55, test_score=0.54, naive_baseline_score=0.60)
    assert diag3["diagnostic_label"] == "Worse than Baseline"

# -------------------------------------------------------------
# 5. Explainability Tests
# -------------------------------------------------------------

def test_native_feature_importance():
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    feature_names = ["num__age", "num__income", "cat__city_NYC", "cat__city_LA"]
    feature_map = {
        "num__age": "age",
        "num__income": "income",
        "cat__city_NYC": "city",
        "cat__city_LA": "city"
    }

    res = extract_native_feature_importance(rf, feature_names, feature_map)
    assert res["available"] is True
    assert len(res["importances"]) == 4
    assert res["importances"][0]["original_feature"] in ["age", "income", "city"]
    assert sum(item["importance"] for item in res["importances"]) == pytest.approx(1.0, 0.01)

def test_permutation_importance_calc():
    X = np.random.randn(60, 3)
    y = 2.0 * X[:, 0] + np.random.randn(60) * 0.1
    ridge = Ridge()
    ridge.fit(X, y)

    res = compute_permutation_importance(
        model=ridge,
        X=X,
        y=y,
        feature_names=["f0", "f1", "f2"],
        feature_map={"f0": "f0", "f1": "f1", "f2": "f2"},
        scoring="r2",
        n_repeats=3
    )
    assert res["available"] is True
    assert len(res["importances"]) == 3
    # f0 should be the most important
    assert res["importances"][0]["feature"] == "f0"
    assert res["importances"][0]["importance_mean"] > 0

def test_shap_computation_fast():
    # Use small synthetic data with sample_size=20
    X = np.random.randn(40, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    rf.fit(X, y)

    eval_data = EvaluationData(
        run_id="test_run",
        dataset_id="test_ds",
        target_col="target",
        problem_type="binary_classification",
        primary_metric="accuracy",
        model_obj=rf,
        model_name="Random Forest",
        engine="sklearn",
        model_bundle={},
        config={},
        X_test_prep=X,
        X_test_raw=pd.DataFrame(X, columns=["f0", "f1", "f2"]),
        y_test=y,
        X_val_prep=X,
        y_val=y,
        feature_names=["f0", "f1", "f2"],
        feature_map={"f0": "f0", "f1": "f1", "f2": "f2"}
    )

    shap_res = explainability_service._compute_shap(eval_data, sample_size=20, max_features=3)
    assert shap_res["available"] is True
    assert len(shap_res["global_importance"]) == 3
    assert len(shap_res["summary_points"]) > 0

def test_shap_graceful_failure_unsupported():
    class DummyNoShapModel:
        pass

    eval_data = EvaluationData(
        run_id="test_run",
        dataset_id="test_ds",
        target_col="target",
        problem_type="binary_classification",
        primary_metric="accuracy",
        model_obj=DummyNoShapModel(),
        model_name="Unsupported Model",
        engine="custom",
        model_bundle={},
        config={},
        X_test_prep=np.random.randn(20, 2),
        X_test_raw=pd.DataFrame(),
        y_test=np.zeros(20),
        X_val_prep=np.random.randn(20, 2),
        y_val=np.zeros(20),
        feature_names=["a", "b"],
        feature_map={"a": "a", "b": "b"}
    )

    shap_res = explainability_service._compute_shap(eval_data, sample_size=10, max_features=2)
    assert shap_res["available"] is False
    assert "reason" in shap_res

# -------------------------------------------------------------
# 5. Continuous Tuning & Human-Readable Metrics Tests
# -------------------------------------------------------------

def test_refined_classification_metrics_and_plain_language():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 1])
    y_prob = np.array([
        [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.15, 0.85], [0.9, 0.1],
        [0.4, 0.6], [0.2, 0.8], [0.6, 0.4], [0.95, 0.05], [0.05, 0.95]
    ])
    res = evaluate_classification(y_true, y_pred, y_prob, problem_type="binary_classification")
    assert "accuracy_pct" in res
    assert res["accuracy_pct"] == "80.0%"
    assert "f1_pct" in res
    assert "confusion_dict" in res
    cd = res["confusion_dict"]
    assert cd["correct_count"] == 8
    assert cd["incorrect_count"] == 2
    assert cd["accuracy_pct"] == 80.0
    assert cd["error_pct"] == 20.0
    assert "8 of 10" in cd["summary_text"]

def test_refined_regression_metrics_and_margins():
    y_true = np.array([100.0, 110.0, 95.0, 120.0, 105.0])
    # Predictions within ±5-8%
    y_pred = np.array([102.0, 108.0, 98.0, 118.0, 104.0])
    res = evaluate_regression(y_true, y_pred)
    assert "r2_pct" in res
    assert "accuracy_within_10_pct" in res
    assert res["accuracy_within_10_pct"] == 100.0  # 100% within 10%
    assert "accuracy_within_20_pct" in res
    assert res["accuracy_within_20_pct"] == 100.0
    assert "human_summary" in res
    assert "r2_explained" in res["human_summary"]
    assert "average_error_mae" in res["human_summary"]

def test_continuous_tuning_execution_reaches_or_logs_benchmark():
    from app.ml.training_service import training_service
    from app.ml.config import AutoMLConfig
    from app.ml.evaluator import ModelResult

    np.random.seed(42)
    X_tr = np.random.randn(80, 4)
    # Target linearly separable with noise
    y_tr = ((X_tr[:, 0] * 2 + X_tr[:, 1]) > 0).astype(int)
    X_val = np.random.randn(30, 4)
    y_val = ((X_val[:, 0] * 2 + X_val[:, 1]) > 0).astype(int)

    config = AutoMLConfig(
        dataset_id="test_ds",
        target="target",
        problem_type="binary_classification",
        training_mode="quick",
        primary_metric="accuracy",
        random_state=42,
        engine_preference="all"
    )

    # Simulate Round 1 with sub-80% score to trigger continuous tuning
    r1_model = ModelResult(
        model_id="m_dummy_r1",
        model_name="Sub80 Baseline",
        engine="sklearn",
        problem_type="binary_classification",
        validation_metrics={"accuracy": 0.65},
        validation_primary_score=0.65,
        training_time_seconds=0.1,
        status="success",
        is_naive_baseline=False
    )
    results = [r1_model]
    from sklearn.linear_model import LogisticRegression
    dummy_clf = LogisticRegression()
    dummy_clf.fit(X_tr, y_tr)
    trained_objs = {"m_dummy_r1": dummy_clf}

    tuning_hist = training_service._run_continuous_tuning(
        config=config,
        primary_metric="accuracy",
        X_tr_prep=X_tr,
        y_tr=y_tr,
        X_val_prep=X_val,
        y_val=y_val,
        results=results,
        trained_model_objs=trained_objs,
        run_id="test_tuning_run"
    )

    assert tuning_hist["target_threshold"] == 0.80
    assert tuning_hist["rounds_run"] >= 2
    assert len(tuning_hist["round_scores"]) >= 2
    # Verify tuned models were added to results
    assert len(results) > 1
    assert any(m.tuning_round == 2 for m in results)

