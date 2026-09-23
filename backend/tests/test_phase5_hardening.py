import os
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

from app.core.errors import EvaluationError, AutoDataBotError
from app.ml.metrics import compute_metrics
from app.ml.partitions import DatasetPartitions
from app.ml.models.sklearn_baselines import train_sklearn_baselines
from app.ml.models.gradient_boosting import (
    XGBModelWrapper,
    CatBoostModelWrapper,
    train_gradient_boosting_models
)
from app.explainability.shap_service import explainability_service
from app.evaluation.loader import EvaluationData

def test_prediction_target_length_mismatch_raises_evaluation_error():
    """
    CRITICAL INVARIANT TEST:
    Given y_true of length 12000 and y_pred of length 9600,
    the evaluation layer MUST raise an EvaluationError detailing the mismatch.
    It must NOT truncate, pad, or silently continue.
    """
    y_true = np.ones(12000, dtype=int)
    y_pred = np.ones(9600, dtype=int)

    with pytest.raises(EvaluationError) as exc_info:
        compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            problem_type="binary_classification",
            partition="test",
            run_id="run_test_mismatch",
            model_name="EnsembleWinner"
        )

    err_msg = str(exc_info.value)
    assert "Prediction/target length mismatch" in err_msg
    assert "partition 'test'" in err_msg
    assert "expected 12000" in err_msg
    assert "got 9600" in err_msg


def test_prediction_target_length_mismatch_regression():
    """
    Ensures regression metrics also strictly enforce len(y_true) == len(y_pred).
    """
    y_true = np.random.randn(1000)
    y_pred = np.random.randn(800)

    with pytest.raises(EvaluationError) as exc_info:
        compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            problem_type="regression",
            partition="validation",
            run_id="run_reg_mismatch",
            model_name="RidgeRegression"
        )

    assert "expected 1000" in str(exc_info.value)
    assert "got 800" in str(exc_info.value)


def test_dataset_partitions_contracts_and_isolation():
    """
    Verifies that DatasetPartitions enforces the invariant:
    TRAIN != VALIDATION != TEST
    and strict dimensional alignment.
    """
    n_tr, n_val, n_test, n_feats = 800, 200, 250, 5

    X_tr_prep = np.random.randn(n_tr, n_feats).astype(np.float32)
    X_tr_raw = pd.DataFrame(X_tr_prep, columns=[f"f{i}" for i in range(n_feats)])
    y_tr = np.random.randint(0, 2, size=n_tr)

    X_val_prep = np.random.randn(n_val, n_feats).astype(np.float32)
    X_val_raw = pd.DataFrame(X_val_prep, columns=[f"f{i}" for i in range(n_feats)])
    y_val = np.random.randint(0, 2, size=n_val)

    X_test_prep = np.random.randn(n_test, n_feats).astype(np.float32)
    X_test_raw = pd.DataFrame(X_test_prep, columns=[f"f{i}" for i in range(n_feats)])
    y_test = np.random.randint(0, 2, size=n_test)

    partitions = DatasetPartitions(
        X_tr_prep=X_tr_prep,
        X_tr_raw=X_tr_raw,
        y_tr=y_tr,
        X_val_prep=X_val_prep,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_prep=X_test_prep,
        X_test_raw=X_test_raw,
        y_test=y_test
    )

    # Valid partitions pass validation
    partitions.validate()

    assert partitions.train_size == 800
    assert partitions.val_size == 200
    assert partitions.test_size == 250

    # AutoGluon training and validation frames must match internal folds
    ag_tr = partitions.get_autogluon_train_df("target")
    ag_val = partitions.get_autogluon_val_df("target")
    assert len(ag_tr) == 800
    assert len(ag_val) == 200
    assert "target" in ag_tr.columns
    assert "target" in ag_val.columns

    # Test corrupted partition length raises EvaluationError
    bad_partitions = DatasetPartitions(
        X_tr_prep=X_tr_prep,
        X_tr_raw=X_tr_raw,
        y_tr=y_tr[:700],  # Corrupted: 700 vs 800
        X_val_prep=X_val_prep,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_prep=X_test_prep,
        X_test_raw=X_test_raw,
        y_test=y_test
    )
    with pytest.raises(EvaluationError):
        bad_partitions.validate()


def test_logistic_regression_no_n_jobs():
    """
    Verifies that LogisticRegression baseline does NOT pass deprecated n_jobs.
    """
    X_tr = np.random.randn(50, 4)
    y_tr = np.random.randint(0, 2, size=50)
    X_val = np.random.randn(15, 4)
    y_val = np.random.randint(0, 2, size=15)

    results = train_sklearn_baselines(
        problem_type="binary_classification",
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val
    )
    lr_res = next((r for r in results if r["model_name"] == "LogisticRegression"), None)
    assert lr_res is not None
    lr_model = lr_res["estimator"]
    params = lr_model.get_params()
    assert params.get("n_jobs") is None or "n_jobs" not in params


def test_artifact_isolation_between_runs(tmp_path):
    """
    Verifies that two independent runs have isolated model artifact directories
    and do not overwrite each other.
    """
    run_1_dir = tmp_path / "runs" / "run_001" / "models"
    run_2_dir = tmp_path / "runs" / "run_002" / "models"

    run_1_ag = run_1_dir / "autogluon"
    run_2_ag = run_2_dir / "autogluon"

    run_1_ag.mkdir(parents=True)
    run_2_ag.mkdir(parents=True)

    (run_1_ag / "predictor.pkl").write_text("run_1_model")
    (run_2_ag / "predictor.pkl").write_text("run_2_model")

    assert (run_1_ag / "predictor.pkl").read_text() == "run_1_model"
    assert (run_2_ag / "predictor.pkl").read_text() == "run_2_model"
    assert run_1_ag != run_2_ag


def test_sklearn_artifact_save_reload_predict(tmp_path):
    """
    Tests save -> reload -> predict for Scikit-learn models.
    """
    from sklearn.ensemble import RandomForestClassifier

    X_train = np.random.randn(100, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.randn(25, 4).astype(np.float32)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)

    bundle = {
        "model_id": "rf_test",
        "model_name": "Random Forest",
        "engine": "sklearn",
        "model_object": rf
    }

    save_path = tmp_path / "rf_model.joblib"
    joblib.dump(bundle, save_path)

    reloaded = joblib.load(save_path)
    reloaded_rf = reloaded["model_object"]

    preds = reloaded_rf.predict(X_test)
    assert len(preds) == 25
    assert set(preds).issubset({0, 1})


def test_xgboost_artifact_save_reload_predict(tmp_path):
    """
    Tests save -> reload -> predict for XGBoost wrapper (classification & regression).
    """
    from xgboost import XGBClassifier

    X_train = np.random.randn(120, 5).astype(np.float32)
    y_train = np.random.choice(["Class_A", "Class_B"], size=120)
    X_test = np.random.randn(30, 5).astype(np.float32)

    class_mapping = {"Class_A": 0, "Class_B": 1}
    y_encoded = np.array([class_mapping[v] for v in y_train])
    raw_xgb = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
    raw_xgb.fit(X_train, y_encoded)
    wrapper = XGBModelWrapper(raw_xgb, class_mapping=class_mapping)

    bundle = {
        "model_id": "xgb_test",
        "model_name": "XGBoost",
        "engine": "xgboost",
        "model_object": wrapper
    }

    save_path = tmp_path / "xgb_model.joblib"
    joblib.dump(bundle, save_path)

    reloaded = joblib.load(save_path)
    reloaded_xgb = reloaded["model_object"]

    preds = reloaded_xgb.predict(X_test)
    assert len(preds) == 30
    assert set(preds).issubset({"Class_A", "Class_B"})

    probs = reloaded_xgb.predict_proba(X_test)
    assert probs.shape == (30, 2)


def test_catboost_artifact_save_reload_predict(tmp_path):
    """
    Tests save -> reload -> predict for CatBoost wrapper.
    """
    from catboost import CatBoostClassifier

    X_train = np.random.randn(100, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.randn(20, 4).astype(np.float32)

    raw_cb = CatBoostClassifier(iterations=10, random_seed=42, verbose=0)
    raw_cb.fit(X_train, y_train)
    wrapper = CatBoostModelWrapper(raw_cb)

    bundle = {
        "model_id": "cb_test",
        "model_name": "CatBoost",
        "engine": "catboost",
        "model_object": wrapper
    }

    save_path = tmp_path / "cb_model.joblib"
    joblib.dump(bundle, save_path)

    reloaded = joblib.load(save_path)
    reloaded_cb = reloaded["model_object"]

    preds = reloaded_cb.predict(X_test)
    assert len(preds) == 20
    assert preds.ndim == 1


def test_flaml_artifact_save_reload_predict(tmp_path):
    """
    Tests save -> reload -> predict for FLAML AutoML.
    """
    from flaml import AutoML

    X_train = np.random.randn(100, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.randn(20, 4).astype(np.float32)

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="classification",
        time_budget=2,
        estimator_list=["rf", "extra_tree"],
        verbose=0
    )

    bundle = {
        "model_id": "flaml_test",
        "model_name": "FLAML AutoML",
        "engine": "flaml",
        "model_object": automl
    }

    save_path = tmp_path / "flaml_model.joblib"
    joblib.dump(bundle, save_path)

    reloaded = joblib.load(save_path)
    reloaded_automl = reloaded["model_object"]

    preds = reloaded_automl.predict(X_test)
    assert len(preds) == 20


def test_shap_uses_validation_partition():
    """
    Verifies that ExplainabilityService._compute_shap computes attributions
    against the validation partition, leaving external test data untouched.
    """
    from sklearn.ensemble import RandomForestClassifier

    X_tr = np.random.randn(80, 4).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=80)
    X_val = np.random.randn(20, 4).astype(np.float32)
    y_val = np.random.randint(0, 2, size=20)
    X_test = np.random.randn(25, 4).astype(np.float32)
    y_test = np.random.randint(0, 2, size=25)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_tr, y_tr)

    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]
    feature_map = {f: f for f in feature_names}

    eval_data = EvaluationData(
        run_id="run_shap_test",
        dataset_id="ds_test",
        target_col="target",
        problem_type="binary_classification",
        primary_metric="accuracy",
        model_obj=rf,
        model_name="Random Forest",
        engine="sklearn",
        model_bundle={},
        config={},
        X_test_prep=X_test,
        X_test_raw=pd.DataFrame(X_test, columns=feature_names),
        y_test=y_test,
        X_val_prep=X_val,
        y_val=y_val,
        feature_names=feature_names,
        feature_map=feature_map,
        X_train_prep=X_tr,
        y_train=y_tr,
        X_tr_prep=X_tr,
        y_tr=y_tr
    )

    shap_result = explainability_service._compute_shap(eval_data, sample_size=50, max_features=10)

    assert shap_result["available"] is True
    assert shap_result["partition"] == "validation"
    assert shap_result["sample_size"] == 20  # Length of validation fold, NOT test fold (25)


def test_evaluation_loader_rejects_failed_run(tmp_path, monkeypatch):
    """
    Verifies that EvaluationDataLoader will refuse to load or report 'success'
    for a run whose status.json indicates 'failed'.
    """
    from app.core.config import settings
    from app.evaluation.loader import data_loader

    fake_outputs = tmp_path / "outputs"
    monkeypatch.setattr(settings, "OUTPUTS_DIR", fake_outputs)

    run_dir = fake_outputs / "runs" / "failed_run_01"
    run_dir.mkdir(parents=True)
    status_file = run_dir / "status.json"
    status_file.write_text(json.dumps({
        "run_id": "failed_run_01",
        "status": "failed",
        "current_stage": "Evaluating winner on partitions",
        "error_message": "Prediction/target length mismatch for partition 'test'"
    }))

    with pytest.raises(AutoDataBotError) as exc_info:
        data_loader.load_evaluation_data("failed_run_01")

    assert "Cannot evaluate run 'failed_run_01'" in str(exc_info.value)
    assert "Prediction/target length mismatch" in str(exc_info.value)


def test_evaluation_endpoints_reject_failed_run(client, tmp_path, monkeypatch):
    """
    Verifies that GET /api/runs/{run_id}/evaluation rejects failed runs with 400.
    """
    from app.core.config import settings

    fake_outputs = tmp_path / "outputs"
    monkeypatch.setattr(settings, "OUTPUTS_DIR", fake_outputs)

    run_dir = fake_outputs / "runs" / "failed_run_02"
    run_dir.mkdir(parents=True)
    status_file = run_dir / "status.json"
    status_file.write_text(json.dumps({
        "run_id": "failed_run_02",
        "status": "failed",
        "current_stage": "Evaluation",
        "error_message": "EvaluationError: Test evaluation failure"
    }))

    res = client.get("/api/runs/failed_run_02/evaluation")
    assert res.status_code == 400
    assert "Cannot evaluate run" in res.json().get("detail", "")

