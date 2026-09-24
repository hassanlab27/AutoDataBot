import pytest
from pathlib import Path
from app.core.config import settings

def test_html_report_generation_classification():
    from app.reporting.report_service import report_service
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    html = report_service.generate_html_report(run_id, force_refresh=True)

    # 1. Output string check
    assert isinstance(html, str)
    assert len(html) > 1000
    assert "<!DOCTYPE html>" in html
    assert "AutoDataBot" in html

    # 2. Verify all 15 sections are present in HTML headings
    assert "Executive Summary" in html
    assert "Dataset Overview" in html
    assert "Data Quality" in html
    assert "Exploratory Data Analysis" in html
    assert "Target & Problem Definition" in html
    assert "Preprocessing" in html
    assert "Training Configuration" in html
    assert "Model Leaderboard" in html
    assert "Selected Model" in html
    assert "Evaluation & Performance" in html
    assert "Generalization & Diagnostics" in html
    assert "Error Analysis" in html
    assert "Feature Importance" in html
    assert "SHAP" in html
    assert "Limitations & Assumptions" in html

    # 3. Verify specific values from the run
    assert run_id in html
    assert "diabetes_risk" in html
    assert "AutoGluon_CatBoost" in html
    assert "0.8013" in html or "80.1%" in html

    # 4. Check cached file on disk
    report_file = run_dir / "report" / "report.html"
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8") == html

    # 5. Determinism: generating twice produces identical content
    html_second = report_service.generate_html_report(run_id, force_refresh=True)
    assert html == html_second


def test_html_report_generation_regression():
    from app.reporting.report_service import report_service
    run_id = "run_20260924_111902_7d89a3"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    html = report_service.generate_html_report(run_id, force_refresh=True)
    assert "price" in html
    assert "Regression" in html
    assert "Mean Absolute Error" in html or "MAE" in html
    assert "Residual" in html
