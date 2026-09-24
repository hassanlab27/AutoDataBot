import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_api_get_report_html_success():
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    res = client.get(f"/api/runs/{run_id}/report")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "<!DOCTYPE html>" in res.text
    assert "AutoDataBot Technical Report" in res.text


def test_api_get_report_pdf_success():
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    res = client.get(f"/api/runs/{run_id}/report/pdf")
    assert res.status_code == 200
    assert "application/pdf" in res.headers["content-type"]
    assert res.content.startswith(b"%PDF-")


def test_api_get_export_zip_success():
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    res = client.get(f"/api/runs/{run_id}/export")
    assert res.status_code == 200
    assert "application/zip" in res.headers["content-type"]
    assert len(res.content) > 1000
    assert res.content.startswith(b"PK")  # ZIP magic bytes


def test_api_report_non_existent_run():
    res = client.get("/api/runs/non_existent_run_9999/report")
    assert res.status_code in [400, 404]


def test_api_report_path_traversal():
    res = client.get("/api/runs/..%2F..%2Fetc%2Fpasswd/report")
    assert res.status_code in [400, 404]
