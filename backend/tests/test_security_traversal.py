import os
import zipfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.core.errors import AutoDataBotError
from app.reporting.report_service import report_service, ReportService
from app.reporting.export_service import export_service, ExportService

client = TestClient(app)

MALICIOUS_RUN_IDS = [
    "../../etc/passwd",
    "../..",
    "..%2F..%2Fetc%2Fpasswd",
    "/etc/passwd",
    "../../../../tmp",
    "run_123/../../../etc",
    "..\\..\\windows\\system32",
]


@pytest.mark.parametrize("bad_id", MALICIOUS_RUN_IDS)
def test_path_traversal_report_api(bad_id):
    res = client.get(f"/api/runs/{bad_id}/report")
    assert res.status_code in [400, 404]


@pytest.mark.parametrize("bad_id", MALICIOUS_RUN_IDS)
def test_path_traversal_pdf_api(bad_id):
    res = client.get(f"/api/runs/{bad_id}/report/pdf")
    assert res.status_code in [400, 404]


@pytest.mark.parametrize("bad_id", MALICIOUS_RUN_IDS)
def test_path_traversal_export_api(bad_id):
    res = client.get(f"/api/runs/{bad_id}/export")
    assert res.status_code in [400, 404]


@pytest.mark.parametrize("bad_id", MALICIOUS_RUN_IDS)
def test_report_service_rejects_traversal(bad_id):
    with pytest.raises((ValueError, AutoDataBotError)):
        report_service.generate_html_report(bad_id)


@pytest.mark.parametrize("bad_id", MALICIOUS_RUN_IDS)
def test_export_service_rejects_traversal(bad_id):
    with pytest.raises((ValueError, AutoDataBotError)):
        export_service.create_export_zip(bad_id)


def test_export_symlink_safety(tmp_path):
    """Verify that symlinks pointing outside the run directory are safely rejected."""
    mock_run_dir = tmp_path / "mock_run"
    mock_run_dir.mkdir(parents=True)
    
    meta_dir = mock_run_dir / "metadata"
    meta_dir.mkdir()
    (meta_dir / "run.json").write_text('{"run_id": "mock_run", "status": "completed"}')
    
    secret_dir = tmp_path / "outside_secrets"
    secret_dir.mkdir()
    secret_file = secret_dir / "passwords.txt"
    secret_file.write_text("SUPER_SECRET_PASSWORD")
    
    symlink_path = mock_run_dir / "metadata" / "leaked_secret.txt"
    try:
        os.symlink(secret_file, symlink_path)
    except OSError:
        pytest.skip("Symlinks not supported on this filesystem")
    
    assert not ExportService.is_safe_file(symlink_path, allowed_root=mock_run_dir)


def test_no_raw_csv_in_export_bundle(tmp_path):
    """Verify that raw CSV files placed inside run directory are excluded from export."""
    mock_run_dir = tmp_path / "runs" / "test_csv_exclusion_run"
    mock_run_dir.mkdir(parents=True)
    
    (mock_run_dir / "dataset.json").write_text('{"dataset_name": "data.csv"}')
    csv_file = mock_run_dir / "raw_data.csv"
    csv_file.write_text("id,secret_value\n1,42\n2,99\n")
    
    assert not ExportService.is_safe_file(csv_file, allowed_root=mock_run_dir)


def test_no_secrets_in_generated_report_or_export():
    """Verify that reports and exports do not expose private credentials or environment secrets."""
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    # 1. HTML Report
    html_content = report_service.generate_html_report(run_id)
    forbidden_terms = [
        "AWS_SECRET_ACCESS_KEY",
        "DATABASE_PASSWORD",
        "PRIVATE_KEY",
        "BEGIN RSA PRIVATE KEY",
        "API_SECRET",
        "/etc/passwd",
        "/home/hassan/.bashrc",
    ]
    for term in forbidden_terms:
        assert term.lower() not in html_content.lower()

    # 2. Export ZIP
    zip_path = export_service.create_export_zip(run_id)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for file_info in zf.infolist():
            assert ".." not in file_info.filename
            assert not file_info.filename.startswith("/")
            assert not file_info.filename.endswith(".csv")
            
            content = zf.read(file_info.filename)
            for term in forbidden_terms:
                assert term.lower().encode() not in content.lower()
