import sys
from unittest.mock import patch
import pytest
from app.core.config import settings
from app.core.errors import AutoDataBotError
from app.reporting.report_service import report_service

def test_generate_pdf_report_success():
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    pdf_bytes = report_service.generate_pdf_report(run_id, force_refresh=True)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")

    # Verify cached file on disk
    pdf_file = run_dir / "report" / "report.pdf"
    assert pdf_file.is_file()
    assert pdf_file.read_bytes() == pdf_bytes


def test_pdf_generation_fallback_when_weasyprint_missing():
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    # Simulate WeasyPrint missing
    with patch.dict(sys.modules, {"weasyprint": None}):
        with pytest.raises(AutoDataBotError) as exc_info:
            report_service.generate_pdf_report(run_id, force_refresh=True)
        assert "PDF generation is unavailable" in str(exc_info.value)

    # Verify HTML report generation is still unaffected
    html = report_service.generate_html_report(run_id)
    assert "<!DOCTYPE html>" in html
