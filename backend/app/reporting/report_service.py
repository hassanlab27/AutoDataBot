from pathlib import Path
from typing import Dict, Any, Optional

from app.core.config import settings
from app.core.errors import AutoDataBotError
from app.core.logging import logger
from app.reporting.report_context import build_report_context, _sanitize_id
from app.reporting.report_builder import report_builder

class ReportService:
    """Coordinates deterministic report generation and persistence."""

    def get_report_dir(self, run_id: str) -> Path:
        clean_id = _sanitize_id(run_id)
        r_dir = settings.OUTPUTS_DIR / "runs" / clean_id / "report"
        r_dir.mkdir(parents=True, exist_ok=True)
        return r_dir

    def generate_html_report(self, run_id: str, force_refresh: bool = False) -> str:
        """
        Builds or returns cached deterministic HTML report for a run.
        """
        clean_id = _sanitize_id(run_id)
        report_dir = self.get_report_dir(clean_id)
        report_file = report_dir / "report.html"

        if not force_refresh and report_file.is_file():
            try:
                return report_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read cached report {report_file}: {e}")

        # Build report context
        context = build_report_context(clean_id)

        # Render HTML
        html_content = report_builder.render_html(context)

        # Persist to disk
        try:
            report_file.write_text(html_content, encoding="utf-8")
            logger.info(f"Report generated and saved to {report_file}")
        except Exception as e:
            logger.warning(f"Failed to cache report to disk: {e}")

        return html_content

    def generate_pdf_report(self, run_id: str, force_refresh: bool = False) -> bytes:
        """
        Builds or returns cached PDF report for a run.
        """
        clean_id = _sanitize_id(run_id)
        report_dir = self.get_report_dir(clean_id)
        pdf_file = report_dir / "report.pdf"

        if not force_refresh and pdf_file.is_file():
            try:
                return pdf_file.read_bytes()
            except Exception as e:
                logger.warning(f"Could not read cached PDF {pdf_file}: {e}")

        # Ensure HTML exists
        html_content = self.generate_html_report(clean_id, force_refresh=force_refresh)

        # Render PDF
        pdf_bytes = report_builder.render_pdf(html_content)

        # Persist to disk
        try:
            pdf_file.write_bytes(pdf_bytes)
            logger.info(f"PDF report generated and saved to {pdf_file} ({len(pdf_bytes)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to cache PDF to disk: {e}")

        return pdf_bytes

report_service = ReportService()
