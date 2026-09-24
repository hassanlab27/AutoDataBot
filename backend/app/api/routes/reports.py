import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import HTMLResponse, FileResponse

from app.core.errors import AutoDataBotError
from app.reporting.report_service import report_service
from app.reporting.export_service import export_service
from app.reporting.report_context import _sanitize_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs/{run_id}", tags=["Reports & Export"])

@router.get("/report", response_class=HTMLResponse)
def get_run_report_html(run_id: str, refresh: bool = Query(False)) -> HTMLResponse:
    """
    Renders and returns a deterministic, standalone HTML report for a completed run.
    """
    try:
        clean_id = _sanitize_id(run_id)
        html_content = report_service.generate_html_report(clean_id, force_refresh=refresh)
        return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error generating report for run {run_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate report: {str(e)}")

@router.get("/report/pdf", response_class=Response)
def get_run_report_pdf(run_id: str, refresh: bool = Query(False)) -> Response:
    """
    Renders and returns a print-ready PDF report for a completed run.
    """
    try:
        clean_id = _sanitize_id(run_id)
        pdf_bytes = report_service.generate_pdf_report(clean_id, force_refresh=refresh)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="autodatabot_{clean_id}_report.pdf"'
            }
        )
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error generating PDF report for run {run_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate PDF: {str(e)}")

@router.get("/export", response_class=FileResponse)
def export_run_zip(run_id: str, refresh: bool = Query(False)) -> FileResponse:
    """
    Generates and downloads a secure ZIP bundle containing all run metadata,
    metrics, explainability, reports, model artifacts, and manifest.
    Excludes raw dataset CSVs by default.
    """
    try:
        clean_id = _sanitize_id(run_id)
        zip_path = export_service.create_export_zip(clean_id, force_refresh=refresh)
        if not zip_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export bundle file not found.")

        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=f"autodatabot_{clean_id}.zip"
        )
    except AutoDataBotError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error generating export ZIP for run {run_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create export bundle: {str(e)}")
