from pathlib import Path
from typing import Dict, Any, Optional
import jinja2

from app.core.errors import AutoDataBotError
from app.core.logging import logger

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

def format_number(val: Any) -> str:
    if val is None or val == "Not available":
        return "Not available"
    if isinstance(val, (int, float)):
        return f"{int(val):,}" if isinstance(val, int) or val.is_integer() else f"{val:,.2f}"
    return str(val)

def format_metric(val: Any) -> str:
    if val is None or val == "Not available":
        return "—"
    if isinstance(val, float):
        if abs(val) < 0.0001 and val != 0:
            return f"{val:.6e}"
        return f"{val:.4f}"
    return str(val)

def format_percent(val: Any) -> str:
    if val is None or val == "Not available":
        return "—"
    if isinstance(val, (int, float)):
        return f"{float(val) * 100:.1f}%" if float(val) <= 1.0 else f"{float(val):.1f}%"
    return str(val)

ACRONYMS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "f1": "F1",
    "f1_macro": "F1 (Macro)",
    "iqr": "IQR",
    "id": "ID",
    "std": "Std Dev"
}

def format_metric_name(val: str) -> str:
    if not val:
        return ""
    val_lower = str(val).lower()
    if val_lower in ACRONYMS:
        return ACRONYMS[val_lower]
    return str(val).replace("_", " ").title()

def get_embedded_css() -> str:
    css_path = STATIC_DIR / "report.css"
    if css_path.is_file():
        try:
            return css_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read static CSS from {css_path}: {e}")
    return "body { font-family: sans-serif; }"

class ReportBuilder:
    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            undefined=jinja2.ChainableUndefined
        )
        self.jinja_env.filters["format_number"] = format_number
        self.jinja_env.filters["format_metric"] = format_metric
        self.jinja_env.filters["format_percent"] = format_percent
        self.jinja_env.filters["format_metric_name"] = format_metric_name

    def render_html(self, context: Dict[str, Any]) -> str:
        """Deterministically render standalone HTML report."""
        try:
            template = self.jinja_env.get_template("report.html")
            embedded_css = get_embedded_css()
            # Context copy with embedded CSS
            render_context = dict(context)
            render_context["embedded_css"] = embedded_css
            return template.render(**render_context)
        except Exception as e:
            logger.exception(f"Failed to render HTML report: {e}")
            raise AutoDataBotError(f"Report rendering error: {str(e)}")

    def render_pdf(self, html_content: str) -> bytes:
        """Render PDF from HTML using WeasyPrint with graceful error handling."""
        try:
            import weasyprint
            doc = weasyprint.HTML(string=html_content)
            return doc.write_pdf()
        except ImportError:
            raise AutoDataBotError(
                "PDF generation is unavailable in this environment (weasyprint is not installed). "
                "Please view the HTML report or use browser print to save as PDF."
            )
        except Exception as e:
            logger.exception(f"WeasyPrint PDF rendering failed: {e}")
            raise AutoDataBotError(f"PDF generation failed: {str(e)}")

report_builder = ReportBuilder()
