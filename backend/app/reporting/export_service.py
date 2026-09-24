import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.core.config import settings
from app.core.errors import AutoDataBotError
from app.core.logging import logger
from app.reporting.report_context import _sanitize_id, _load_json
from app.reporting.report_service import report_service

EXCLUDED_EXTENSIONS = {".csv", ".pyc", ".log", ".tmp"}
EXCLUDED_PATTERNS = {"__pycache__", ".venv", ".git"}

README_TEMPLATE = """AutoDataBot Model & Analysis Export Bundle
===========================================
Run ID: {run_id}
Dataset ID: {dataset_id}
Problem Type: {problem_type}
Selected Model: {winning_model} ({winning_engine})
Created At: {created_at}

Contents Overview:
------------------
1. manifest.json: Complete export manifest with checksums and provenance.
2. report/:
   - report.html: Standalone deterministic HTML report (open in any browser).
   - report.pdf: Print-ready PDF report (if generated).
3. metadata/:
   - run.json: Run execution timing, stage logs, and parameters.
   - dataset.json: Dataset dimensions, column types, and ingestion report.
   - training_config.json: AutoML tuning hyperparameters and seeds.
4. metrics/:
   - leaderboard.json: Candidate model leaderboard scores and training times.
   - evaluation.json: Full holdout test evaluation payload.
   - diagnostics.json: Generalization gap and under/overfitting analysis.
5. explainability/:
   - feature_importance.json: Native model feature importances.
   - permutation.json: Validation partition permutation importance scores.
   - shap_summary.json: Local and global SHAP attributions (if available).
6. model/:
   - Trained model weights, transformers, and serialization artifacts.

Note: Raw dataset CSVs are excluded by default for security and privacy.
"""

class ExportService:
    """Manages secure, verified ZIP bundling of run artifacts and reports."""

    def get_export_dir(self, run_id: str) -> Path:
        clean_id = _sanitize_id(run_id)
        exp_dir = settings.OUTPUTS_DIR / "runs" / clean_id / "export"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def create_export_zip(self, run_id: str, force_refresh: bool = False) -> Path:
        """
        Creates a secure ZIP archive containing all verified report, metric,
        metadata, and model artifacts for a completed run.
        """
        clean_id = _sanitize_id(run_id)
        run_dir = settings.OUTPUTS_DIR / "runs" / clean_id
        if not run_dir.is_dir():
            raise AutoDataBotError(f"Run {clean_id} not found on disk.")

        export_dir = self.get_export_dir(clean_id)
        zip_path = export_dir / f"autodatabot_{clean_id}.zip"

        if not force_refresh and zip_path.is_file():
            return zip_path

        # 1. Verify status is completed
        status_data = _load_json(run_dir / "status.json") or {}
        run_status = status_data.get("status", "unknown")
        if run_status != "completed":
            raise AutoDataBotError(f"Cannot export run {clean_id}: status is '{run_status}' (must be 'completed').")

        summary_data = _load_json(run_dir / "summary.json") or {}
        dataset_id = summary_data.get("dataset_id") or status_data.get("dataset_id")
        if not dataset_id:
            raise AutoDataBotError(f"Run {clean_id} missing dataset association.")
        clean_dataset_id = _sanitize_id(dataset_id)

        # 2. Ensure HTML report exists
        try:
            report_service.generate_html_report(clean_id)
        except Exception as e:
            logger.warning(f"Failed to generate HTML report for export: {e}")

        # Try PDF report generation (optional, don't fail if weasyprint missing)
        try:
            report_service.generate_pdf_report(clean_id)
        except Exception:
            pass

        # 3. Assemble Files & Validate Path Security
        prefix = f"autodatabot_{clean_id}/"
        files_to_pack: List[tuple[Path, str]] = []
        files_excluded: List[str] = ["*.csv (raw datasets excluded by policy)", "logs/*", ".venv/*"]

        def _is_safe_file(p: Path) -> bool:
            if not p.is_file():
                return False
            name_lower = p.name.lower()
            if any(name_lower.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
                return False
            for part in p.parts:
                if part in EXCLUDED_PATTERNS:
                    return False
            return True

        # Metadata files
        summary_file = run_dir / "summary.json"
        if _is_safe_file(summary_file):
            files_to_pack.append((summary_file, f"{prefix}metadata/run.json"))

        config_file = run_dir / "config.json"
        if _is_safe_file(config_file):
            files_to_pack.append((config_file, f"{prefix}metadata/training_config.json"))

        dataset_meta_file = settings.UPLOAD_DIR / f"{clean_dataset_id}_meta.json"
        if _is_safe_file(dataset_meta_file):
            files_to_pack.append((dataset_meta_file, f"{prefix}metadata/dataset.json"))

        # Preprocessing metadata if available
        prep_meta_file = settings.OUTPUTS_DIR / "datasets" / clean_dataset_id / "preprocessing" / "metadata.json"
        if _is_safe_file(prep_meta_file):
            files_to_pack.append((prep_meta_file, f"{prefix}metadata/preprocessing_metadata.json"))

        # Metrics files
        leaderboard_file = run_dir / "leaderboard.json"
        if _is_safe_file(leaderboard_file):
            files_to_pack.append((leaderboard_file, f"{prefix}metrics/leaderboard.json"))

        eval_dir = run_dir / "evaluation"
        if eval_dir.is_dir():
            for p in eval_dir.glob("*.json"):
                if _is_safe_file(p):
                    files_to_pack.append((p, f"{prefix}metrics/{p.name}"))

        # Explainability files
        explain_dir = run_dir / "explainability"
        if explain_dir.is_dir():
            for p in explain_dir.glob("*.json"):
                if _is_safe_file(p):
                    files_to_pack.append((p, f"{prefix}explainability/{p.name}"))

        # Report files
        rep_dir = run_dir / "report"
        if rep_dir.is_dir():
            for p in rep_dir.iterdir():
                if _is_safe_file(p):
                    files_to_pack.append((p, f"{prefix}report/{p.name}"))

        # Model artifacts
        models_dir = run_dir / "models"
        if models_dir.is_dir():
            for root, _, files in os.walk(models_dir):
                for f in files:
                    p = Path(root) / f
                    if _is_safe_file(p):
                        rel = p.relative_to(models_dir)
                        files_to_pack.append((p, f"{prefix}model/{rel.as_posix()}"))

        # 4. Generate README.txt
        readme_content = README_TEMPLATE.format(
            run_id=clean_id,
            dataset_id=clean_dataset_id,
            problem_type=summary_data.get("problem_type", "unknown"),
            winning_model=summary_data.get("winning_model", "unknown"),
            winning_engine=summary_data.get("winning_engine", "unknown"),
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # 5. Generate manifest.json
        manifest_data = {
            "application": "AutoDataBot",
            "app_version": settings.VERSION,
            "report_version": "1.0",
            "run_id": clean_id,
            "dataset_id": clean_dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "problem_type": summary_data.get("problem_type"),
            "selected_model": summary_data.get("winning_model"),
            "winning_engine": summary_data.get("winning_engine"),
            "validation_score": summary_data.get("validation_score"),
            "test_score": summary_data.get("test_score"),
            "files_included": [arcname for _, arcname in files_to_pack] + [f"{prefix}README.txt", f"{prefix}manifest.json"],
            "files_excluded": files_excluded
        }
        manifest_content = json.dumps(manifest_data, indent=2)

        # 6. Write to ZIP
        temp_zip = export_dir / f"temp_{clean_id}.zip"
        try:
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # Add README
                zf.writestr(f"{prefix}README.txt", readme_content)
                # Add manifest
                zf.writestr(f"{prefix}manifest.json", manifest_content)
                # Add all artifacts with path validation
                for src_path, arcname in files_to_pack:
                    # Traversal guard
                    clean_arcname = os.path.normpath(arcname).replace("\\", "/")
                    if clean_arcname.startswith("/") or clean_arcname.startswith("..") or "/../" in clean_arcname:
                        logger.error(f"Dangerous path blocked from export: {arcname}")
                        continue
                    zf.write(src_path, arcname=clean_arcname)

            # 7. Post-creation integrity verification
            with zipfile.ZipFile(temp_zip, "r") as zf:
                corrupted = zf.testzip()
                if corrupted:
                    raise AutoDataBotError(f"Corrupted file inside generated ZIP: {corrupted}")
                # Verify manifest parses
                m_raw = zf.read(f"{prefix}manifest.json")
                json.loads(m_raw.decode("utf-8"))

            # Atomic move into final position
            temp_zip.replace(zip_path)
            logger.info(f"Export ZIP created successfully at {zip_path} ({zip_path.stat().st_size} bytes)")
            return zip_path

        except Exception as e:
            if temp_zip.exists():
                try:
                    temp_zip.unlink()
                except Exception:
                    pass
            logger.exception(f"Failed to create export ZIP: {e}")
            raise AutoDataBotError(f"Export bundle generation failed: {str(e)}")

export_service = ExportService()
