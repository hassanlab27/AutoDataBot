import json
import zipfile
import pytest
from pathlib import Path
from app.core.config import settings
from app.core.errors import AutoDataBotError

def test_export_zip_creation_and_integrity():
    from app.reporting.export_service import export_service
    run_id = "run_20260924_112411_ecd3fe"
    run_dir = settings.OUTPUTS_DIR / "runs" / run_id
    if not run_dir.exists():
        pytest.skip(f"Test run {run_id} not available")

    zip_path = export_service.create_export_zip(run_id, force_refresh=True)

    # 1. File exists and is non-empty
    assert zip_path.is_file()
    assert zip_path.stat().st_size > 1000

    # 2. Open and test ZIP integrity
    with zipfile.ZipFile(zip_path, "r") as zf:
        # testzip returns None if all file CRC checks pass
        assert zf.testzip() is None

        file_list = zf.namelist()
        prefix = f"autodatabot_{run_id}/"

        # 3. Required top-level files exist
        assert f"{prefix}README.txt" in file_list
        assert f"{prefix}manifest.json" in file_list
        assert f"{prefix}report/report.html" in file_list

        # 4. Manifest is valid JSON with required metadata
        manifest_raw = zf.read(f"{prefix}manifest.json").decode("utf-8")
        manifest = json.loads(manifest_raw)
        assert manifest["run_id"] == run_id
        assert manifest["report_version"] == "1.0"
        assert "files_included" in manifest
        assert "files_excluded" in manifest
        assert "dataset_id" in manifest
        assert "selected_model" in manifest

        # 5. Security & Isolation checks
        # Raw CSV must NEVER be included
        for f in file_list:
            assert not f.endswith(".csv"), f"Raw CSV file found in export: {f}"
            assert ".." not in f, f"Path traversal string found in zip entry: {f}"
            assert not f.startswith("/"), f"Absolute path found in zip entry: {f}"
            assert ".venv" not in f
            assert "__pycache__" not in f


def test_export_path_traversal_protection():
    from app.reporting.export_service import export_service
    # Attempt malicious path traversal in run_id
    with pytest.raises(AutoDataBotError):
        export_service.create_export_zip("../../etc/passwd")

    with pytest.raises(AutoDataBotError):
        export_service.create_export_zip("../../../tmp")
