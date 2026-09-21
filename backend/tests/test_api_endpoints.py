import io
from fastapi.testclient import TestClient

def test_upload_valid_csv(client: TestClient, sample_clean_csv_bytes):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("clean.csv", io.BytesIO(sample_clean_csv_bytes), "text/csv")}
    )
    assert response.status_code == 201
    data = response.json()
    assert "dataset_id" in data
    assert data["rows_count"] == 5
    assert data["columns_count"] == 4
    assert "summary" in data
    assert "quality" in data
    assert "ingestion" in data

    # Test retrieval using dataset_id
    dataset_id = data["dataset_id"]
    
    # 1. Summary endpoint
    res_summary = client.get(f"/api/datasets/{dataset_id}/summary")
    assert res_summary.status_code == 200
    assert res_summary.json()["row_count"] == 5

    # 2. Quality endpoint
    res_quality = client.get(f"/api/datasets/{dataset_id}/quality")
    assert res_quality.status_code == 200
    assert "quality_score" in res_quality.json()

    # 3. Preview endpoint
    res_preview = client.get(f"/api/datasets/{dataset_id}/preview?limit=3")
    assert res_preview.status_code == 200
    preview_data = res_preview.json()
    assert preview_data["total_rows"] == 5
    assert preview_data["returned_rows"] == 3
    assert len(preview_data["rows"]) == 3

def test_upload_invalid_file_type(client: TestClient):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("document.txt", io.BytesIO(b"Hello world"), "text/plain")}
    )
    assert response.status_code == 415
    assert response.json()["error"] == "INVALID_FILE_TYPE"

def test_upload_empty_csv(client: TestClient):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "EMPTY_DATASET"

def test_get_nonexistent_dataset(client: TestClient):
    response = client.get("/api/datasets/ds_non_existent_12345/summary")
    assert response.status_code == 404
    assert response.json()["error"] == "DATASET_NOT_FOUND"
