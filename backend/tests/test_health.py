from fastapi.testclient import TestClient

def test_health_check(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "app" in data

def test_root_route(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]
