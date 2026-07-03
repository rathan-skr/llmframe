"""
API integration tests — no real LLM or vector store calls.
Tests: health check, authentication enforcement, session endpoints.
"""
import pytest
import src.llmframe.api.app as api_mod
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_config():
    """Reset the config singleton before each test so env changes take effect."""
    api_mod._config = None
    yield
    api_mod._config = None


def test_health_returns_ok():
    client = TestClient(api_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_health_requires_no_auth(monkeypatch):
    monkeypatch.setenv("API_SECRET_KEY", "super-secret")
    client = TestClient(api_mod.app)
    # /health must be reachable without a key
    r = client.get("/health")
    assert r.status_code == 200


def test_protected_endpoint_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("API_SECRET_KEY", "my-test-key")
    client = TestClient(api_mod.app)
    r = client.post("/query", json={"question": "test"})
    assert r.status_code == 401


def test_protected_endpoint_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("API_SECRET_KEY", "my-test-key")
    client = TestClient(api_mod.app)
    r = client.post("/query", json={"question": "test"}, headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


def test_protected_endpoint_passes_correct_key(monkeypatch):
    monkeypatch.setenv("API_SECRET_KEY", "my-test-key")
    client = TestClient(api_mod.app)
    r = client.post("/query", json={"question": "test"}, headers={"X-API-Key": "my-test-key"})
    # Auth passed — response will be 500 (no real RAG set up) but NOT 401
    assert r.status_code != 401


def test_no_secret_key_allows_all_requests(monkeypatch):
    monkeypatch.setenv("API_SECRET_KEY", "")
    client = TestClient(api_mod.app)
    # Should reach the endpoint (will 500 without real LLM, but auth is open)
    r = client.post("/query", json={"question": "test"})
    assert r.status_code != 401


def test_session_not_found_returns_404(monkeypatch):
    monkeypatch.setenv("API_SECRET_KEY", "")
    client = TestClient(api_mod.app)
    r = client.get("/sessions/non-existent-id")
    assert r.status_code == 404


def test_delete_nonexistent_session_returns_404(monkeypatch):
    monkeypatch.setenv("API_SECRET_KEY", "")
    client = TestClient(api_mod.app)
    r = client.delete("/sessions/non-existent-id")
    assert r.status_code == 404
