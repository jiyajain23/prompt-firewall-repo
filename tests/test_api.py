import os
import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Force testing mode so models/engines are skipped at startup
os.environ["TESTING"] = "1"
os.environ["FIREWALL_CONFIG"] = "configs/model_config.yaml"

from src.api.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["device"] == "n/a"
        assert data["faiss_vectors"] == 0

def test_auth_no_key_required():
    # By default, require_api_key is false, but API_KEY defaults to "" (so no auth is enforced)
    with TestClient(app) as client:
        # We expect a 503 Service Unavailable because the engine is not initialized in testing mode
        response = client.post("/v1/classify", json={"prompt": "test", "include_shap": False})
        assert response.status_code == 503

def test_auth_key_validation(monkeypatch):
    # Set FIREWALL_API_KEY and REQUIRE_API_KEY
    monkeypatch.setenv("FIREWALL_API_KEY", "test-secret-key")
    monkeypatch.setenv("REQUIRE_API_KEY", "true")
    
    with TestClient(app) as client:
        # No header -> 401 Unauthorized
        response = client.post("/v1/classify", json={"prompt": "test"})
        assert response.status_code == 401
        
        # Wrong key -> 401 Unauthorized
        response = client.post("/v1/classify", headers={"x-api-key": "wrong"}, json={"prompt": "test"})
        assert response.status_code == 401
        
        # Correct key -> 503 Service Unavailable (auth passed, fails at engine check)
        response = client.post("/v1/classify", headers={"x-api-key": "test-secret-key"}, json={"prompt": "test"})
        assert response.status_code == 503

def test_auth_key_missing_fails_startup(monkeypatch):
    # REQUIRE_API_KEY=true but FIREWALL_API_KEY is empty/missing
    monkeypatch.setenv("FIREWALL_API_KEY", "")
    monkeypatch.setenv("REQUIRE_API_KEY", "true")
    
    with pytest.raises(ValueError) as excinfo:
        with TestClient(app):
            pass
    assert "FIREWALL_API_KEY environment variable is missing" in str(excinfo.value)

def test_rate_limiting(monkeypatch):
    # Reduce rate limit to 2 for quick testing
    # Note: we can mock app.state.rate_limit directly after startup
    with TestClient(app) as client:
        client.app.state.rate_limit = 2
        
        # Request 1: OK (503 because engine is None)
        response = client.post("/v1/classify", json={"prompt": "test"})
        assert response.status_code == 503
        
        # Request 2: OK (503)
        response = client.post("/v1/classify", json={"prompt": "test"})
        assert response.status_code == 503
        
        # Request 3: Rate limited -> 429
        response = client.post("/v1/classify", json={"prompt": "test"})
        assert response.status_code == 429
        assert response.json()["detail"] == "Rate limit exceeded"

def test_classify_with_mock_engine():
    with TestClient(app) as client:
        # Mock engine and store in state
        mock_engine = MagicMock()
        mock_engine.classify.return_value = {
            "verdict": "SAFE",
            "is_adversarial": False,
            "ensemble_score": 0.1,
            "xgb_score": 0.1,
            "transformer_score": 0.1,
            "faiss": {"hit": False},
            "top_families": [],
            "signals": [],
            "shap_top5": [],
            "latency_ms": 1.5,
            "prompt_hash": "123456"
        }
        client.app.state.engine = mock_engine
        
        response = client.post("/v1/classify", json={"prompt": "hello world", "include_shap": False})
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "SAFE"
        assert data["is_adversarial"] is False
        mock_engine.classify.assert_called_once_with("hello world", include_shap=False)

def test_proxy_ip_resolution():
    from src.api.main import _get_client_ip
    from fastapi import Request
    
    # Mock FastAPI request object
    mock_request = MagicMock(spec=Request)
    
    # 1. No proxy headers -> falls back to client host
    mock_request.headers = {}
    mock_request.client = MagicMock()
    mock_request.client.host = "192.168.1.50"
    assert _get_client_ip(mock_request) == "192.168.1.50"
    
    # 2. X-Real-IP set -> uses X-Real-IP
    mock_request.headers = {"x-real-ip": "10.0.0.1"}
    assert _get_client_ip(mock_request) == "10.0.0.1"
    
    # 3. X-Forwarded-For set (multiple IPs) -> uses first client IP
    mock_request.headers = {"x-forwarded-for": "203.0.113.195, 70.41.3.18, 150.172.238.178"}
    assert _get_client_ip(mock_request) == "203.0.113.195"

def test_rate_limiting_cleanup_prevents_leak():
    with TestClient(app) as client:
        # Populate rate buckets with some mock client IPs
        buckets = client.app.state.rate_buckets
        
        # IP 1: active (recent request)
        buckets["1.1.1.1"] = [time.time()]
        
        # IP 2: inactive (older than 60s)
        buckets["2.2.2.2"] = [time.time() - 70]
        
        # Force last_cleanup_time to be old to trigger the sweep on the next request
        client.app.state.last_cleanup_time = time.time() - 100
        
        # Trigger rate limit check on a new IP (3.3.3.3)
        client.post("/v1/classify", json={"prompt": "hello"})
        
        # Verify that the sweep occurred:
        # - "1.1.1.1" should still exist because it was active
        # - "2.2.2.2" should be completely deleted from the dictionary
        # - "3.3.3.3" should exist
        assert "1.1.1.1" in buckets
        assert "2.2.2.2" not in buckets
        assert "3.3.3.3" in buckets

def test_cors_restrictions():
    with TestClient(app) as client:
        # 1. Check preflight OPTIONS request with valid method and headers
        headers = {
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key, content-type",
        }
        response = client.options("/v1/classify", headers=headers)
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://example.com"
        assert "POST" in response.headers.get("access-control-allow-methods", "")
        assert "x-api-key" in response.headers.get("access-control-allow-headers", "").lower()
        assert "content-type" in response.headers.get("access-control-allow-headers", "").lower()
        
        # 2. Check preflight OPTIONS with a disallowed method (PUT)
        headers_disallowed_method = {
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "PUT",
        }
        response_disallowed = client.options("/v1/classify", headers=headers_disallowed_method)
        assert "PUT" not in response_disallowed.headers.get("access-control-allow-methods", "")


