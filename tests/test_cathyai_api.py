import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import httpx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ollama-api'))
from app import app

client = TestClient(app)


def test_health_backend_up():
    with patch("app.httpx.AsyncClient") as mock_client:
        mock_get = Mock()
        mock_client.return_value.__aenter__.return_value.get = mock_get
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["emotion_enabled"] is False


def test_health_backend_down():
    with patch("app.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["backend_status"] == "down"


def test_models_success():
    with patch("app.httpx.AsyncClient") as mock_client:
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "llama2"}, {"name": "gemma2:2b"}]}
        mock_response.raise_for_status = Mock()
        
        async def mock_get(*args, **kwargs):
            return mock_response
        
        mock_client.return_value.__aenter__.return_value.get = mock_get
        
        response = client.get("/models")
        assert response.status_code == 200
        assert response.json() == {"models": ["llama2", "gemma2:2b"]}


def test_models_raw():
    with patch("app.httpx.AsyncClient") as mock_client:
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "llama2", "size": 123}]}
        mock_response.raise_for_status = Mock()
        
        async def mock_get(*args, **kwargs):
            return mock_response
        
        mock_client.return_value.__aenter__.return_value.get = mock_get
        
        response = client.get("/models/raw")
        assert response.status_code == 200
        assert "models" in response.json()


def test_models_backend_unavailable():
    with patch("app.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.side_effect = httpx.RequestError("Connection failed")
        response = client.get("/models")
        assert response.status_code == 503
        assert "Backend unavailable" in response.json()["detail"]
