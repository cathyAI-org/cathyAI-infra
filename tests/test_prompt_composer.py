import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TestPromptComposer:
    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["CHAR_API_URL"] = "http://test"
        os.environ["CHAR_API_KEY"] = "test_key"
        os.environ["REQUEST_TIMEOUT"] = "10"

    @pytest.fixture
    def mock_char_api(self):
        return {
            "char_id": "catherine",
            "system_prompt": "You are Catherine, a helpful AI assistant.",
            "character_background": "Catherine is knowledgeable and friendly.",
            "matrix": {"append_rules_text": "Be concise in Matrix."},
            "webui": {"append_rules_text": "Be detailed in webui."},
        }

    def test_health(self):
        import sys
        sys.path.insert(0, 'prompt-composer')
        from app import app
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_compose_chat(self, mock_char_api):
        import sys
        sys.path.insert(0, 'prompt-composer')
        from app import app
        
        with patch("app.fetch_character_private") as mock_fetch:
            mock_fetch.return_value = mock_char_api
            
            client = TestClient(app)
            response = client.post(
                "/v1/prompt/compose",
                json={
                    "character_id": "catherine",
                    "platform": "webui",
                    "task": "chat",
                    "person_context": {"person_id": "p_test", "preferred_name": "Test"},
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["character_id"] == "catherine"
            assert data["platform"] == "webui"
            assert data["task"] == "chat"
            assert len(data["system_sections"]) > 0
            assert "Catherine" in data["system_text"]

    @pytest.mark.asyncio
    async def test_compose_with_memory(self, mock_char_api):
        import sys
        sys.path.insert(0, 'prompt-composer')
        from app import app
        
        with patch("app.fetch_character_private") as mock_fetch:
            mock_fetch.return_value = mock_char_api
            
            client = TestClient(app)
            response = client.post(
                "/v1/prompt/compose",
                json={
                    "character_id": "catherine",
                    "platform": "matrix",
                    "task": "chat",
                    "memory_context": {
                        "facts": ["User prefers Python"],
                        "snippets": ["Previous conversation about coding"],
                    },
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "Python" in data["system_text"]
