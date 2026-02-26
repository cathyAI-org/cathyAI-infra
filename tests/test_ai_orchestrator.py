import os
import tempfile

import pytest
from fastapi.testclient import TestClient


class TestAIOrchestrator:
    @pytest.fixture(autouse=True)
    def setup(self):
        import sys
        if 'app' in sys.modules:
            del sys.modules['app']
        
        os.environ["PROMPT_COMPOSER_URL"] = "http://test:8110"
        os.environ["OLLAMA_API_URL"] = "http://test:8000"
        os.environ["STATE_DB"] = tempfile.mktemp(suffix=".sqlite")
        os.environ["GLOBAL_WORKERS"] = "1"
        os.environ["MAX_QUEUE_PER_SESSION"] = "3"
        os.environ["MAX_TOTAL_QUEUED"] = "50"
        
        sys.path.insert(0, 'ai-orchestrator')
        from app import app, on_startup
        import asyncio
        asyncio.run(on_startup())
        self.app = app

    def test_health(self):
        client = TestClient(self.app)
        
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_submit_job(self):
        client = TestClient(self.app)
        
        response = client.post(
            "/v1/jobs/submit",
            json={
                "source": "webui",
                "session_id": "test-session",
                "task_type": "chat",
                "character_id": "catherine",
                "platform": "webui",
                "user_message": "Hello",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_get_job(self):
        client = TestClient(self.app)
        
        submit_response = client.post(
            "/v1/jobs/submit",
            json={
                "source": "webui",
                "session_id": "test-session",
                "task_type": "chat",
                "character_id": "catherine",
                "platform": "webui",
                "user_message": "Hello",
            },
        )
        
        job_id = submit_response.json()["job_id"]
        
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"

    def test_coalescing(self):
        client = TestClient(self.app)
        
        response1 = client.post(
            "/v1/jobs/submit",
            json={
                "source": "webui",
                "session_id": "test-coalesce",
                "task_type": "chat",
                "character_id": "catherine",
                "platform": "webui",
                "user_message": "First message",
                "coalesce": True,
            },
        )
        
        job_id_1 = response1.json()["job_id"]
        
        response2 = client.post(
            "/v1/jobs/submit",
            json={
                "source": "webui",
                "session_id": "test-coalesce",
                "task_type": "chat",
                "character_id": "catherine",
                "platform": "webui",
                "user_message": "Second message",
                "coalesce": True,
            },
        )
        
        data2 = response2.json()
        assert data2["coalesced_into"] == job_id_1

    def test_queue_limits(self):
        import sys
        if 'app' in sys.modules:
            del sys.modules['app']
        
        os.environ["MAX_QUEUE_PER_SESSION"] = "2"
        
        sys.path.insert(0, 'ai-orchestrator')
        from app import app, on_startup
        import asyncio
        asyncio.run(on_startup())
        client = TestClient(app)
        
        for i in range(2):
            response = client.post(
                "/v1/jobs/submit",
                json={
                    "source": "webui",
                    "session_id": "test-limits",
                    "task_type": "chat",
                    "character_id": "catherine",
                    "platform": "webui",
                    "user_message": f"Message {i}",
                    "coalesce": False,
                },
            )
            assert response.status_code == 200
        
        response = client.post(
            "/v1/jobs/submit",
            json={
                "source": "webui",
                "session_id": "test-limits",
                "task_type": "chat",
                "character_id": "catherine",
                "platform": "webui",
                "user_message": "Too many",
                "coalesce": False,
            },
        )
        assert response.status_code == 429
