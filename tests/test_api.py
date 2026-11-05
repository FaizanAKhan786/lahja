"""API tests against the mock backend (no GPU required)."""
import os

os.environ["LAHJA_MOCK_MODEL"] = "1"

from fastapi.testclient import TestClient

from lahja.serve.api import app


def client() -> TestClient:
    return TestClient(app)


def test_health():
    with client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_info_reports_backend():
    with client() as c:
        assert c.get("/info").json()["backend"] == "mock"


def test_generate_returns_text_and_latency():
    with client() as c:
        r = c.post("/generate", json={"prompt": "شلونك؟", "max_new_tokens": 32})
        assert r.status_code == 200
        body = r.json()
        assert body["text"]
        assert body["latency_ms"] >= 0


def test_generate_validates_input():
    with client() as c:
        assert c.post("/generate", json={"prompt": ""}).status_code == 422
        assert c.post("/generate", json={"prompt": "hi", "max_new_tokens": 99999}).status_code == 422


def test_streaming_emits_sse_and_done():
    with client() as c:
        r = c.post("/generate", json={"prompt": "مرحبا", "stream": True})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "[DONE]" in r.text
