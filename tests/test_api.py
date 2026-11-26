from fastapi.testclient import TestClient
import pytest
import os
import io

# Set mock API key for tests
os.environ["OPENAI_API_KEY"] = "mock"
os.environ["GROQ_API_KEY"] = "mock"

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "groq_configured" in data
    assert "openai_configured" in data

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "uploaded_files" in data

def test_config_keys_endpoint():
    response = client.post("/config/keys", json={"groq_api_key": "gsk_test123", "persist": False})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_query_endpoint():
    payload = {"query": "How do I fix my CloudStore database?"}
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "documents_retrieved" in data
    assert "hallucination_detected" in data
    assert "iterations" in data

def test_hallucination_trigger():
    payload = {"query": "Test hallucination: Why is my AuthMagic failing?"}
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["iterations"] >= 1

def test_upload_endpoint():
    sample_text = b"Product: CloudSync\nIssue: File conflicts on sync.\nResolution: Ensure file locking is turned on in settings."
    files = [
        ("files", ("test_doc.txt", io.BytesIO(sample_text), "text/plain"))
    ]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_added"] >= 1
