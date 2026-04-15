from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    print("✅ Root endpoint works")

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check works")

def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "This is amazing! I love it!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert data["sentiment"] in ["positive", "negative"]
    print(f"✅ Prediction works: {data['sentiment']}")

def test_batch_predict():
    response = client.post(
        "/batch-predict",
        json={"texts": ["Great!", "Terrible!", "Okay"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3
    print("✅ Batch prediction works")

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "accuracy" in response.json()
    print("✅ Metrics endpoint works")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])