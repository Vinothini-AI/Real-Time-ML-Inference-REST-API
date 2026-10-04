from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 35,
    "tenure_months": 12,
    "monthly_charges": 79.90,
    "support_tickets": 3,
    "late_payments": 1,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "paperless_billing": "Yes"
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"]
    assert body["model_version"]
    assert "metrics" in body

def test_predict_schema_and_response():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ["CHURN", "NO_CHURN"]
    assert 0 <= body["churn_probability"] <= 1
    assert 0 <= body["confidence"] <= 1
    assert body["model_name"]
    assert body["model_version"]

def test_predict_rejects_invalid_contract():
    payload = dict(VALID_PAYLOAD)
    payload["contract_type"] = "Invalid Contract"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_rejects_missing_feature():
    payload = dict(VALID_PAYLOAD)
    del payload["age"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
