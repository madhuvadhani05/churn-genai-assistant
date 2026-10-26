from fastapi.testclient import TestClient

from api.main import app
from tests.test_model import HIGH_RISK

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_expected_fields():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability", "will_churn", "threshold"}
    assert body["will_churn"] is True


def test_predict_rejects_bad_input():
    r = client.post("/predict", json={"tenure": "abc"})
    assert r.status_code == 422