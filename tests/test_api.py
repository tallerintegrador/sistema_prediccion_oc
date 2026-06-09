"""
Pruebas del API (FastAPI) con TestClient.

Requieren que el ETL y el export de métricas ya se ejecutaron:
    python -m ml.metrics_export
    python -m ml.load_to_db
"""

from fastapi.testclient import TestClient

from api.app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "service" in r.json()


def test_summary():
    r = client.get("/summary")
    assert r.status_code == 200
    j = r.json()
    assert j["best_model"]
    assert j["alerts"]["total"] >= j["alerts"]["high"]


def test_forecast():
    r = client.get("/forecast")
    assert r.status_code == 200
    j = r.json()
    assert len(j["history"]) > 0
    assert len(j["forecast"]) > 0


def test_metrics_forecast():
    r = client.get("/metrics/forecast")
    assert r.status_code == 200
    j = r.json()
    assert j["primary_metric"] == "WAPE"
    assert j["best_model"]
    assert len(j["model_comparison"]) > 0


def test_metrics_anomalies():
    r = client.get("/metrics/anomalies")
    assert r.status_code == 200
    j = r.json()
    assert "precision_at_k" in j
    assert "level_distribution" in j
    assert j["concordance"]["jaccard"]


def test_models():
    r = client.get("/models")
    assert r.status_code == 200
    assert len(r.json()["models"]) > 0


def test_alerts_pagination():
    r = client.get("/alerts", params={"level": "high", "page": 1, "page_size": 5})
    assert r.status_code == 200
    j = r.json()
    assert j["page_size"] == 5
    assert len(j["items"]) <= 5
    assert j["total"] >= 0


def test_alerts_invalid_level():
    r = client.get("/alerts", params={"level": "bogus"})
    assert r.status_code == 422
