from fastapi.testclient import TestClient

from app.data.synthetic import generate_region
from app.main import app

client = TestClient(app)


def test_health_and_regions():
    assert client.get("/api/health").status_code == 200
    r = client.get("/api/regions").json()
    assert len(r) == 3


def test_stored_results_exist():
    for kind in ("forecast", "kernel", "noise", "simulator"):
        assert client.get(f"/api/results/{kind}").status_code == 200


def test_series_endpoint_returns_the_expected_generated_observations():
    start = "2019-01-01 00:00"
    response = client.get("/api/series?region=C&seed=3&start=2019-01-01%2000:00&hours=24")
    assert response.status_code == 200
    body = response.json()
    assert body["region"] == "C"
    assert len(body["t"]) == len(body["y"]) == len(body["wind_ms"]) == 24
    expected = generate_region("C", seed=3).loc[start:].iloc[:24]
    assert body["y"] == expected["y"].round(4).tolist()
    assert body["wind_ms"] == expected["wind_ms"].round(4).tolist()


def test_explanation_uses_accurate_model_name_and_cautious_comparison():
    response = client.get("/api/explain?region=C")
    assert response.status_code == 200
    text = response.json()["text"]
    assert "ARIMA" not in text
    assert "AR(24)" in text
    assert "does not establish statistical significance or real-world performance" in text
