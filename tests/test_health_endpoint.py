"""Teste integrare pentru GET /api/health."""


class TestHealthEndpoint:
    """Verifica structura status-ului si parametri runtime."""

    def test_health_returneaza_ok(self, client):
        """Dupa startup complet, status='ok'."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_contine_versiune_modul2(self, client):
        """models_loaded include versiunea modulului 2 (pentru screenshot teza)."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "modul2_classifier" in data["models_loaded"]
        assert data["models_loaded"]["modul2_classifier"] == "xlmr_baseline_v2"

    def test_health_threshold_constant(self, client):
        """Threshold-ul productie expus in /health = -0.0073."""
        resp = client.get("/api/health")
        data = resp.json()
        assert data["threshold_modul3"] == -0.0073

    def test_health_seed_constant(self, client):
        """Seed reproducibilitate = 42."""
        resp = client.get("/api/health")
        data = resp.json()
        assert data["seed"] == 42

    def test_health_lime_lazy(self, client):
        """LIME e marcat ca lazy_not_loaded pana la primul /explain_lime."""
        resp = client.get("/api/health")
        data = resp.json()
        # Imediat dupa startup, LIME nu e inca initializat
        assert data["models_loaded"]["lime_explainer"] in [
            "lazy_not_loaded", "loaded"
        ]
