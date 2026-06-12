"""Teste integrare pentru POST /api/explain_lime."""


class TestExplainEndpoint:
    """Verifica restrictia arhitecturala: LIME doar pentru cls0."""

    def test_lime_pe_articol_credibil_returneaza_cuvinte(self, client):
        """Articol cls0 → 200 + cuvinte evidentiate."""
        text = (
            "Comisia Europeană a anunțat astăzi un nou pachet de sancțiuni. "
            "Măsurile vor intra în vigoare imediat după publicare oficială."
        )
        resp = client.post("/api/explain_lime", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()

        assert "cuvinte_evidentiate" in data
        assert len(data["cuvinte_evidentiate"]) > 0
        for c in data["cuvinte_evidentiate"]:
            assert "cuvant" in c
            assert "pondere" in c
            assert isinstance(c["pondere"], float)

        assert "fidelity_lime" in data
        assert data["metoda"] == "LIME"
        assert "faith_auc" in data["nota_validare"] or "validat" in data["nota_validare"].lower()

    def test_lime_pe_articol_propaganda_refuzat(self, client):
        """Articol cls1 → 400 cu mesaj clar despre justificarea empirica."""
        text = (
            "Rusia a fost forțată să intervină militar pentru a proteja Donbas. "
            "Putin a declarat că NATO a încălcat promisiunile făcute. "
            "Operațiunea este o măsură de autoapărare legitimă pentru securitate."
        )
        resp = client.post("/api/explain_lime", json={"text": text})
        assert resp.status_code == 400
        detail = resp.json()["detail"].lower()
        # Mesajul trebuie sa explice de ce — referire la cls1 / nefiabil / faith_auc
        assert "cls1" in detail or "dezinformare" in detail or \
               "nefiabil" in detail or "misleading" in detail

    def test_lime_text_gol_returneaza_validare(self, client):
        """Text empty → 422."""
        resp = client.post("/api/explain_lime", json={"text": ""})
        assert resp.status_code == 422
