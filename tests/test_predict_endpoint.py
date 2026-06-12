"""Teste integrare pentru POST /api/predict."""

import pytest


class TestPredictEndpoint:
    """Verifica contractul HTTP + logica de orchestrare."""

    def test_articol_credibil_returneaza_cls0(self, client):
        """Text credibil cu vocabular tematic → decizie tehnica cls0.

        NOTA (audit iunie 2026): pe acest text XLM-R e saturat de vocabularul
        tematic (prob_cls1 ≈ 0.999), deci dezacordul M2=cls1 + M3=cls0
        declanseaza INTENTIONAT verdictul borderline (reported speech trap) —
        decizia tehnica ramane cls0, dar display-ul avertizeaza utilizatorul.
        """
        text = (
            "Comisia Europeană a anunțat astăzi un nou pachet de sancțiuni economice "
            "împotriva entităților implicate în finanțarea războiului. Măsurile vor "
            "intra în vigoare la data publicării în Jurnalul Oficial al Uniunii Europene. "
            "Statele membre au confirmat sprijinul unanim pentru aceste decizii."
        )
        resp = client.post("/api/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()

        assert data["decizie"] == "stire_credibila"
        assert data["decizie_incerta"] is False
        assert data["scor_modul3_diff_mean"] is not None
        assert len(data["propozitii_top_cls0"]) > 0
        assert len(data["propozitii_top_cls1"]) > 0

        if data["is_borderline"]:
            # Dezacordul inter-modular trebuie SEMNALAT, nu mascat
            assert data["decizie_display"] == "Verdict incert — caz limită"
            assert data["motiv_borderline"] is not None
            assert "manual" in data["nota_metodologica"].lower()
        else:
            assert data["decizie_display"] == "Știre credibilă"
            # Nota metodologica pentru cls0 trebuie sa mentioneze butonul LIME
            assert "explica" in data["nota_metodologica"].lower() or \
                   "lime" in data["nota_metodologica"].lower()

    def test_dezacord_m2_cls1_m3_cls0_semnalat_borderline(self, client):
        """Reported speech trap (fix CRITIC din audit, iunie 2026).

        Text credibil pe care XLM-R il clasifica cls1 cu probabilitate foarte
        mare (vocabular tematic), dar modulul 3 il clasifica cls0. Inainte de
        fix, sistemul afisa „Stire credibila" verde fara avertisment — exact
        cazul Test 4 (citate Putin verbatim) documentat in Cap. 5 al lucrarii.
        """
        text = (
            "Comisia Europeană a anunțat astăzi un nou pachet de sancțiuni economice "
            "împotriva entităților implicate în finanțarea războiului. Măsurile vor "
            "intra în vigoare la data publicării în Jurnalul Oficial al Uniunii Europene. "
            "Statele membre au confirmat sprijinul unanim pentru aceste decizii."
        )
        resp = client.post("/api/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()

        # Premisele dezacordului (dependente de model — verificate explicit)
        assert data["decizie"] == "stire_credibila"          # M3 → cls0
        assert data["scor_baseline_prob_cls1"] > 0.80         # M2 → cls1 confident

        # Comportamentul corectat: banda ambra + motiv explicit
        assert data["is_borderline"] is True
        assert data["motiv_borderline"] == "dezacord_m2_cls1_m3_cls0"
        assert data["decizie_display"] == "Verdict incert — caz limită"

    def test_articol_propagandistic_returneaza_cls1(self, client):
        """Text cu 'rusia' / 'putin' → cls1 + nota despre structuri distribuite."""
        text = (
            "Rusia a fost forțată să intervină militar pentru a proteja populația "
            "vorbitoare de limbă rusă din Donbas. Putin a declarat că NATO a încălcat "
            "promisiunile făcute la sfârșitul Războiului Rece. Operațiunea specială "
            "este o măsură de autoapărare legitimă."
        )
        resp = client.post("/api/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()

        assert data["decizie"] == "dezinformare_pro_rusa"
        assert data["decizie_incerta"] is False
        assert data["scor_modul3_diff_mean"] > -0.0073  # peste threshold
        # Pe cls1 nota metodologica trebuie sa explice de ce nu afisam LIME
        assert "distribu" in data["nota_metodologica"].lower() or \
               "structur" in data["nota_metodologica"].lower()

    def test_text_scurt_returneaza_incert(self, client):
        """Text fara propozitii in [7,54] cuvinte → decizie_incerta=True."""
        # Trei propozitii foarte scurte (sub 7 cuvinte fiecare)
        text = "Hello world. Salut. Test scurt aici."
        resp = client.post("/api/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()

        assert data["decizie"] == "incert"
        assert data["decizie_display"] == "Incert"
        assert data["decizie_incerta"] is True
        assert data["scor_modul3_diff_mean"] is None
        assert data["incredere"] is None
        assert data["propozitii_top_cls0"] == []
        assert data["propozitii_top_cls1"] == []
        assert "scurt" in data["nota_metodologica"].lower() or \
               "granular" in data["nota_metodologica"].lower()

    def test_text_gol_returneaza_eroare_validare(self, client):
        """Text empty string → 422 Unprocessable Entity (Pydantic validation)."""
        resp = client.post("/api/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_text_lipsa_returneaza_eroare_validare(self, client):
        """Lipsa camp 'text' → 422."""
        resp = client.post("/api/predict", json={})
        assert resp.status_code == 422

    def test_metadata_contine_telemetry(self, client):
        """Raspunsul trebuie sa contina metadata cu lungimi + timp."""
        text = (
            "Acesta este un articol de test cu propoziții suficient de lungi "
            "pentru a respecta filtrul de lungime impus de modulul 3 al sistemului. "
            "Articolul nu conține nicio narațiune pro-rusă identificabilă."
        )
        resp = client.post("/api/predict", json={"text": text})
        data = resp.json()
        m = data["metadata"]
        assert m["lungime_input_caractere"] == len(text)
        assert m["n_propozitii_total"] >= 1
        assert m["timp_inferenta_ms"] >= 0
        assert isinstance(m["input_truncat_xlmr"], bool)

    def test_threshold_constant_in_response(self, client):
        """Threshold-ul productie trebuie sa fie constant -0.0073."""
        text = "Acesta este un articol scurt pentru testul threshold-ului."
        resp = client.post("/api/predict", json={"text": text})
        data = resp.json()
        # Pydantic alias e cu diacritica
        assert data["threshold_producție"] == -0.0073
