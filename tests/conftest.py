"""
Fixture-uri comune pentru testele de integrare API.

NOTA: testele folosesc modelele REALE (XLM-R, mpnet, Stanza) — context
manager-ul TestClient declanseaza lifespan-ul FastAPI, deci modelele se
incarca o singura data per sesiune de teste (~15-20s pe MPS).
Ruleaza din radacina proiectului: `pytest tests/ -v`
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """TestClient cu lifespan activ — modelele se incarca o data per sesiune."""
    with TestClient(app) as c:
        yield c
