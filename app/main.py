"""
Entry point FastAPI pentru sistemul de detectie a dezinformarii pro-ruse.

Modulul 5 al lucrarii de licenta „Sistem de Detectie Automata si Explicabila
a Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din
Ucraina" (Informatica, anul III, 2025-2026).

Acest modul orchestreaza:
- Incarcarea modelelor la startup (lifespan handler) — NU per request
- Inregistrarea endpoint-urilor (/predict, /explain_lime, /health)
- Servirea frontend-ului static (HTML + Tailwind CDN + Vanilla JS)

Threshold productie modul 3: -0.0073 (calibrat CV 5-fold, F1=0.9454).
Coloana text input peste tot: text_curat.
Comentarii si docstring-uri in romana.

Rulare:
    python -m app.main
sau
    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import APP_TITLE, APP_VERSION, STATIC_DIR, TEMPLATES_DIR
from app.core.model_loader import model_loader
from app.routes import explain, health, predict


# Logging structurat pentru toate modulele aplicatiei (loggere "app.*").
# basicConfig e no-op daca root logger-ul are deja handlers — compatibil
# atat cu `python -m app.main` cat si cu `uvicorn app.main:app`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app.main")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan handler — incarca modelele O SINGURA DATA la startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Hook executat la pornirea si oprirea aplicatiei.

    La startup:
    - Stanza pipeline (segmentare romana) — ~5s
    - XLM-RoBERTa baseline v2 (clasificare modul 2) — ~5s pe MPS
    - Sentence-transformers mpnet (encoder modul 3) — ~5s pe MPS
    - Embeddings cls0 + cls1 din cache .npy (sau recalculate daca lipsesc)

    LIME explainer (modul 4) e LAZY: se initializeaza la primul request
    /explain_lime, nu la startup. Motivul: daca utilizatorul testeaza doar
    articole cls1, LIME nu se foloseste niciodata.

    Total startup time estimat: 15-20 secunde pe MacBook M2 Pro.
    """
    logger.info("=" * 60)
    logger.info("%s v%s", APP_TITLE, APP_VERSION)
    logger.info("Încarc modelele... (15-20s pe MPS)")
    model_loader.load_all()
    logger.info("✓ Toate modulele critice încărcate.")
    logger.info("Status: %s", model_loader.status_summary())
    logger.info("=" * 60)
    yield
    # La shutdown — pe MPS nu avem nevoie de eliberare explicita
    logger.info("Aplicație oprită.")


# ─────────────────────────────────────────────────────────────────────────────
# Instantiere FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Sistem de detecție automată și explicabilă a dezinformării pro-ruse "
        "în presa românească. Folosește XLM-RoBERTa (modul 2 — clasificare "
        "globală) + similaritate semantică per propoziție (modul 3 — explicabilitate "
        "principală, F1=0.9454) + LIME (modul 4 — explicabilitate lexicală, "
        "doar pe articole prezise ca credibile)."
    ),
    lifespan=lifespan,
)

# Montare fisiere statice (CSS, JS) la /static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Inregistrare rute API la prefix /api
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(explain.router, prefix="/api", tags=["explain"])
app.include_router(health.router, prefix="/api", tags=["health"])

# Templates Jinja2 pentru servirea index.html
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Root — serveste single-page UI
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """Serveste frontend-ul (HTML + Tailwind via CDN + Vanilla JS)."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_title": APP_TITLE,
            "version": APP_VERSION,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Punct de intrare pentru rulare directa: python -m app.main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # NU reload — modelele se reincarca la fiecare schimbare
        log_level="info",
    )
