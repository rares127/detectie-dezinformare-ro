"""
Endpoint GET /api/health — verificare status modele incarcate.

Util pentru:
- Debugging local (vezi ce s-a incarcat dupa startup)
- Screenshot-uri in capitolul „Implementare" al tezei
- Healthcheck pentru deployment (returneaza 200 daca totul OK, 503 altfel)
"""

from fastapi import APIRouter, status

from app.config import DEVICE, SEED, THRESHOLD_MODUL3
from app.core.model_loader import model_loader
from app.schemas.responses import HealthResponse


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Status modele încărcate",
)
async def health() -> HealthResponse:
    """
    Returneaza status-ul fiecarui modul + parametri runtime.

    Status:
    - 'ok': toate modulele critice (modul 2, modul 3, Stanza) sunt incarcate
    - 'degraded': startup nu e complet (LIME e ignorat, fiind lazy)
    """
    statusul = "ok" if model_loader.este_gata else "degraded"
    return HealthResponse(
        status=statusul,
        models_loaded=model_loader.status_summary(),
        threshold_modul3=THRESHOLD_MODUL3,
        device=DEVICE,
        seed=SEED,
    )
