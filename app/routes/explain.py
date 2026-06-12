"""
Endpoint POST /api/explain_lime — explicabilitate lexicala LIME.

Restrictie arhitecturala: rulam LIME EXCLUSIV pe predictii cls0 (stiri credibile).

Justificare empirica (findings_xai_l4.md, Tabel 4-way faithfulness deletion AUC):
- Cls0 (Grup A): faith_auc = +0.169 → cuvinte cu impact cauzal real
- Cls1 (Grup B baseline): faith_auc = -0.0001 → stergere cuvinte ≈ no-op
- Cls1 (Grup D LOSO TP): faith_auc = -0.002 → uneori predictia creste
  (modelul nu se baza pe acele cuvinte)

CRITERIU DE ELIGIBILITATE — corectat la audit (iunie 2026):
  diff_mean (modulul 3) e RECALCULAT server-side pe textul primit, NU mai e
  acceptat de la client. Versiunea anterioara avea incredere in diff_mean-ul
  trimis de client — un client putea trimite orice valoare ≤ threshold si
  obtinea o explicatie LIME pe un text cls1, exact cazul in care propria
  noastra analiza o califica drept misleading. Recalcularea costa ~1s,
  neglijabil fata de cele 10-30s ale LIME, si garanteaza ca explicatia
  corespunde textului curent (nu unui /predict anterior, posibil editat).

  Criteriul ramane modulul 3 (F1=0.9454), NU modulul 2 izolat — acesta are
  LOSO-V drop 70.65pp si clasifica gresit ca cls1 stiri credibile cu
  vocabular tematic.

NOTA TEHNICA: endpoint-ul e `def` (sync), NU `async def` — FastAPI il ruleaza
in threadpool. Cu `async def`, cele 10-30s de LIME sincron ar bloca event
loop-ul: NICIUN alt request (/predict, /health) nu ar primi raspuns intre timp.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.config import THRESHOLD_MODUL3
from app.core.model_loader import model_loader
from app.schemas.requests import ExplainRequest
from app.schemas.responses import CuvantEvidentiat, ExplainResponse


logger = logging.getLogger("app.explain")

router = APIRouter()


# Justificare empirica afisata in UI (text constant)
_NOTA_VALIDARE_LIME = (
    "Cuvintele afișate au impact cauzal validat empiric pe articole credibile "
    "(faith_auc = +0.169, n=25). Pe articolele propagandistice această metodă "
    "ar fi nefiabilă (faith_auc ≈ 0 sau negativ) — vezi capitolul 4 al lucrării."
)


@router.post(
    "/explain_lime",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Explicație LIME (cuvinte cu impact) — doar pentru predicții cls0",
)
def explain_lime(req: ExplainRequest) -> ExplainResponse:
    """
    Genereaza explicatia LIME pentru un articol a carui decizie finala este cls0.

    Pasii:
    1. Verifica ca modelele sunt incarcate (503 daca nu)
    2. RECALCULEAZA decizia modulului 3 pe textul primit (segmentare Stanza +
       similaritate semantica, ~1s). Refuza LIME daca:
       - textul nu are propozitii valide (verdict INCERT — nu exista decizie cls0)
       - diff_mean > THRESHOLD_MODUL3 (decizie cls1 — faith_auc ≈ 0, explicatia
         ar induce in eroare)
    3. Lazy-init LIME explainer (la primul request)
    4. Ruleaza LIME (~10-30 sec pe MPS) cu predict_proba_batch al modulului 2
    5. Returneaza cuvintele evidentiate + fidelity-ul modelului-surogat

    Returns:
        ExplainResponse cu top cuvinte ordonate dupa impact absolut.

    Raises:
        503: daca modelele nu sunt incarcate
        400: daca decizia recalculata nu este cls0 (incert sau cls1)
    """
    if not model_loader.este_gata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelele nu sunt încărcate. Așteaptă finalizarea startup-ului.",
        )

    text = req.text.strip()

    # ─────────────────────────────────────────────────────────────────────
    # Pas 1: Recalculam decizia modulului 3 SERVER-SIDE (sursa de adevar)
    # ─────────────────────────────────────────────────────────────────────
    propozitii_valide, _ = model_loader.preprocesor.segmenteaza_si_filtreaza(text)

    if not propozitii_valide:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "LIME nu e disponibil: textul nu conține nicio propoziție validă "
                "(7-54 cuvinte), deci verdictul sistemului este INCERT, nu cls0. "
                "LIME e validat empiric doar pe articole cu decizie finală cls0."
            ),
        )

    rezultat_m3 = model_loader.scorer.calculeaza_scor(propozitii_valide)
    diff_mean = rezultat_m3["diff_mean"]

    # Criteriu: diff_mean > THRESHOLD_MODUL3 inseamna cls1 conform sistemului
    # complet — refuzam LIME (faith_auc ≈ 0 pe cls1, deci afisarea ar induce
    # in eroare utilizatorul).
    if diff_mean > THRESHOLD_MODUL3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"LIME nu e disponibil: decizia finală a sistemului este cls1 "
                f"(diff_mean = {diff_mean:.4f} > threshold {THRESHOLD_MODUL3}). "
                f"Pe articole propagandistice, faith_auc al LIME este ≈ 0 sau negativ — "
                f"afișarea cuvintelor colorate ar fi misleading. "
                f"Folosește propozițiile similare din corpus cls1 returnate de /predict."
            ),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Pas 2: Lazy-init LIME (doar la primul request)
    # ─────────────────────────────────────────────────────────────────────
    model_loader.asigura_lime_incarcat()

    # ─────────────────────────────────────────────────────────────────────
    # Pas 3: Ruleaza LIME (LENT — 10-30 sec)
    # ─────────────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    rezultat_lime = model_loader.explainer_lime.explica_cls0(
        text=text,
        predict_proba_fn=model_loader.clasificator.predict_proba_batch,
    )
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "explain_lime: diff_mean=%+.4f (recalculat server-side) | "
        "fidelity=%.3f | %dms",
        diff_mean, rezultat_lime["fidelity_lime"], elapsed_ms,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Pas 4: Construire raspuns Pydantic
    # ─────────────────────────────────────────────────────────────────────
    cuvinte = [
        CuvantEvidentiat(cuvant=c["cuvant"], pondere=c["pondere"])
        for c in rezultat_lime["cuvinte_evidentiate"]
    ]

    return ExplainResponse(
        cuvinte_evidentiate=cuvinte,
        fidelity_lime=rezultat_lime["fidelity_lime"],
        metoda="LIME",
        nota_validare=_NOTA_VALIDARE_LIME,
    )
