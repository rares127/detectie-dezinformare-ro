"""
Endpoint POST /api/predict — clasificare globala + similaritate semantica.

Orchestreaza:
1. Preprocessing (Stanza segmentare + filtru lungime [7, 54] cuvinte)
2. Modul 2 (XLM-R baseline) — predictie globala + telemetry trunchiere
3. Modul 3 (similaritate semantica) — diff_mean + top propozitii
4. Decizie finala bazata pe modulul 3 (threshold = -0.0073)

Edge case INCERT: daca articolul nu are nicio propozitie in [7, 54] cuvinte,
modulul 3 e imposibil → returnam doar baseline + flag `decizie_incerta=True`.

Cazuri BORDERLINE (verdict cu incredere redusa, vezi `motiv_borderline`):
- dezacord_m2_cls1_m3_cls0: M3 zice credibil, XLM-R indica >80% dezinformare.
  Pattern „reported speech trap" (Test 4, citate Putin verbatim) — singurul
  mod de eroare in care sistemul poate valida propaganda reala.
- dezacord_m3_cls1_m2_cls0: M3 zice dezinformare, XLM-R indica <20%.
  Vocabular tematic suprapus cu propaganda, in cadru jurnalistic legitim.
- proximitate_threshold: |diff_mean − threshold| < 0.003.
- esantion_mic: sub MIN_PROPOZITII_FIABIL (3) propozitii valide → diff_mean
  statistic fragil.

NOTA DE TRANSPARENTA: decizia finala vine de la modul 3 (mai robust cross-source,
F1=0.9454 vs LOSO-V drop 7.7pp). Modul 2 e afisat ca semnal indicativ, dar nu
dicteaza verdictul.

NOTA TEHNICA: endpoint-ul e `def` (sync), NU `async def` — FastAPI il ruleaza
automat in threadpool. Cu `async def`, inferenta sincrona (Stanza + torch)
ar bloca event loop-ul pentru toate celelalte request-uri.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.config import (
    BORDERLINE_MARGIN,
    LABEL_DISPLAY,
    MIN_PROPOZITII_FIABIL,
    PRAG_DEZACORD_M2_JOS,
    PRAG_DEZACORD_M2_SUS,
    THRESHOLD_MODUL3,
)
from app.core.model_loader import model_loader
from app.schemas.requests import PredictRequest
from app.schemas.responses import (
    MetadataPredictie,
    PredictResponse,
    PropozitieDetaliu,
    PropozitieSimilara,
)


logger = logging.getLogger("app.predict")

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Helper-i locali
# ─────────────────────────────────────────────────────────────────────────────
def _calculeaza_incredere(diff_mean: float) -> float:
    """
    Mapeaza diff_mean la o valoare ∈ [0, 1] pentru bara de incredere din UI.

    Distanta fata de threshold e normalizata folosind tanh, care:
    - Satureaza gratios la valori extreme (nu explodeaza)
    - Returneaza 0.5 chiar la threshold (nici o tabara)
    - Se apropie de 0 sau 1 pe masura ce ne indepartam de threshold

    Factor de scala: 50.0 — calibrat astfel incat diferenta ~0.05 fata de
    threshold (tipica pentru articole clar-clasificate) sa dea ~0.85 incredere.
    """
    import math

    distanta = diff_mean - THRESHOLD_MODUL3
    # tanh(50 * 0.05) ≈ 0.987; tanh(50 * 0.01) ≈ 0.46
    confidence = (math.tanh(50.0 * distanta) + 1.0) / 2.0
    return float(confidence)


def _determina_motiv_borderline(
    decizie_pred: int,
    prob_cls1: float,
    diff_mean: float,
    n_propozitii_valide: int,
) -> Optional[str]:
    """
    Verifica toate conditiile de verdict borderline, in ordinea prioritatii.

    Ordinea reflecta gravitatea: dezacordul M2=cls1 + M3=cls0 e primul pentru
    ca e singurul caz in care sistemul poate valida propaganda reala
    (reported speech trap, Test 4 — vezi Cap. 5 al lucrarii).

    Returns:
        Identificatorul motivului sau None daca verdictul nu e borderline.
    """
    if decizie_pred == 0 and prob_cls1 > PRAG_DEZACORD_M2_SUS:
        return "dezacord_m2_cls1_m3_cls0"
    if decizie_pred == 1 and prob_cls1 < PRAG_DEZACORD_M2_JOS:
        return "dezacord_m3_cls1_m2_cls0"
    if abs(diff_mean - THRESHOLD_MODUL3) < BORDERLINE_MARGIN:
        return "proximitate_threshold"
    if n_propozitii_valide < MIN_PROPOZITII_FIABIL:
        return "esantion_mic"
    return None


def _genereaza_nota_metodologica(
    decizie_incerta: bool,
    label_pred: int,
    motiv_borderline: Optional[str] = None,
    prob_cls1: float = 0.0,
    n_propozitii_valide: int = 0,
) -> str:
    """
    Genereaza mesaj de transparenta afisat in UI sub verdict.

    Strategia diferentiata pe clasa (din PROMPT_chat_nou_modul5_revizuit.md):
    - Pe cls0: oferim buton LIME (faith_auc validat empiric +0.169)
    - Pe cls1: explicam DE CE nu coloram cuvinte (structuri distribuite)
    - Pe INCERT: explicam ca e nevoie de mai multe propozitii
    - Pe BORDERLINE: mesaj specific per motiv (vezi _determina_motiv_borderline)
    """
    if decizie_incerta:
        return (
            "Articolul este prea scurt pentru analiza granulară modulul 3 "
            "(necesită propoziții cu 7-54 cuvinte). Verdictul afișat se "
            "bazează doar pe clasificarea globală XLM-R baseline."
        )
    if motiv_borderline == "dezacord_m2_cls1_m3_cls0":
        return (
            f"Dezacord între modulele de clasificare: analiza semantică "
            f"(modulul 3) indică o știre credibilă, însă clasificatorul global "
            f"XLM-R estimează probabilitate foarte mare de dezinformare "
            f"({prob_cls1:.1%}). Acest tipar apare la articole care citează "
            f"declarații propagandistice fără a le susține (reported speech), "
            f"dar și la propagandă reală pe care analiza semantică o ratează. "
            f"Recomandăm insistent verificare manuală."
        )
    if motiv_borderline == "dezacord_m3_cls1_m2_cls0":
        return (
            f"Dezacord între modulele de clasificare: baseline XLM-R indică "
            f"probabilitate scăzută de dezinformare ({prob_cls1:.1%}), însă "
            f"analiza semantică per propoziție depășește pragul. Articolul "
            f"conține probabil vocabular tematic suprapus cu propaganda "
            f"(Rusia, război, NATO) într-un cadru jurnalistic legitim. "
            f"Recomandăm verificare manuală."
        )
    if motiv_borderline == "proximitate_threshold":
        return (
            "Scor modulul 3 foarte aproape de threshold (marjă < 0.003). "
            "Verdict de încredere redusă — recomandăm verificare manuală."
        )
    if motiv_borderline == "esantion_mic":
        return (
            f"Verdictul se bazează pe doar {n_propozitii_valide} "
            f"propoziți{'e' if n_propozitii_valide == 1 else 'i'} valid"
            f"{'ă' if n_propozitii_valide == 1 else 'e'} (sub pragul de "
            f"{MIN_PROPOZITII_FIABIL}). Pe eșantioane atât de mici, scorul "
            f"semantic e statistic fragil — recomandăm un text mai lung sau "
            f"verificare manuală."
        )
    if label_pred == 1:  # dezinformare
        return (
            "Pentru articole de tip propagandistic, explicabilitatea lexicală "
            "(cuvinte colorate) NU este afișată — modelul folosește structuri "
            "distribuite cross-token, nu vocabular localizat (vezi documentația "
            "metodologică, capitol 4). Explicabilitatea se oferă prin "
            "propozițiile similare din corpusul de referință (modul 3)."
        )
    # cls0 — stire credibila
    return (
        "Pentru articole credibile, explicabilitatea lexicală este validată "
        "empiric (faith_auc=+0.169). Apăsați butonul „Generează explicație” "
        "pentru a vedea cuvintele cu impact major asupra clasificării."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint principal
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Clasificare articol + analiză granulară per propoziție",
)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Endpoint principal de inferenta.

    Pipeline:
    1. Verificare modele incarcate (503 daca startup nu e complet)
    2. Preprocessing: segmentare Stanza + filtru lungime
    3. Modul 2: clasificare globala XLM-R (rapid, ~100ms)
    4. Modul 3: daca exista propozitii valide → similaritate semantica
                altfel → flag INCERT
    5. Combinare in raspuns Pydantic

    Returneaza: PredictResponse cu toate scorurile + propozitii top + nota
    metodologica pentru afisare in UI.
    """
    if not model_loader.este_gata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelele nu sunt încărcate. Așteaptă finalizarea startup-ului.",
        )

    t_start = time.perf_counter()
    text = req.text.strip()

    # ─────────────────────────────────────────────────────────────────────
    # Pas 1: Preprocessing — segmentare + filtrare
    # ─────────────────────────────────────────────────────────────────────
    propozitii_valide, n_propozitii_total = (
        model_loader.preprocesor.segmenteaza_si_filtreaza(text)
    )

    # ─────────────────────────────────────────────────────────────────────
    # Pas 2: Modul 2 — clasificare globala XLM-R
    # ─────────────────────────────────────────────────────────────────────
    rezultat_baseline = model_loader.clasificator.predict_text_unic(text)
    label_baseline = rezultat_baseline["label_pred"]
    prob_cls1 = rezultat_baseline["prob_cls1"]
    input_truncat = rezultat_baseline["input_truncat"]

    # ─────────────────────────────────────────────────────────────────────
    # Pas 3: Modul 3 — scor combinat granular (DACA avem propozitii valide)
    # ─────────────────────────────────────────────────────────────────────
    decizie_incerta = len(propozitii_valide) == 0

    if decizie_incerta:
        # EDGE CASE: zero propozitii valide → fallback la baseline doar.
        # Decizie afisata = "incert" (NU folosim label_baseline ca verdict,
        # pentru ca modul 2 are LOSO-V drop 70.65pp; e doar indicativ).
        nota = _genereaza_nota_metodologica(
            decizie_incerta=True, label_pred=label_baseline
        )
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info(
            "predict: INCERT (0 propoziții valide din %d) | prob_cls1=%.4f | %dms",
            n_propozitii_total, prob_cls1, elapsed_ms,
        )
        return PredictResponse(
            decizie="incert",
            decizie_display="Incert",
            decizie_incerta=True,
            scor_baseline_prob_cls1=prob_cls1,
            label_baseline=LABEL_DISPLAY[label_baseline],
            scor_modul3_diff_mean=None,
            scor_modul3_cls0_mean=None,
            scor_modul3_cls1_mean=None,
            threshold_producție=THRESHOLD_MODUL3,
            incredere=None,
            is_borderline=False,
            motiv_borderline=None,
            propozitii_top_cls0=[],
            propozitii_top_cls1=[],
            nota_metodologica=nota,
            metadata=MetadataPredictie(
                lungime_input_caractere=len(text),
                n_propozitii_total=n_propozitii_total,
                n_propozitii_valide=0,
                input_truncat_xlmr=input_truncat,
                timp_inferenta_ms=elapsed_ms,
            ),
        )

    # Caz normal: rulam modulul 3
    rezultat_m3 = model_loader.scorer.calculeaza_scor(propozitii_valide)
    diff_mean = rezultat_m3["diff_mean"]
    decizie_pred = rezultat_m3["decizie_pred"]
    incredere = _calculeaza_incredere(diff_mean)

    # Detectie caz limita — toate conditiile, in ordinea prioritatii
    # (vezi _determina_motiv_borderline pentru semantica fiecarui motiv)
    motiv_borderline = _determina_motiv_borderline(
        decizie_pred=decizie_pred,
        prob_cls1=prob_cls1,
        diff_mean=diff_mean,
        n_propozitii_valide=len(propozitii_valide),
    )
    is_borderline = motiv_borderline is not None

    # Decizia finala vine de la modul 3 (mai robust decat modul 2).
    # Campul 'decizie' ramane tehnic (cls0/cls1); display-ul se schimba pentru borderline.
    decizie_label = "dezinformare_pro_rusa" if decizie_pred == 1 else "stire_credibila"
    decizie_display = (
        "Verdict incert — caz limită" if is_borderline else LABEL_DISPLAY[decizie_pred]
    )

    nota = _genereaza_nota_metodologica(
        decizie_incerta=False,
        label_pred=decizie_pred,
        motiv_borderline=motiv_borderline,
        prob_cls1=prob_cls1,
        n_propozitii_valide=len(propozitii_valide),
    )

    # Convertim dict-urile la modele Pydantic
    propozitii_top_cls0 = [
        PropozitieSimilara(**p) for p in rezultat_m3["propozitii_top_cls0"]
    ]
    propozitii_top_cls1 = [
        PropozitieSimilara(**p) for p in rezultat_m3["propozitii_top_cls1"]
    ]
    propozitii_detalii = [
        PropozitieDetaliu(**p) for p in rezultat_m3["propozitii_detalii"]
    ]

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "predict: %s%s | diff_mean=%+.4f | prob_cls1=%.4f | %d/%d propoziții | %dms",
        decizie_label,
        f" [borderline: {motiv_borderline}]" if is_borderline else "",
        diff_mean, prob_cls1, len(propozitii_valide), n_propozitii_total, elapsed_ms,
    )

    return PredictResponse(
        decizie=decizie_label,
        decizie_display=decizie_display,
        decizie_incerta=False,
        scor_baseline_prob_cls1=prob_cls1,
        label_baseline=LABEL_DISPLAY[label_baseline],
        scor_modul3_diff_mean=diff_mean,
        scor_modul3_cls0_mean=rezultat_m3["scor_cls0_mean"],
        scor_modul3_cls1_mean=rezultat_m3["scor_cls1_mean"],
        threshold_producție=THRESHOLD_MODUL3,
        incredere=incredere,
        is_borderline=is_borderline,
        motiv_borderline=motiv_borderline,
        propozitii_top_cls0=propozitii_top_cls0,
        propozitii_top_cls1=propozitii_top_cls1,
        propozitii_detalii=propozitii_detalii,
        nota_metodologica=nota,
        metadata=MetadataPredictie(
            lungime_input_caractere=len(text),
            n_propozitii_total=n_propozitii_total,
            n_propozitii_valide=len(propozitii_valide),
            input_truncat_xlmr=input_truncat,
            timp_inferenta_ms=elapsed_ms,
        ),
    )
