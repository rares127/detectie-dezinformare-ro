"""
Scheme de validare output pentru endpoint-uri.

Structurile reflecta strategia diferentiata pe clasa din PROMPT_chat_nou_modul5_revizuit.md:
- Modul 3 (similaritate semantica) — afisat MEREU
- LIME — disponibil DOAR pe cls0 (endpoint separat /explain_lime)
- DeepLift/GradShap/IG — NU apar in UI (justificate empiric prin faith_auc ≈ 0)
"""

from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Componente reutilizabile
# ─────────────────────────────────────────────────────────────────────────────
class PropozitieSimilara(BaseModel):
    """O propozitie din corpusul de referinta, similara cu o propozitie din input."""

    propozitie_input: str = Field(
        ..., description="Propoziția din articolul utilizatorului"
    )
    propozitie_referinta: str = Field(
        ..., description="Propoziția similară din corpusul de referință"
    )
    sursa: str = Field(
        ..., description="Sursa propoziției de referință (digi24.ro, veridica.ro etc.)"
    )
    scor_similaritate: float = Field(
        ..., description="Cosine similarity (∈ [-1, 1], normalizat → [0, 1])"
    )


class PropozitieDetaliu(BaseModel):
    """
    Detalii per propozitie valida — folosit pentru vizualizarea colorata din UI (Modul 3).

    Fiecare camp `match_cls*` reflecta propozitia din corpus cu cea mai mare
    similaritate cosine fata de propozitia curenta din input.
    """

    text: str = Field(..., description="Textul propoziției din articolul utilizatorului")
    diff: float = Field(..., description="scor_cls1 − scor_cls0 per propoziție individuală")
    scor_cls0: float = Field(
        ..., description="Similaritate cosine max cu corpusul cls0 (credibil)"
    )
    scor_cls1: float = Field(
        ..., description="Similaritate cosine max cu corpusul cls1 (propagandă)"
    )
    match_cls0_text: str = Field(
        ..., description="Propoziția cea mai similară din corpusul credibil"
    )
    match_cls0_sursa: str = Field(..., description="Sursa propoziției match cls0")
    match_cls0_sim: float = Field(..., description="Similaritate cosine cu matchul cls0")
    match_cls1_text: str = Field(
        ..., description="Propoziția cea mai similară din corpusul propagandă"
    )
    match_cls1_sursa: str = Field(..., description="Sursa propoziției match cls1")
    match_cls1_sim: float = Field(..., description="Similaritate cosine cu matchul cls1")


class CuvantEvidentiat(BaseModel):
    """Cuvant cu impact LIME pe predictie (doar pentru cls0)."""

    cuvant: str
    pondere: float = Field(
        ..., description="Pondere LIME (pozitiv = pro clasa prezisă)"
    )


class MetadataPredictie(BaseModel):
    """Informatii de transparenta pentru utilizator."""

    lungime_input_caractere: int
    n_propozitii_total: int
    n_propozitii_valide: int = Field(
        ...,
        description=f"Propoziții care respectă filtrul de lungime [7, 54] cuvinte",
    )
    input_truncat_xlmr: bool = Field(
        ...,
        description="True dacă textul depășește 256 tokens și a fost trunchiat pentru XLM-R",
    )
    timp_inferenta_ms: int


# ─────────────────────────────────────────────────────────────────────────────
# Raspuns principal: POST /api/predict
# ─────────────────────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    """
    Raspunsul endpoint-ului /predict.

    Strategia diferentiata:
    - Modul 2 (XLM-R) → scor global + label + prob_cls1
    - Modul 3 (semantic) → diff_mean + propozitii top + decizie finala
    - Decizia finala vine de la modul 3 (threshold = -0.0073), modul 2 e indicativ.

    Edge case "INCERT": cand articolul nu are propozitii in [7, 54] cuvinte,
    modul 3 e imposibil → afisam doar baseline + flag `decizie_incerta=True`.
    """

    decizie: str = Field(
        ...,
        description=(
            "Una dintre: 'dezinformare_pro_rusa', 'stire_credibila', sau 'incert'. "
            "'incert' apare când modul 3 nu poate calcula scor (zero propoziții valide)."
        ),
    )
    decizie_display: str = Field(
        ...,
        description="Versiune lizibilă pentru UI ('Dezinformare pro-rusă' / 'Știre credibilă' / 'Incert')",
    )
    decizie_incerta: bool = Field(
        ...,
        description="True dacă scorul modul 3 nu a putut fi calculat (sub 1 propoziție validă)",
    )

    # Modul 2 — clasificare globala XLM-R
    scor_baseline_prob_cls1: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilitatea cls1 (dezinformare) după softmax XLM-R",
    )
    label_baseline: str = Field(
        ..., description="Predicția modulului 2 (independent de modul 3)"
    )

    # Modul 3 — similaritate semantica (None daca incert)
    scor_modul3_diff_mean: Optional[float] = Field(
        None,
        description=(
            "Scor combinat = mean(scor_cls1) - mean(scor_cls0) per propoziție. "
            "Decizie cls1 dacă > -0.0073. None dacă incert."
        ),
    )
    scor_modul3_cls0_mean: Optional[float] = Field(
        None, description="Similaritatea medie cu corpusul cls0 (credibil). None dacă incert."
    )
    scor_modul3_cls1_mean: Optional[float] = Field(
        None, description="Similaritatea medie cu corpusul cls1 (propagandă). None dacă incert."
    )
    threshold_producție: float = Field(
        -0.0073, description="Threshold calibrat CV 5-fold (constant)"
    )

    # Confidence pentru bara orizontala din UI
    # (distanta fata de threshold, normalizata la [0, 1])
    incredere: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Distanța față de threshold normalizată. None dacă incert.",
    )

    is_borderline: bool = Field(
        False,
        description=(
            "True dacă verdictul are încredere redusă. Cauzele posibile sunt "
            "enumerate în câmpul 'motiv_borderline'. "
            "Afectează doar afișarea UI — câmpul 'decizie' rămâne neschimbat."
        ),
    )

    motiv_borderline: Optional[str] = Field(
        None,
        description=(
            "Motivul pentru care verdictul e borderline (None dacă nu e). "
            "Valori posibile, în ordinea priorității: "
            "'dezacord_m2_cls1_m3_cls0' — modulul 3 zice credibil, dar XLM-R "
            "indică >80% dezinformare (pattern reported speech trap / Test 4); "
            "'dezacord_m3_cls1_m2_cls0' — modulul 3 zice dezinformare, dar "
            "XLM-R indică <20% (vocabular tematic în cadru jurnalistic); "
            "'proximitate_threshold' — |diff_mean − threshold| < 0.003; "
            "'esantion_mic' — sub 3 propoziții valide pentru modulul 3."
        ),
    )

    # Top propozitii similare (pastrate pentru compatibilitate si debug)
    propozitii_top_cls0: list[PropozitieSimilara] = Field(
        default_factory=list,
        description="Top 3 propoziții din input cu cea mai mare similaritate cu corpusul cls0",
    )
    propozitii_top_cls1: list[PropozitieSimilara] = Field(
        default_factory=list,
        description="Top 3 propoziții din input cu cea mai mare similaritate cu corpusul cls1",
    )

    # Detalii per propozitie valida — folosit pentru vizualizarea colorata din UI
    propozitii_detalii: list[PropozitieDetaliu] = Field(
        default_factory=list,
        description=(
            "Detalii per fiecare propoziție validă (filtrată [7, 54] cuvinte): "
            "diff individual, scoruri cls0/cls1, best match din fiecare corpus. "
            "Folosit pentru vizualizarea colorată din UI (Modul 3)."
        ),
    )

    # Nota metodologica afisata in UI (transparenta)
    nota_metodologica: str = Field(
        ...,
        description=(
            "Mesaj contextual pentru utilizator: explică limitările explicabilității "
            "lexicale pe cls1 (vezi findings_xai_l4.md, Secțiunea 7bis)."
        ),
    )

    metadata: MetadataPredictie


# ─────────────────────────────────────────────────────────────────────────────
# Raspuns endpoint LIME: POST /api/explain_lime
# ─────────────────────────────────────────────────────────────────────────────
class ExplainResponse(BaseModel):
    """
    Raspuns LIME — disponibil DOAR pentru predictii cls0.

    Justificare empirica (findings_xai_l4.md, Tabel 4-way):
    - Cls0 (Grup A): faith_auc LIME = +0.169 → cuvinte cu impact cauzal real
    - Cls1 (Grup B): faith_auc LIME = -0.0001 → stergere cuvinte ≈ no-op
    - Cls1 LOSO TP (Grup D): faith_auc LIME = -0.002 → uneori anti-corect
    Rulam LIME doar acolo unde e validat empiric.
    """

    cuvinte_evidentiate: list[CuvantEvidentiat] = Field(
        ...,
        description=(
            "Top cuvinte cu impact pe predicția cls0, ordonate descrescător după "
            "pondere absolută. Pondere pozitivă = sprijină cls0 (credibil)."
        ),
    )
    fidelity_lime: float = Field(
        ...,
        description=(
            "Scor R² al modelului-surogat LIME. Pe cls0 baseline v2 ≈ 0.50 "
            "(suficient pentru atribuții lexicale fiabile)."
        ),
    )
    metoda: str = Field(
        "LIME",
        description="Metoda XAI folosită (constant)",
    )
    nota_validare: str = Field(
        ...,
        description="Justificare empirică a folosirii LIME (faith_auc validat)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Raspuns /api/health
# ─────────────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Status modele incarcate — util pentru debugging si screenshot teza."""

    status: str = Field(..., description="'ok' dacă toate modulele critice sunt încărcate")
    models_loaded: dict = Field(
        ...,
        description=(
            "Dicționar cu identificatorii modelelor încărcate. "
            "LIME apare ca 'lazy_not_loaded' până la primul request /explain_lime."
        ),
    )
    threshold_modul3: float
    device: str
    seed: int
