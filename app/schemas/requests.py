"""
Scheme de validare input pentru endpoint-uri.

Lungime maxima: MAX_INPUT_CARACTERE (50.000) — peste orice articol real,
dar protejeaza Stanza + encoderul de input-uri abuzive care ar bloca
serverul minute intregi. Texte peste ~1500 caractere se trunchiaza doar
pentru XLM-R (cu warning in UI), nu se resping.
"""

from pydantic import BaseModel, Field

from app.config import MAX_INPUT_CARACTERE


class PredictRequest(BaseModel):
    """Cerere pentru endpoint-ul /api/predict."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_CARACTERE,
        description=(
            "Textul articolului de analizat. Recomandare: titlu + corp în română. "
            "Texte mai lungi de ~1500 caractere se trunchiază pentru clasificarea "
            "globală (XLM-R, max_length=256 tokens). Modul 3 lucrează pe propoziții "
            "individuale și nu e afectat de trunchiere."
        ),
    )


class ExplainRequest(BaseModel):
    """
    Cerere pentru endpoint-ul /api/explain_lime.

    Conform deciziei arhitecturale: rulam LIME DOAR daca decizia FINALA
    a sistemului este cls0 (stire credibila). Decizia finala vine din
    modulul 3 (diff_mean vs. THRESHOLD_MODUL3 = -0.0073), NU din modulul 2.

    IMPORTANT (corectat la audit, iunie 2026): diff_mean NU mai e acceptat
    de la client. Backend-ul recalculeaza scorul modulului 3 server-side
    (~1s, neglijabil fata de cele 10-30s ale LIME). Motivele:
    - trust boundary: un client putea trimite diff_mean fals si obtinea
      o explicatie LIME pe un text cls1 — exact cazul in care propria
      noastra analiza (findings_xai_l4.md) o califica drept misleading;
    - staleness: textul putea fi editat intre /predict si /explain_lime,
      iar diff_mean-ul primit nu mai corespundea textului trimis.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_CARACTERE,
        description=(
            "Textul articolului pentru care se cere explicația LIME. "
            "Backend-ul recalculează decizia modulului 3 pe acest text; "
            "dacă decizia e cls1 (diff_mean > -0.0073), răspunde 400."
        ),
    )
