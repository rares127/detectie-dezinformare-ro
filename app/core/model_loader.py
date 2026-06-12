"""
Singleton pentru orchestrarea modelelor.

Toate modelele se incarca O SINGURA DATA la startup (lifespan handler din main.py).
Endpoint-urile FastAPI acceseaza modelele prin acest singleton — NU re-incarcam
nimic per request.

Modele incarcate la startup:
  - Stanza pipeline (RO, segmentare propozitii) — ~700MB pe disc, ~200MB RAM
  - XLM-R baseline v2 (clasificare globala, modul 2) — ~500MB
  - Sentence-transformers mpnet (encoder modul 3) — ~1.1GB
  - Embeddings precalculate cls0 + cls1 (din cache .npy) — cativa MB

Modele lazy (la primul request relevant):
  - LimeTextExplainer (modul 4) — light, dar inutil daca nu se cere /explain_lime
"""

import logging

from app.core.modul2_classifier import ClasificatorModul2
from app.core.modul3_semantic import ScorerModul3
from app.core.modul4_lime import ExplainerLIME
from app.core.preprocessing import PreprocessorStanza
from app.config import DEVICE, SEED, THRESHOLD_MODUL3


logger = logging.getLogger("app.loader")


class ModelLoader:
    """
    Singleton centralizat pentru toate modelele.

    Folosit prin instanta globala `model_loader` (vezi finalul fisierului).
    Endpoint-urile importa: `from app.core.model_loader import model_loader`.
    """

    def __init__(self):
        # Module critice (incarcate la startup)
        self.preprocesor = PreprocessorStanza()
        self.clasificator = ClasificatorModul2()
        self.scorer = ScorerModul3()
        # Module lazy (incarcate la primul request)
        self.explainer_lime = ExplainerLIME()

        # Flag global de stare
        self._startup_complet = False

    def load_all(self) -> None:
        """
        Incarca toate modulele critice (NU si LIME — acela e lazy).

        Apelat o singura data din lifespan handler-ul FastAPI.
        Ordine: preprocessor → clasificator → scorer (din considerente de
        diagnostic — daca crapa ceva, vedem unde).
        """
        if self._startup_complet:
            return

        logger.info("Device detectat: %s | Seed: %s | Threshold modul 3: %s",
                    DEVICE, SEED, THRESHOLD_MODUL3)

        # 1. Stanza (segmentare propozitii)
        logger.info("(1/3) Inițializez Stanza pipeline...")
        self.preprocesor.initializeaza()

        # 2. XLM-R baseline (clasificare globala)
        logger.info("(2/3) Încarc XLM-R baseline v2...")
        self.clasificator.initializeaza()

        # 3. Encoder mpnet + corpusuri + embeddings
        logger.info("(3/3) Încarc encoder modul 3 + corpusuri...")
        self.scorer.initializeaza()

        self._startup_complet = True

    def asigura_lime_incarcat(self) -> None:
        """
        Lazy-init pentru LIME. Apelat la primul request /explain_lime.

        Idempotent — al doilea apel e no-op.
        """
        if not self.explainer_lime.este_initializat:
            logger.info("(lazy) Inițializez LIME explainer...")
            self.explainer_lime.initializeaza()

    def status_summary(self) -> dict:
        """
        Returneaza status-ul fiecarui modul. Folosit in /api/health.

        Returns:
            Dict cu identificatorii fiecarui modul. LIME apare ca
            'lazy_not_loaded' pana la primul request /explain_lime.
        """
        return {
            "modul2_classifier": (
                self.clasificator.model_version
                if self.clasificator.este_initializat
                else "neîncărcat"
            ),
            "modul3_encoder": (
                "paraphrase-multilingual-mpnet-base-v2"
                if self.scorer.este_initializat
                else "neîncărcat"
            ),
            "stanza_pipeline": (
                "ro_tokenize"
                if self.preprocesor.este_initializat
                else "neîncărcat"
            ),
            "lime_explainer": (
                "loaded"
                if self.explainer_lime.este_initializat
                else "lazy_not_loaded"
            ),
        }

    @property
    def este_gata(self) -> bool:
        """True daca toate modelele critice sunt incarcate."""
        return self._startup_complet


# ─────────────────────────────────────────────────────────────────────────────
# Instanta globala — folosita peste tot prin import
# ─────────────────────────────────────────────────────────────────────────────
model_loader = ModelLoader()
