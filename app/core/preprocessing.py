"""
Preprocessing NLP pentru pipeline-ul de inferenta.

Functionalitate:
- Segmentare propozitii cu Stanza (ro, tokenize)
- Filtrare propozitii pe lungime [MIN_CUVINTE, MAX_CUVINTE]
- Identic cu pipeline-ul din `calibrare_threshold_v2.py::segmenteaza_articole_val()`,
  pentru a garanta ca scorurile productie sunt comparabile cu cele calibrate.
"""

from typing import Optional

import stanza

from app.config import (
    MIN_CUVINTE,
    MAX_CUVINTE,
    STANZA_LANG,
    STANZA_PROCESSORS,
    STANZA_USE_GPU,
)


class PreprocessorStanza:
    """
    Wrapper pentru pipeline-ul Stanza romanesc.

    Pipeline-ul e incarcat o singura data (la initializare) si reutilizat
    pentru toate request-urile. Stanza ruleaza pe CPU (consistent cu
    antrenarea modulului 3) — vezi config.STANZA_USE_GPU = False.
    """

    def __init__(self):
        """Initializeaza pipeline-ul Stanza pentru romana."""
        # NOTA: la primul run, Stanza descarca modelul ro (~700MB).
        # Documentat in README_app.md.
        self._nlp: Optional[stanza.Pipeline] = None

    def initializeaza(self) -> None:
        """Incarca pipeline-ul Stanza. Apelat o singura data la startup."""
        if self._nlp is not None:
            return
        self._nlp = stanza.Pipeline(
            lang=STANZA_LANG,
            processors=STANZA_PROCESSORS,
            verbose=False,
            use_gpu=STANZA_USE_GPU,
            download_method=None,  # presupunem ca modelul e deja descarcat
        )

    @property
    def este_initializat(self) -> bool:
        """True daca pipeline-ul Stanza e gata de folosit."""
        return self._nlp is not None

    def segmenteaza(self, text: str) -> list[str]:
        """
        Segmenteaza textul in propozitii brute (inainte de filtrare).

        Args:
            text: Textul articolului (string lung, posibil cu titlu + corp).

        Returns:
            Lista de propozitii ca string-uri, in ordinea aparitiei in text.
        """
        if self._nlp is None:
            raise RuntimeError(
                "Pipeline Stanza neinițializat. Apelează initializeaza() la startup."
            )
        if not text or not text.strip():
            return []
        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sentences if sent.text.strip()]

    def segmenteaza_si_filtreaza(self, text: str) -> tuple[list[str], int]:
        """
        Segmenteaza SI filtreaza propozitiile pe lungime.

        Identic cu logica din `calibrare_threshold_v2.py` linia 220-221:
        pastram doar propozitiile cu MIN_CUVINTE ≤ nr_cuvinte ≤ MAX_CUVINTE.

        Args:
            text: Textul articolului.

        Returns:
            (propozitii_valide, n_total_brute):
              - propozitii_valide: lista propozitiilor care trec filtrul
              - n_total_brute: numarul total de propozitii inainte de filtrare
                (util pentru telemetry: cate propozitii am pierdut din cauza
                lungimii)
        """
        propozitii_brute = self.segmenteaza(text)
        propozitii_valide = []
        for prop in propozitii_brute:
            n_cuvinte = len(prop.split())
            if MIN_CUVINTE <= n_cuvinte <= MAX_CUVINTE:
                propozitii_valide.append(prop)
        return propozitii_valide, len(propozitii_brute)
