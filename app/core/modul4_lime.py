"""
Modul 4 — LIME (Local Interpretable Model-agnostic Explanations).

EXTRAS direct din `06_lime_xlmr_v2.py`, cu urmatoarele decizii arhitecturale:

1. **Lazy loading**: LimeTextExplainer se initializeaza la primul request
   /explain_lime, NU la startup. Motivul: daca utilizatorul testeaza doar
   articole cls1, LIME nu se foloseste niciodata — economisim memorie.

2. **Doar pentru cls0**: rulam LIME EXCLUSIV pe predictii cls0 (stiri credibile).
   Justificare empirica (findings_xai_l4.md, Tabel 4-way):
     - Cls0 (Grup A) faith_auc = +0.169 → cuvinte cu impact cauzal real
     - Cls1 (Grup B/D) faith_auc ≈ 0 sau NEGATIV → stergerea nu schimba
       predictia (sau o creste) → afisarea ar fi misleading
   Restrictia e impusa la nivel de endpoint (HTTP 400 daca pred=1).

3. **Softmax (nu logits)**: contractul LimeTextExplainer cere `predict_proba`
   care returneaza probabilitati. Pe cls0 unde faith_auc e validat, softmax
   functioneaza corect. Logits sunt folosite doar in diagnosticul Modul 4
   (07_lime_l1a_diagnostic.py), NU in productie.
"""

from typing import Optional

import numpy as np
from lime.lime_text import LimeTextExplainer

from app.config import (
    LIME_BOW,
    LIME_NUM_FEATURES,
    LIME_NUM_SAMPLES,
    SEED,
)


class ExplainerLIME:
    """
    Wrapper LIME pentru articole prezise ca cls0 (credibil).

    Pe MPS, o explicatie LIME dureaza ~10-30 secunde (1000 perturbari).
    Acesta e motivul pentru care endpoint-ul /explain_lime e separat de
    /predict — utilizatorul il declanseaza manual, nu automat.
    """

    def __init__(self):
        self._explainer: Optional[LimeTextExplainer] = None

    def initializeaza(self) -> None:
        """
        Creeaza LimeTextExplainer cu configuratia identica cu modulul 4 diagnostic.

        Numele claselor sunt in ordinea conventionala (id2label din modul 2):
        [0]=stire_credibila, [1]=dezinformare_pro_rusa.
        """
        if self._explainer is not None:
            return
        # Setam seed-ul numpy pentru reproducibilitatea perturbarilor LIME
        np.random.seed(SEED)
        self._explainer = LimeTextExplainer(
            class_names=["stire_credibila", "dezinformare_pro_rusa"],
            bow=LIME_BOW,  # False — pastreaza ordinea token-urilor (critic pe transformere)
            random_state=SEED,
        )

    @property
    def este_initializat(self) -> bool:
        return self._explainer is not None

    def explica_cls0(self, text: str, predict_proba_fn) -> dict:
        """
        Genereaza explicatia LIME pentru o predictie cls0.

        Args:
            text: Textul articolului (acelasi cu cel pasat la /predict).
            predict_proba_fn: Functie de tipul `texts -> ndarray (N, 2)`
                — exact cum cere LimeTextExplainer.explain_instance.
                In productie, se paseaza `clasificator_modul2.predict_proba_batch`.

        Returns:
            Dict cu:
              - cuvinte_evidentiate: lista de dict {cuvant, pondere}
                (top LIME_NUM_FEATURES, sortate descrescator dupa pondere absoluta)
              - fidelity_lime: scorul R² al modelului-surogat LIME
        """
        if self._explainer is None:
            raise RuntimeError("LIME neinițializat. Apelează initializeaza().")

        # Rulam LIME pe label-ul cls0 (=0) — conventie identica cu 06_lime_xlmr_v2.py
        exp = self._explainer.explain_instance(
            text,
            predict_proba_fn,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
            labels=[0],  # cls0 (stire credibila)
        )

        # Fidelity = R² locala a modelului-surogat — informativ pentru utilizator
        fidelity = float(exp.score) if hasattr(exp, "score") else 0.0
        # Pe versiuni LIME mai noi, exp.score e dict {label: r2}; tratam ambele cazuri
        if hasattr(exp, "score") and isinstance(exp.score, dict):
            fidelity = float(exp.score.get(0, 0.0))

        # exp.as_list(label=0) → [(cuvant, pondere), ...]
        # Pondere pozitiva = sprijina cls0 (credibil), negativa = sprijina cls1
        cuvinte = []
        for cuvant, pondere in exp.as_list(label=0):
            cuvant_norm = cuvant.strip()
            if not cuvant_norm:
                continue
            cuvinte.append({"cuvant": cuvant_norm, "pondere": float(pondere)})

        # Sortam dupa valoare absoluta descrescator (cuvintele cu impact mai mare prima)
        cuvinte.sort(key=lambda d: abs(d["pondere"]), reverse=True)

        return {
            "cuvinte_evidentiate": cuvinte,
            "fidelity_lime": fidelity,
        }
