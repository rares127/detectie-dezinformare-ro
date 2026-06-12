"""
Modul 2 — Wrapper inferenta pentru XLM-RoBERTa baseline v2.

Logica e extrasa DIRECT din `03_eval_xlmr_baseline_v2.py::predict_dataframe()`,
adaptata pentru text unic (nu DataFrame). Configuratia de tokenizare e identica
cu antrenarea (max_length=256, truncation=True).

IMPORTANT: pe articole > 256 tokens, XLM-R primeste text trunchiat.
Acesta e comportamentul EXACT cu care a fost antrenat si calibrat — NU
schimbam strategia (ex. chunking) fara re-evaluare completa.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from app.config import (
    DEVICE,
    LABEL_NAMES,
    MODEL_BASELINE_DIR,
    XLMR_BATCH_SIZE,
    XLMR_MAX_LENGTH,
)


class ClasificatorModul2:
    """
    Wrapper pentru XLM-RoBERTa baseline v2 (clasificare globala).

    Modelul e incarcat o singura data la startup. Inferenta e thread-safe
    (nu modificam state intern), deci poate fi apelat din endpoint-uri
    sync FastAPI fara locking suplimentar.
    """

    def __init__(self, model_dir: Path = MODEL_BASELINE_DIR):
        """
        Args:
            model_dir: Path catre folder-ul `final/` al modelului baseline v2.
        """
        self.model_dir = Path(model_dir)
        self.device = DEVICE
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._model: Optional[PreTrainedModel] = None
        # Identificator versiune model — folosit in /health
        self.model_version: Optional[str] = None

    def initializeaza(self) -> None:
        """Incarca tokenizer + model pe device. Idempotent."""
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Folder model XLM-R baseline lipsă: {self.model_dir}. "
                f"Asigură-te că ai antrenat modelul (vezi 02_train_xlmr_baseline_v2.py)."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        ).to(self.device)
        self._model.eval()
        # Folosim numele directorului ca identificator versiune
        # (ex: "xlmr_baseline_v2" pentru models/xlmr_baseline_v2/final/)
        self.model_version = self.model_dir.parent.name

    @property
    def este_initializat(self) -> bool:
        """True daca modelul e incarcat si gata de inferenta."""
        return self._model is not None

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Acces tokenizer (folosit de modulul 4 LIME pentru tokenizare consistenta)."""
        if self._tokenizer is None:
            raise RuntimeError("Modul 2 neinițializat.")
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """Acces model (folosit intern de LIME prin predict_proba_batch)."""
        if self._model is None:
            raise RuntimeError("Modul 2 neinițializat.")
        return self._model

    # ─────────────────────────────────────────────────────────────────────
    # Inferenta pe text unic
    # ─────────────────────────────────────────────────────────────────────
    def predict_text_unic(self, text: str) -> dict:
        """
        Ruleaza clasificarea pe un singur text.

        Args:
            text: Articolul de clasificat (string).

        Returns:
            Dict cu:
              - prob_cls0, prob_cls1: probabilitati post-softmax
              - label_pred: 0 sau 1 (argmax)
              - label_name: 'stire_credibila' sau 'dezinformare_pro_rusa'
              - input_truncat: True daca textul a fost taiat la max_length
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Modul 2 neinițializat. Apelează initializeaza().")

        # Verificam daca textul ar fi trunchiat (pentru telemetry UI)
        # Tokenizam fara truncation ca sa comparam cu max_length real
        n_tokens_raw = len(self._tokenizer.encode(text, add_special_tokens=True))
        input_truncat = n_tokens_raw > XLMR_MAX_LENGTH

        # Inferenta cu truncation (consistent cu antrenarea)
        with torch.no_grad():
            enc = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=XLMR_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            logits = self._model(**enc).logits  # shape (1, 2)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (2,)

        prob_cls0 = float(probs[0])
        prob_cls1 = float(probs[1])
        label_pred = int(np.argmax(probs))

        return {
            "prob_cls0": prob_cls0,
            "prob_cls1": prob_cls1,
            "label_pred": label_pred,
            "label_name": LABEL_NAMES[label_pred],
            "input_truncat": input_truncat,
            "n_tokens_raw": n_tokens_raw,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Inferenta batch — folosita de LIME (predict_proba)
    # ─────────────────────────────────────────────────────────────────────
    def predict_proba_batch(self, texts: list[str]) -> np.ndarray:
        """
        Inferenta batch pentru LIME.

        EXTRAS DIRECT din `06_lime_xlmr_v2.py::predict_proba()` (linii 75-85).
        Aceeasi semantica: input lista texte, output matrice (N, 2) cu
        probabilitati dupa softmax. Asta e contractul pe care il asteapta
        LimeTextExplainer.

        Args:
            texts: Lista de texte (LIME genereaza ~num_samples=1000 perturbari).

        Returns:
            Matrice numpy (N, 2) cu probabilitati [prob_cls0, prob_cls1].
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Modul 2 neinițializat.")

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), XLMR_BATCH_SIZE):
                batch = texts[i:i + XLMR_BATCH_SIZE]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=XLMR_MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)
