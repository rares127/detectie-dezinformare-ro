"""
Modul 3 — Similaritate semantica per propozitie.

Implementare 1:1 cu pipeline-ul din `calibrare_threshold_v2.py`:
- Encoder: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- Embeddings normalizate L2 (cosine = dot product)
- Pentru fiecare propozitie: scor = max(cosine cu toate propozitiile din corpus)
- Agregare la articol: mean(scor_cls0), mean(scor_cls1)
- Decizie: diff_mean = mean(cls1) − mean(cls0); pred=1 daca > THRESHOLD_MODUL3

Threshold productie: -0.0073 (calibrat CV 5-fold pe test set, F1=0.9454).
NU se modifica fara re-calibrare.

Optimizare cache:
- Embeddings corpusurilor (cls0 + cls1) sunt precalculate si salvate ca .npy
- La startup, incercam sa incarcam din cache; fallback: reproducem din parquet
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import (
    CORPUS_CLS0_PATH,
    CORPUS_CLS1_PATH,
    DEVICE,
    DOWNSAMPLE_CLS1_LA,
    EMBEDDINGS_CACHE_DIR,
    SBERT_BATCH_SIZE,
    SBERT_MODEL_NAME,
    SEED,
    THRESHOLD_MODUL3,
    TOP_K_PROPOZITII_UI,
)


logger = logging.getLogger("app.modul3")


def _hash_corpus(texts: list[str], model_name: str) -> str:
    """
    Hash determinist pentru cache embeddings.

    Identic cu `calibrare_threshold_v2.py::calculeaza_hash_corpus()`.
    Returneaza primii 16 caractere din SHA256 — folosit ca sufix in .npy.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\n")
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


class ScorerModul3:
    """
    Calculeaza scorul combinat (diff_mean) si top propozitii similare.

    Workflow startup:
    1. Incarca encoder mpnet (~1.1GB)
    2. Incarca corpus cls0 + cls1 (parquet)
    3. Downsample cls1 la 5290 propozitii (paritate cu cls0, seed=42)
    4. Incearca sa incarce embeddings din cache .npy; daca lipsesc, calculeaza
       si salveaza cache.

    Workflow inferenta (per request):
    1. Primeste lista propozitii valide (deja filtrate [7, 54] cuvinte)
    2. Encode propozitiile (pe device-ul disponibil)
    3. Calculeaza cosine similarity vs corpus cls0 si cls1
    4. Returneaza diff_mean + top-K propozitii similare
    """

    def __init__(self):
        """Initializare lazy — modelul se incarca in initializeaza()."""
        self.device = DEVICE
        self._encoder: Optional[SentenceTransformer] = None
        # Embeddings precalculate (n_corpus, 768)
        self._emb_cls0: Optional[np.ndarray] = None
        self._emb_cls1: Optional[np.ndarray] = None
        # Texte propozitii corpus (pentru afisare in UI)
        self._texte_cls0: Optional[list[str]] = None
        self._texte_cls1: Optional[list[str]] = None
        self._surse_cls0: Optional[list[str]] = None
        self._surse_cls1: Optional[list[str]] = None

    # ─────────────────────────────────────────────────────────────────────
    # Incarcare corpus + embeddings (cu cache)
    # ─────────────────────────────────────────────────────────────────────
    def _incarca_corpus_cls0(self) -> tuple[list[str], list[str]]:
        """Incarca corpus cls0 din parquet. Returneaza (texte, surse)."""
        if not CORPUS_CLS0_PATH.exists():
            raise FileNotFoundError(
                f"Corpus cls0 lipsă: {CORPUS_CLS0_PATH}. "
                f"Vezi pipeline preprocessing modul 3."
            )
        df = pd.read_parquet(CORPUS_CLS0_PATH)
        return df["propozitie"].tolist(), df["sursa_site"].tolist()

    def _incarca_corpus_cls1(self) -> tuple[list[str], list[str]]:
        """
        Incarca corpus cls1 din parquet si aplica downsample la 5290.

        Replica exacta a logicii din calibrare_threshold_v2.py — folosim
        pandas.sample(n=5290, random_state=42) pentru paritate cu cls0.
        """
        if not CORPUS_CLS1_PATH.exists():
            raise FileNotFoundError(
                f"Corpus cls1 lipsă: {CORPUS_CLS1_PATH}. "
                f"Vezi pipeline preprocessing modul 3."
            )
        df = pd.read_parquet(CORPUS_CLS1_PATH)
        # Downsample seed=42 — IDENTIC cu calibrarea threshold-ului
        if len(df) > DOWNSAMPLE_CLS1_LA:
            df = df.sample(n=DOWNSAMPLE_CLS1_LA, random_state=SEED).reset_index(drop=True)
        return df["propozitie"].tolist(), df["sursa_site"].tolist()

    def _incarca_sau_calculeaza_embeddings(
        self, texts: list[str], nume: str
    ) -> np.ndarray:
        """
        Incarca embeddings din cache .npy sau le calculeaza daca lipsesc.

        Replica logica din `calibrare_threshold_v2.py::incarca_sau_calculeaza_embeddings()`.
        Asta inseamna: daca cache-ul exista din rularea precedenta a calibrarii,
        il reutilizam direct (zero cost startup).
        """
        EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        h = _hash_corpus(texts, SBERT_MODEL_NAME)
        cache_path = EMBEDDINGS_CACHE_DIR / f"{nume}_{h}.npy"

        if cache_path.exists():
            emb = np.load(cache_path)
            if emb.shape[0] == len(texts):
                logger.info("Cache HIT: %s %s", cache_path.name, emb.shape)
                return emb
            logger.warning("Cache CORUPT pentru %s, recalculez...", cache_path.name)

        logger.info("Cache MISS: calculez %s embeddings pe %s...",
                    f"{len(texts):,}", self.device)
        if self._encoder is None:
            raise RuntimeError("Encoder mpnet neinițializat.")
        emb = self._encoder.encode(
            texts,
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # esential: cosine = dot product
            device=self.device,
        )
        np.save(cache_path, emb)
        logger.info("Cache SAVE: %s", cache_path.name)
        return emb

    def initializeaza(self) -> None:
        """
        Incarca encoder + corpus + embeddings. Idempotent.

        Ordine:
        1. Incarca encoder mpnet (consuma ~1.1GB pe device)
        2. Incarca corpus cls0 (parquet)
        3. Incarca corpus cls1 (parquet) + downsample la 5290
        4. Cache hit/miss embeddings cls0
        5. Cache hit/miss embeddings cls1
        """
        if self._emb_cls0 is not None and self._emb_cls1 is not None:
            return

        # 1. Encoder
        logger.info("Încarc encoder: %s", SBERT_MODEL_NAME)
        self._encoder = SentenceTransformer(SBERT_MODEL_NAME, device=self.device)

        # 2-3. Corpusuri
        logger.info("Încarc corpus cls0 din %s", CORPUS_CLS0_PATH.name)
        self._texte_cls0, self._surse_cls0 = self._incarca_corpus_cls0()
        logger.info("Corpus cls0: %s propoziții", f"{len(self._texte_cls0):,}")

        logger.info("Încarc corpus cls1 din %s", CORPUS_CLS1_PATH.name)
        self._texte_cls1, self._surse_cls1 = self._incarca_corpus_cls1()
        logger.info("Corpus cls1 (post-downsample): %s propoziții",
                    f"{len(self._texte_cls1):,}")

        # 4-5. Embeddings (cu cache fallback)
        self._emb_cls0 = self._incarca_sau_calculeaza_embeddings(
            self._texte_cls0, "cls0_corpus"
        )
        self._emb_cls1 = self._incarca_sau_calculeaza_embeddings(
            self._texte_cls1, "cls1_corpus_v2_downsampled"
        )

    @property
    def este_initializat(self) -> bool:
        return self._emb_cls0 is not None and self._emb_cls1 is not None

    # ─────────────────────────────────────────────────────────────────────
    # Inferenta
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _scor_cosine_max(
        emb_query: np.ndarray, emb_corpus: np.ndarray, batch: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pentru fiecare propozitie query, calculeaza:
          - scor_max: max(cosine cu fiecare propozitie din corpus)
          - idx_max: indexul propozitiei din corpus cu cea mai mare similaritate

        Logica e extrasa din `calibrare_threshold_v2.py::scor_cosine_max()`,
        cu modificarea ca returnam si indicii (necesari pentru afisarea
        propozitiilor de referinta in UI).

        Args:
            emb_query: (n_query, 768) — embeddings propozitii input
            emb_corpus: (n_corpus, 768) — embeddings precalculate corpus
            batch: dimensiune batch pentru a nu satura memoria

        Returns:
            (scoruri_max, indici_max), ambele de shape (n_query,)
        """
        n_query = emb_query.shape[0]
        scoruri = np.zeros(n_query, dtype=np.float32)
        indici = np.zeros(n_query, dtype=np.int64)
        for i in range(0, n_query, batch):
            b = emb_query[i:i + batch]
            sim_matrix = b @ emb_corpus.T  # (batch, n_corpus)
            scoruri[i:i + batch] = sim_matrix.max(axis=1)
            indici[i:i + batch] = sim_matrix.argmax(axis=1)
        return scoruri, indici

    def calculeaza_scor(self, propozitii_valide: list[str]) -> dict:
        """
        Calculeaza scorul combinat si top propozitii pentru un articol.

        Args:
            propozitii_valide: Lista de propozitii care au trecut filtrul
                de lungime [7, 54] cuvinte.

        Returns:
            Dict cu:
              - scor_cls0_mean, scor_cls1_mean, diff_mean: scorurile agregate
              - decizie_pred: 1 daca diff_mean > THRESHOLD_MODUL3, altfel 0
              - propozitii_top_cls0: top-K propozitii (input + match corpus + scor)
              - propozitii_top_cls1: top-K propozitii (input + match corpus + scor)

        Raises:
            ValueError: daca lista e goala (caller trebuie sa gestioneze
                edge case-ul "incert" — vezi route /predict).
        """
        if not self.este_initializat:
            raise RuntimeError("Modul 3 neinițializat.")
        if not propozitii_valide:
            raise ValueError(
                "Listă de propoziții goală — modul 3 nu poate calcula scor. "
                "Caller-ul trebuie să returneze decizie_incerta=True."
            )

        # 1. Encode propozitiile articolului (cu normalize)
        emb_query = self._encoder.encode(
            propozitii_valide,
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )

        # 2. Scoruri max + indici match per propozitie
        scoruri_cls0, idx_cls0 = self._scor_cosine_max(emb_query, self._emb_cls0)
        scoruri_cls1, idx_cls1 = self._scor_cosine_max(emb_query, self._emb_cls1)

        # 3. Agregare la nivel articol (mean) — IDENTIC cu agrega_la_articol()
        scor_cls0_mean = float(scoruri_cls0.mean())
        scor_cls1_mean = float(scoruri_cls1.mean())
        diff_mean = scor_cls1_mean - scor_cls0_mean

        # 4. Decizie binara pe baza threshold-ului calibrat
        decizie_pred = 1 if diff_mean > THRESHOLD_MODUL3 else 0

        # 5. Top-K propozitii similare cu corpus cls0 (sortate descrescator)
        top_idx_cls0_query = np.argsort(-scoruri_cls0)[:TOP_K_PROPOZITII_UI]
        propozitii_top_cls0 = [
            {
                "propozitie_input": propozitii_valide[i],
                "propozitie_referinta": self._texte_cls0[idx_cls0[i]],
                "sursa": self._surse_cls0[idx_cls0[i]],
                # Cosine pe embeddings normalizate ∈ [-1, 1] → afisam ca atare
                "scor_similaritate": float(scoruri_cls0[i]),
            }
            for i in top_idx_cls0_query
        ]

        # 6. Top-K propozitii similare cu corpus cls1
        top_idx_cls1_query = np.argsort(-scoruri_cls1)[:TOP_K_PROPOZITII_UI]
        propozitii_top_cls1 = [
            {
                "propozitie_input": propozitii_valide[i],
                "propozitie_referinta": self._texte_cls1[idx_cls1[i]],
                "sursa": self._surse_cls1[idx_cls1[i]],
                "scor_similaritate": float(scoruri_cls1[i]),
            }
            for i in top_idx_cls1_query
        ]

        # 7. Detalii per propozitie — pentru vizualizarea colorata din UI
        #    Ordinea respecta ordinea originala a propozitiilor din articol.
        propozitii_detalii = [
            {
                "text": propozitii_valide[i],
                "diff": float(scoruri_cls1[i] - scoruri_cls0[i]),
                "scor_cls0": float(scoruri_cls0[i]),
                "scor_cls1": float(scoruri_cls1[i]),
                "match_cls0_text": self._texte_cls0[idx_cls0[i]],
                "match_cls0_sursa": self._surse_cls0[idx_cls0[i]],
                "match_cls0_sim": float(scoruri_cls0[i]),
                "match_cls1_text": self._texte_cls1[idx_cls1[i]],
                "match_cls1_sursa": self._surse_cls1[idx_cls1[i]],
                "match_cls1_sim": float(scoruri_cls1[i]),
            }
            for i in range(len(propozitii_valide))
        ]

        return {
            "scor_cls0_mean": scor_cls0_mean,
            "scor_cls1_mean": scor_cls1_mean,
            "diff_mean": diff_mean,
            "decizie_pred": decizie_pred,
            "threshold": THRESHOLD_MODUL3,
            "propozitii_top_cls0": propozitii_top_cls0,
            "propozitii_top_cls1": propozitii_top_cls1,
            "propozitii_detalii": propozitii_detalii,
        }
