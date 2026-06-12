"""
Configuratie centralizata pentru API-ul de detectie a dezinformarii.

Toate constantele sunt extrase din scripturile de antrenare si calibrare
din modulele 2-4. NU modifica fara re-calibrare completa a threshold-ului.

Referinte:
- Threshold: calibrare_threshold_v2.py (CV 5-fold pe test set, F1=0.9454)
- Encoder: paraphrase-multilingual-mpnet-base-v2 (ales dupa benchmark v3)
- Filtru lungime: consistent cu corpus cls0 + cls1 (audit propozitii)
- max_length=256: identic cu antrenarea XLM-R baseline v2
"""

from pathlib import Path

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Identitate aplicatie
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE = "Detector Dezinformare Pro-Rusă"
# Versiunea apare si ca query param pe fisierele statice (cache-busting):
# orice modificare in app.js/custom.css trebuie insotita de bump aici,
# altfel browserele pot servi versiunea veche din cache.
APP_VERSION = "1.1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Path-uri relative la radacina proiectului (de unde se ruleaza uvicorn)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

# Modele (artefacte produse de modulele 2 si 3)
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_BASELINE_DIR = MODELS_DIR / "xlmr_baseline_v2" / "final"

# Date si cache (pentru modul 3)
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_CLS0_PATH = PROCESSED_DIR / "propozitii_cls0_corpus.parquet"
CORPUS_CLS1_PATH = PROCESSED_DIR / "propozitii_cls1_corpus_v2.parquet"
EMBEDDINGS_CACHE_DIR = PROCESSED_DIR / "embeddings_cache"


# ─────────────────────────────────────────────────────────────────────────────
# Modul 2 — XLM-RoBERTa baseline (clasificare globala)
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: max_length=256 e consistent cu antrenarea (02_train_xlmr_baseline_v2.py).
# Schimbarea ar invalida statusul modelului.
XLMR_MAX_LENGTH = 256
XLMR_BATCH_SIZE = 16  # pentru tokenizare LIME (multi-sample inference)

LABEL_NAMES = {0: "stire_credibila", 1: "dezinformare_pro_rusa"}
LABEL_DISPLAY = {
    0: "Știre credibilă",
    1: "Dezinformare pro-rusă",
}


# ─────────────────────────────────────────────────────────────────────────────
# Modul 3 — Similaritate semantica (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SBERT_BATCH_SIZE = 32  # consistent cu calibrare_threshold_v2.py

# Filtru lungime propozitie — IDENTIC cu corpus train (audit_corpus_cls0)
MIN_CUVINTE = 7
MAX_CUVINTE = 54

# Downsample cls1 la paritate cu cls0 (5290 propozitii, seed=42)
# Conform benchmark_v4_post_curatare.py — corpusul productie.
DOWNSAMPLE_CLS1_LA = 5290

# Threshold productie — CALIBRAT, NU se modifica fara re-calibrare CV.
# Sursa: calibrare_threshold_v2.py, sectiunea CV 5-fold, mean across folds.
THRESHOLD_MODUL3 = -0.0073

# Marja borderline — articole cu |diff_mean − threshold| < aceasta valoare
# sau cu dezacord modul2 vs modul3 sunt afisate cu verdict de incredere redusa.
# Afecteaza DOAR UI-ul; decizia tehnica (cls0/cls1) ramane neschimbata.
# Cohen's d = +2.41 intre distributii → risc minim de over-flagging cu 0.003.
BORDERLINE_MARGIN = 0.003

# Praguri dezacord inter-modular (AMBELE directii):
# - M3=cls1 dar XLM-R sub 20% cls1 → vocabular tematic in cadru jurnalistic
# - M3=cls0 dar XLM-R peste 80% cls1 → pattern „reported speech trap" (Test 4,
#   citate Putin verbatim) — singurul mod de eroare in care sistemul poate
#   valida propaganda reala; trebuie semnalat ca verdict incert.
PRAG_DEZACORD_M2_JOS = 0.20
PRAG_DEZACORD_M2_SUS = 0.80

# Sub acest numar de propozitii valide, diff_mean e statistic fragil
# (metricile F1=0.9454 sunt masurate pe articole cu distributie normala
# de propozitii) → verdictul se afiseaza ca borderline.
MIN_PROPOZITII_FIABIL = 3

# Plafon lungime input (caractere). Peste orice articol real (~10-15k),
# sub orice abuz care ar tine Stanza + encoderul ocupate minute intregi.
MAX_INPUT_CARACTERE = 50_000

# Top-K propozitii similare afisate in UI per corpus
TOP_K_PROPOZITII_UI = 3


# ─────────────────────────────────────────────────────────────────────────────
# Modul 4 — LIME (lazy, doar pe cls0)
# ─────────────────────────────────────────────────────────────────────────────
# Configuratie IDENTICA cu 06_lime_xlmr_v2.py si 07_lime_l1a_diagnostic.py
# pentru consistenta cu cifrele din capitolul XAI al tezei.
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 1000
LIME_BOW = False  # bow=False pentru transformere (pastreaza ordinea token-urilor)


# ─────────────────────────────────────────────────────────────────────────────
# Reproductibilitate
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Detectie device (MPS pe macOS M2, CUDA daca exista, altfel CPU)
# ─────────────────────────────────────────────────────────────────────────────
def detect_device() -> str:
    """
    Detecteaza automat device-ul disponibil.

    Ordine preferinta: MPS (Apple Silicon) > CUDA > CPU.
    Consistent cu scripturile de antrenare (02, 03, 06).
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = detect_device()


# ─────────────────────────────────────────────────────────────────────────────
# Stanza — segmentare propozitii (CPU, consistent cu calibrare_threshold_v2.py)
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: use_gpu=False pastrat pentru consistenta cu pipeline-ul de
# antrenare. Stanza pe MPS are uneori probleme cu modele romanesti.
STANZA_LANG = "ro"
STANZA_PROCESSORS = "tokenize"
STANZA_USE_GPU = False
