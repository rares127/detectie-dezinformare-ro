"""
clean_veridica_v5.py — Re-cleaning Veridica v5 pentru simetrie cu G4M/D24

Parte din proiectul de licenta „Sistem de Detectie Automata si Explicabila a
Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din
Ucraina".

Input:  data/processed/veridica_clean_v4.csv  (510 articole, deja curatate v4)
Output: data/processed/veridica_clean_v5.csv

CONTEXT METODOLOGIC:
    La validarea adversariala intermediara, Test 1 (LogReg pe lungime bruta)
    a obtinut 65.4% acuratete — peste pragul de 62%. Cauza: asimetrie de
    truncate intre clase:
        - G4Media v2 + Digi24 v1: median text_curat = 250 cw (truncate sistematic)
        - Veridica v4: median text_curat = 196 cw (fara truncate)
    Veridica avea 22.7% articole > 250 cw (pana la 1034 cw), creand o
    distributie bimodala distincta fata de clasa 0.

    Solutia aleasa (Directia 3 revizuita): aplicam pe Veridica EXACT aceleasi
    reguli de normalizare aplicate la G4M/D24, dar FARA a re-scrape sau
    modifica `stire_citata` (care e deja un camp pur propagandistic, fara
    leakage de vocabular fact-checker — verificat empiric pe sample-uri).

REGULI APLICATE (in ordine):
    1. Drop articole fara data parsabila (3 articole)
    2. Drop ianuarie 2026 (consistent G4M/D24, decizie consolidata in recap)
    3. Drop too_short < 64 cw pe text_curat (consistent G4M/D24)
       — coincide cu cele 8 articole care nu aveau stire_citata
    4. Truncate text_curat la 250 cw (consistent G4M/D24)
    5. Recalculare hash_continut pe text_curat truncat (pentru dedup la merge)

NIMIC altceva nu se modifica:
    - stire_citata, naratiuni_false, obiective_propaganda raman ca in v4
    - text_curat_v3 (corpul cu prefix categorial) raman ca in v4
    - label, label_numeric raman (clasa 1)

OBIECTIV: aducem median text_curat de la 196 → ~220-230 (apropiat de
G4M/D24 224-242), reducand shortcut-ul Test 1 de la 65% spre 58-60%.
Test 4 (char n-grams) va ramane la ~80-85% — semnal stilistic real, nu
artefact, consistent cu obiectivul lucrarii de a detecta si stilul retoric.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuratie
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path.cwd()
INPUT_CSV = ROOT / "data" / "processed" / "veridica_clean_v4.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "veridica_clean_v5.csv"
LOG_FILE = ROOT / "data" / "raw" / "clean_veridica_v5.log"

# Praguri — strict aliniate cu G4M/D24
WORDS_TOO_SHORT = 64
WORDS_TRUNCATE = 250

YEARMONTH_DROP = "2026-01"


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clean_veridica_v5")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Functii de cleaning
# ─────────────────────────────────────────────────────────────────────────────


def truncate_words(text: str, max_words: int) -> str:
    """
    Trunchiaza text la max_words cuvinte. Cuvant = secventa non-whitespace.

    Identic cu functia din clean_digi24_v1.py si clean_g4media_v2.py
    (consistenta stricta inter-sursa).
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def count_words(text: str) -> int:
    """Numara cuvintele intr-un text."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def compute_hash(text: str) -> str:
    """SHA1 pe text — pentru dedup post-clean si verificare unicitate la merge."""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────


def run_cleaning(logger: logging.Logger) -> None:
    """
    Pipeline de re-cleaning Veridica v4 → v5.

    Toate dropurile sunt logate cu count + procentaj din total initial,
    consistent cu clean_digi24_v1.py si clean_g4media_v2.py.
    """
    if not INPUT_CSV.exists():
        logger.error("Input CSV inexistent: %s", INPUT_CSV)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("RE-CLEANING VERIDICA V4 → V5 — START")
    logger.info("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)
    logger.info("Citit input v4: %d articole", n0)

    # Parsare data
    df["data_pub"] = pd.to_datetime(df["data"], errors="coerce")
    df["an"] = df["data_pub"].dt.year
    df["luna"] = df["data_pub"].dt.to_period("M").astype(str)

    # ─── REGULA 1: Drop articole fara data parsabila ────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["data_pub"]).copy()
    logger.info(
        "Regula 1 — Drop fără dată parsabilă: -%d (rămas: %d)",
        n_before - len(df),
        len(df),
    )

    # ─── REGULA 2: Drop ianuarie 2026 ───────────────────────────────────────
    n_before = len(df)
    df = df[df["luna"] != YEARMONTH_DROP].copy()
    logger.info(
        "Regula 2 — Drop %s: -%d (rămas: %d)",
        YEARMONTH_DROP,
        n_before - len(df),
        len(df),
    )

    # ─── REGULA 3: Drop too_short < 64 cw (pe text_curat v4) ───────────────
    n_before = len(df)
    df["nr_cuvinte_v4"] = df["text_curat"].apply(count_words)
    df = df[df["nr_cuvinte_v4"] >= WORDS_TOO_SHORT].copy()
    logger.info(
        "Regula 3 — Drop too_short < %d cw: -%d (rămas: %d)",
        WORDS_TOO_SHORT,
        n_before - len(df),
        len(df),
    )

    # ─── REGULA 4: Truncate text_curat la 250 cw ────────────────────────────
    df["text_curat"] = df["text_curat"].apply(
        lambda t: truncate_words(t, WORDS_TRUNCATE)
    )
    df["nr_cuvinte_truncat"] = df["text_curat"].apply(count_words)
    n_truncated = (df["nr_cuvinte_v4"] > WORDS_TRUNCATE).sum()
    logger.info(
        "Regula 4 — Truncate la %d cw: %d articole truncate (%.1f%%)",
        WORDS_TRUNCATE,
        n_truncated,
        100 * n_truncated / len(df),
    )
    logger.info(
        "  median nr_cuvinte_truncat: %d (era %d în v4)",
        int(df["nr_cuvinte_truncat"].median()),
        int(df["nr_cuvinte_v4"].median()),
    )

    # ─── REGULA 5: Recalculare hash_continut pe text_curat post-truncate ────
    df["hash_continut"] = df["text_curat"].apply(compute_hash)

    # Verificare dedup
    n_before = len(df)
    df = df.drop_duplicates(subset=["hash_continut"], keep="first")
    if n_before - len(df) > 0:
        logger.info(
            "Dedup post-clean pe hash_continut nou: -%d (rămas: %d)",
            n_before - len(df),
            len(df),
        )

    # ─── Schema output (pastram toate coloanele relevante din v4) ───────────
    output_cols = [
        "id",
        "url",
        "titlu",
        "data",
        "an",
        "luna",
        "sursa_site",
        "sectiune",
        "text_curat",          # post-truncate
        "stire_citata",        # neschimbat
        "naratiuni_false",     # neschimbat
        "obiective_propaganda",  # neschimbat
        "nr_cuvinte_v4",       # lungimea originala
        "nr_cuvinte_truncat",  # lungimea finala
        "calitate_extractie",
        "label",
        "label_numeric",
        "hash_continut",
    ]
    # Doar coloanele care exista
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    # ─── Audit final ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("RE-CLEANING TERMINAT")
    logger.info("=" * 70)
    logger.info("Input v4:  %d articole", n0)
    logger.info("Output v5: %d articole (%.1f%% retenție)", len(df_out), 100 * len(df_out) / n0)
    logger.info("Salvat: %s", OUTPUT_CSV)
    logger.info("")
    logger.info("=== Distribuție pe an (post-clean) ===")
    for an, n in df_out["an"].value_counts().sort_index().items():
        logger.info("  %d: %d", int(an), n)
    logger.info("")
    logger.info("=== Stats nr_cuvinte_truncat ===")
    stats = df_out["nr_cuvinte_truncat"].describe()
    logger.info(
        "  median=%.0f mean=%.0f min=%d max=%d",
        stats["50%"],
        stats["mean"],
        int(stats["min"]),
        int(stats["max"]),
    )
    logger.info("  pct la 250 cw (truncate): %d/%d (%.1f%%)",
                int((df_out["nr_cuvinte_truncat"] == 250).sum()),
                len(df_out),
                100 * (df_out["nr_cuvinte_truncat"] == 250).sum() / len(df_out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-cleaning Veridica v4 → v5 pentru simetrie G4M/D24."
    )
    parser.parse_args()
    logger = setup_logging()
    run_cleaning(logger)


if __name__ == "__main__":
    main()
