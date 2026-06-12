"""
clean_veridica_2022.py
──────────────────────
Cleaning pentru veridica_2022_raw.csv — articolele din 2022 colectate
cu scraper_veridica_2022.py (paginile 21–28 din Veridica).

Parte din proiectul de licenta „Sistem de Detectie Automata si Explicabila
a Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din
Ucraina".

REGULI APLICATE (aliniate cu clean_veridica_v5.py pentru simetrie):

    Regula 1 — Drop articole din 2023
        Paginile 21–22 contineau un mic overlap cu ianuarie–februarie 2023.
        Aceste articole exista deja in veridica_clean_v5.csv (colectate
        anterior). Le eliminam pentru a evita duplicate la merge.

    Regula 2 — Drop fallback_verificare_manuala
        Structura HTML nestandard — scraper-ul nu a extras STIRE/NARATIUNI.
        text_curat nu e utilizabil ca input pentru clasificator.

    Regula 3 — Drop suspect_contaminare
        stire_citata contine markeri de fact-check — leak confirmat.
        text_curat e gol pentru aceste articole (protectie din scraper).

    Regula 4 — Promovare suspect_dimensiune → excelenta
        XLM-RoBERTa trunchiaza la 512 tokens — lungimea nu e problema.
        (Consistent cu clean_veridica_v4_2.py)

    Regula 5 — Drop nerelevante tematic
        relevanta_ucraina == False → articole despre alte subiecte de
        dezinformare (oculta, ecodictatura etc.), nu pro-ruse.

    Regula 6 — Drop too_short < 64 cuvinte pe text_curat
        Consistent cu clean_veridica_v5.py si clean_g4media_v2.py.

    Regula 7 — Truncate text_curat la 250 cuvinte
        Simetrie cu clasa 0 (G4Media/Digi24) si cu veridica_clean_v5.csv.

    Regula 8 — Deduplicare pe hash_continut (recalculat post-truncate)

    Regula 9 — Selectie coloane identice cu veridica_clean_v5.csv
        Permite merge direct fara redenumiri.

OUTPUT:
    veridica_2022_clean.csv    → dataset curat, gata pentru merge
    veridica_2022_clean_report.txt → raport cu statistici

PASUL URMATOR:
    python merge_veridica_final.py
    (concateneaza veridica_clean_v5.csv + veridica_2022_clean.csv)
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Configurare ────────────────────────────────────────────────────────────────

INPUT_CSV   = "veridica_2022_raw.csv"
OUTPUT_CSV  = "veridica_2022_clean.csv"
OUTPUT_RPT  = "veridica_2022_clean_report.txt"

# Aliniate strict cu clean_veridica_v5.py
WORDS_TOO_SHORT = 64
WORDS_TRUNCATE  = 250
AN_DROP         = 2023   # eliminam overlap-ul cu 2023


# ── Functii helper ─────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Numara cuvintele unui text."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def truncate_words(text: str, max_words: int) -> str:
    """
    Trunchiaza text la max_words cuvinte.
    Identic cu functia din clean_veridica_v5.py si clean_g4media_v2.py
    (consistenta stricta inter-sursa).
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def compute_hash(text: str) -> str:
    """SHA1 pe text — consistent cu clean_veridica_v5.py."""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def setup_logging() -> logging.Logger:
    """Configurare logging catre consola si fisier."""
    logger = logging.getLogger("clean_veridica_2022")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(OUTPUT_RPT, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("CLEANING VERIDICA 2022 RAW → CLEAN")
    logger.info("=" * 70)

    if not Path(INPUT_CSV).exists():
        logger.error("Input CSV inexistent: %s", INPUT_CSV)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)
    logger.info("Citit input: %d articole din %s", n0, INPUT_CSV)

    # Parsare data
    df["data_pub"] = pd.to_datetime(df["data"], errors="coerce")
    df["an"]       = df["data_pub"].dt.year

    # ── Regula 1: Drop 2023 ────────────────────────────────────────────────────
    n_before = len(df)
    df = df[df["an"] != AN_DROP].copy()
    logger.info(
        "Regula 1 — Drop an=%d (overlap cu v5): -%d (rămas: %d)",
        AN_DROP, n_before - len(df), len(df),
    )

    # ── Regula 2: Drop fallback_verificare_manuala ─────────────────────────────
    n_before = len(df)
    df = df[df["calitate_extractie"] != "fallback_verificare_manuala"].copy()
    logger.info(
        "Regula 2 — Drop fallback_verificare_manuala: -%d (rămas: %d)",
        n_before - len(df), len(df),
    )

    # ── Regula 3: Drop suspect_contaminare ─────────────────────────────────────
    n_before = len(df)
    df = df[df["calitate_extractie"] != "suspect_contaminare"].copy()
    logger.info(
        "Regula 3 — Drop suspect_contaminare: -%d (rămas: %d)",
        n_before - len(df), len(df),
    )

    # ── Regula 4: Promovare suspect_dimensiune → excelenta ─────────────────────
    mask_dim = df["calitate_extractie"] == "suspect_dimensiune"
    n_dim    = mask_dim.sum()
    if n_dim > 0:
        df.loc[mask_dim, "calitate_extractie"] = "excelenta"
        logger.info(
            "Regula 4 — Promovare suspect_dimensiune → excelenta: %d", n_dim
        )

    # ── Regula 5: Drop nerelevante tematic ────────────────────────────────────
    n_before = len(df)
    df = df[df["relevanta_ucraina"] == True].copy()
    logger.info(
        "Regula 5 — Drop nerelevante tematic (relevanta_ucraina=False): -%d (rămas: %d)",
        n_before - len(df), len(df),
    )

    # ── Regula 6: Drop too_short < 64 cuvinte ─────────────────────────────────
    n_before     = len(df)
    df["_wc_raw"] = df["text_curat"].apply(count_words)
    df = df[df["_wc_raw"] >= WORDS_TOO_SHORT].copy()
    logger.info(
        "Regula 6 — Drop too_short < %d cw: -%d (rămas: %d)",
        WORDS_TOO_SHORT, n_before - len(df), len(df),
    )

    # ── Regula 7: Truncate text_curat la 250 cuvinte ──────────────────────────
    df["text_curat"]  = df["text_curat"].apply(
        lambda t: truncate_words(t, WORDS_TRUNCATE)
    )
    df["_wc_truncat"] = df["text_curat"].apply(count_words)
    n_truncated       = (df["_wc_raw"] > WORDS_TRUNCATE).sum()
    logger.info(
        "Regula 7 — Truncate la %d cw: %d articole truncate (%.1f%%)",
        WORDS_TRUNCATE, n_truncated, 100 * n_truncated / len(df),
    )
    logger.info(
        "  median nr_cuvinte: %d → %d (post-truncate)",
        int(df["_wc_raw"].median()),
        int(df["_wc_truncat"].median()),
    )

    # ── Regula 8: Recalculare hash + deduplicare ──────────────────────────────
    df["hash_continut"] = df["text_curat"].apply(compute_hash)
    n_before            = len(df)
    df = df.drop_duplicates(subset=["hash_continut"], keep="first")
    if n_before - len(df) > 0:
        logger.info(
            "Regula 8 — Dedup post-truncate: -%d (rămas: %d)",
            n_before - len(df), len(df),
        )

    # ── Regula 9: Coloane identice cu veridica_clean_v5.csv ───────────────────
    # Adaugam coloanele derivate necesare pentru simetrie la merge
    df["luna"]              = df["data_pub"].dt.to_period("M").astype(str)
    df["nr_cuvinte_v4"]     = df["_wc_raw"]       # lungimea originala
    df["nr_cuvinte_truncat"] = df["_wc_truncat"]  # lungimea finala

    OUTPUT_COLS = [
        "id", "url", "titlu", "data", "an", "luna",
        "sursa_site", "sectiune",
        "text_curat",
        "stire_citata", "naratiuni_false", "obiective_propaganda",
        "nr_cuvinte_v4", "nr_cuvinte_truncat",
        "calitate_extractie",
        "label", "label_numeric",
        "hash_continut",
    ]

    # Generam ID stabil (prefix vrd22_ pentru a distinge de v5 la audit)
    df = df.sort_values("data_pub", na_position="last").reset_index(drop=True)
    df["id"] = [f"vrd22_{i:04d}" for i in range(len(df))]

    df_out = df[[c for c in OUTPUT_COLS if c in df.columns]].copy()

    # ── Salvare ────────────────────────────────────────────────────────────────
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info("")
    logger.info("=" * 70)
    logger.info("CLEANING TERMINAT")
    logger.info("=" * 70)
    logger.info("Input raw:    %d articole", n0)
    logger.info("Output clean: %d articole (%.1f%% retenție)",
                len(df_out), 100 * len(df_out) / n0)
    logger.info("Salvat: %s", OUTPUT_CSV)

    # ── Statistici finale ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== Distribuție pe an ===")
    for an, n in df_out["an"].value_counts().sort_index().items():
        if pd.notna(an):
            logger.info("  %d: %d articole", int(an), n)

    logger.info("")
    logger.info("=== Distribuție pe lună (2022) ===")
    for luna, n in df_out["luna"].value_counts().sort_index().items():
        logger.info("  %s: %d", luna, n)

    logger.info("")
    logger.info("=== Distribuție calitate_extractie ===")
    for q, n in df_out["calitate_extractie"].value_counts().items():
        logger.info("  %-30s: %d", q, n)

    logger.info("")
    logger.info("=== Stats nr_cuvinte_truncat ===")
    stats = df_out["nr_cuvinte_truncat"].describe()
    logger.info(
        "  median=%.0f  mean=%.1f  min=%d  max=%d",
        stats["50%"], stats["mean"], int(stats["min"]), int(stats["max"]),
    )
    pct250 = (df_out["nr_cuvinte_truncat"] == 250).sum()
    logger.info(
        "  articole truncate la 250 cw: %d (%.1f%%)",
        pct250, 100 * pct250 / len(df_out),
    )

    logger.info("")
    logger.info("=== Sanity checks ===")
    logger.info("  Toate label=1?      %s", (df_out["label_numeric"] == 1).all())
    logger.info("  text_curat OK?      %s",
                (df_out["text_curat"].fillna("").str.len() > 0).all())
    logger.info("  Hash unic?          %s", df_out["hash_continut"].is_unique)
    logger.info("  ID unic?            %s", df_out["id"].is_unique)

    logger.info("")
    logger.info("PASUL URMĂTOR: rulează merge_veridica_final.py")
    logger.info("  Input 1: veridica_clean_v5.csv")
    logger.info("  Input 2: veridica_2022_clean.csv  ← acest fișier")
    logger.info("  Output:  veridica_clean_final.csv")


if __name__ == "__main__":
    main()
