"""
merge_veridica_final.py
───────────────────────
Merge intre veridica_clean_v5.csv (2023–2025) si veridica_2022_clean.csv
(2022) → veridica_clean_final.csv (dataset complet clasa 1).

Parte din proiectul de licenta „Sistem de Detectie Automata si Explicabila
a Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din
Ucraina".

CE FACE:
    1. Incarca ambele CSV-uri si verifica ca au coloane compatibile
    2. Concateneaza — v5 primul (2023–2025), 2022 al doilea
    3. Deduplicare pe hash_continut (protectie impotriva overlap-urilor)
    4. Deduplicare pe url (a doua linie de aparare)
    5. Re-sortare cronologica + re-generare ID stabil (vrd_XXXX)
    6. Sanity checks finale
    7. Salvare veridica_clean_final.csv

OUTPUT:
    veridica_clean_final.csv      → dataset final clasa 1, gata pentru
                                    antrenare XLM-RoBERTa
    merge_veridica_final_report.txt → raport audit
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Configurare ────────────────────────────────────────────────────────────────

INPUT_V5   = "veridica_clean_v5.csv"
INPUT_2022 = "veridica_2022_clean.csv"
OUTPUT_CSV = "veridica_clean_final.csv"
OUTPUT_RPT = "merge_veridica_final_report.txt"

# Coloane care trebuie sa existe in ambele fisiere pentru merge valid
COLS_REQUIRED = [
    "url", "titlu", "data", "text_curat",
    "label", "label_numeric", "hash_continut",
]

# Coloane pastrate in output final — intersectie intre cele doua surse
COLS_OUTPUT = [
    "id", "url", "titlu", "data", "an", "luna",
    "sursa_site", "sectiune",
    "text_curat",
    "stire_citata", "naratiuni_false", "obiective_propaganda",
    "nr_cuvinte_v4", "nr_cuvinte_truncat",
    "calitate_extractie",
    "label", "label_numeric",
    "hash_continut",
]


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("merge_veridica_final")
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


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("MERGE VERIDICA FINAL: v5 + 2022 → clean_final")
    logger.info("=" * 70)

    # ── Verificare existenta fisiere ──────────────────────────────────────────
    for path in [INPUT_V5, INPUT_2022]:
        if not Path(path).exists():
            logger.error("Fișier lipsă: %s", path)
            sys.exit(1)

    # ── Incarcare ─────────────────────────────────────────────────────────────
    df_v5   = pd.read_csv(INPUT_V5)
    df_2022 = pd.read_csv(INPUT_2022)

    logger.info("Încărcat %s: %d articole", INPUT_V5,   len(df_v5))
    logger.info("Încărcat %s: %d articole", INPUT_2022, len(df_2022))

    # ── Verificare coloane obligatorii ────────────────────────────────────────
    for col in COLS_REQUIRED:
        if col not in df_v5.columns:
            logger.error("Coloană lipsă în v5: %s", col)
            sys.exit(1)
        if col not in df_2022.columns:
            logger.error("Coloană lipsă în 2022: %s", col)
            sys.exit(1)

    # ── Verificare ca ambele sunt clasa 1 ─────────────────────────────────────
    assert (df_v5["label_numeric"]   == 1).all(), "v5 conține label != 1!"
    assert (df_2022["label_numeric"] == 1).all(), "2022 conține label != 1!"
    logger.info("Verificare label=1: OK pentru ambele surse")

    # ── Concatenare (v5 primul — cronologie 2023–2025 inainte) ───────────────
    df = pd.concat([df_v5, df_2022], ignore_index=True)
    logger.info("Post-concatenare: %d articole", len(df))

    # ── Deduplicare pe hash_continut ─────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset=["hash_continut"], keep="first")
    n_hash_dup = n_before - len(df)
    if n_hash_dup > 0:
        logger.info("Dedup pe hash_continut: -%d duplicate (rămas: %d)",
                    n_hash_dup, len(df))
    else:
        logger.info("Dedup pe hash_continut: 0 duplicate — OK")

    # ── Deduplicare pe url ───────────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    n_url_dup = n_before - len(df)
    if n_url_dup > 0:
        logger.info("Dedup pe url: -%d duplicate (rămas: %d)",
                    n_url_dup, len(df))
    else:
        logger.info("Dedup pe url: 0 duplicate — OK")

    # ── Parsare data + sortare cronologica ────────────────────────────────────
    df["data_pub"] = pd.to_datetime(df["data"], errors="coerce")
    df["an"]       = df["data_pub"].dt.year
    df["luna"]     = df["data_pub"].dt.to_period("M").astype(str)
    df = df.sort_values("data_pub", na_position="last").reset_index(drop=True)

    # ── Re-generare ID stabil ─────────────────────────────────────────────────
    df["id"] = [f"vrd_{i:04d}" for i in range(len(df))]

    # ── Selectie coloane output ───────────────────────────────────────────────
    df_out = df[[c for c in COLS_OUTPUT if c in df.columns]].copy()

    # ── Salvare ───────────────────────────────────────────────────────────────
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # ── Raport final ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("MERGE TERMINAT")
    logger.info("=" * 70)
    logger.info("Input v5   : %d articole", len(df_v5))
    logger.info("Input 2022 : %d articole", len(df_2022))
    logger.info("Total merge: %d articole", len(df_v5) + len(df_2022))
    logger.info("Duplicate eliminate: %d", n_hash_dup + n_url_dup)
    logger.info("Output final: %d articole", len(df_out))
    logger.info("Salvat: %s", OUTPUT_CSV)

    logger.info("")
    logger.info("=== Distribuție pe an ===")
    for an, n in df_out["an"].value_counts().sort_index().items():
        if pd.notna(an):
            logger.info("  %d: %d articole", int(an), n)

    logger.info("")
    logger.info("=== Distribuție calitate_extractie ===")
    for q, n in df_out["calitate_extractie"].value_counts().items():
        logger.info("  %-30s: %d", q, n)

    logger.info("")
    logger.info("=== Stats nr_cuvinte_truncat ===")
    if "nr_cuvinte_truncat" in df_out.columns:
        stats = df_out["nr_cuvinte_truncat"].describe()
        logger.info(
            "  median=%.0f  mean=%.1f  min=%d  max=%d",
            stats["50%"], stats["mean"],
            int(stats["min"]), int(stats["max"]),
        )

    logger.info("")
    logger.info("=== Sanity checks ===")
    logger.info("  Toate label=1?   %s", (df_out["label_numeric"] == 1).all())
    logger.info("  text_curat OK?   %s",
                (df_out["text_curat"].fillna("").str.len() > 0).all())
    logger.info("  Hash unic?       %s", df_out["hash_continut"].is_unique)
    logger.info("  URL unic?        %s", df_out["url"].is_unique)
    logger.info("  ID unic?         %s", df_out["id"].is_unique)

    logger.info("")
    logger.info("=== Distribuție pe sursă (din URL) ===")
    df_out["_sursa"] = df_out["url"].str.extract(r"https?://(?:www\.)?([^/]+)")
    for sursa, n in df_out["_sursa"].value_counts().items():
        logger.info("  %-25s: %d", sursa, n)

    logger.info("")
    logger.info("Dataset clasa 1 complet — gata pentru merge cu clasa 0")
    logger.info("(G4Media + Digi24 → dataset_final.csv pentru XLM-RoBERTa)")


if __name__ == "__main__":
    main()
