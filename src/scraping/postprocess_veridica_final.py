"""
postprocess_veridica_final.py
─────────────────────────────
Post-processing final pe veridica_clean_final.csv — doua fix-uri:

    FIX 1 — Strip prefixe jurnalistice din text_curat
        191 articole aveau text_curat incepand cu "DEZINFORMARE:",
        "FAKE NEWS:", "PROPAGANDA DE RAZBOI:" etc. — prefix-ul categoriei
        editoriale Veridica, nu parte din naratiunea pro-Kremlin.
        Daca clasificatorul il vede, poate invata eticheta din prefix
        (shortcut lexical) in loc de continutul naratiunii.
        Fix: strip regex la inceputul text_curat.

    FIX 2 — Drop articole fara termeni Ucraina/Rusia in text_curat
        18 articole nu contin niciun termen tematic relevant
        (vaccin/COVID, Hidroelectrica, Olimpiada/satanism, UE produse,
        parteneriate SUA etc.). Sunt dezinformari generale, nu pro-ruse
        legate de conflictul din Ucraina — off-topic fata de obiectivul
        proiectului. Drop inclusiv cele 5 borderline Moldova
        (Partidul Sor, federalizare, Maia Sandu/Epstein etc.) —
        clasificatorul trebuie sa invete pe semnal lingvistic explicit,
        nu pe context geopolitic implicit.

    FIX 3 — Recalculare hash_continut post-strip
        text_curat s-a schimbat → hash-urile vechi sunt invalide.
        Recalculam SHA1 si re-deduplicam.

    FIX 4 — Re-generare ID stabil dupa drop + re-sortare cronologica

Input:  veridica_clean_final.csv
Output: veridica_clean_final_v2.csv
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys

import pandas as pd

# ── Configurare ────────────────────────────────────────────────────────────────

INPUT_CSV  = "veridica_clean_final.csv"
OUTPUT_CSV = "veridica_clean_final_v2.csv"
OUTPUT_RPT = "postprocess_veridica_final_report.txt"

# Regex pentru strip prefix jurnalistic la inceputul text_curat.
# Acopera toate variantele observate in dataset:
#   "DEZINFORMARE: ", "FAKE NEWS: ", "PROPAGANDA DE RAZBOI: ",
#   "PROPAGANDA: ", "PROPAGANDA: ", "Propaganda de razboi: "
# Greedy pe spatiu/punct/liniuta dupa prefix.
PREFIX_STRIP_RE = re.compile(
    r"^(DEZINFORMARE|FAKE\s*NEWS|PROPAGANDĂ(\s+DE\s+RĂZBOI)?|"
    r"PROPAGANDA(\s+DE\s+RAZBOI)?|PROPAGANDĂ|PROPAGANDA)\s*[:–\-]?\s*",
    re.IGNORECASE,
)

# Termeni tematici minimi — cel putin unul trebuie sa apara in text_curat
# dupa strip, altfel articolul e off-topic fata de conflictul din Ucraina
UKRAINE_TEMATIC_RE = re.compile(
    r"\b(ucrain|rusi[ae]|ruș|ruși|rus(esc|ească|ilor)|putin|zelenski|kremlin|"
    r"donbas|crimeea|mariupol|invazie|invazia|război|razboi|wagner|"
    r"transnistri|herson|kherson|kiev|kyiv|moscova|operați\w*\s+special|"
    r"nato|republica\s+moldova|r\.\s*moldova|luhansk|lugansk|donețk|"
    r"zaporijia|zaporizhzhia|harkov|harkiv|odesa|lviv|bahmut|azov|"
    r"sputnik|pro[- ]?rus|pro[- ]?kremlin|biolaborator|denazific)\w*\b",
    re.IGNORECASE,
)


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("postprocess_veridica")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in [logging.StreamHandler(sys.stdout),
              logging.FileHandler(OUTPUT_RPT, encoding="utf-8")]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ── Functii helper ─────────────────────────────────────────────────────────────

def strip_prefix(text: str) -> str:
    """Elimina prefixul jurnalistic de la inceputul text_curat."""
    if not isinstance(text, str):
        return text
    return PREFIX_STRIP_RE.sub("", text).strip()


def compute_hash(text: str) -> str:
    """SHA1 pe text — consistent cu clean_veridica_v5.py."""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def count_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("POST-PROCESSING VERIDICA CLEAN FINAL → V2")
    logger.info("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)
    logger.info("Input: %d articole din %s", n0, INPUT_CSV)

    # ── FIX 1: Strip prefixe jurnalistice ─────────────────────────────────────
    df["_are_prefix"] = df["text_curat"].str.match(PREFIX_STRIP_RE, na=False)
    n_prefix = df["_are_prefix"].sum()

    df["text_curat"] = df["text_curat"].apply(strip_prefix)

    # Verificare ca strip-ul a functionat
    df["_are_prefix_after"] = df["text_curat"].str.match(PREFIX_STRIP_RE, na=False)
    n_prefix_ramas = df["_are_prefix_after"].sum()

    logger.info(
        "FIX 1 — Strip prefix jurnalistic: %d articole procesate, %d rămase cu prefix",
        n_prefix, n_prefix_ramas,
    )
    if n_prefix_ramas > 0:
        logger.warning("  Articole cu prefix nestripped:")
        for _, r in df[df["_are_prefix_after"]].iterrows():
            logger.warning("  [%s] %s", r["id"], str(r["text_curat"])[:80])

    # ── FIX 2: Drop off-topic ─────────────────────────────────────────────────
    n_before = len(df)
    mask_relevant = df["text_curat"].str.contains(UKRAINE_TEMATIC_RE, na=False)
    df_drop = df[~mask_relevant].copy()
    df = df[mask_relevant].copy()
    n_dropped = n_before - len(df)

    logger.info(
        "FIX 2 — Drop off-topic (fără termeni Ucraina/Rusia): -%d (rămas: %d)",
        n_dropped, len(df),
    )
    if n_dropped > 0:
        logger.info("  Articole droppate:")
        for _, r in df_drop.iterrows():
            logger.info("  [%s][%s] %s", r["id"], r.get("an", "?"), r["titlu"][:70])

    # ── FIX 3: Recalculare hash post-strip ────────────────────────────────────
    df["hash_continut"] = df["text_curat"].apply(compute_hash)

    n_before = len(df)
    df = df.drop_duplicates(subset=["hash_continut"], keep="first")
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        logger.info("FIX 3 — Dedup post-strip pe hash: -%d", n_dedup)
    else:
        logger.info("FIX 3 — Dedup post-strip: 0 duplicate — OK")

    # ── FIX 4: Re-sortare + re-generare ID ────────────────────────────────────
    df["data_pub"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values("data_pub", na_position="last").reset_index(drop=True)
    df["id"] = [f"vrd_{i:04d}" for i in range(len(df))]

    # Actualizare nr_cuvinte_truncat dupa strip (unele articole au acum mai
    # putine cuvinte — prefixul putea fi 2–4 cuvinte)
    df["nr_cuvinte_truncat"] = df["text_curat"].apply(count_words)

    # ── Curatare coloane helper ────────────────────────────────────────────────
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")],
                 errors="ignore")

    # ── Salvare ────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # ── Raport final ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("REZULTAT FINAL")
    logger.info("=" * 70)
    logger.info("Input:  %d articole", n0)
    logger.info("Output: %d articole (%.1f%% retenție)",
                len(df), 100 * len(df) / n0)
    logger.info("Salvat: %s", OUTPUT_CSV)

    logger.info("")
    logger.info("=== Distribuție pe an ===")
    df["_an"] = pd.to_datetime(df["data"], errors="coerce").dt.year
    for an, n in df["_an"].value_counts().sort_index().items():
        if pd.notna(an):
            logger.info("  %d: %d articole", int(an), n)

    logger.info("")
    logger.info("=== Stats nr_cuvinte_truncat (post-strip) ===")
    stats = df["nr_cuvinte_truncat"].describe()
    logger.info(
        "  median=%.0f  mean=%.1f  min=%d  max=%d",
        stats["50%"], stats["mean"], int(stats["min"]), int(stats["max"]),
    )

    logger.info("")
    logger.info("=== Sanity checks ===")
    # Verificare ca nu mai exista prefixe
    n_prefix_final = df["text_curat"].str.match(PREFIX_STRIP_RE, na=False).sum()
    logger.info("  Prefix jurnalistic rămas?  %s  (%d articole)",
                "NU ✓" if n_prefix_final == 0 else f"DA ⚠ {n_prefix_final}",
                n_prefix_final)
    # Verificare ca toate au termeni tematici
    n_offtopic_final = (~df["text_curat"].str.contains(
        UKRAINE_TEMATIC_RE, na=False)).sum()
    logger.info("  Articole off-topic rămase? %s  (%d articole)",
                "NU ✓" if n_offtopic_final == 0 else f"DA ⚠ {n_offtopic_final}",
                n_offtopic_final)
    logger.info("  Toate label=1?             %s",
                "DA ✓" if (df["label_numeric"] == 1).all() else "NU ⚠")
    logger.info("  text_curat OK?             %s",
                "DA ✓" if (df["text_curat"].fillna("").str.len() > 0).all() else "NU ⚠")
    logger.info("  Hash unic?                 %s",
                "DA ✓" if df["hash_continut"].is_unique else "NU ⚠")
    logger.info("  ID unic?                   %s",
                "DA ✓" if df["id"].is_unique else "NU ⚠")

    logger.info("")
    logger.info("Dataset clasa 1 — gata pentru merge cu clasa 0 (G4Media + Digi24)")


if __name__ == "__main__":
    main()
