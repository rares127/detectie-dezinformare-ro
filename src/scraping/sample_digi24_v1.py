"""
sample_digi24_v1.py — Sampling stratificat pentru Digi24 v1, clasa 0

Parte din proiectul de licenta „Sistem de Detectie Automata si Explicabila a
Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din
Ucraina".

Input:  data/processed/digi24_v1_clean.csv  (~6348 articole post-clean)
Output: data/processed/digi24_v1_sampled.csv  (240 articole sampled stratificat)

Strategie de sampling:
    Stratificare an × luna uniform, cu tinta fixa per an, distribuita aproximativ
    egal pe lunile disponibile. Seed 42 pentru reproductibilitate.

    Tinta totala: 240 articole, distribuite:
        2023:  80 articole (~7/luna × 12 luni)
        2024:  80 articole (~7/luna × 12 luni)
        2025:  50 articole (~4/luna × 12 luni)
        2026:  30 articole (10/luna × 3 luni: feb, mar, apr)

    Justificare proportionala cu G4Media v2 (250/250/137/66) — pastram acelasi
    profil temporal pentru ca merge-ul cu G4Media sa nu introduca asimetrie
    temporala inter-sursa pe clasa 0.

Algoritm:
    1. Calculeaza cota bruta per luna = floor(an_target / num_luni_an)
    2. Distribuie restul aleator (cu seed) pe lunile cu pool mai mare
    3. Pentru fiecare bucket (an, luna): sampling random fara inlocuire
    4. Daca o luna are pool < cota, ia tot pool-ul si redistribuie diferenta

Output: acelasi schema ca digi24_v1_clean.csv, cu o coloana suplimentara
    `bucket_sampling` (string „YYYY-MM") pentru auditing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuratie
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path.cwd()
INPUT_CSV = ROOT / "data" / "processed" / "digi24_v1_clean.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "digi24_v1_sampled.csv"
LOG_FILE = ROOT / "data" / "raw" / "sample_digi24_v1.log"

SEED = 42  # consistent cu G4Media v2

# Tintele per an (decizia 80/80/50/30 = 240 total, proportional cu G4Media)
TARGETS_PER_YEAR: dict[int, int] = {
    2023: 80,
    2024: 80,
    2025: 50,
    2026: 30,
}

TOTAL_TARGET = sum(TARGETS_PER_YEAR.values())  # 240


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:
    """Configureaza logging dual: consola + fisier."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sample_digi24")
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
# Sampling
# ─────────────────────────────────────────────────────────────────────────────


def compute_quotas_per_month(
    df: pd.DataFrame, year: int, target: int, logger: logging.Logger
) -> dict[str, int]:
    """
    Calculeaza cota per luna pentru un an, distribuita cat mai uniform.

    Strategie:
    1. cota_baza = target // num_luni
    2. rest = target - cota_baza * num_luni → distribuit pe lunile cu cel mai
       mare pool disponibil (capacitate de a absorbi mai mult)
    3. Daca o luna are pool < cota propusa, plafonam la pool si redistribuim
       diferenta pe celelalte luni cu capacitate (max 3 iteratii).
    """
    # Lunile reale disponibile pentru anul respectiv
    pool_per_luna = (
        df[df["an"] == year]
        .groupby("luna")
        .size()
        .sort_index()
        .to_dict()
    )
    luni = sorted(pool_per_luna.keys())
    n_luni = len(luni)
    if n_luni == 0:
        logger.warning("An %d: zero luni disponibile, skip", year)
        return {}

    cota_baza = target // n_luni
    rest = target - cota_baza * n_luni
    quotas = {luna: cota_baza for luna in luni}

    # Distribuie rest-ul pe lunile cu pool mai mare (descrescator)
    # Folosim seed-ul random.seed implicit prin sortare determinista
    luni_by_pool = sorted(luni, key=lambda l: -pool_per_luna[l])
    for i in range(rest):
        quotas[luni_by_pool[i % n_luni]] += 1

    # Verifica plafonarea: daca vreo luna cere mai mult decat are
    for _ in range(3):  # max 3 redistribuiri
        deficit = 0
        for luna in luni:
            if quotas[luna] > pool_per_luna[luna]:
                deficit += quotas[luna] - pool_per_luna[luna]
                quotas[luna] = pool_per_luna[luna]
        if deficit == 0:
            break
        # Redistribuie deficit-ul pe lunile cu capacitate
        capacitate = [
            (luna, pool_per_luna[luna] - quotas[luna]) for luna in luni
        ]
        capacitate = [(l, c) for l, c in capacitate if c > 0]
        if not capacitate:
            logger.warning(
                "An %d: deficit %d nedistribuibil — pool insuficient total",
                year,
                deficit,
            )
            break
        capacitate.sort(key=lambda x: -x[1])
        for i in range(deficit):
            luna, _ = capacitate[i % len(capacitate)]
            quotas[luna] += 1

    return quotas


def sample_year(
    df: pd.DataFrame, year: int, target: int, seed: int, logger: logging.Logger
) -> pd.DataFrame:
    """Sampling stratificat pe luna pentru un an."""
    quotas = compute_quotas_per_month(df, year, target, logger)
    if not quotas:
        return pd.DataFrame()

    sampled_parts = []
    actual_total = 0
    for luna in sorted(quotas.keys()):
        cota = quotas[luna]
        pool = df[(df["an"] == year) & (df["luna"] == luna)]
        if cota > len(pool):
            cota = len(pool)
        sampled = pool.sample(n=cota, random_state=seed)
        sampled_parts.append(sampled)
        actual_total += len(sampled)
        logger.info(
            "  %s: %d/%d sampled (pool=%d)",
            luna,
            len(sampled),
            quotas[luna],
            len(pool),
        )

    logger.info(
        "An %d: total sampled = %d / țintă %d", year, actual_total, target
    )
    return pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()


def run_sampling(logger: logging.Logger) -> None:
    """Pipeline complet de sampling."""
    if not INPUT_CSV.exists():
        logger.error("Input CSV inexistent: %s", INPUT_CSV)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("SAMPLING DIGI24 V1 — START (seed=%d)", SEED)
    logger.info("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    logger.info("Citit input: %d articole post-clean", len(df))
    logger.info("Țintă totală: %d articole", TOTAL_TARGET)
    logger.info("")

    sampled_parts = []
    for year, target in TARGETS_PER_YEAR.items():
        logger.info("=== An %d (țintă: %d) ===", year, target)
        sampled_year_df = sample_year(df, year, target, SEED, logger)
        if len(sampled_year_df) > 0:
            sampled_parts.append(sampled_year_df)
        logger.info("")

    df_sampled = pd.concat(sampled_parts, ignore_index=True)

    # Adaugam coloana bucket_sampling pentru audit
    df_sampled["bucket_sampling"] = df_sampled["luna"]

    # Re-ordonam cronologic pentru un output curat
    df_sampled = df_sampled.sort_values("data_publicarii").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_sampled.to_csv(OUTPUT_CSV, index=False)

    # ─── Audit final ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("SAMPLING TERMINAT")
    logger.info("=" * 70)
    logger.info("Total sampled: %d / țintă %d", len(df_sampled), TOTAL_TARGET)
    logger.info("Salvat: %s", OUTPUT_CSV)
    logger.info("")
    logger.info("=== Distribuție pe an (final) ===")
    for an, n in df_sampled["an"].value_counts().sort_index().items():
        logger.info("  %d: %d", int(an), n)
    logger.info("")
    logger.info("=== Distribuție pe lună (final) ===")
    for luna, n in df_sampled["luna"].value_counts().sort_index().items():
        logger.info("  %s: %d", luna, n)
    logger.info("")
    logger.info("=== Stats nr_cuvinte_truncat ===")
    stats = df_sampled["nr_cuvinte_truncat"].describe()
    logger.info(
        "  median=%.0f mean=%.0f min=%d max=%d",
        stats["50%"],
        stats["mean"],
        int(stats["min"]),
        int(stats["max"]),
    )
    logger.info("")
    logger.info("=== Distribuție secțiuni (top 10) ===")
    for sect, n in df_sampled["sectiune"].value_counts().head(10).items():
        logger.info("  %s: %d", sect, n)
    logger.info("")
    logger.info("=== Verificare unicitate ===")
    logger.info("  hash unice: %d / %d", df_sampled["hash_continut"].nunique(), len(df_sampled))
    logger.info("  id unice:   %d / %d", df_sampled["id_articol"].nunique(), len(df_sampled))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sampling stratificat an × lună pentru Digi24 v1."
    )
    parser.parse_args()
    logger = setup_logging()
    run_sampling(logger)


if __name__ == "__main__":
    main()
