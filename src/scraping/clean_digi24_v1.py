"""
clean_digi24_v1.py — Cleaning pentru Digi24 v1, clasa 0 (stiri credibile)

Parte din proiectul de licenta „Sistem de Detectie Automata si Explicabila a
Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din
Ucraina".

Input:  data/raw/digi24_v1_raw.csv  (6846 articole fetched)
Output: data/processed/digi24_v1_clean.csv

Reguli aplicate (in ordine), validate empiric pe audit raw:
    1. Drop year=2022 (decizie metodologica, drop definitiv)
    2. Drop ianuarie 2026 (consistent cu G4Media v2)
    3. Drop LiveText — union 4 criterii:
        a. „Live Text" in titlu
        b. marker „LiveText-ul Digi24.ro" in corp
        c. 3+ ocurente „ACTUALIZARE XX:XX"
        d. diff data_actualizarii − data_publicarii > 30 zile
    4. Strip prefix editorial din titlu — Video&Foto, Galerie Foto, Live Text,
       Analiza, Exclusiv, Video, Foto (in ordine descrescatoare a specificitatii)
    5. Strip boilerplate final corp: regex `\\s*Editor\\s*:.*$` cu DOTALL
    6. Strip imagini / captions inline (Sursa foto, FOTO: captura, Imagine cu
       caracter ilustrativ)
    7. Strip „LiveText-ul Digi24.ro ... AICI." inline pentru articole non-dropped
    8. Drop too_long > 1100 cuvinte pre-clean (consistent G4Media v2)
    9. Drop too_short < 64 cuvinte post-clean (consistent G4Media v2)
   10. Truncate la 250 cuvinte (consistent G4Media v2)

Output schema (aliniat cu g4media_v2_sampled.csv pentru merge ulterior):
    id_dataset, id_articol, url, sursa, tag_sursa, sectiune,
    data_publicarii, an, luna,
    titlu, titlu_clean, corp_clean, text_curat,
    nr_cuvinte_raw, nr_cuvinte_clean, nr_cuvinte_truncat,
    label, label_numeric, hash_continut
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuratie
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path.cwd()
RAW_CSV = ROOT / "data" / "raw" / "digi24_v1_2022_raw.csv"
OUT_CSV = ROOT / "data" / "processed" / "digi24_v1_2022_clean.csv"
LOG_FILE = ROOT / "data" / "raw" / "clean_digi24_v1.log"

# Praguri
# NOTA pe WORDS_TOO_LONG_PRECLEAN: G4Media v2 a folosit 1100 cw. Pe Digi24
# (wire-heavy + multe analize/investigatii lungi de calitate) pragul 1100 ar
# taia 470 articole non-LiveText legitime (analize Politico/NYT, interviuri,
# investigatii, format „Exclusiv"/„Analiza"). Ridicam pragul la 2000 cw doar
# pentru Digi24 — taie outlierii extremi care sunt cu mare probabilitate
# LiveText nedetectati sau dossiere maraton, dar pastreaza corpul de analize
# legitime. Truncate-ul la 250 cw normalizeaza oricum input-ul clasificatorului.
# Asimetria de prag e justificata metodologic (sursa diferita, profil stilistic
# diferit) si va fi notata in capitolul Metodologie al lucrarii.
WORDS_TOO_LONG_PRECLEAN = 2000  # vs 1100 la G4Media — vezi nota
WORDS_TOO_SHORT_POSTCLEAN = 64  # consistent G4Media v2
WORDS_TRUNCATE = 250  # consistent G4Media v2

# Drop temporal
YEAR_DROP = 2022
YEARMONTH_DROP = "2026-01"  # ianuarie 2026 — gap editorial G4Media

# LiveText: diff act-pub mai mare de N zile = LiveText
LIVETEXT_DIFF_DAYS = 30

# Prefixe editoriale lipite de titlu — ORDINE CRITICA
# Cele compuse („Video&Foto", „Galerie Foto", „Live Text") TREBUIE testate
# inaintea celor simple („Video", „Foto"), altfel „Video" prinde primul si
# lasa „&Foto" agatat de titlu.
EDITORIAL_PREFIXES = [
    "Video&Foto",
    "Galerie Foto",
    "Live Text",
    "LiveText",  # varianta fara spatiu, observata sporadic
    "Analiză",
    "Exclusiv",
    "Video",
    "Foto",
]

# ─────────────────────────────────────────────────────────────────────────────
# Regex compilate
# ─────────────────────────────────────────────────────────────────────────────

# Marker boilerplate final — strip de la „Editor :" pana la EOF
RE_EDITOR_TAIL = re.compile(r"\s*Editor\s*:.*$", re.DOTALL | re.IGNORECASE)

# LiveText markers
RE_LIVETEXT_TITLE = re.compile(r"Live\s*Text", re.IGNORECASE)
RE_LIVETEXT_CORP_MARKER = re.compile(
    r"LiveText-ul\s+Digi24\.ro", re.IGNORECASE
)
RE_ACTUALIZARE = re.compile(r"ACTUALIZARE\s+\d{1,2}[:\.]\d{2}")

# Strip imagini/captions inline (line-level, pastram restul liniei goal)
RE_SURSA_FOTO = re.compile(r"^\s*Sursa\s+foto\s*:[^\n]*$", re.MULTILINE | re.IGNORECASE)
RE_FOTO_CAPTURA = re.compile(
    r"^\s*FOTO\s*:\s*captur[ăa][^\n]*$", re.MULTILINE | re.IGNORECASE
)
RE_IMAGINE_ILUSTRATIV = re.compile(
    r"^\s*Imagine\s+cu\s+caracter\s+ilustrativ\.[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)

# Strip propozitie „LiveText-ul Digi24.ro care a acoperit ... AICI."
RE_LIVETEXT_INLINE = re.compile(
    r"LiveText-ul\s+Digi24\.ro[^\.]*?AICI\.\s*",
    re.IGNORECASE | re.DOTALL,
)

# „Citeste si:" — block de related links la final de articol.
# Audit empiric: 11.5% din articole, median pozitie 91.7% din text, dar
# exista cazuri inline (10-30%) care NU trebuie strippate. Strip doar daca
# „Citeste si:" e in treimea finala a textului — atunci e clar boilerplate.
RE_CITESTE_SI = re.compile(r"Cite[șs]te\s+(?:și|si)\s*:", re.IGNORECASE)

# Whitespace cleanup final
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_MULTI_SPACE = re.compile(r"[ \t]+")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:
    """Configureaza logging dual: consola + fisier."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clean_digi24")
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
# Functii cleaning — fiecare e independenta, testabila, idempotenta
# ─────────────────────────────────────────────────────────────────────────────


def is_livetext(row: pd.Series) -> bool:
    """
    Determina daca un articol e LiveText prin union de 4 criterii.

    Suficient sa se declanseze ORICARE pentru drop. La audit empiric:
    224 candidati total (3.3%) cu overlap masiv intre criterii.
    """
    titlu = row["titlu"] if isinstance(row["titlu"], str) else ""
    corp = row["corp_articol"] if isinstance(row["corp_articol"], str) else ""

    # 1. „Live Text" in titlu
    if RE_LIVETEXT_TITLE.search(titlu):
        return True

    # 2. Marker in corp
    if RE_LIVETEXT_CORP_MARKER.search(corp):
        return True

    # 3. 3+ ACTUALIZARE XX:XX
    if len(RE_ACTUALIZARE.findall(corp)) >= 3:
        return True

    # 4. Diff > 30 zile
    diff_days = row.get("diff_zile", 0)
    if pd.notna(diff_days) and diff_days > LIVETEXT_DIFF_DAYS:
        return True

    return False


def strip_editorial_prefix(titlu: str) -> str:
    """
    Elimina prefixul editorial lipit de titlu.

    Verifica prefixele in ORDINE descrescatoare a specificitatii (compuse
    inaintea celor simple), ca sa nu lase resturi agatate. Doar primul match
    se aplica (un titlu nu are simultan „Video" + „Foto" lipite).

    Verificare „caracterul urmator": prefix-ul e considerat valid (= se
    strippeaza) DOAR DACA ce urmeaza NU e litera mica. Asta inseamna ca:
        - „VideoVolodimir..." → strippeaza (V e mare)
        - „Foto„Creaturi..." → strippeaza („ e ghilimea, nu litera mica)
        - „Videoclip nou..." → NU strippeaza (c e litera mica, e cuvant natural)
    Aceasta regula mai laxa permite si ghilimelele romanesti „...", cifrele,
    si orice alt caracter non-alfabetic, dar respinge cuvintele naturale.

    Exemple validate empiric:
        „VideoVolodimir Zelenski..." → „Volodimir Zelenski..."
        „Video&FotoSummitul..." → „Summitul..."
        „Live TextRusii au atacat..." → „Rusii au atacat..."
        „AnalizaVa mai rezista..." → „Va mai rezista..."
        „Foto„Creaturi complet bolnave"" → „„Creaturi complet bolnave""
        „Analiza„Spiritul si litera" summitului..." → „„Spiritul si litera" summitului..."
    """
    if not isinstance(titlu, str):
        return titlu

    for prefix in EDITORIAL_PREFIXES:
        if titlu.startswith(prefix):
            rest = titlu[len(prefix):]
            if not rest:
                continue
            # Strippeaza DOAR daca ce urmeaza NU e litera mica (care ar insemna
            # ca prefix-ul e parte dintr-un cuvant natural, ex. „Videoclip")
            if not rest[0].islower():
                return rest
    return titlu


def strip_corp(corp: str) -> str:
    """
    Aplica toate strip-urile pe corpul articolului, in ordine.

    Ordine importanta:
    1. Editor tail PRIMUL — eliminam semnatura editorului „Editor : X.X." la
       finalul articolului. La audit empiric: 97.9% din articole au markerul,
       median pozitie 99.4% din text, taie ~3 cw in medie.
       Nota: alte boilerplate-uri din recapitulare (Etichete, Top Citite,
       Digi Sport, Te-ar putea interesa, Urmareste stirile) NU sunt prezente
       in corp_articol — scraper-ul a fost suficient de specific la selectori
       si a exclus deja sidebar-ul/footer-ul. Verificat empiric: 0/6846 hits.
    2. „Citeste si:" tail — strip de la „Citeste si:" la EOF DOAR daca pozitia
       e > 75% din text (block de related links la final). Daca e inline (in
       primele 75% ale textului), e referinta legitima in corp si o pastram.
    3. LiveText inline — pentru articolele care au scapat de drop dar au
       totusi o trimitere reziduala.
    4. Captions inline (Sursa foto, FOTO: captura, Imagine ilustrativ).
    5. Whitespace cleanup.
    """
    if not isinstance(corp, str):
        return ""

    # 1. Editor tail — strip de la „Editor :" pana la sfarsit
    corp = RE_EDITOR_TAIL.sub("", corp)

    # 2. „Citeste si:" tail — strip conditionat (doar daca e in ultimul sfert)
    m = RE_CITESTE_SI.search(corp)
    if m and len(corp) > 0 and m.start() / len(corp) > 0.75:
        corp = corp[: m.start()].rstrip()

    # 3. LiveText inline (sentence-level)
    corp = RE_LIVETEXT_INLINE.sub("", corp)

    # 4. Captions
    corp = RE_SURSA_FOTO.sub("", corp)
    corp = RE_FOTO_CAPTURA.sub("", corp)
    corp = RE_IMAGINE_ILUSTRATIV.sub("", corp)

    # 5. Whitespace cleanup
    corp = RE_MULTI_NEWLINE.sub("\n\n", corp)
    corp = RE_MULTI_SPACE.sub(" ", corp)
    corp = corp.strip()

    return corp


def truncate_words(text: str, max_words: int) -> str:
    """
    Trunchiaza textul la max_words cuvinte, pastrand cuvintele intregi.

    Cuvant = orice secventa separata de whitespace. Nu rupe in mijlocul
    cuvintelor. Pastreaza newline-urile relative din primele max_words cuvinte.
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    # Reconstruim textul truncat — pierdem newline-urile pentru simplitate
    return " ".join(words[:max_words])


def count_words(text: str) -> int:
    """Numara cuvintele intr-un text. Cuvant = secventa non-whitespace."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────


def run_cleaning(logger: logging.Logger) -> None:
    """
    Pipeline complet: citire raw → 10 reguli → output processed.

    Toate dropurile sunt logate cu count + procentaj din total initial, ca sa
    avem un audit reproductibil.
    """
    if not RAW_CSV.exists():
        logger.error("Raw CSV inexistent: %s", RAW_CSV)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("CLEANING DIGI24 V1 — START")
    logger.info("=" * 70)

    df = pd.read_csv(RAW_CSV)
    n0 = len(df)
    logger.info("Citit raw CSV: %d articole", n0)

    # Sanity check: doar fetch_ok=True
    df = df[df["fetch_ok"] == True].copy()  # noqa: E712
    logger.info("După filtrare fetch_ok=True: %d (-%d)", len(df), n0 - len(df))

    # Parsare date
    df["data_pub"] = pd.to_datetime(df["data_publicarii"], errors="coerce")
    df["data_act"] = pd.to_datetime(df["data_actualizarii"], errors="coerce")
    df["diff_zile"] = (
        df["data_act"] - df["data_pub"]
    ).dt.total_seconds() / 86400
    df["an"] = df["data_pub"].dt.year
    df["luna"] = df["data_pub"].dt.to_period("M").astype(str)

    n_no_date = df["data_pub"].isna().sum()
    if n_no_date > 0:
        logger.warning("Articole fără data_publicarii parsabilă: %d", n_no_date)
        df = df.dropna(subset=["data_pub"])

    # ─── REGULA 1: Drop year=2022 ───────────────────────────────────────────
    n_before = len(df)
    df = df[df["an"] != YEAR_DROP].copy()
    logger.info(
        "Regula 1 — Drop year=%d: -%d articole (rămas: %d)",
        YEAR_DROP,
        n_before - len(df),
        len(df),
    )

    # ─── REGULA 2: Drop ianuarie 2026 ───────────────────────────────────────
    n_before = len(df)
    df = df[df["luna"] != YEARMONTH_DROP].copy()
    logger.info(
        "Regula 2 — Drop %s: -%d articole (rămas: %d)",
        YEARMONTH_DROP,
        n_before - len(df),
        len(df),
    )

    # ─── REGULA 3: Drop LiveText (4 criterii union) ─────────────────────────
    n_before = len(df)
    df["is_livetext"] = df.apply(is_livetext, axis=1)
    n_livetext = int(df["is_livetext"].sum())
    df = df[~df["is_livetext"]].copy()
    logger.info(
        "Regula 3 — Drop LiveText: -%d articole (rămas: %d)",
        n_before - len(df),
        len(df),
    )
    logger.info("  (LiveText flagged: %d)", n_livetext)
    df = df.drop(columns=["is_livetext"])

    # ─── REGULA 8: Drop too_long pre-clean ──────────────────────────────────
    # NOTA: ordinea reala e 8 inainte de strip, ca sa nu mai facem strip pe
    # articole care oricum se duc. Pre-clean = nr_cuvinte din raw CSV.
    n_before = len(df)
    df = df[df["nr_cuvinte"] <= WORDS_TOO_LONG_PRECLEAN].copy()
    logger.info(
        "Regula 8 — Drop too_long > %d cw pre-clean: -%d (rămas: %d)",
        WORDS_TOO_LONG_PRECLEAN,
        n_before - len(df),
        len(df),
    )

    # ─── REGULA 4: Strip prefix editorial din titlu ─────────────────────────
    df["titlu_clean"] = df["titlu"].apply(strip_editorial_prefix)
    n_stripped = (df["titlu"] != df["titlu_clean"]).sum()
    logger.info(
        "Regula 4 — Strip prefix editorial titlu: %d titluri modificate",
        n_stripped,
    )

    # ─── REGULA 5+6+7: Strip boilerplate corp ───────────────────────────────
    df["corp_clean"] = df["corp_articol"].apply(strip_corp)
    df["nr_cuvinte_clean"] = df["corp_clean"].apply(count_words)

    # Audit cleaning: cat a taiat strip-ul in medie
    df["delta_strip"] = df["nr_cuvinte"] - df["nr_cuvinte_clean"]
    logger.info(
        "Reguli 5+6+7 — Strip corp: median tăiat %d cw, mean %d cw",
        int(df["delta_strip"].median()),
        int(df["delta_strip"].mean()),
    )

    # ─── REGULA 9: Drop too_short post-clean ────────────────────────────────
    n_before = len(df)
    df = df[df["nr_cuvinte_clean"] >= WORDS_TOO_SHORT_POSTCLEAN].copy()
    logger.info(
        "Regula 9 — Drop too_short < %d cw post-clean: -%d (rămas: %d)",
        WORDS_TOO_SHORT_POSTCLEAN,
        n_before - len(df),
        len(df),
    )

    # ─── REGULA 10: Truncate la 250 cuvinte ─────────────────────────────────
    df["text_curat"] = (
        df["titlu_clean"] + ". " + df["corp_clean"]
    ).apply(lambda t: truncate_words(t, WORDS_TRUNCATE))
    df["nr_cuvinte_truncat"] = df["text_curat"].apply(count_words)
    logger.info(
        "Regula 10 — Truncate la %d cw: median final %d cw",
        WORDS_TRUNCATE,
        int(df["nr_cuvinte_truncat"].median()),
    )

    # ─── Adaugam label-urile (clasa 0) ──────────────────────────────────────
    df["label"] = "stire_credibila"
    df["label_numeric"] = 0

    # ─── Recalculam hash post-clean (pentru dedup la merge cu G4Media) ──────
    df["hash_continut"] = (df["titlu_clean"] + "||" + df["corp_clean"]).apply(
        lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()
    )

    # Verificare dedup post-clean (poate strip-ul a transformat 2 articole
    # diferite in acelasi continut — improbabil, dar verificam)
    n_before = len(df)
    df = df.drop_duplicates(subset=["hash_continut"], keep="first")
    if n_before - len(df) > 0:
        logger.info(
            "Dedup post-clean pe hash_continut: -%d (rămas: %d)",
            n_before - len(df),
            len(df),
        )

    # ─── Schema finala pentru output ────────────────────────────────────────
    output_cols = [
        "id_dataset",
        "id_articol",
        "url",
        "sursa",
        "tag_sursa",
        "sectiune",
        "data_publicarii",
        "an",
        "luna",
        "titlu",
        "titlu_clean",
        "corp_clean",
        "text_curat",
        "nr_cuvinte",  # raw
        "nr_cuvinte_clean",
        "nr_cuvinte_truncat",
        "label",
        "label_numeric",
        "hash_continut",
    ]
    df_out = df[output_cols].rename(columns={"nr_cuvinte": "nr_cuvinte_raw"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    # ─── Audit final ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CLEANING TERMINAT")
    logger.info("=" * 70)
    logger.info("Input:  %d articole", n0)
    logger.info("Output: %d articole (%.1f%% retenție)", len(df_out), 100 * len(df_out) / n0)
    logger.info("Salvat: %s", OUT_CSV)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cleaning pentru Digi24 v1 raw CSV."
    )
    parser.parse_args()  # nu avem flag-uri, dar pastram consistent cu scraper
    logger = setup_logging()
    run_cleaning(logger)


if __name__ == "__main__":
    main()
