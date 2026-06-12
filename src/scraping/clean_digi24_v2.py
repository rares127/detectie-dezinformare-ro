import argparse
import csv
import logging
import re
import sys
from pathlib import Path
import pandas as pd

# ─── Configurare ─────────────────────────────────────────────────────────────

RAW_CSV = Path("data/raw/digi24_v1_2022_raw.csv") # UPDATE FISIER 2022
OUT_CSV = Path("data/processed/digi24_v1_2022_clean.csv") # UPDATE FISIER 2022
LOG_FILE = Path("data/processed/clean_digi24_v1_2022.log")

# ... Regex-urile tale raman la fel ...
RE_TITLE_PREFIX = re.compile(
    r"^(?:Video&Foto|Galerie Foto|Live Text|Analiză|Exclusiv|Video|Foto)\s*",
    flags=re.IGNORECASE,
)
RE_BOILERPLATE_EDITOR = re.compile(r"\s*Editor\s*:.*$", flags=re.DOTALL)
RE_INLINE_SURSA = re.compile(
    r"(?:Sursa foto:|FOTO:|Imagine cu caracter ilustrativ).*?(?:\n|$)",
    flags=re.IGNORECASE,
)
RE_LIVETEXT_MARKER = re.compile(
    r"(?:Urmăriți LiveText-ul Digi24\.ro care a acoperit evenimentele din Ucraina|Desfășurarea evenimentelor.*?:)",
    flags=re.IGNORECASE,
)

def setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clean_digi24_2022")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def is_livetext(row: pd.Series) -> bool:
    titlu = str(row.get("titlu", "")).lower()
    if "live text" in titlu:
        return True
    corp = str(row.get("corp_articol", "")).lower()
    if "livetext-ul digi24.ro" in corp:
        return True
    if corp.count("actualizare") >= 3:
        return True
    return False

def clean_titlu(titlu: str) -> str:
    if not isinstance(titlu, str):
        return ""
    t = RE_TITLE_PREFIX.sub("", titlu)
    return t.strip()

def clean_corp(corp: str) -> str:
    if not isinstance(corp, str):
        return ""
    c = RE_BOILERPLATE_EDITOR.sub("", corp)
    c = RE_INLINE_SURSA.sub(" ", c)
    c = RE_LIVETEXT_MARKER.sub(" ", c)
    c = re.sub(r"\s+", " ", c)
    return c.strip()

def main() -> None:
    logger = setup_logging()

    if not RAW_CSV.exists():
        logger.error("Lipsă input: %s", RAW_CSV)
        return

    logger.info("Încărcare %s...", RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    n0 = len(df)
    
    # Doar un sanity check - pastram DOAR fetch_ok
    df = df[df["fetch_ok"] == True]
    n_ok = len(df)
    logger.info("Retinut %d articole fetch_ok=True (aruncat %d erori HTTP/Parse)", n_ok, n0 - n_ok)

    # 1. Parsare date
    df["dt_pub"] = pd.to_datetime(df["data_publicarii"], format="%Y-%m-%d %H:%M", errors="coerce")
    df = df.dropna(subset=["dt_pub"])
    df["an"] = df["dt_pub"].dt.year

    # 2. STRICT 2022 (Aici era greseala in vechiul script)
    n_pre_an = len(df)
    df = df[df["an"] == 2022]
    logger.info("Filtru an == 2022: eliminat %d articole out-of-range", n_pre_an - len(df))

    # 3. Drop LiveText
    mask_live = df.apply(is_livetext, axis=1)
    n_pre_live = len(df)
    df = df[~mask_live]
    logger.info("Filtru LiveText: eliminat %d articole", n_pre_live - len(df))

    # 4. Curatare NLP (Titlu + Corp)
    df["titlu_clean"] = df["titlu"].apply(clean_titlu)
    df["corp_clean"] = df["corp_articol"].apply(clean_corp)

    # Filtram articolele prea scurte dupa curatare
    df["nr_cuvinte_clean"] = df["corp_clean"].apply(lambda x: len(x.split()))
    n_pre_len = len(df)
    df = df[df["nr_cuvinte_clean"] >= 50]
    logger.info("Filtru cuvinte < 50: eliminat %d articole goale/scurte", n_pre_len - len(df))

    # 5. Generare text final si limitare la 512 cuvinte (pentru RoBERTa)
    df["text_curat"] = df["titlu_clean"] + ". " + df["corp_clean"]
    df["text_curat"] = df["text_curat"].apply(lambda x: " ".join(x.split()[:512]))
    df["nr_cuvinte_truncat"] = df["text_curat"].apply(lambda x: len(x.split()))

    # 6. Adaugam Label 0 (Stiri Adevarate)
    df["label"] = "Adevărat"
    df["label_numeric"] = 0

    # 7. Salvare
    output_cols = [
        "id_dataset", "url", "sursa", "data_publicarii", "an", "titlu", 
        "titlu_clean", "text_curat", "nr_cuvinte_raw", "nr_cuvinte_truncat", 
        "label", "label_numeric", "hash_continut"
    ]
    df_out = df.rename(columns={"nr_cuvinte": "nr_cuvinte_raw"})[output_cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    logger.info("=" * 70)
    logger.info("CLEANING 2022 TERMINAT")
    logger.info("Input:  %d articole brute", n0)
    logger.info("Output: %d articole curate (Gata de unire)", len(df_out))
    logger.info("Salvat: %s", OUT_CSV)

if __name__ == "__main__":
    main()