"""
clean_g4media_v1.py
───────────────────
Cleaning post-collection pentru CSV-ul G4Media.

Aplica urmatoarele fix-uri descoperite empiric pe primul batch de 500
articole G4Media:

1. **Filtru "subiect principal Ucraina"** (Pasul 2.G.1)
   Drop articole care mentioneaza Iran/Israel/Orient Mijlociu MAI MULT
   decat Ucraina/Rusia/Moldova. 25% din primul batch erau pure off-topic.

2. **Strip "Sursa: ..." boilerplate** (Pasul 2.G.2)
   G4Media adauga la sfarsitul a ~5% din articole signature de tip
   "Sursa: KYIV POST / Rador Radio Romania / Traducerea: Sergiu Dan."
   Asta e boilerplate de redactie, nu continut.

3. **Strip signature reziduale** (Pasul 2.G.3)
   Variante extinse ale ATTRIBUTION_PATTERN care scapa in primul scraper:
   - "transmite Agerpres ." (cu spatiu inainte de punct)
   - "transmite Agerpres citand AFP."
   - "relateaza AFP, preluata de Agerpres."

4. **Strip "Citeste si" inline** (Pasul 2.G.4)
   3 articole din 500 aveau "Citeste si" in interiorul paragrafelor
   (link-uri inline catre articole conexe), nu in div separat.

5. **Recalculare nr_cuvinte** dupa strip + recalculare hash_continut

Filtre de validare post-cleaning:
- Re-aplica filtrul de lungime (64 ≤ nr_cuvinte ≤ 1100) — articole
  care devin prea scurte dupa strip sunt drop
- Re-aplica filtrul tematic (is_ukraine_related) — articole care nu mai
  trec dupa strip sunt drop

Output: g4media_clean_v1.csv (pastreaza coloanele initiale + adauga coloane
de audit: drop_reason, n_chars_stripped, etc.)

Utilizare:
    python clean_g4media_v1.py
    # Citeste data/raw/g4media_raw.csv
    # Scrie data/processed/g4media_clean_v1.csv
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

# Importam filtrul tematic partajat
sys.path.insert(0, str(Path(__file__).parent))
from common.thematic_filters import is_ukraine_related, topic_match_details  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Configurare
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "g4media_raw.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "g4media_clean_v1.csv"
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "g4media_clean_v1_report.txt"

# Filtre lungime (simetrice cu Veridica + buffer)
MIN_TEXT_WORDS = 64
MAX_TEXT_WORDS = 1100


# ══════════════════════════════════════════════════════════════════════════════
# Pattern-uri pentru subiect principal — Iran/Israel vs Ucraina/Rusia/Moldova
# ══════════════════════════════════════════════════════════════════════════════
#
# Validate empiric: pe 500 articole G4Media, 125 (25%) au fost detectate ca
# fiind dominant despre Iran/Israel. Sample manual de 10 a confirmat zero
# false pozitive — toate 10 erau genuin off-topic (Dubai turism, Ierusalim
# Floriilor, BCE inflatie, etc.).

UKRAINE_RUSSIA_PATTERN = re.compile(
    r"\b("
    r"ucrain\w*|kiev|kyiv|zelenski|zelensky|"
    r"rus(ia|ă|ești|ească|esc|ilor)|ruse\w+|kremlin\w*|moscova|"
    r"putin|lavrov|medvedev|șoigu|prigojin|"
    r"donbas|donețk|lugansk|luhansk|crimeea|sevastopol|"
    r"mariupol|herson|kherson|harkov|kharkiv|odesa|odessa|lvov|lviv|"
    r"moldov\w*|chișinău|chisinau|transnistr\w*|"
    r"sandu|dodon|șor|plahotniuc|"
    r"comrat|găgăuz\w*|gagauz\w*|tiraspol"
    r")\b",
    re.IGNORECASE
)

IRAN_ISRAEL_PATTERN = re.compile(
    r"\b("
    r"iran|iranian\w*|teheran|"
    r"israel|israelian\w*|tel\s+aviv|netanyahu|"
    r"hamas|hezbollah|gaza|cisiordan|"
    r"orient\w+\s+mijloci\w+|"
    r"liban|libanez\w*|sirian\w*|sirian"
    r")\b",
    re.IGNORECASE
)


def topic_dominance(text: str) -> tuple[int, int, str]:
    """
    Returneaza (count_ucraine, count_iran, decision).

    Decision:
    - "keep_no_iran"          → zero mentionari Iran/Israel
    - "keep_ukraine_dominant" → count_ua >= count_iran > 0
    - "drop_iran_dominant"    → count_iran > count_ua
    """
    n_ua = len(UKRAINE_RUSSIA_PATTERN.findall(text))
    n_ir = len(IRAN_ISRAEL_PATTERN.findall(text))

    if n_ir == 0:
        return n_ua, n_ir, "keep_no_iran"
    if n_ua >= n_ir:
        return n_ua, n_ir, "keep_ukraine_dominant"
    return n_ua, n_ir, "drop_iran_dominant"


# ══════════════════════════════════════════════════════════════════════════════
# Strip "Sursa: ..." boilerplate Rador
# ══════════════════════════════════════════════════════════════════════════════
#
# Pattern observat in 25/500 articole din primul batch:
#   "...textul articolului. Sursa: KYIV POST / Rador Radio Romania / Traducerea: Sergiu Dan."
#   "...textul. Sursa: GAZETA / Rador Radio Romania."
#   "...text. Sursa: Defense Express / Rador Radio Romania/ Traducere: Andrei Suba."
#
# Forma generala: "Sursa:" urmata de orice pana la sfarsitul textului.
# E un marker de boilerplate aproape garantat ca vine la sfarsit.

SURSA_BOILERPLATE_PATTERN = re.compile(
    r"\s*Sursa:\s*[^.]*?(Rador|Traducerea?|Traducere)[^.]*\.\s*$",
    re.IGNORECASE,
)


def strip_sursa_boilerplate(text: str) -> tuple[str, bool]:
    """
    Elimina signature 'Sursa: ... Rador ... Traducerea: ...' de la sfarsit.
    Returneaza (text_curat, was_stripped).
    """
    if not text:
        return text, False
    new_text = SURSA_BOILERPLATE_PATTERN.sub("", text).strip()
    return new_text, new_text != text


# ══════════════════════════════════════════════════════════════════════════════
# Strip signature reziduale extinse
# ══════════════════════════════════════════════════════════════════════════════
#
# Variante care au scapat scraperului v1 din cauza:
# 1. Spatiu inainte de punct: "transmite Agerpres ." (in loc de "Agerpres.")
# 2. Modificator dupa sursa: "transmite Agerpres citand AFP."
# 3. Atribuire dubla: "relateaza AFP, preluata de Agerpres."
#
# Aceste forme apar inline (in mijlocul textului intre propozitii), nu la
# sfarsit, deci scapa stripului final din scraper.

EXTENDED_ATTRIBUTION_SOURCES = (
    r"AFP|Reuters|BBC|CNN|Bloomberg|Associated\s+Press|AP|"
    r"Agerpres|Mediafax|EFE|DPA|TASS|Interfax|Sky\s+News|"
    r"Financial\s+Times|FT|Wall\s+Street\s+Journal|WSJ|"
    r"New\s+York\s+Times|NYT|Washington\s+Post|Guardian|"
    r"Politico|Deutsche\s+Welle|DW|Euronews|Der\s+Spiegel|"
    r"Kyiv\s+Post|Defense\s+Express"
)

EXTENDED_ATTRIBUTION_VERBS = (
    r"relatează|transmite|potrivit|conform|preluat[ăa]?\s+de|citat[ăa]?\s+de|"
    r"informează|notează|scrie|menționează|anunță|cit(?:ând|at)"
)

# Varianta cu spatiu optional inainte de punct si optional modificator
EXTENDED_INLINE_PATTERN = re.compile(
    rf",\s*({EXTENDED_ATTRIBUTION_VERBS})\s+({EXTENDED_ATTRIBUTION_SOURCES})"
    rf"(?:\s*,?\s*(?:{EXTENDED_ATTRIBUTION_VERBS})\s+({EXTENDED_ATTRIBUTION_SOURCES}))?"
    rf"\s*\.\s*",
    re.IGNORECASE,
)


def strip_extended_signatures(text: str) -> tuple[str, int]:
    """
    Elimina signature reziduale (variante extinse).
    Returneaza (text_curat, n_chars_stripped).
    """
    if not text:
        return text, 0
    n_before = len(text)
    new_text = EXTENDED_INLINE_PATTERN.sub(" ", text)
    new_text = re.sub(r"\s+", " ", new_text).strip()
    return new_text, n_before - len(new_text)


# ══════════════════════════════════════════════════════════════════════════════
# Strip "Citeste si" inline
# ══════════════════════════════════════════════════════════════════════════════
#
# Pattern observat in 3/500 articole:
#   "...textul anterior. Citeste si SURSE Discutii despre colaborarea..."
#   "...textul. Citeste si George Simion defileaza la brat..."
#
# Strip: "Citeste si" + restul textului care urmeaza (pana la sfarsitul
# textului sau pana la urmatorul segment cu litera mica = cuvant-titlu nou).
# CONSERVATOR: stripam doar pana la finalul textului daca "Citeste si" e
# aproape de final (in ultimele 200 caractere), pentru ca poate fi un
# trailer de articole conexe.

CITESTE_SI_TRAILER_PATTERN = re.compile(
    r"\s*Cite[șs]te\s+și\s+.{0,500}$",
    re.IGNORECASE,
)


def strip_citeste_si(text: str) -> tuple[str, bool]:
    """
    Elimina 'Citeste si ...' care apare la sfarsitul textului.
    """
    if not text:
        return text, False
    new_text = CITESTE_SI_TRAILER_PATTERN.sub("", text).strip()
    return new_text, new_text != text


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Citire: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    n_initial = len(df)
    print(f"  → {n_initial} articole")
    print()

    # Audit: pastram valorile originale inainte de cleaning
    df["text_curat_raw"] = df["text_curat"].copy()
    df["nr_cuvinte_raw"] = df["nr_cuvinte"].copy()

    # Statistici per pas
    stats = {
        "iran_dominant_dropped": 0,
        "sursa_boilerplate_stripped": 0,
        "extended_signature_chars_stripped": 0,
        "citeste_si_stripped": 0,
        "too_short_after_clean": 0,
        "too_long_after_clean": 0,
        "no_topic_after_clean": 0,
    }

    drop_reasons = []

    for idx, row in df.iterrows():
        text = str(row["text_curat"])

        # ── Pas 1: filtru subiect principal
        n_ua, n_ir, decision = topic_dominance(text)
        if decision == "drop_iran_dominant":
            drop_reasons.append((idx, "iran_dominant", f"UA={n_ua} IR={n_ir}"))
            stats["iran_dominant_dropped"] += 1
            continue

        # ── Pas 2: strip Sursa: ... Rador
        text, sursa_stripped = strip_sursa_boilerplate(text)
        if sursa_stripped:
            stats["sursa_boilerplate_stripped"] += 1

        # ── Pas 3: strip signature reziduale extinse
        text, n_chars_stripped = strip_extended_signatures(text)
        if n_chars_stripped > 0:
            stats["extended_signature_chars_stripped"] += n_chars_stripped

        # ── Pas 4: strip Citeste si inline
        text, citeste_stripped = strip_citeste_si(text)
        if citeste_stripped:
            stats["citeste_si_stripped"] += 1

        # ── Re-validare dupa cleaning
        n_words = len(text.split())

        if n_words < MIN_TEXT_WORDS:
            drop_reasons.append((idx, "too_short_after_clean", f"{n_words} cuvinte"))
            stats["too_short_after_clean"] += 1
            continue

        if n_words > MAX_TEXT_WORDS:
            drop_reasons.append((idx, "too_long_after_clean", f"{n_words} cuvinte"))
            stats["too_long_after_clean"] += 1
            continue

        if not is_ukraine_related(text):
            drop_reasons.append((idx, "no_topic_after_clean", "filtru tematic"))
            stats["no_topic_after_clean"] += 1
            continue

        # ── Update CSV
        df.at[idx, "text_curat"] = text
        df.at[idx, "nr_cuvinte"] = n_words
        df.at[idx, "hash_continut"] = hashlib.md5(text.encode()).hexdigest()[:16]

        # Recalculam si matched_core / matched_hybrid
        det = topic_match_details(text)
        df.at[idx, "matched_core"] = int(det.matched_core)
        df.at[idx, "matched_hybrid"] = int(det.matched_hybrid)

    # Drop randurile marcate
    drop_indices = [d[0] for d in drop_reasons]
    df_clean = df.drop(index=drop_indices).reset_index(drop=True)

    # ── Output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Output: {OUTPUT_CSV}")
    print(f"  → {len(df_clean)} articole (din {n_initial} inițiale)")
    print()

    # ── Raport
    print("=" * 72)
    print("RAPORT CLEANING G4MEDIA v1")
    print("=" * 72)
    print(f"  Inițial:                                  {n_initial}")
    print(f"  Drop subiect Iran/Israel dominant:        {stats['iran_dominant_dropped']}")
    print(f"  Drop prea scurt după strip:               {stats['too_short_after_clean']}")
    print(f"  Drop prea lung după strip:                {stats['too_long_after_clean']}")
    print(f"  Drop fără topic după strip:               {stats['no_topic_after_clean']}")
    print(f"  RĂMASE:                                   {len(df_clean)}")
    print()
    print(f"  Sursa: ... boilerplate stripat:           {stats['sursa_boilerplate_stripped']}")
    print(f"  Caractere stripate (signature extinse):   {stats['extended_signature_chars_stripped']}")
    print(f"  Citește și inline stripat:                {stats['citeste_si_stripped']}")
    print()

    # Statistici lungime post-cleaning
    print(f"Lungime text_curat post-cleaning:")
    print(f"  min:    {df_clean['nr_cuvinte'].min():.0f}")
    print(f"  p25:    {df_clean['nr_cuvinte'].quantile(0.25):.0f}")
    print(f"  median: {df_clean['nr_cuvinte'].median():.0f}")
    print(f"  p75:    {df_clean['nr_cuvinte'].quantile(0.75):.0f}")
    print(f"  p95:    {df_clean['nr_cuvinte'].quantile(0.95):.0f}")
    print(f"  max:    {df_clean['nr_cuvinte'].max():.0f}")
    print()
    print("Comparație cu Veridica clasa 1:")
    print("  min 64 / p25 152 / median 196 / p75 243 / p95 442 / max 1034")
    print()

    # Distributie temporala
    df_clean['data_parsed'] = pd.to_datetime(df_clean['data'], errors='coerce', utc=True)
    df_clean['an'] = df_clean['data_parsed'].dt.year
    print("Distribuție temporală:")
    print(df_clean['an'].value_counts().sort_index().to_string())
    print()

    # Distributie tematica
    n = len(df_clean)
    only_core = ((df_clean['matched_core']==1) & (df_clean['matched_hybrid']==0)).sum()
    only_hybrid = ((df_clean['matched_core']==0) & (df_clean['matched_hybrid']==1)).sum()
    overlap = ((df_clean['matched_core']==1) & (df_clean['matched_hybrid']==1)).sum()
    print("Distribuție tematică:")
    print(f"  doar core:    {only_core:>3} ({100*only_core/n:.1f}%)")
    print(f"  doar hybrid:  {only_hybrid:>3} ({100*only_hybrid/n:.1f}%)")
    print(f"  overlap:      {overlap:>3} ({100*overlap/n:.1f}%)")
    print()
    print("Comparație cu Veridica clasa 1:")
    print("  doar core: 63.0% / doar hybrid: 6.6% / overlap: 30.4%")
    print()

    # Salvare raport in fisier
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Cleaning G4Media v1 — raport\n")
        f.write(f"Input:  {INPUT_CSV}\n")
        f.write(f"Output: {OUTPUT_CSV}\n\n")
        f.write(f"Total inițial:  {n_initial}\n")
        f.write(f"Total final:    {len(df_clean)}\n")
        f.write(f"Pierdere:       {n_initial - len(df_clean)} ({100*(n_initial-len(df_clean))/n_initial:.1f}%)\n\n")

        f.write("Statistici drop:\n")
        for k, v in stats.items():
            f.write(f"  {k:<40} {v}\n")
        f.write("\n")

        f.write("Drop detalii (primele 30):\n")
        for idx, reason, info in drop_reasons[:30]:
            f.write(f"  idx={idx} {reason}: {info}\n")

    print(f"Raport detaliat: {REPORT_FILE}")
    print()
    print("✓ Cleaning complet")


if __name__ == "__main__":
    main()
