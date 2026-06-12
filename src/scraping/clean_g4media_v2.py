"""
Clean G4Media v2 — cleaning final pe CSV-ul combinat.

Input: g4media_v2_complet_raw.csv (2725 articole, 3 surse: principal + supliment + v1)
Output: g4media_v2_clean.csv (~2575 articole post-drop)

Pasi de cleaning, in ordine:

1. Strip boilerplate final tip "sursa: News.ro [, Sursa foto: ...]"
   - regex: [Ss]ursa(\\s+foto)?\\s*: orice pana la final
   - acopera AMBELE: 'sursa:' editorial (~400 cazuri) si 'Sursa foto:' standalone (~135)
   - decizie metodologica: e shortcut stilistic 100% absent din Veridica

2. Strip "Citeste [ss]i ..." si tot ce vine dupa
   - regex: Cite[ss]te\\s+[ss]i.*$
   - link-uri inline catre alte articole, extrase ca text de scraper

3. Strip prefixe editoriale cu WHITELIST EXPLICIT (NU regex generic pe all-caps)
   - Lista: VIDEO|UPDATE|BREAKING|FOTO|LIVETEXT|EXCLUSIV|ANALIZA|INTERVIU|
           GALERIE FOTO|OPINIE|EDITORIAL|LIVE|VIDEOREPORTAJ
   - NU strip-uim acronime legitime (SUA, NATO, ONU, UE, BBC, CNN) — verificat
     ca 49 articole incep cu astfel de acronime ca subiect, nu ca prefix
   - Aplicare RECURSIVA pentru combinatii (BREAKING UPDATE, FOTO VIDEO, etc.)
   - Aplicare oglinda a Pasului 0 Veridica (FAKE NEWS:, PROPAGANDA DE RAZBOI:)
   - Justificare: label leakage editorial — apar predominant pe G4Media,
     absent din Veridica → shortcut stilistic daca nu strip-uim

4. Strip "Rador" / "Radio Romania" standalone la final
   - cazuri marginale care au scapat de pattern-ul de la pasul 1

5. Strip extended attribution chains ramase
   - pattern imbunatatit pentru ', relateaza AFP, citata de Agerpres.ro'
   - lantul dublu de atribuire pe care pattern-ul existent din scraper nu-l prinde

6. Cleanup whitespace + punctuatie finala izolata

7. Drop too_long (> 1100 cuvinte) — INAINTE de truncate, pe lungimea originala
   - Argument: articolele lungi sunt analize/longreads, stil diferit de stiri
   - Impact asteptat: ~137 articole drop (5.0%, majoritar din 2022)

8. Drop too_short (< 64 cuvinte) — siguranta, zero cazuri asteptate

9. Truncate la 250 cuvinte — primele 250 dupa strip
   - Validat empiric in recapitulare: LogReg pe lungime → 56.8% (≈ baseline)
   - Rezolva asimetria de lungime cu Veridica (median 196 vs G4Media 297)

10. Recalculeaza nr_cuvinte si hash_continut post-cleaning

Output CSV: schema identica cu input + doua coloane noi pentru audit:
  - nr_cuvinte_pre_clean: lungimea inainte de cleaning
  - cleaning_actions: lista regulilor aplicate (ex: "sursa|prefix_editorial|truncate")
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Path-uri ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = WORK_DIR / "g4media_v2_complet_raw.csv"
OUTPUT_CSV = PROCESSED_DIR / "g4media_v2_clean.csv"
REPORT_FILE = PROCESSED_DIR / "g4media_v2_clean_report.txt"

# ── Praguri ───────────────────────────────────────────────────────────────────
MIN_WORDS = 64
MAX_WORDS_PRE_TRUNCATE = 1100
TRUNCATE_TARGET = 250


# ══════════════════════════════════════════════════════════════════════════════
# REGULI DE CLEANING
# ══════════════════════════════════════════════════════════════════════════════

# 1. Boilerplate final 'sursa:' / 'Sursa foto:'
# Match: orice varianta de [sS]ursa, optional " foto", apoi ":" si totul pana la final
# Verificat empiric: 100% din cele 406 ocurente sunt in ultimele 250 caractere
RE_SURSA_FINAL = re.compile(
    r"\s*[sS]ursa(\s+foto)?\s*:.*$",
    re.DOTALL,
)

# 2. 'Citeste [ss]i' si tot ce vine dupa
RE_CITESTE_SI = re.compile(
    r"\s*Cite[șs]te\s+[șs]i\b.*$",
    re.DOTALL,
)

# 3. Prefixe editoriale — WHITELIST EXPLICIT
# Ordine in alternativa: cele compuse PRIMELE (GALERIE FOTO inainte de FOTO/GALERIE)
EDITORIAL_PREFIXES = [
    "GALERIE FOTO",
    "VIDEOREPORTAJ",
    "LIVETEXT",
    "BREAKING",
    "EXCLUSIV",
    "EDITORIAL",
    "INTERVIU",
    "ANALIZĂ",
    "UPDATE",
    "OPINIE",
    "VIDEO",
    "FOTO",
    "LIVE",
]
RE_EDITORIAL_PREFIX = re.compile(
    r"^(?:" + "|".join(EDITORIAL_PREFIXES) + r")\b[\s:/]*",
)

# 4. Rador / Radio Romania standalone la final
# (cazuri care nu au fost prinse de pattern-ul sursa: pentru ca nu era precedat
#  de "sursa:")
RE_RADOR_FINAL = re.compile(
    r"\s*(potrivit\s+)?Rador(\s+Radio\s+Rom[âa]nia)?\s*[/.]?\s*$",
    re.IGNORECASE,
)

# 5. Extended attribution chains ramase
# Pattern compozit: ", relateaza AFP, citata de Agerpres.ro" la final
# Construit ca extensie a celui existent in scraper (care prinde doar single)
ATTRIBUTION_VERBS = (
    r"relatează|transmite|potrivit|conform|preluat\s+de|citat[ăe]?\s+de|"
    r"informează|notează|scrie|menționează|anunță"
)
ATTRIBUTION_SOURCES = (
    r"AFP|Reuters|BBC|CNN|Bloomberg|Associated\s+Press|AP|"
    r"Agerpres(?:\.ro)?|Mediafax|EFE|DPA|TASS|Interfax|Sky\s+News|"
    r"Financial\s+Times|FT|Wall\s+Street\s+Journal|WSJ|"
    r"New\s+York\s+Times|NYT|Washington\s+Post|Guardian|"
    r"Politico|Deutsche\s+Welle|DW|Euronews|Der\s+Spiegel|News\.ro"
)
# Lant de atribuire dublu sau simplu, la final
RE_ATTRIBUTION_CHAIN = re.compile(
    rf",?\s*({ATTRIBUTION_VERBS})\s+({ATTRIBUTION_SOURCES})"
    rf"(?:\s*,\s*({ATTRIBUTION_VERBS})\s+({ATTRIBUTION_SOURCES}))?"
    rf"\s*\.?\s*$",
    re.IGNORECASE,
)

# 6. Cleanup whitespace + punctuatie finala izolata
RE_MULTIPLE_SPACES = re.compile(r"\s+")
RE_TRAILING_PUNCT = re.compile(r"[\s,;:/.\-]+$")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTII DE CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def strip_sursa_final(text: str) -> tuple[str, bool]:
    """Strip boilerplate 'sursa:' / 'Sursa foto:' pana la final."""
    new_text, n = RE_SURSA_FINAL.subn("", text)
    return new_text, n > 0


def strip_citeste_si(text: str) -> tuple[str, bool]:
    """Strip 'Citeste si ...' si tot ce vine dupa."""
    new_text, n = RE_CITESTE_SI.subn("", text)
    return new_text, n > 0


def strip_editorial_prefixes(text: str) -> tuple[str, bool]:
    """
    Strip prefixele editoriale RECURSIV.
    Aplica pattern-ul repetat pana cand nu mai matches → handle combinatii
    (BREAKING UPDATE, FOTO VIDEO, GALERIE FOTO, etc.)
    """
    changed = False
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        new_text, n = RE_EDITORIAL_PREFIX.subn("", cur, count=1)
        if n > 0:
            changed = True
            cur = new_text.lstrip()
    return cur, changed


def strip_rador_final(text: str) -> tuple[str, bool]:
    """Strip Rador / Radio Romania standalone la final."""
    new_text, n = RE_RADOR_FINAL.subn("", text)
    return new_text, n > 0


def strip_attribution_chain(text: str) -> tuple[str, bool]:
    """
    Strip lant de atribuire la final, repetitiv.
    Acopera atat 'relateaza AFP' simplu cat si 'relateaza AFP, citata de Agerpres.ro'.
    """
    changed = False
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        new_text, n = RE_ATTRIBUTION_CHAIN.subn("", cur)
        if n > 0:
            changed = True
            cur = new_text.strip()
    return cur, changed


def cleanup_whitespace(text: str) -> str:
    """Collapse multiple spaces si strip punctuatie finala izolata."""
    text = RE_MULTIPLE_SPACES.sub(" ", text).strip()
    text = RE_TRAILING_PUNCT.sub("", text).strip()
    # Re-adaugam un punct final daca textul nu se termina cu punctuatie
    if text and text[-1] not in ".!?":
        text += "."
    return text


def truncate_to_words(text: str, max_words: int) -> tuple[str, bool]:
    """Truncate textul la primele max_words cuvinte. Returneaza (text, was_truncated)."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    truncated = " ".join(words[:max_words])
    # Asigura ca se termina cu punctuatie (fara sa rupem la mijloc de cuvant)
    if truncated and truncated[-1] not in ".!?":
        truncated += "."
    return truncated, True


def clean_article(text: str) -> tuple[str, list[str]]:
    """
    Aplica toate regulile de cleaning in ordine.
    Returneaza (text_curatat, lista_actiunilor_aplicate).
    """
    actions = []

    # 1. sursa: / Sursa foto:
    text, hit = strip_sursa_final(text)
    if hit:
        actions.append("sursa")

    # 2. Citeste si
    text, hit = strip_citeste_si(text)
    if hit:
        actions.append("citeste_si")

    # 3. Prefixe editoriale (recursiv)
    text, hit = strip_editorial_prefixes(text)
    if hit:
        actions.append("prefix_editorial")

    # 4. Rador final
    text, hit = strip_rador_final(text)
    if hit:
        actions.append("rador")

    # 5. Attribution chains (recursiv)
    text, hit = strip_attribution_chain(text)
    if hit:
        actions.append("attribution")

    # 6. Whitespace cleanup
    text = cleanup_whitespace(text)

    return text, actions


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def src_of(id_str: str) -> str:
    if id_str.startswith("g4m_v2s_"):
        return "supliment"
    if id_str.startswith("g4m_v2_"):
        return "principal"
    if id_str.startswith("g4m_v1r_"):
        return "v1_recovered"
    return "unknown"


def main() -> None:
    print("=" * 70)
    print("CLEAN G4MEDIA v2 — cleaning + truncate + drop")
    print("=" * 70)

    if not INPUT_CSV.exists():
        sys.exit(f"EROARE: lipsește {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"\nÎncărcat: {len(df)} articole")

    df["src"] = df["id"].apply(src_of)
    print(f"Per sursă: {df['src'].value_counts().to_dict()}")

    # Salvam lungimea originala pentru audit
    df["nr_cuvinte_pre_clean"] = df["nr_cuvinte"].astype(int)

    # ── Aplicam cleaning text pe rand ─────────────────────────────────────────
    print("\nAplicare reguli de cleaning text...")
    cleaned_texts = []
    actions_per_row = []
    action_counter: Counter = Counter()

    for _, row in df.iterrows():
        text = str(row["text_curat"])
        new_text, actions = clean_article(text)
        cleaned_texts.append(new_text)
        actions_per_row.append("|".join(actions) if actions else "")
        for a in actions:
            action_counter[a] += 1

    df["text_curat"] = cleaned_texts
    df["cleaning_actions"] = actions_per_row
    df["nr_cuvinte"] = df["text_curat"].str.split().str.len()

    # ── Raport reguli aplicate ────────────────────────────────────────────────
    print("\nReguli de cleaning text aplicate (atinge text):")
    for action, count in sorted(action_counter.items(), key=lambda x: -x[1]):
        print(f"  {count:5} × {action}")

    # ── Drop too_long ─────────────────────────────────────────────────────────
    n_before = len(df)
    too_long_mask = df["nr_cuvinte_pre_clean"] > MAX_WORDS_PRE_TRUNCATE
    too_long_per_src = df[too_long_mask]["src"].value_counts().to_dict()
    df = df[~too_long_mask].reset_index(drop=True)
    print(f"\nDrop too_long (> {MAX_WORDS_PRE_TRUNCATE} cuvinte pre-clean): "
          f"{n_before - len(df)} eliminate")
    print(f"  Per sursă: {too_long_per_src}")

    # ── Drop too_short (post-cleaning) ────────────────────────────────────────
    # Important: verificam DUPA cleaning, pentru ca strip-urile pot reduce text
    n_before = len(df)
    too_short_mask = df["nr_cuvinte"] < MIN_WORDS
    too_short_per_src = df[too_short_mask]["src"].value_counts().to_dict()
    df = df[~too_short_mask].reset_index(drop=True)
    print(f"\nDrop too_short (< {MIN_WORDS} cuvinte post-clean): "
          f"{n_before - len(df)} eliminate")
    if too_short_per_src:
        print(f"  Per sursă: {too_short_per_src}")

    # ── Truncate la 250 cuvinte ────────────────────────────────────────────────
    print(f"\nTruncate la {TRUNCATE_TARGET} cuvinte...")
    n_truncated = 0
    new_texts = []
    for text in df["text_curat"]:
        new_text, was_trunc = truncate_to_words(text, TRUNCATE_TARGET)
        new_texts.append(new_text)
        if was_trunc:
            n_truncated += 1
    df["text_curat"] = new_texts
    df["nr_cuvinte"] = df["text_curat"].str.split().str.len()
    print(f"  Articole truncate: {n_truncated}/{len(df)}")

    # ── Recalculeaza hash ──────────────────────────────────────────────────────
    df["hash_continut"] = df["text_curat"].apply(
        lambda t: hashlib.md5(t.encode()).hexdigest()[:16]
    )

    # ── Statistici finale ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STATISTICI FINALE")
    print("=" * 70)
    print(f"\nTotal articole post-clean: {len(df)}")

    print(f"\nDistribuție pe ani:")
    df["an"] = df["data"].astype(str).str[:4]
    for an, count in df["an"].value_counts().sort_index().items():
        print(f"  {an}: {count:5}")

    print(f"\nDistribuție pe sursă:")
    print(f"  {df['src'].value_counts().to_dict()}")

    print(f"\nLungime post-clean (cuvinte):")
    nw = df["nr_cuvinte"]
    print(f"  min={nw.min()}, median={int(nw.median())}, "
          f"p75={int(nw.quantile(0.75))}, max={nw.max()}, mean={nw.mean():.1f}")
    print(f"  (Veridica median pentru comparație: 196)")

    # ── Audit thematic post-clean ──────────────────────────────────────────────
    pass_rate = df["audit_thematic_pass"].astype(int).mean() * 100
    print(f"\nAudit thematic pass post-clean: {pass_rate:.1f}%")

    # ── Drop coloane temp + salvare ────────────────────────────────────────────
    df = df.drop(columns=["src", "an"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Salvat: {OUTPUT_CSV}")

    # ── Sample audit: top 10 articole cu cea mai mare reducere ────────────────
    df["delta"] = df["nr_cuvinte_pre_clean"] - df["nr_cuvinte"]
    top_reduced = df.nlargest(5, "delta")[["id", "nr_cuvinte_pre_clean",
                                            "nr_cuvinte", "cleaning_actions",
                                            "titlu"]]
    print("\n── Top 5 articole cu cea mai mare reducere (audit manual) ──")
    for _, r in top_reduced.iterrows():
        print(f"  [{r['id']}] {r['nr_cuvinte_pre_clean']} → {r['nr_cuvinte']} "
              f"({r['cleaning_actions']})")
        print(f"    {r['titlu'][:90]}")


if __name__ == "__main__":
    main()
