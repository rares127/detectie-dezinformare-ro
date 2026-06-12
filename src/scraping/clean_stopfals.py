"""
Cleaning si audit pentru stopfals_raw_v1.csv.

Probleme cunoscute din log:
  - Articole cu text prea scurt (46-104 cuvinte): "Cronici dezinformarii",
    rezumate video, anunturi de concurs — nu au continut trainabil
  - Articole fara stire_citata (naratiunea falsa nu e izolabila)
  - Potentiale duplicate pe continut

Output:
  - stopfals_clean_v1.csv   ← dataset curat, gata pentru merge
  - stopfals_audit_v1.md    ← raport cu statistici si exemple eliminate
"""

import pandas as pd
import re
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────
INPUT  = Path("data/processed/stopfals_raw_v1.csv")
OUTPUT = Path("data/processed/stopfals_clean_v1.csv")
AUDIT  = Path("findings/stopfals_audit_v1.md")

AUDIT.parent.mkdir(parents=True, exist_ok=True)

# Praguri de calitate
MIN_CUVINTE    = 200   # sub acest prag → articol prea scurt, probabil rezumat/anunt
MIN_CUVINTE_WARN = 300 # intre 200-300 → warning, pastram dar notam

# Tipare de titluri pentru articole non-trainabile (rezumate, cronici, anunturi)
TITLURI_EXCLUDE_REGEX = [
    r"^cronica dezinform",          # "CRONICA DEZINFORMARII (16-28 feb...)"
    r"^anticorpi la fals",          # rezumate scurte video
    r"câștigătorii.*test",          # anunturi concurs
    r"^cele mai răspândite falsuri.*\d{4}$",   # colectii/top-uri scurte
    r"^falsuri despre.*\d{4}$",     # titluri de compilatie fara corp
    r"^falsuri în contextul",       # compilatii lunare scurte
    r"au fost desemna",             # anunturi castigatori
]

# ─── Incarcare ──────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, encoding="utf-8")
print(f"Articole încărcate: {len(df)}")
print(f"Coloane: {list(df.columns)}")
print(f"\nDistribuție per an (raw):")
print(df["an"].value_counts().sort_index().to_string())

eliminare_log = []

# ─── 1. Elimina articole prea scurte ────────────────────────────────────────
masca_scurte = df["nr_cuvinte"] < MIN_CUVINTE
scurte = df[masca_scurte].copy()
if len(scurte):
    print(f"\n[1] Articole sub {MIN_CUVINTE} cuvinte: {len(scurte)}")
    for _, r in scurte.iterrows():
        print(f"    [{r['an']}] {r['nr_cuvinte']}w | {r['titlu'][:70]}")
        eliminare_log.append({"motiv": f"sub_{MIN_CUVINTE}_cuvinte", "titlu": r["titlu"], "nr_cuvinte": r["nr_cuvinte"], "an": r["an"]})

df = df[~masca_scurte].copy()

# ─── 2. Elimina articole cu titluri de tip compilatie/rezumat ───────────────
def titlu_exclus(titlu: str) -> bool:
    t = titlu.strip().lower()
    return any(re.search(pat, t) for pat in TITLURI_EXCLUDE_REGEX)

masca_compilatii = df["titlu"].apply(titlu_exclus)
compilatii = df[masca_compilatii].copy()
if len(compilatii):
    print(f"\n[2] Articole compilație/rezumat (titlu pattern): {len(compilatii)}")
    for _, r in compilatii.iterrows():
        print(f"    [{r['an']}] {r['nr_cuvinte']}w | {r['titlu'][:70]}")
        eliminare_log.append({"motiv": "titlu_compilatie", "titlu": r["titlu"], "nr_cuvinte": r["nr_cuvinte"], "an": r["an"]})

df = df[~masca_compilatii].copy()

# ─── 3. Deduplica pe hash_continut ──────────────────────────────────────────
inainte = len(df)
df = df.drop_duplicates(subset=["hash_continut"]).copy()
dupa = len(df)
if inainte != dupa:
    print(f"\n[3] Duplicate eliminate pe hash: {inainte - dupa}")

# ─── 4. Warning articole intre 200-300 cuvinte ──────────────────────────────
masca_warn = (df["nr_cuvinte"] >= MIN_CUVINTE) & (df["nr_cuvinte"] < MIN_CUVINTE_WARN)
if masca_warn.sum():
    print(f"\n[WARN] Articole 200-300 cuvinte (păstrate, verifică manual): {masca_warn.sum()}")
    for _, r in df[masca_warn].iterrows():
        print(f"    [{r['an']}] {r['nr_cuvinte']}w | {r['titlu'][:70]}")

# ─── 5. Statistici stire_citata ─────────────────────────────────────────────
fara_citat = df["stire_citata"].isna() | (df["stire_citata"].str.strip() == "")
print(f"\n[INFO] Articole fără stire_citata: {fara_citat.sum()}/{len(df)}")
print(f"[INFO] Articole cu stire_citata: {(~fara_citat).sum()}/{len(df)}")

# ─── 6. Verificare camp text_curat ──────────────────────────────────────────
# Redenumire consistenta cu schema proiectului
# Schema proiect: id, text, label, sursa, an, luna, titlu, url, hash_continut
# stopfals are: text_curat (= "text" in schema finala)
if "text_curat" not in df.columns and "text" in df.columns:
    df = df.rename(columns={"text": "text_curat"})

# ─── 7. Rezultat final ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"REZULTAT FINAL după cleaning:")
print(f"  Articole raw:      {len(pd.read_csv(INPUT))}")
print(f"  Articole eliminate: {len(eliminare_log)}")
print(f"  Articole curate:   {len(df)}")
print(f"\nDistribuție per an (clean):")
print(df["an"].value_counts().sort_index().to_string())
print(f"\nStatistici nr_cuvinte:")
print(df["nr_cuvinte"].describe().round(0).to_string())

# ─── 8. Salvare ─────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"\nCSV curat salvat: {OUTPUT}")

# ─── 9. Raport audit Markdown ────────────────────────────────────────────────
with open(AUDIT, "w", encoding="utf-8") as f:
    f.write("# Audit Cleaning — stopfals_raw_v1.csv\n\n")
    f.write(f"**Data**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
    f.write(f"## Sumar\n\n")
    f.write(f"| Metric | Valoare |\n|---|---|\n")
    f.write(f"| Articole raw | {len(pd.read_csv(INPUT))} |\n")
    f.write(f"| Articole eliminate | {len(eliminare_log)} |\n")
    f.write(f"| Articole curate | {len(df)} |\n")
    f.write(f"| Cu stire_citata | {(~fara_citat).sum()} |\n")
    f.write(f"| Fără stire_citata | {fara_citat.sum()} |\n\n")
    f.write(f"## Distribuție per an (clean)\n\n")
    for an, cnt in df["an"].value_counts().sort_index().items():
        f.write(f"- **{an}**: {cnt} articole\n")
    f.write(f"\n## Articole eliminate ({len(eliminare_log)})\n\n")
    f.write("| An | Cuvinte | Motiv | Titlu |\n|---|---|---|---|\n")
    for e in eliminare_log:
        f.write(f"| {e['an']} | {e['nr_cuvinte']} | {e['motiv']} | {e['titlu'][:70]} |\n")
    f.write(f"\n## Note\n\n")
    f.write("- Articolele cu `stire_citata` gol nu sunt eliminate — stopfals.md\n")
    f.write("  nu are întotdeauna citat izolat în `<em>`, dar `text_curat` conține\n")
    f.write("  oricum narațiunea falsă integrată în corpul articolului.\n")
    f.write("- Pragul `nr_cuvinte >= 200` elimină cronicile scurte și anunțurile.\n")
    f.write("- Input-ul corect pentru clasificator: `text_curat` (corpul articolului),\n")
    f.write("  NU `stire_citata` (care e mai degrabă util pentru modulul granular).\n")

print(f"Raport audit salvat: {AUDIT}")
