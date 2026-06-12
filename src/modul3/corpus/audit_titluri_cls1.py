"""
Audit titluri cls1 — decidem daca le includem in corpus propagandistic.

Intrebarea de cercetare:
    Titlurile articolelor Veridica/Stopfals contin:
    - (a) Naratiunea propagandistica in forma bruta (OK pentru corpus cls1)?
    - (b) Framing jurnalistic de demascare (POATE contamina corpusul cls1)?

Pattern-uri periculoase de cautat:
    - Prefixe: „Naratiune falsa:", „Fake:", „Verdict:", „Dezinformare:",
      „Stire falsa:", „Minciuna:", „Manipulare:", „Stopfals:", etc.
    - Verbe de demascare la inceput: „Demontam...", „Nu e adevarat ca...",
      „De ce este fals...", etc.
    - Cuvinte cheie de fact-checking: „verificare", „analiza", „demonteaza"

Daca majoritatea titlurilor contin astfel de pattern-uri, decizia e clara:
    NU includem titlurile in corpus cls1 (ar contamina cu voce jurnalistica).

Daca putine titluri au pattern-uri (ex. <20%), putem:
    - (i) Exclude titlurile problematice si pastra restul.
    - (ii) Exclude TOATE titlurile pentru siguranta.
    - (iii) Normaliza titlurile (taind prefixele) si pastra ce ramane.

Input:
    - data/processed/dataset_v2_train.csv
    - data/processed/dataset_v2_val.csv

Output:
    - findings/audit_titluri_cls1.md
    - findings/audit_titluri_cls1.json

Rulare:
    python scripts/audit_titluri_cls1.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd


# -----------------------------------------------------------------------------
# Configuratie
# -----------------------------------------------------------------------------
CALE_TRAIN = Path("data/processed/dataset_v2_train.csv")
CALE_VAL = Path("data/processed/dataset_v2_val.csv")

CALE_OUT_MD = Path("findings/audit_titluri_cls1.md")
CALE_OUT_JSON = Path("findings/audit_titluri_cls1.json")


# Pattern-uri periculoase — indicatori de framing jurnalistic de demascare
# Le clasific in categorii pentru a avea o analiza mai clara.

# categoria 1: prefixe tipice de fact-checking (cel mai serios semnal)
PATTERNS_PREFIX_FACTCHECKING = [
    (r"^narațiune\s+fals[ăa]:?", "prefix_naratiune_falsa"),
    (r"^fake:?", "prefix_fake"),
    (r"^verdict:?", "prefix_verdict"),
    (r"^dezinformare:?", "prefix_dezinformare"),
    (r"^minciun[ăa]:?", "prefix_minciuna"),
    (r"^manipulare:?", "prefix_manipulare"),
    (r"^știre\s+fals[ăa]:?", "prefix_stire_falsa"),
    (r"^stopfals:?", "prefix_stopfals"),
    (r"^veridica:?", "prefix_veridica"),
    (r"^propagand[ăa]:?", "prefix_propaganda"),
    (r"^fals:?", "prefix_fals"),
    (r"^intox:?", "prefix_intox"),
]

# categoria 2: verbe/constructii de demascare la inceput
PATTERNS_VERBE_DEMASCARE = [
    (r"^demont[ăa]m\b", "verb_demontam"),
    (r"^analiz[ăa]m\b", "verb_analizam"),
    (r"^verific[ăa]m\b", "verb_verificam"),
    (r"^nu\s+e\s+adev[ăa]rat", "constr_nu_e_adevarat"),
    (r"^nu\s+este\s+adev[ăa]rat", "constr_nu_este_adevarat"),
    (r"^de\s+ce\s+(e|este)\s+fals", "constr_de_ce_este_fals"),
    (r"^cum\s+(a|au)\s+fost\s+manipulat", "constr_cum_manipulat"),
    (r"^adev[ăa]rul\s+despre", "constr_adevarul_despre"),
    (r"\bintox\b", "cuvant_intox"),
]

# categoria 3: substantive/constructii care sugereaza voce jurnalistica
# (mai blande — pot aparea si in titluri propagandistice autentice)
PATTERNS_VOCE_JURNALISTICA = [
    (r"\bfact[-\s]?check", "fact_check"),
    (r"\bmit\s+vs\s+realitate", "mit_vs_realitate"),
    (r"\bcontext:?\s", "cuvant_context"),
    (r"\brealitatea:?\s", "cuvant_realitatea"),
]


def verifica_patterns(titlu: str, categ: list) -> list[str]:
    """Returneaza lista etichetelor de pattern-uri detectate in titlu."""
    if not isinstance(titlu, str):
        return []
    txt = titlu.strip().lower()
    detectate = []
    for regex, eticheta in categ:
        if re.search(regex, txt, flags=re.IGNORECASE):
            detectate.append(eticheta)
    return detectate


def main() -> None:
    """Pipeline audit titluri cls1."""
    print("=" * 70)
    print("AUDIT TITLURI cls1 — decizie includere în corpus propagandistic")
    print("=" * 70)

    if not CALE_TRAIN.exists() or not CALE_VAL.exists():
        raise FileNotFoundError(f"Lipsește train sau val în data/processed/")

    # 1. incarcare si filtrare cls1
    df_train = pd.read_csv(CALE_TRAIN)
    df_val = pd.read_csv(CALE_VAL)
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    df_cls1 = df_all[df_all["label_numeric"] == 1].copy()
    print(f"\nTotal cls1 (train+val): {len(df_cls1)} articole")
    print(f"Distribuție per sursă:")
    for sursa, n in df_cls1["sursa_site"].value_counts().items():
        print(f"  {sursa}: {n}")

    # 2. statistici de baza pe titluri
    print(f"\n--- Statistici titluri ---")
    lungimi = df_cls1["titlu"].fillna("").str.split().str.len()
    print(f"Titluri vide: {(lungimi == 0).sum()}")
    print(f"Lungime min: {lungimi.min()}, mediană: {lungimi.median():.0f}, "
          f"max: {lungimi.max()}, medie: {lungimi.mean():.1f}")

    # 3. detectare pattern-uri per titlu
    print(f"\n--- Detectare pattern-uri de framing jurnalistic ---")
    df_cls1["patterns_prefix"] = df_cls1["titlu"].apply(
        lambda t: verifica_patterns(t, PATTERNS_PREFIX_FACTCHECKING)
    )
    df_cls1["patterns_verbe"] = df_cls1["titlu"].apply(
        lambda t: verifica_patterns(t, PATTERNS_VERBE_DEMASCARE)
    )
    df_cls1["patterns_voce"] = df_cls1["titlu"].apply(
        lambda t: verifica_patterns(t, PATTERNS_VOCE_JURNALISTICA)
    )
    df_cls1["are_orice_pattern"] = (
        df_cls1["patterns_prefix"].str.len()
        + df_cls1["patterns_verbe"].str.len()
        + df_cls1["patterns_voce"].str.len()
    ) > 0

    # 4. sumar global
    n_total = len(df_cls1)
    n_cu_prefix = (df_cls1["patterns_prefix"].str.len() > 0).sum()
    n_cu_verbe = (df_cls1["patterns_verbe"].str.len() > 0).sum()
    n_cu_voce = (df_cls1["patterns_voce"].str.len() > 0).sum()
    n_cu_orice = df_cls1["are_orice_pattern"].sum()

    print(f"\nRezultate globale:")
    print(f"  Titluri cu prefix fact-checking: {n_cu_prefix} "
          f"({n_cu_prefix/n_total*100:.1f}%)")
    print(f"  Titluri cu verbe demascare: {n_cu_verbe} "
          f"({n_cu_verbe/n_total*100:.1f}%)")
    print(f"  Titluri cu voce jurnalistică: {n_cu_voce} "
          f"({n_cu_voce/n_total*100:.1f}%)")
    print(f"  Titluri cu ORICE pattern: {n_cu_orice} "
          f"({n_cu_orice/n_total*100:.1f}%)")

    # 5. breakdown per sursa
    print(f"\n--- Breakdown per sursă ---")
    per_sursa = {}
    for sursa in df_cls1["sursa_site"].unique():
        df_s = df_cls1[df_cls1["sursa_site"] == sursa]
        n_s = len(df_s)
        n_s_prefix = (df_s["patterns_prefix"].str.len() > 0).sum()
        n_s_verbe = (df_s["patterns_verbe"].str.len() > 0).sum()
        n_s_voce = (df_s["patterns_voce"].str.len() > 0).sum()
        n_s_orice = df_s["are_orice_pattern"].sum()
        per_sursa[sursa] = {
            "total": int(n_s),
            "cu_prefix": int(n_s_prefix),
            "cu_verbe": int(n_s_verbe),
            "cu_voce": int(n_s_voce),
            "cu_orice": int(n_s_orice),
            "pct_orice": float(n_s_orice / n_s * 100) if n_s else 0,
        }
        print(f"  {sursa} ({n_s} titluri):")
        print(f"    cu prefix: {n_s_prefix} ({n_s_prefix/n_s*100:.1f}%)")
        print(f"    cu verbe: {n_s_verbe} ({n_s_verbe/n_s*100:.1f}%)")
        print(f"    cu voce: {n_s_voce} ({n_s_voce/n_s*100:.1f}%)")
        print(f"    cu ORICE: {n_s_orice} ({n_s_orice/n_s*100:.1f}%)")

    # 6. top pattern-uri individuale
    toate_patterns = []
    for col in ["patterns_prefix", "patterns_verbe", "patterns_voce"]:
        for lst in df_cls1[col]:
            toate_patterns.extend(lst)
    counter_patterns = Counter(toate_patterns)
    print(f"\n--- Top pattern-uri individuale ---")
    for pattern, n in counter_patterns.most_common(15):
        print(f"  {pattern}: {n}")

    # 7. exemple concrete per categorie (3-5 per categorie)
    exemple = {
        "prefix_factchecking": [],
        "verbe_demascare": [],
        "voce_jurnalistica": [],
        "fara_pattern": [],
    }
    for _, r in df_cls1.iterrows():
        titlu = str(r["titlu"])[:150]
        sursa = r["sursa_site"]
        if r["patterns_prefix"] and len(exemple["prefix_factchecking"]) < 10:
            exemple["prefix_factchecking"].append(f"[{sursa}] {titlu}")
        elif r["patterns_verbe"] and len(exemple["verbe_demascare"]) < 10:
            exemple["verbe_demascare"].append(f"[{sursa}] {titlu}")
        elif r["patterns_voce"] and len(exemple["voce_jurnalistica"]) < 10:
            exemple["voce_jurnalistica"].append(f"[{sursa}] {titlu}")
        elif not r["are_orice_pattern"] and len(exemple["fara_pattern"]) < 10:
            exemple["fara_pattern"].append(f"[{sursa}] {titlu}")

    print(f"\n--- Exemple per categorie ---")
    for categ, lista in exemple.items():
        print(f"\n  [{categ}]")
        for ex in lista[:5]:
            print(f"    {ex}")

    # 8. recomandare automata
    print(f"\n--- Recomandare ---")
    pct_orice = n_cu_orice / n_total * 100
    if pct_orice > 50:
        recomandare = (
            f"❌ NU INCLUDE TITLURILE. {pct_orice:.1f}% au pattern de "
            f"framing jurnalistic — risc mare de contaminare."
        )
        decizie = "exclude_toate"
    elif pct_orice > 20:
        recomandare = (
            f"⚠️  CURĂȚARE SELECTIVĂ: {pct_orice:.1f}% au pattern. "
            f"Opțiune: exclude titlurile cu pattern, păstrează restul, "
            f"sau normalizează prefixele."
        )
        decizie = "curatare_selectiva"
    else:
        recomandare = (
            f"✓ INCLUDE TITLURILE. Doar {pct_orice:.1f}% au pattern, "
            f"risc mic. Opțional: normalizează cele {n_cu_prefix} cu prefix."
        )
        decizie = "include_cu_curatare_minima"
    print(f"  {recomandare}")

    # 9. salvare raport markdown
    CALE_OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    linii = [
        "# Audit titluri cls1 — decizie includere în corpus propagandistic",
        "",
        f"**Articole analizate:** {n_total} (train + val)",
        "",
        "## Sumar global",
        "",
        "| Categorie | Nr. titluri | % din total |",
        "|---|---|---|",
        f"| Cu prefix fact-checking | {n_cu_prefix} | {n_cu_prefix/n_total*100:.1f}% |",
        f"| Cu verbe de demascare | {n_cu_verbe} | {n_cu_verbe/n_total*100:.1f}% |",
        f"| Cu voce jurnalistică | {n_cu_voce} | {n_cu_voce/n_total*100:.1f}% |",
        f"| Cu ORICE pattern | {n_cu_orice} | {n_cu_orice/n_total*100:.1f}% |",
        f"| **Fără niciun pattern** | **{n_total - n_cu_orice}** | "
        f"**{(n_total - n_cu_orice)/n_total*100:.1f}%** |",
        "",
        "## Breakdown per sursă",
        "",
        "| Sursă | Total | Cu prefix | Cu verbe | Cu voce | Cu ORICE | % ORICE |",
        "|---|---|---|---|---|---|---|",
    ]
    for sursa, d in per_sursa.items():
        linii.append(
            f"| {sursa} | {d['total']} | {d['cu_prefix']} | "
            f"{d['cu_verbe']} | {d['cu_voce']} | {d['cu_orice']} | "
            f"{d['pct_orice']:.1f}% |"
        )

    linii += [
        "",
        "## Top pattern-uri individuale",
        "",
        "| Pattern | Nr. titluri |",
        "|---|---|",
    ]
    for pattern, n in counter_patterns.most_common(20):
        linii.append(f"| `{pattern}` | {n} |")

    # exemple
    linii += ["", "## Exemple concrete", ""]
    titluri_categ = {
        "prefix_factchecking": "### Titluri cu prefix fact-checking (RISC MARE)",
        "verbe_demascare": "### Titluri cu verbe de demascare (RISC MEDIU)",
        "voce_jurnalistica": "### Titluri cu voce jurnalistică (RISC SCĂZUT)",
        "fara_pattern": "### Titluri FĂRĂ niciun pattern (OK pentru corpus)",
    }
    for cheie, heading in titluri_categ.items():
        linii.append(heading)
        linii.append("")
        for ex in exemple[cheie][:5]:
            linii.append(f"- {ex}")
        linii.append("")

    linii += [
        "## Recomandare",
        "",
        f"**Decizie automată:** `{decizie}`",
        "",
        recomandare,
        "",
        "## Opțiuni de continuare",
        "",
        "1. **Exclude toate titlurile** — cel mai sigur, cost = pierdem ~15-20% din propoziții.",
        "2. **Exclude selectiv** (doar cele cu pattern) — compromis rezonabil.",
        "3. **Normalizare prefixe** — tăiem prefixele gen „Narațiune falsă:” și păstrăm restul titlului.",
        "",
        "*Generat automat.*",
    ]
    CALE_OUT_MD.write_text("\n".join(linii), encoding="utf-8")
    print(f"\n✅ Raport: {CALE_OUT_MD}")

    # 10. salvare JSON cu detalii
    CALE_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_total": int(n_total),
        "n_cu_prefix": int(n_cu_prefix),
        "n_cu_verbe": int(n_cu_verbe),
        "n_cu_voce": int(n_cu_voce),
        "n_cu_orice": int(n_cu_orice),
        "pct_orice": float(pct_orice),
        "per_sursa": per_sursa,
        "top_patterns": dict(counter_patterns.most_common(30)),
        "decizie_automata": decizie,
        "recomandare": recomandare,
    }
    CALE_OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"✅ JSON: {CALE_OUT_JSON}")


if __name__ == "__main__":
    main()
