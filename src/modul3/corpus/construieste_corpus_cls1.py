"""
Construire corpus propagandistic cls1 — Modulul 3, Optiunea A.

Scop:
    Construieste corpusul de referinta cls1 (propozitii propagandistice
    din articolele Veridica + Stopfals), pentru a testa ipoteza:
    „dezinformarea pro-Kremlin foloseste tipare semantice recurente
    pe care un model de similaritate le poate detecta".

    Simetric cu corpusul cls0, dar cu conventie inversata in benchmark:
    scor mai mare cu corpus cls1 → articolul evaluat seamana cu naratiuni
    pro-Kremlin cunoscute → mai probabil dezinformare.

Surse:
    - Veridica.ro — citatele pro-Kremlin din coloana `stire_citata`
      (NU text_curat — acela contine analiza jurnalistului, opusul propagandei)
    - Stopfals.md — idem

Strategia titlurilor (din audit_titluri_cls1):
    - 89.3% din titluri sunt naratiuni propagandistice brute — le pastram.
    - ~66 titluri Veridica au prefix „PROPAGANDA DE RAZBOI:" — taiem prefixul.
    - 2 titluri Stopfals cu pattern jurnalistic — excludem din prudenta.
    - Titlul concatenat cu stire_citata, la fel ca pipeline-ul cls0.

Validari critice:
    - Articolele cls1 folosite sunt DOAR din train+val (NU din test).
    - Verifica explicit ca niciun id nu apare in dataset_v2_test.csv.
    - Filtrare lungime [7, 54]w pentru consistenta cu cls0.

Input:
    - data/processed/dataset_v2_train.csv
    - data/processed/dataset_v2_val.csv
    - data/processed/dataset_v2_test.csv (pentru validare anti-contaminare)

Output:
    - data/processed/propozitii_cls1_raw.parquet (inainte de curatare)
    - data/processed/propozitii_cls1_corpus.parquet (final, filtrat)
    - findings/construire_corpus_cls1.md

Rulare:
    python scripts/construieste_corpus_cls1.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import stanza


# -----------------------------------------------------------------------------
# Configuratie
# -----------------------------------------------------------------------------
CALE_TRAIN = Path("data/processed/dataset_v2_train.csv")
CALE_VAL = Path("data/processed/dataset_v2_val.csv")
CALE_TEST = Path("data/processed/dataset_v2_test.csv")

CALE_OUT_RAW = Path("data/processed/propozitii_cls1_raw.parquet")
CALE_OUT_CORPUS = Path("data/processed/propozitii_cls1_corpus.parquet")
CALE_OUT_MD = Path("findings/construire_corpus_cls1.md")

LUNG_MIN_CUVINTE = 7
LUNG_MAX_CUVINTE = 54


# Prefixe Veridica de taiat (detectate in audit)
# Regex-uri case-insensitive pe startul titlului.
PREFIXE_VERIDICA_DE_TAIAT = [
    r"^propagand[ăa]\s+de\s+r[ăa]zboi\s*:\s*",
    # adaug si variante mai conservative in caz ca sunt formulari similare
    r"^propagand[ăa]\s+rus[ăa]\s*:\s*",
    r"^propagand[ăa]\s*:\s*",
]

# Titluri Stopfals cu pattern — le excludem din corpus (din audit)
# Le identificam prin prezenta pattern-urilor specifice.
PATTERNS_EXCLUDE_TITLU = [
    r"^fals\s*:\s*",
    r"^context\s*:\s*",
    r"\bfact[-\s]?check\b",
]


def incarca_stanza() -> stanza.Pipeline:
    """Incarca pipeline-ul Stanza pentru romana."""
    stanza.download("ro", processors="tokenize", verbose=False)
    return stanza.Pipeline(
        lang="ro", processors="tokenize",
        tokenize_no_ssplit=False, verbose=False,
    )


def normalizeaza_titlu(titlu: str) -> tuple[str, str]:
    """Normalizeaza titlul conform strategiei din audit.

    Returneaza (titlu_normalizat, actiune) unde actiune este:
    - 'ok' — titlul e OK, foloseste-l integral
    - 'prefix_taiat' — am taiat un prefix Veridica
    - 'exclude' — titlul contine pattern de excludere, NU folosi
    """
    if not isinstance(titlu, str):
        return "", "exclude"

    txt = titlu.strip()
    if not txt:
        return "", "exclude"

    # verificare exclude (Stopfals meta-jurnalistic)
    for pattern in PATTERNS_EXCLUDE_TITLU:
        if re.search(pattern, txt, flags=re.IGNORECASE):
            return "", "exclude"

    # verificare prefix Veridica — taiem prefixul
    for pattern in PREFIXE_VERIDICA_DE_TAIAT:
        match = re.match(pattern, txt, flags=re.IGNORECASE)
        if match:
            txt_nou = txt[match.end():].strip()
            if txt_nou:  # doar daca ramane ceva dupa taiere
                return txt_nou, "prefix_taiat"
            else:
                return "", "exclude"

    # titlu OK
    return txt, "ok"


def segmenteaza_articol(nlp: stanza.Pipeline, articol_id: str, titlu_norm: str,
                        stire_citata: str, sursa: str, an: int) -> list[dict]:
    """Segmenteaza articol in propozitii cu Stanza.

    Input: titlu_norm (deja curatat) + stire_citata.
    """
    titlu_norm = (titlu_norm or "").strip()
    corp = (stire_citata or "").strip()
    if titlu_norm and titlu_norm[-1] not in ".!?":
        titlu_norm = titlu_norm + "."
    text_integral = (titlu_norm + " " + corp).strip() if titlu_norm else corp

    if not text_integral:
        return []

    doc = nlp(text_integral)
    propozitii = []
    for poz, sent in enumerate(doc.sentences):
        txt = sent.text.strip()
        nw = len(txt.split())
        propozitii.append({
            "articol_id": articol_id,
            "sursa_site": sursa,
            "an": an,
            "pozitie_in_articol": poz,
            "propozitie": txt,
            "nr_cuvinte": nw,
            "nr_caractere": len(txt),
        })
    return propozitii


def main() -> None:
    """Pipeline complet construire corpus cls1."""
    print("=" * 70)
    print("CONSTRUIRE CORPUS cls1 (PROPAGANDISTIC) — Modulul 3 Opțiunea A")
    print("=" * 70)

    # 1. verificare fisiere
    for cale in [CALE_TRAIN, CALE_VAL, CALE_TEST]:
        if not cale.exists():
            raise FileNotFoundError(f"Lipsește: {cale}")

    # 2. incarcare train+val cls1
    df_train = pd.read_csv(CALE_TRAIN)
    df_val = pd.read_csv(CALE_VAL)
    df_test = pd.read_csv(CALE_TEST)

    df_all_trainval = pd.concat([df_train, df_val], ignore_index=True)
    df_cls1 = df_all_trainval[df_all_trainval["label_numeric"] == 1].copy()

    print(f"\nArticole cls1 din train+val: {len(df_cls1)}")
    print(f"Distribuție per sursă:")
    for sursa, n in df_cls1["sursa_site"].value_counts().items():
        print(f"  {sursa}: {n}")
    print(f"Distribuție per an:")
    for an, n in df_cls1["an"].value_counts().sort_index().items():
        print(f"  {an}: {n}")

    # 3. VALIDARE ANTI-CONTAMINARE — critica!
    print(f"\n--- Validare anti-contaminare cu test set ---")
    ids_test = set(df_test["id"].astype(str))
    ids_cls1 = set(df_cls1["id"].astype(str))
    suprapuneri = ids_cls1 & ids_test
    if suprapuneri:
        print(f"❌ CONTAMINARE DETECTATĂ! {len(suprapuneri)} id-uri cls1 "
              f"suprapuse cu test set: {sorted(suprapuneri)[:5]}...")
        print("Nu pot continua — există risc de data leakage în benchmark.")
        sys.exit(1)
    print(f"✓ Zero suprapuneri cu test set "
          f"({len(ids_cls1)} cls1 × {len(ids_test)} test)")

    # 4. normalizare titluri
    print(f"\n--- Normalizare titluri ---")
    stats_titluri = {"ok": 0, "prefix_taiat": 0, "exclude": 0}
    df_cls1["titlu_norm"] = ""
    df_cls1["actiune_titlu"] = ""
    for idx, rand in df_cls1.iterrows():
        titlu_n, actiune = normalizeaza_titlu(rand["titlu"])
        df_cls1.at[idx, "titlu_norm"] = titlu_n
        df_cls1.at[idx, "actiune_titlu"] = actiune
        stats_titluri[actiune] += 1
    print(f"Titluri OK (folosite direct): {stats_titluri['ok']}")
    print(f"Titluri cu prefix tăiat: {stats_titluri['prefix_taiat']}")
    print(f"Titluri excluse (folosim doar stire_citata): {stats_titluri['exclude']}")

    # 5. segmentare
    print(f"\n--- Segmentare cu Stanza ---")
    nlp = incarca_stanza()
    toate_prop = []
    for i, rand in df_cls1.iterrows():
        props = segmenteaza_articol(
            nlp=nlp,
            articol_id=str(rand["id"]),
            titlu_norm=str(rand.get("titlu_norm", "")),
            stire_citata=str(rand.get("stire_citata", "")),
            sursa=str(rand["sursa_site"]),
            an=int(rand["an"]) if pd.notna(rand.get("an")) else 0,
        )
        toate_prop.extend(props)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(df_cls1)} articole procesate")

    df_prop_raw = pd.DataFrame(toate_prop)
    total_brut = len(df_prop_raw)
    print(f"Propoziții brute: {total_brut}")

    # salvam raw pentru referinta
    CALE_OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    df_prop_raw.to_parquet(CALE_OUT_RAW, index=False)
    print(f"✓ Raw salvat: {CALE_OUT_RAW}")

    # 6. filtrare lungime [7, 54]
    inainte = len(df_prop_raw)
    df_prop = df_prop_raw[
        (df_prop_raw["nr_cuvinte"] >= LUNG_MIN_CUVINTE)
        & (df_prop_raw["nr_cuvinte"] <= LUNG_MAX_CUVINTE)
    ].copy().reset_index(drop=True)
    eliminate_lungime = inainte - len(df_prop)
    print(f"\nFiltru lungime [{LUNG_MIN_CUVINTE}, {LUNG_MAX_CUVINTE}]w: "
          f"eliminate {eliminate_lungime}, rămase {len(df_prop)}")

    # 7. salvare corpus final
    CALE_OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    df_prop.to_parquet(CALE_OUT_CORPUS, index=False)
    print(f"✅ Corpus cls1 salvat: {CALE_OUT_CORPUS} ({len(df_prop)} propoziții)")

    # 8. statistici finale
    print(f"\n--- Statistici corpus final ---")
    print(f"Propoziții: {len(df_prop)}")
    print(f"Per sursă:")
    for sursa, n in df_prop["sursa_site"].value_counts().items():
        print(f"  {sursa}: {n}")
    print(f"Per an:")
    for an, n in df_prop["an"].value_counts().sort_index().items():
        print(f"  {an}: {n}")
    print(f"Lungime propoziție (cuvinte):")
    nw = df_prop["nr_cuvinte"]
    print(f"  min={nw.min()}, p5={nw.quantile(0.05):.0f}, "
          f"mediană={nw.median():.0f}, p95={nw.quantile(0.95):.0f}, max={nw.max()}, "
          f"medie={nw.mean():.1f}")

    # 9. raport markdown
    linii = [
        "# Construire corpus cls1 (propagandistic) — raport",
        "",
        "**Opțiunea A** — test dacă dezinformarea pro-Kremlin folosește "
        "tipare semantice recurente detectabile prin similaritate.",
        "",
        "## Sursă și validare",
        "",
        f"- Articole cls1 din train+val: **{len(df_cls1)}**",
        f"- Validare anti-contaminare cu test set: **✓ Zero suprapuneri**",
        "",
        "## Compoziție",
        "",
        "| Sursă | Nr. articole |",
        "|---|---|",
    ]
    for sursa, n in df_cls1["sursa_site"].value_counts().items():
        linii.append(f"| {sursa} | {n} |")

    linii += [
        "",
        "## Tratament titluri (din audit)",
        "",
        "| Acțiune | Nr. articole |",
        "|---|---|",
        f"| OK, folosit integral | {stats_titluri['ok']} |",
        f"| Prefix tăiat (Veridica `PROPAGANDĂ DE RĂZBOI:`) | "
        f"{stats_titluri['prefix_taiat']} |",
        f"| Exclus (pattern meta-jurnalistic) | {stats_titluri['exclude']} |",
        "",
        "## Procesare",
        "",
        f"- Propoziții brute (post-Stanza): **{total_brut}**",
        f"- După filtru lungime [{LUNG_MIN_CUVINTE}, {LUNG_MAX_CUVINTE}]w: **{len(df_prop)}**",
        f"- Eliminate (prea scurte/lungi): **{eliminate_lungime}** "
        f"({eliminate_lungime/total_brut*100:.1f}%)",
        "",
        "## Corpus final",
        "",
        f"- **{len(df_prop)} propoziții**",
        "",
        "### Distribuție per sursă",
        "",
        "| Sursă | Nr. propoziții |",
        "|---|---|",
    ]
    for sursa, n in df_prop["sursa_site"].value_counts().items():
        linii.append(f"| {sursa} | {n} |")

    linii += [
        "",
        "### Distribuție per an",
        "",
        "| An | Nr. propoziții |",
        "|---|---|",
    ]
    for an, n in df_prop["an"].value_counts().sort_index().items():
        linii.append(f"| {an} | {n} |")

    linii += [
        "",
        "### Statistici lungime (cuvinte)",
        "",
        f"- Min: **{int(nw.min())}**",
        f"- p5: **{int(nw.quantile(0.05))}**",
        f"- Mediană: **{int(nw.median())}**",
        f"- p95: **{int(nw.quantile(0.95))}**",
        f"- Max: **{int(nw.max())}**",
        f"- Medie: **{nw.mean():.1f}**",
        "",
        "## Note importante pentru benchmark-ul v4",
        "",
        "- **Convenție scor cls1**: scor MAI MARE = articol similar cu narațiuni "
        "propagandistice cunoscute = mai probabil dezinformare.",
        "- **Direcție inversă** față de corpus cls0 (acolo scor mare = credibil).",
        "- **Opțiunea A izolat**: AUC doar pe `scor_cls1` (fără cls0). Dacă ≥0.75, "
        "viabil ca clasificator.",
        "- **Opțiunea D combinat**: AUC pe `scor_cls1 - scor_cls0`, folosind ambele "
        "corpusuri. Scorul combinat ar trebui să dea separabilitate mai bună decât "
        "oricare singur.",
        "",
        "## Recomandări pentru pași următori",
        "",
        "1. **Audit rezidual** pe propozițiile cls1 — posibil să existe zgomot specific "
        "(cookie banners, etichete vorbitor, link-uri reziduale) pe care îl curățăm "
        "cu un pipeline similar cu cel pentru cls0.",
        "2. **Benchmark v4** pe subset-ul v3 existent (167 articole) — fără recolecting.",
        "3. **Decizie finală** după benchmark: Opțiunea A sau D sau C.",
        "",
        "*Generat automat.*",
    ]
    CALE_OUT_MD.write_text("\n".join(linii), encoding="utf-8")
    print(f"\n✅ Raport: {CALE_OUT_MD}")


if __name__ == "__main__":
    main()
