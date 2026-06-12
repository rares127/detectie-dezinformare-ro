"""
Selectare subset v3 pentru benchmark embeddings (Modulul 3).

Schimbari fata de v2:
    - cls0: ~75 articole din test_cls0_external_v2.csv (dupa scraping).
    - cls1: TOT cls1 din dataset_v2_test.csv (~110 articole), fara sub-sampling.
    - Total: ~185 articole, aliniate temporal cu corpusul.

Input:
    - data/raw/test_cls0_external_v2.csv (produs de scraping_cls0_extern.py)
    - data/processed/dataset_v2_test.csv
    - data/processed/propozitii_cls0_corpus.parquet (validare anti-contaminare)

Output:
    - data/processed/subset_benchmark_v3.parquet
    - findings/subset_benchmark_v3.md

Rulare:
    python scripts/selecteaza_subset_benchmark_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import stanza

CALE_CLS0_EXTERN = Path("data/raw/test_cls0_external_v2.csv")
CALE_DATASET_TEST = Path("data/processed/dataset_v2_test.csv")
CALE_CORPUS = Path("data/processed/propozitii_cls0_corpus.parquet")

CALE_OUT_PARQUET = Path("data/processed/subset_benchmark_v3.parquet")
CALE_OUT_MD = Path("findings/subset_benchmark_v3.md")

LUNG_MIN_CUVINTE = 7
LUNG_MAX_CUVINTE = 54


def incarca_stanza() -> stanza.Pipeline:
    """Pipeline Stanza pentru segmentare romana."""
    stanza.download("ro", processors="tokenize", verbose=False)
    return stanza.Pipeline(lang="ro", processors="tokenize",
                           tokenize_no_ssplit=False, verbose=False)


def valideaza_cls0_extern(df_extern: pd.DataFrame, df_corpus: pd.DataFrame) -> None:
    """Validari anti-contaminare si consistenta."""
    erori = []
    if not (df_extern["label_numeric"] == 0).all():
        erori.append("Nu toate articolele cls0 extern au label_numeric = 0")
    if not df_extern["id"].is_unique:
        erori.append("id-uri duplicate în fișierul extern")

    ids_corpus = set(df_corpus["articol_id"].astype(str))
    ids_extern = set(df_extern["id"].astype(str))
    suprapuneri = ids_extern & ids_corpus
    if suprapuneri:
        erori.append(
            f"CONTAMINARE: {len(suprapuneri)} id-uri suprapuse cu corpus: "
            f"{sorted(suprapuneri)[:5]}..."
        )

    nw = df_extern["stire_citata"].fillna("").str.split().str.len()
    prea_scurte = (nw < 100).sum()
    if prea_scurte > 0:
        erori.append(f"{prea_scurte} articole cu < 100 cuvinte")

    if erori:
        print("❌ VALIDARE EȘUATĂ:", file=sys.stderr)
        for e in erori:
            print(f"   - {e}", file=sys.stderr)
        sys.exit(1)

    print("✓ Validare cls0 extern trecută:")
    print(f"  - {len(df_extern)} articole")
    print(f"  - {df_extern['sursa_site'].nunique()} surse distincte")
    print(f"  - Distribuție pe ani:")
    for an, n in df_extern["an"].value_counts().sort_index().items():
        print(f"      {an}: {n}")


def segmenteaza_articol(nlp, articol_id: str, titlu: str, stire_citata: str,
                        sursa: str, label: int) -> list[dict]:
    """Segmenteaza articol in propozitii cu Stanza."""
    titlu = (titlu or "").strip()
    corp = (stire_citata or "").strip()
    if titlu and titlu[-1] not in ".!?":
        titlu = titlu + "."
    text_integral = (titlu + " " + corp).strip() if titlu else corp
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
            "label_numeric": label,
            "pozitie_in_articol": poz,
            "propozitie": txt,
            "nr_cuvinte": nw,
            "nr_caractere": len(txt),
        })
    return propozitii


def main() -> None:
    """Pipeline complet: validare → selectie → segmentare → save."""
    print("=" * 70)
    print("SELECTARE SUBSET BENCHMARK v3 — scale up")
    print("=" * 70)

    for cale in [CALE_CLS0_EXTERN, CALE_DATASET_TEST, CALE_CORPUS]:
        if not cale.exists():
            raise FileNotFoundError(f"Lipsește: {cale}")

    # cls0 extern (scraped)
    print(f"\n--- cls0 extern: {CALE_CLS0_EXTERN} ---")
    df_cls0 = pd.read_csv(CALE_CLS0_EXTERN)
    df_corpus = pd.read_parquet(CALE_CORPUS)
    valideaza_cls0_extern(df_cls0, df_corpus)

    # cls1 complet din test set
    print(f"\n--- cls1 TOT din test set: {CALE_DATASET_TEST} ---")
    df_test = pd.read_csv(CALE_DATASET_TEST)
    df_cls1 = df_test[df_test["label_numeric"] == 1].copy()
    print(f"Total cls1 din test set: {len(df_cls1)}")
    print(f"Distribuție per sursă:")
    for s, n in df_cls1["sursa_site"].value_counts().items():
        print(f"  {s}: {n}")
    print(f"Distribuție per an:")
    for an, n in df_cls1["an"].value_counts().sort_index().items():
        print(f"  {an}: {n}")

    # combin
    df_cls0_comp = df_cls0[["id", "titlu", "stire_citata", "sursa_site", "label_numeric"]].copy()
    df_cls1_comp = df_cls1[["id", "titlu", "stire_citata", "sursa_site", "label_numeric"]].copy()
    df_subset = pd.concat([df_cls0_comp, df_cls1_comp], ignore_index=True)
    print(f"\nSubset total: {len(df_subset)} articole "
          f"({len(df_cls0_comp)} cls0 + {len(df_cls1_comp)} cls1)")

    # segmentare
    print(f"\n--- Segmentare cu Stanza ---")
    nlp = incarca_stanza()
    toate = []
    for i, rand in df_subset.iterrows():
        props = segmenteaza_articol(
            nlp=nlp,
            articol_id=str(rand["id"]),
            titlu=str(rand.get("titlu", "")),
            stire_citata=str(rand.get("stire_citata", "")),
            sursa=str(rand["sursa_site"]),
            label=int(rand["label_numeric"]),
        )
        toate.extend(props)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(df_subset)} articole segmentate")

    df_prop = pd.DataFrame(toate)
    total_brut = len(df_prop)
    print(f"Propoziții brute: {total_brut}")

    # filtrare lungime
    inainte = len(df_prop)
    df_prop = df_prop[
        (df_prop["nr_cuvinte"] >= LUNG_MIN_CUVINTE)
        & (df_prop["nr_cuvinte"] <= LUNG_MAX_CUVINTE)
    ].reset_index(drop=True)
    eliminate = inainte - len(df_prop)
    print(f"Filtru [{LUNG_MIN_CUVINTE}, {LUNG_MAX_CUVINTE}]w: "
          f"eliminate {eliminate}, rămase {len(df_prop)}")

    # save
    CALE_OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_prop.to_parquet(CALE_OUT_PARQUET, index=False)
    print(f"\n✅ Salvat: {CALE_OUT_PARQUET}")

    # raport
    CALE_OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    linii = [
        "# Subset benchmark v3 — raport selecție",
        "",
        f"**Reparație față de v2:** scale up de la 30 la ~{len(df_subset)} "
        "articole, cu aliniere temporală corpus↔cls0.",
        "",
        "## Compoziție",
        "",
        "| Categorie | Nr. articole |",
        "|---|---|",
        f"| cls0 (extern scraped + manual) | {len(df_cls0_comp)} |",
        f"| cls1 (tot test set) | {len(df_cls1_comp)} |",
        f"| **Total** | **{len(df_subset)}** |",
        "",
        "## Distribuție temporală cls0 vs cls1",
        "",
        "| An | cls0 ext | cls1 test | Propoziții corpus |",
        "|---|---|---|---|",
    ]
    per_an_cls0 = df_cls0["an"].value_counts().to_dict()
    per_an_cls1 = df_cls1["an"].value_counts().to_dict()
    per_an_corp = df_corpus["an"].value_counts().to_dict()
    for an in sorted(set(per_an_cls0) | set(per_an_cls1) | set(per_an_corp)):
        linii.append(
            f"| {an} | {per_an_cls0.get(an, 0)} | "
            f"{per_an_cls1.get(an, 0)} | {per_an_corp.get(an, 0)} |"
        )
    linii += [
        "",
        "## Propoziții",
        "",
        f"- Brute: **{total_brut}**",
        f"- După filtru [{LUNG_MIN_CUVINTE}, {LUNG_MAX_CUVINTE}]w: **{len(df_prop)}**",
        "",
        "*Generat automat.*",
    ]
    CALE_OUT_MD.write_text("\n".join(linii), encoding="utf-8")
    print(f"✅ Raport: {CALE_OUT_MD}")


if __name__ == "__main__":
    main()
