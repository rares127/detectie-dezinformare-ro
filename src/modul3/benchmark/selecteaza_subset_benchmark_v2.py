"""
Selectare subset pentru benchmark embeddings v2 (Modulul 3, Pasul 2 reparat).

Schimbari fata de v1:
    - cls0 (15 articole) vin dintr-un fisier EXTERN (`test_cls0_external.csv`),
      cu articole din surse NE-vazute in corpus (Pro TV, HotNews, etc.).
      Asta elimina contaminarea train/test descoperita in v1.
    - cls1 (15 articole) extins la 12 Veridica + 3 Stopfals, pastrand
      proportia distributiei reale (~80/20).
    - Validare explicita: nici un articol cls0 din subset nu trebuie sa
      aiba articol_id prezent in corpus.

Stratificare:
    - cls0 (15): toate din test_cls0_external.csv (stratificate pe sursa
      dupa cum au fost colectate manual)
    - cls1 (15): 12 veridica.ro + 3 stopfals.md din dataset_v2_test.csv

Input:
    - data/raw/test_cls0_external.csv            (colectat manual, 15 articole)
    - data/processed/dataset_v2_test.csv         (pentru cls1)
    - data/processed/propozitii_cls0_corpus.parquet (pentru validarea anti-contaminare)

Output:
    - data/processed/subset_benchmark_v2.parquet
    - findings/subset_benchmark_v2.md

Seed: 42.

Rulare:
    python scripts/selecteaza_subset_benchmark_v2.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import stanza

# -----------------------------------------------------------------------------
# Configuratie cablata
# -----------------------------------------------------------------------------
CALE_CLS0_EXTERN = Path("data/raw/test_cls0_external.csv")
CALE_DATASET_TEST = Path("data/processed/dataset_v2_test.csv")
CALE_CORPUS = Path("data/processed/propozitii_cls0_corpus.parquet")

CALE_OUT_PARQUET = Path("data/processed/subset_benchmark_v2.parquet")
CALE_OUT_MD = Path("findings/subset_benchmark_v2.md")

SEED = 42
LUNG_MIN_CUVINTE = 7
LUNG_MAX_CUVINTE = 54

# Cote cls1 (cls0 vine integral din fisierul extern, fara sampling)
COTE_CLS1 = {
    ("veridica.ro", 1): 12,
    ("stopfals.md", 1): 3,
}
NR_CLS0_ASTEPTAT = 15


# -----------------------------------------------------------------------------
# Functii utilitare
# -----------------------------------------------------------------------------
def incarca_stanza() -> stanza.Pipeline:
    """Incarca pipeline-ul Stanza pentru romana, doar segmentare."""
    stanza.download("ro", processors="tokenize", verbose=False)
    return stanza.Pipeline(
        lang="ro",
        processors="tokenize",
        tokenize_no_ssplit=False,
        verbose=False,
    )


def valideaza_cls0_extern(df_extern: pd.DataFrame, df_corpus: pd.DataFrame) -> None:
    """Validari pe articolele cls0 externe.

    - Toate au label_numeric = 0.
    - Toate au id-uri unice, cu prefix `ext_`.
    - Niciun articol_id nu exista in corpus (anti-contaminare critica).
    - Nr. articole = 15 (NR_CLS0_ASTEPTAT).
    - Toate au stire_citata ne-vida si minim 100 cuvinte.
    """
    erori = []
    if len(df_extern) != NR_CLS0_ASTEPTAT:
        erori.append(
            f"Număr articole: {len(df_extern)}, așteptat {NR_CLS0_ASTEPTAT}"
        )
    if not (df_extern["label_numeric"] == 0).all():
        erori.append("Nu toate articolele au label_numeric = 0")
    if not df_extern["id"].is_unique:
        erori.append("id-uri duplicate în fișierul extern")
    if not df_extern["id"].str.startswith("ext_").all():
        erori.append("Nu toate id-urile au prefix `ext_`")

    # validare anti-contaminare: articol_id din corpus e tipul str
    ids_corpus = set(df_corpus["articol_id"].astype(str))
    ids_extern = set(df_extern["id"].astype(str))
    suprapuneri = ids_extern & ids_corpus
    if suprapuneri:
        erori.append(
            f"CONTAMINARE! {len(suprapuneri)} id-uri cls0 externe există "
            f"și în corpus: {sorted(suprapuneri)[:5]}..."
        )

    # lungime minima stire_citata
    nw = df_extern["stire_citata"].fillna("").str.split().str.len()
    prea_scurte = df_extern[nw < 100]
    if len(prea_scurte) > 0:
        erori.append(
            f"{len(prea_scurte)} articole au < 100 cuvinte în stire_citata"
        )

    if erori:
        print("\n❌ VALIDARE EȘUATĂ:", file=sys.stderr)
        for e in erori:
            print(f"   - {e}", file=sys.stderr)
        sys.exit(1)

    print("✓ Validare cls0 extern trecută:")
    print(f"  - {len(df_extern)} articole, toate cu label_numeric=0")
    print(f"  - {df_extern['sursa_site'].nunique()} surse distincte")
    print(f"  - Zero suprapuneri cu corpus")
    print(f"  - Lungime medie stire_citata: {nw.mean():.0f} cuvinte")


def selecteaza_cls1_stratificat(df_test: pd.DataFrame) -> pd.DataFrame:
    """Selecteaza cele 15 articole cls1 conform cotelor, cu seed reproductibil."""
    random.seed(SEED)
    bucati = []
    rapoarte = []
    for (sursa, label), cota in COTE_CLS1.items():
        pool = df_test[
            (df_test["sursa_site"] == sursa) & (df_test["label_numeric"] == label)
        ].copy()
        disponibile = len(pool)
        if disponibile == 0:
            rapoarte.append(
                f"  ⚠️  {sursa} label={label}: 0 disponibile — cota nu poate fi respectată"
            )
            continue
        cota_efectiva = min(cota, disponibile)
        if cota_efectiva < cota:
            rapoarte.append(
                f"  ⚠️  {sursa} label={label}: cerut {cota}, disponibil {disponibile}, "
                f"luat {cota_efectiva}"
            )
        ales = pool.sample(n=cota_efectiva, random_state=SEED)
        bucati.append(ales)
        rapoarte.append(f"  ✓ {sursa} label={label}: {cota_efectiva}/{cota} articole")

    rezultat = pd.concat(bucati, ignore_index=True)
    print("Selecție cls1 stratificată:")
    for r in rapoarte:
        print(r)
    return rezultat


def segmenteaza_articol(nlp: stanza.Pipeline, articol_id: str, titlu: str,
                        stire_citata: str, sursa: str, label: int) -> list[dict]:
    """Segmenteaza un articol in propozitii cu Stanza.

    Input canonic: titlu + stire_citata. Consistent cu conventia proiectului
    pentru clasificator.
    """
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    """Pipeline complet: validare → selectie → segmentare → save."""
    print("=" * 70)
    print("SELECTARE SUBSET BENCHMARK v2 — CU cls0 EXTERN")
    print("=" * 70)

    # 1. verific prezenta fisierelor
    for cale in [CALE_CLS0_EXTERN, CALE_DATASET_TEST, CALE_CORPUS]:
        if not cale.exists():
            raise FileNotFoundError(f"Lipsește: {cale}")

    # 2. incarc cls0 extern si validez
    print(f"\n--- Încarc cls0 extern: {CALE_CLS0_EXTERN} ---")
    df_cls0 = pd.read_csv(CALE_CLS0_EXTERN)
    df_corpus = pd.read_parquet(CALE_CORPUS)
    valideaza_cls0_extern(df_cls0, df_corpus)

    # 3. incarc cls1 din test set standard
    print(f"\n--- Încarc cls1 din test set: {CALE_DATASET_TEST} ---")
    df_test = pd.read_csv(CALE_DATASET_TEST)
    df_cls1 = selecteaza_cls1_stratificat(df_test)
    print(f"Total cls1 selectat: {len(df_cls1)} articole")

    # 4. combin
    print(f"\n--- Combinare ---")
    df_cls0_comp = df_cls0[["id", "titlu", "stire_citata", "sursa_site", "label_numeric"]].copy()
    df_cls1_comp = df_cls1[["id", "titlu", "stire_citata", "sursa_site", "label_numeric"]].copy()
    df_subset = pd.concat([df_cls0_comp, df_cls1_comp], ignore_index=True)
    print(f"Subset total: {len(df_subset)} articole "
          f"({len(df_cls0_comp)} cls0 + {len(df_cls1_comp)} cls1)")

    # 5. segmentare Stanza
    print(f"\n--- Segmentare cu Stanza (ro) ---")
    nlp = incarca_stanza()
    toate_propozitiile = []
    for _, rand in df_subset.iterrows():
        props = segmenteaza_articol(
            nlp=nlp,
            articol_id=str(rand["id"]),
            titlu=str(rand.get("titlu", "")),
            stire_citata=str(rand.get("stire_citata", "")),
            sursa=str(rand["sursa_site"]),
            label=int(rand["label_numeric"]),
        )
        toate_propozitiile.extend(props)

    df_prop = pd.DataFrame(toate_propozitiile)
    total_brut = len(df_prop)
    print(f"Propoziții brute: {total_brut}")

    # 6. filtrare lungime [7, 54]
    inainte = len(df_prop)
    df_prop = df_prop[
        (df_prop["nr_cuvinte"] >= LUNG_MIN_CUVINTE)
        & (df_prop["nr_cuvinte"] <= LUNG_MAX_CUVINTE)
    ].reset_index(drop=True)
    eliminate = inainte - len(df_prop)
    print(f"Filtru [{LUNG_MIN_CUVINTE}, {LUNG_MAX_CUVINTE}]w: "
          f"eliminate {eliminate}, rămase {len(df_prop)}")

    # 7. save parquet
    CALE_OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_prop.to_parquet(CALE_OUT_PARQUET, index=False)
    print(f"\n✅ Salvat: {CALE_OUT_PARQUET}")

    # 8. raport markdown
    CALE_OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    linii = [
        "# Subset benchmark v2 — raport selecție",
        "",
        f"**Seed:** {SEED}",
        f"**Reparație față de v1:** cls0 din surse externe (anti-contaminare).",
        "",
        "## Compoziție subset",
        "",
        "| Categorie | Sursă | Nr. articole |",
        "|---|---|---|",
    ]
    for sursa, n in df_cls0["sursa_site"].value_counts().items():
        linii.append(f"| cls0 (extern) | {sursa} | {n} |")
    for (sursa, label), cota in COTE_CLS1.items():
        realizat = len(df_cls1[(df_cls1["sursa_site"] == sursa)
                               & (df_cls1["label_numeric"] == label)])
        linii.append(f"| cls1 | {sursa} | {realizat} |")
    linii.append(f"| **Total** | | **{len(df_subset)}** |")

    linii += [
        "",
        "## Validare anti-contaminare",
        "",
        f"- Articole cls0 cu id în corpus: **0** ✓",
        f"- Surse cls0 externe: {df_cls0['sursa_site'].nunique()}",
        "",
        "## Statistici propoziții",
        "",
        f"- Brute (post-Stanza): **{total_brut}**",
        f"- După filtru [{LUNG_MIN_CUVINTE}, {LUNG_MAX_CUVINTE}]w: **{len(df_prop)}**",
        f"- Eliminate: **{eliminate}** ({eliminate / total_brut * 100:.1f}%)",
        "",
    ]

    per_articol = df_prop.groupby(["articol_id", "sursa_site", "label_numeric"]).size()
    linii += [
        "## Distribuție propoziții per articol",
        "",
        f"- Min: **{per_articol.min()}**",
        f"- Mediană: **{per_articol.median():.0f}**",
        f"- Max: **{per_articol.max()}**",
        f"- Medie: **{per_articol.mean():.1f}**",
        "",
        "## Distribuție propoziții per sursă × clasă",
        "",
        "| Sursă | Label | Nr. propoziții |",
        "|---|---|---|",
    ]
    for (sursa, label), n in df_prop.groupby(["sursa_site", "label_numeric"]).size().items():
        linii.append(f"| {sursa} | {label} | {n} |")
    linii.append("")
    linii.append("*Generat automat.*")

    CALE_OUT_MD.write_text("\n".join(linii), encoding="utf-8")
    print(f"✅ Raport: {CALE_OUT_MD}")


if __name__ == "__main__":
    main()
