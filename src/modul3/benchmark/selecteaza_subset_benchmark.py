"""
Selectare subset pentru benchmark embeddings (Modulul 3, Pasul 2).

Scop:
    Construieste un subset de 20 articole din dataset_v2_test.csv, stratificat
    pe clasa si pe sursa, segmenteaza fiecare articol in propozitii cu Stanza
    (aceeasi logica folosita la construirea corpusului cls0) si salveaza
    rezultatul intr-un parquet gata de embedat.

Stratificare:
    - cls0 (10 articole): 5 digi24.ro + 5 g4media.ro
    - cls1 (10 articole): 8 veridica.ro + 2 stopfals.md
    (reflecta distributia reala aproximativa din dataset)

Input:
    Coloana `titlu + stire_citata` (input-ul canonic al clasificatorului).
    Filtru lungime [7w, 54w] pe propozitii, identic cu filtrul aplicat
    corpusului de referinta in dedup_si_filtrare_cls0.py.

Output:
    - data/processed/subset_benchmark.parquet
    - findings/subset_benchmark.md

Seed: 42 (cablat in config).

Rulare:
    python scripts/selecteaza_subset_benchmark.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import stanza

# -----------------------------------------------------------------------------
# Configuratie cablata (toate caile sunt relative la radacina proiectului)
# -----------------------------------------------------------------------------
CALE_DATASET_TEST = Path("data/processed/dataset_v2_test.csv")
CALE_OUT_PARQUET = Path("data/processed/subset_benchmark.parquet")
CALE_OUT_MD = Path("findings/subset_benchmark.md")

SEED = 42
LUNG_MIN_CUVINTE = 7  # acelasi prag ca la corpus
LUNG_MAX_CUVINTE = 54  # acelasi prag ca la corpus

# Cote stratificate (suma cls0=10, suma cls1=10)
COTE = {
    ("digi24.ro", 0): 5,
    ("g4media.ro", 0): 5,
    ("veridica.ro", 1): 8,
    ("stopfals.md", 1): 2,
}


# -----------------------------------------------------------------------------
# Functii utilitare
# -----------------------------------------------------------------------------
def incarca_stanza() -> stanza.Pipeline:
    """Incarca pipeline-ul Stanza pentru romana, doar tokenizer + segmentare."""
    # descarca modelul daca nu e prezent local (run one-time)
    stanza.download("ro", processors="tokenize", verbose=False)
    return stanza.Pipeline(
        lang="ro",
        processors="tokenize",
        tokenize_no_ssplit=False,
        verbose=False,
    )


def selecteaza_stratificat(df_test: pd.DataFrame) -> pd.DataFrame:
    """Selecteaza articolele conform cotelor, cu seed reproductibil.

    Daca o cota nu poate fi atinsa (sursa e sub-reprezentata in test set),
    raportam explicit si luam maximul disponibil.
    """
    random.seed(SEED)
    bucati = []
    rapoarte = []
    for (sursa, label), cota in COTE.items():
        pool = df_test[
            (df_test["sursa_site"] == sursa) & (df_test["label_numeric"] == label)
        ].copy()
        disponibile = len(pool)
        if disponibile == 0:
            rapoarte.append(f"  ⚠️  {sursa} label={label}: 0 articole disponibile — cota nu poate fi respectată")
            continue
        cota_efectiva = min(cota, disponibile)
        if cota_efectiva < cota:
            rapoarte.append(
                f"  ⚠️  {sursa} label={label}: cerut {cota}, disponibil {disponibile}, luat {cota_efectiva}"
            )
        ales = pool.sample(n=cota_efectiva, random_state=SEED)
        bucati.append(ales)
        rapoarte.append(f"  ✓ {sursa} label={label}: {cota_efectiva}/{cota} articole selectate")

    rezultat = pd.concat(bucati, ignore_index=True)
    print("Selecție stratificată:")
    for r in rapoarte:
        print(r)
    return rezultat


def segmenteaza_articol(nlp: stanza.Pipeline, articol_id: str, titlu: str,
                        stire_citata: str, sursa: str, label: int) -> list[dict]:
    """Segmenteaza un articol in propozitii cu Stanza.

    Input canonic: titlu + stire_citata (conform conventiei proiectului).
    Returneaza o lista de dict-uri, fiecare cu o propozitie si metadate.
    """
    # concatenare titlu + corp, cu punct daca titlul nu se termina in punctuatie
    titlu = (titlu or "").strip()
    corp = (stire_citata or "").strip()
    if titlu and not titlu[-1] in ".!?":
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
    """Ruleaza intregul pipeline: selectie → segmentare → filtrare → save."""
    print("=" * 70)
    print("SELECTARE SUBSET BENCHMARK EMBEDDINGS")
    print("=" * 70)

    # 1. incarcare test set
    if not CALE_DATASET_TEST.exists():
        raise FileNotFoundError(
            f"Nu găsesc {CALE_DATASET_TEST}. Verifică că rulezi din rădăcina proiectului."
        )
    df_test = pd.read_csv(CALE_DATASET_TEST)
    print(f"\nTest set încărcat: {len(df_test)} articole")
    print(f"Distribuție surse × label:\n{pd.crosstab(df_test['sursa_site'], df_test['label_numeric'])}")

    # 2. selectie stratificata
    print("\n--- Selecție stratificată ---")
    df_subset = selecteaza_stratificat(df_test)
    print(f"\nTotal selectat: {len(df_subset)} articole")

    # 3. segmentare cu Stanza
    print("\n--- Segmentare cu Stanza (ro) ---")
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
    print(f"Propoziții brute din segmentare: {total_brut}")

    # 4. filtrare lungime [7, 54] cuvinte — consistent cu corpusul
    inainte = len(df_prop)
    df_prop = df_prop[
        (df_prop["nr_cuvinte"] >= LUNG_MIN_CUVINTE)
        & (df_prop["nr_cuvinte"] <= LUNG_MAX_CUVINTE)
    ].reset_index(drop=True)
    eliminate = inainte - len(df_prop)
    print(f"Filtru lungime [{LUNG_MIN_CUVINTE}, {LUNG_MAX_CUVINTE}]w: "
          f"eliminate {eliminate}, rămase {len(df_prop)}")

    # 5. save parquet
    CALE_OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_prop.to_parquet(CALE_OUT_PARQUET, index=False)
    print(f"\n✅ Salvat: {CALE_OUT_PARQUET} ({len(df_prop)} propoziții)")

    # 6. raport markdown
    CALE_OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    linii = [
        "# Subset benchmark embeddings — raport selecție",
        "",
        f"**Seed:** {SEED}",
        f"**Sursă:** `{CALE_DATASET_TEST}`",
        "",
        "## Cote cerute vs realizate",
        "",
        "| Sursă | Label | Cerut | Realizat |",
        "|---|---|---|---|",
    ]
    for (sursa, label), cota in COTE.items():
        realizat = len(df_subset[(df_subset["sursa_site"] == sursa)
                                 & (df_subset["label_numeric"] == label)])
        linii.append(f"| {sursa} | {label} | {cota} | {realizat} |")

    linii += [
        "",
        f"**Total articole:** {len(df_subset)}",
        "",
        "## Statistici propoziții",
        "",
        f"- Brute (post-Stanza): **{total_brut}**",
        f"- După filtru lungime [{LUNG_MIN_CUVINTE}, {LUNG_MAX_CUVINTE}]w: **{len(df_prop)}**",
        f"- Eliminate: **{eliminate}** ({eliminate / total_brut * 100:.1f}%)",
        "",
        "## Distribuție propoziții per articol",
        "",
    ]
    per_articol = df_prop.groupby(["articol_id", "sursa_site", "label_numeric"]).size()
    linii.append(f"- Min: **{per_articol.min()}**")
    linii.append(f"- Mediană: **{per_articol.median():.0f}**")
    linii.append(f"- Max: **{per_articol.max()}**")
    linii.append(f"- Medie: **{per_articol.mean():.1f}**")
    linii += [
        "",
        "## Distribuție per sursă × clasă (propoziții)",
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
