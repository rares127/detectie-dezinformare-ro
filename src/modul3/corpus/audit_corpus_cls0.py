"""
Modul 3 — Pasul 1.5: Audit calitate corpus cls0.

Scop: inainte de filtrare+dedup+embeddings, identificam exact ce fractiune
din cele 6.047 de propozitii este artefact (boilerplate CMS, titluri
recomandate, fragmente meta) vs continut jurnalistic real.

Abordare:
    1. Detectia pattern-urilor de boilerplate cookie/CMS (regex calibrat)
    2. Detectia blocurilor de titluri recomandate (heuristici structurale)
    3. Detectia fragmentelor meta (vorbitor-fara-citat, lead-in-uri)
    4. Esantion stratificat pe lungime pentru verificare manuala
    5. Breakdown pe sursa (digi24 vs g4media) — ipoteza: zgomotul e
       concentrat pe Digi24

Output:
    findings/audit_corpus_cls0.md  — raport cu EXEMPLE pentru fiecare
                                      categorie (decidem regulile impreuna)
    findings/audit_corpus_cls0.json — cifre brute

Input:
    data/processed/propozitii_cls0_raw.parquet (din pasul anterior)

Rulare:
    python audit_corpus_cls0.py --input data/processed/propozitii_cls0_raw.parquet
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 42
random.seed(SEED)


# ---------------------------------------------------------------------------
# Pattern-uri de boilerplate CMS / cookie
# ---------------------------------------------------------------------------
# Fiecare pattern e construit pe baza exemplelor REALE gasite in top-10
# duplicate. Folosim `re.IGNORECASE` si diacritice optionale, ca sa prindem
# si variante cu/fara diacritice (scraping inconsistent).

BOILERPLATE_PATTERNS = {
    "cookie_afisare": re.compile(
        r"set[aă]rile?\s+tale?\s+privind\s+cookie",
        re.IGNORECASE,
    ),
    "cookie_actualizare": re.compile(
        r"actualiz[ai]\s+set[aă]rile\s+(modulelor?\s+)?cookie",
        re.IGNORECASE,
    ),
    "cookie_accept": re.compile(
        r"e\s+nevoie\s+s[aă]\s+accep[tț]i\s+cookie",
        re.IGNORECASE,
    ),
    "cookie_generic": re.compile(
        r"\bcookie[-\s]?uri(le)?\b",
        re.IGNORECASE,
    ),
    "continut_afisare": re.compile(
        r"afi[sș]area?\s+con[tț]inutul(ui)?\s+din\s+aceast[aă]\s+sec[tț]iune",
        re.IGNORECASE,
    ),
    "social_media_plugin": re.compile(
        r"accep[tț]i\s+cookie[-\s]?urile?\s+social\s+media",
        re.IGNORECASE,
    ),
    "abonare_newsletter": re.compile(
        r"(aboneaz[aă][-\s]te|abonare)\s+(la\s+)?newsletter",
        re.IGNORECASE,
    ),
    "citeste_si": re.compile(
        r"^(cite[sș]te\s+[sș]i|vezi\s+[sș]i|vezi\s+aici|vezi\s+mai\s+(multe|jos))[\s:]",
        re.IGNORECASE,
    ),
    "foto_credit": re.compile(
        r"^foto\s*:\s*",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Heuristici pentru titluri recomandate
# ---------------------------------------------------------------------------

def prop_fara_punct_final(text: str) -> bool:
    """Propozitie care nu se termina cu punct/!/?/: — candidat de titlu."""
    text = text.rstrip()
    if not text:
        return False
    return text[-1] not in ".!?…:;\")"


def prop_incepe_cu_ghilimele(text: str) -> bool:
    """„Ghost Murmur”, „A fost un luptator...” — tipic titluri citate."""
    return text.lstrip().startswith(("„", "\"", "«"))


def prop_contine_multe_ghilimele_deschise(text: str, prag: int = 2) -> bool:
    """Concatenare de titluri citate: mai multe „..." intr-o singura „propozitie".

    Signal clar de scurgere din blocul „articole recomandate" — un paragraf
    real rar are 3+ titluri consecutive citate.
    """
    return text.count("„") >= prag or text.count("\"") >= prag * 2


def prop_termina_colon_fara_urmare(text: str) -> bool:
    """'Zelenski:' sau 'Putin:' — eticheta vorbitor fara citat urmator."""
    text = text.strip()
    if not text.endswith(":"):
        return False
    cuvinte = text.split()
    return len(cuvinte) <= 3  # scurt si termina cu : = cel mai probabil artefact


def prop_multi_caps_consecutive(text: str, prag: int = 4) -> bool:
    """Cuvinte multiple care incep cu majuscula — semnal de titluri lipite.

    Propozitie reala: „Ministerul Apararii al Romaniei a confirmat..." — 2 cuvinte cu majuscula.
    Titluri concatenate: „Foto: ... Submarinul Kronos, dezvoltat de o companie..." — multe entitati
    intamplatoare unite.

    Masuram fractiunea de cuvinte cu majuscula (excluzand primul).
    """
    cuvinte = text.split()
    if len(cuvinte) < 10:
        return False
    cu_majuscula = sum(
        1 for c in cuvinte[1:]
        if c and c[0].isupper() and not c.isupper()  # exclude acronime
    )
    # Daca >25% din cuvintele interne incep cu majuscula, e suspect
    return cu_majuscula / (len(cuvinte) - 1) > 0.25


# ---------------------------------------------------------------------------
# Analiza propriu-zisa
# ---------------------------------------------------------------------------

def clasifica_propozitii(df: pd.DataFrame) -> pd.DataFrame:
    """Adauga coloane booleene pentru fiecare categorie de zgomot detectat."""
    df = df.copy()

    # Boilerplate
    for nume, pattern in BOILERPLATE_PATTERNS.items():
        df[f"boilerplate_{nume}"] = df["propozitie"].apply(
            lambda t: bool(pattern.search(t))
        )

    # Orice boilerplate
    boilerplate_cols = [c for c in df.columns if c.startswith("boilerplate_")]
    df["is_boilerplate"] = df[boilerplate_cols].any(axis=1)

    # Heuristici titluri recomandate
    df["fara_punct_final"] = df["propozitie"].apply(prop_fara_punct_final)
    df["incepe_cu_ghilimele"] = df["propozitie"].apply(prop_incepe_cu_ghilimele)
    df["multe_ghilimele_interne"] = df["propozitie"].apply(prop_contine_multe_ghilimele_deschise)
    df["colon_fara_urmare"] = df["propozitie"].apply(prop_termina_colon_fara_urmare)
    df["multi_caps"] = df["propozitie"].apply(prop_multi_caps_consecutive)

    # Propozitii foarte scurte (<5 cuvinte) care nu se termina cu punct
    # = cel mai probabil titluri in liste
    df["scurta_si_fara_punct"] = (df["nr_cuvinte"] < 5) & df["fara_punct_final"]

    # Scor suspect: cate heuristici de zgomot se aprind
    heuristici_suspecte = [
        "multe_ghilimele_interne",
        "colon_fara_urmare",
        "multi_caps",
        "scurta_si_fara_punct",
    ]
    df["scor_suspect"] = df[heuristici_suspecte].sum(axis=1)

    # Categorie finala
    df["categorie_zgomot"] = "probabil_curat"
    df.loc[df["is_boilerplate"], "categorie_zgomot"] = "boilerplate_cms"
    df.loc[df["scor_suspect"] >= 2, "categorie_zgomot"] = "titluri_concatenate"
    df.loc[df["colon_fara_urmare"] & (df["scor_suspect"] >= 1), "categorie_zgomot"] = "eticheta_vorbitor"
    df.loc[df["scurta_si_fara_punct"] & ~df["is_boilerplate"], "categorie_zgomot"] = "titlu_scurt_probabil"

    return df


def esantion_pentru_verificare_manuala(
    df: pd.DataFrame, n_per_bucket: int = 15
) -> dict:
    """Esantion stratificat pe lungime pentru audit vizual uman.

    Scopul: omul sa verifice pe esantion daca heuristicile au fals pozitive
    (propozitii marcate zgomot care sunt de fapt continut real) sau fals
    negative (propozitii marcate curate care sunt de fapt artefacte).
    """
    rng = random.Random(SEED)
    bucket_map = {
        "foarte_scurte (1-4 cuvinte)": df[df["nr_cuvinte"] < 5],
        "scurte (5-14 cuvinte)": df[(df["nr_cuvinte"] >= 5) & (df["nr_cuvinte"] < 15)],
        "medii (15-34 cuvinte)": df[(df["nr_cuvinte"] >= 15) & (df["nr_cuvinte"] < 35)],
        "lungi (35-59 cuvinte)": df[(df["nr_cuvinte"] >= 35) & (df["nr_cuvinte"] < 60)],
        "foarte_lungi (>=60 cuvinte)": df[df["nr_cuvinte"] >= 60],
    }
    result = {}
    for nume, subset in bucket_map.items():
        if len(subset) == 0:
            result[nume] = []
            continue
        n = min(n_per_bucket, len(subset))
        indici = rng.sample(range(len(subset)), n)
        result[nume] = [
            {
                "sursa": subset.iloc[i]["sursa_site"],
                "nr_cuvinte": int(subset.iloc[i]["nr_cuvinte"]),
                "categorie_detectata": subset.iloc[i]["categorie_zgomot"],
                "scor_suspect": int(subset.iloc[i]["scor_suspect"]),
                "text": subset.iloc[i]["propozitie"][:300],
            }
            for i in indici
        ]
    return result


# ---------------------------------------------------------------------------
# Raport markdown
# ---------------------------------------------------------------------------

def scrie_raport(df: pd.DataFrame, esantion: dict, output_path: Path) -> None:
    total = len(df)
    lines = [
        "# Audit calitate corpus cls0",
        "",
        f"**Input:** {total:,} propoziții segmentate din articolele cls0 (Digi24 + G4Media)",
        "",
        "## 1. Overview categorii de zgomot",
        "",
        "| Categorie | Total | % din corpus | digi24 | g4media |",
        "|---|---|---|---|---|",
    ]
    for cat, df_cat in df.groupby("categorie_zgomot"):
        total_cat = len(df_cat)
        pct = 100 * total_cat / total
        digi = (df_cat["sursa_site"] == "digi24.ro").sum()
        g4m = (df_cat["sursa_site"] == "g4media.ro").sum()
        lines.append(f"| `{cat}` | {total_cat:,} | {pct:.2f}% | {digi:,} | {g4m:,} |")

    lines += [
        "",
        "## 2. Boilerplate CMS — breakdown pe pattern",
        "",
        "| Pattern | Matches | digi24 | g4media |",
        "|---|---|---|---|",
    ]
    for pattern_name in BOILERPLATE_PATTERNS:
        col = f"boilerplate_{pattern_name}"
        total_m = df[col].sum()
        digi = df[df["sursa_site"] == "digi24.ro"][col].sum()
        g4m = df[df["sursa_site"] == "g4media.ro"][col].sum()
        lines.append(f"| `{pattern_name}` | {total_m} | {digi} | {g4m} |")

    lines += [
        "",
        "### Exemple boilerplate detectat (primele 5 per pattern)",
        "",
    ]
    for pattern_name in BOILERPLATE_PATTERNS:
        col = f"boilerplate_{pattern_name}"
        matches = df[df[col]].head(5)
        if len(matches) == 0:
            continue
        lines.append(f"#### `{pattern_name}` — {df[col].sum()} matches total")
        lines.append("")
        for _, row in matches.iterrows():
            txt = row["propozitie"][:200]
            lines.append(f"- [{row['sursa_site']}] *{row['nr_cuvinte']}w*: {txt}")
        lines.append("")

    lines += [
        "## 3. Heuristici structurale",
        "",
        "| Heuristică | Matches | % | digi24 | g4media |",
        "|---|---|---|---|---|",
    ]
    for h in ["fara_punct_final", "incepe_cu_ghilimele", "multe_ghilimele_interne",
              "colon_fara_urmare", "multi_caps", "scurta_si_fara_punct"]:
        total_m = df[h].sum()
        pct = 100 * total_m / total
        digi = df[df["sursa_site"] == "digi24.ro"][h].sum()
        g4m = df[df["sursa_site"] == "g4media.ro"][h].sum()
        lines.append(f"| `{h}` | {total_m} | {pct:.2f}% | {digi} | {g4m} |")

    lines += [
        "",
        "### Exemple — titluri concatenate (scor_suspect >= 2)",
        "",
    ]
    for _, row in df[df["scor_suspect"] >= 2].head(8).iterrows():
        heur = []
        for h in ["multe_ghilimele_interne", "colon_fara_urmare", "multi_caps", "scurta_si_fara_punct"]:
            if row[h]:
                heur.append(h)
        lines.append(f"- [{row['sursa_site']}] *{row['nr_cuvinte']}w* [{', '.join(heur)}]:")
        lines.append(f"  > {row['propozitie'][:250]}")
        lines.append("")

    lines += [
        "## 4. Ipoteza „zgomot concentrat pe digi24\"",
        "",
    ]
    total_zgomot_digi = ((df["sursa_site"] == "digi24.ro") &
                        (df["categorie_zgomot"] != "probabil_curat")).sum()
    total_zgomot_g4m = ((df["sursa_site"] == "g4media.ro") &
                       (df["categorie_zgomot"] != "probabil_curat")).sum()
    total_digi = (df["sursa_site"] == "digi24.ro").sum()
    total_g4m = (df["sursa_site"] == "g4media.ro").sum()
    lines += [
        f"- digi24.ro: **{total_zgomot_digi:,} / {total_digi:,}** propoziții detectate ca zgomot "
        f"({100 * total_zgomot_digi / total_digi:.2f}%)",
        f"- g4media.ro: **{total_zgomot_g4m:,} / {total_g4m:,}** propoziții detectate ca zgomot "
        f"({100 * total_zgomot_g4m / total_g4m:.2f}%)",
        "",
    ]
    if total_zgomot_digi / total_digi > 2 * total_zgomot_g4m / total_g4m:
        lines.append("> **Ipoteză confirmată.** Zgomotul e concentrat pe digi24, probabil "
                     "din scurgeri de boilerplate CMS (cookie banners, blocuri 'articole "
                     "recomandate'). G4Media are scraping mai curat.")
    else:
        lines.append("> Ipoteza nu se confirmă net — zgomotul e distribuit comparabil pe ambele surse.")
    lines.append("")

    lines += [
        "## 5. Eșantion stratificat pentru verificare manuală",
        "",
        "> **Instrucțiuni pentru audit manual.** Citește fiecare propoziție și compară",
        "> coloana `categorie_detectata` cu judecata ta. Ne interesează:",
        "> - **False positives**: propoziții marcate ca zgomot dar care sunt conținut real",
        "> - **False negatives**: propoziții marcate `probabil_curat` dar care sunt artefacte",
        "",
    ]
    for bucket, items in esantion.items():
        lines.append(f"### {bucket}")
        lines.append("")
        for item in items:
            lines.append(
                f"- [{item['sursa']}] *{item['nr_cuvinte']}w* "
                f"`{item['categorie_detectata']}` (scor={item['scor_suspect']}):"
            )
            lines.append(f"  > {item['text']}")
        lines.append("")

    # Sumar final
    curat = (df["categorie_zgomot"] == "probabil_curat").sum()
    zgomot = total - curat
    lines += [
        "## 6. Sumar decizional",
        "",
        f"- Propoziții `probabil_curat`: **{curat:,}** ({100 * curat / total:.2f}%)",
        f"- Propoziții detectate ca zgomot: **{zgomot:,}** ({100 * zgomot / total:.2f}%)",
        "",
        "**Pași următori:**",
        "1. Citește eșantionul manual (Secțiunea 5) și marchează FP/FN observate.",
        "2. Ajustează regulile (lărgire/restrângere pattern, prag scor_suspect) pe baza FP/FN.",
        "3. Abia apoi rulează scriptul de filtrare finală + deduplicare + embeddings.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit calitate corpus cls0")
    parser.add_argument("--input", type=Path, required=True,
                        help="propozitii_cls0_raw.parquet (sau .csv)")
    parser.add_argument("--output-dir", type=Path, default=Path("findings"))
    parser.add_argument("--n-esantion", type=int, default=15,
                        help="Propoziții per bucket în eșantionul manual")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Încarc {args.input}")
    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"      {len(df):,} propoziții")

    print("[2/4] Clasific propozițiile pe categorii de zgomot")
    df = clasifica_propozitii(df)

    print("[3/4] Construiesc eșantion stratificat pentru verificare manuală")
    esantion = esantion_pentru_verificare_manuala(df, n_per_bucket=args.n_esantion)

    print("[4/4] Scriu raportul")
    md_path = args.output_dir / "audit_corpus_cls0.md"
    scrie_raport(df, esantion, md_path)
    print(f"      → {md_path}")

    # JSON cu cifre
    rezumat_json = {
        "total_propozitii": len(df),
        "per_categorie": df["categorie_zgomot"].value_counts().to_dict(),
        "per_pattern_boilerplate": {
            p: int(df[f"boilerplate_{p}"].sum()) for p in BOILERPLATE_PATTERNS
        },
        "per_heuristica": {
            h: int(df[h].sum())
            for h in ["fara_punct_final", "incepe_cu_ghilimele",
                     "multe_ghilimele_interne", "colon_fara_urmare",
                     "multi_caps", "scurta_si_fara_punct"]
        },
        "zgomot_per_sursa": {
            sursa: {
                "total": int((df["sursa_site"] == sursa).sum()),
                "zgomot": int(((df["sursa_site"] == sursa) &
                              (df["categorie_zgomot"] != "probabil_curat")).sum()),
            }
            for sursa in df["sursa_site"].unique()
        },
    }
    json_path = args.output_dir / "audit_corpus_cls0.json"
    json_path.write_text(json.dumps(rezumat_json, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"      → {json_path}")

    # Salvez si dataframe-ul adnotat (util pentru pasul urmator)
    adnotat_path = args.input.parent / "propozitii_cls0_adnotate.parquet"
    try:
        df.to_parquet(adnotat_path, index=False)
        print(f"      → {adnotat_path}")
    except ImportError:
        adnotat_path = args.input.parent / "propozitii_cls0_adnotate.csv"
        df.to_csv(adnotat_path, index=False)
        print(f"      → {adnotat_path}")

    # Sumar consola
    print()
    print("=" * 60)
    print("SUMAR AUDIT")
    print("=" * 60)
    print(f"Total propoziții:         {len(df):,}")
    print(f"Probabil curate:          {(df['categorie_zgomot'] == 'probabil_curat').sum():,}")
    print(f"Boilerplate CMS:          {(df['categorie_zgomot'] == 'boilerplate_cms').sum():,}")
    print(f"Titluri concatenate:      {(df['categorie_zgomot'] == 'titluri_concatenate').sum():,}")
    print(f"Titluri scurte probabil:  {(df['categorie_zgomot'] == 'titlu_scurt_probabil').sum():,}")
    print(f"Eticheta vorbitor:        {(df['categorie_zgomot'] == 'eticheta_vorbitor').sum():,}")


if __name__ == "__main__":
    main()
