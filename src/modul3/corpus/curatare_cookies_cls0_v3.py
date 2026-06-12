"""
Modul 3 — Pasul 2 v3: Curatare cookies (versiune iterativa).

Motivatia v3: v2 lasa 3 banner-e reziduale. Cauza: existau propozitii cu
DOUA banner-e consecutive (B=„Poti actualiza..." urmat imediat de A=
„Setarile tale..."). v2 taia doar primul banner (curatat_prefix), iar
bucata „pastrata" era de fapt al doilea banner, care ramanea in output
ca propozitie falsa credibila.

Fix v3: aplica trateaza_propozitie ITERATIV pe output-ul propriu pana
cand nu mai gaseste niciun banner. Daca dupa iteratie rezultatul final
e sub pragul minim sau inca e banner, il arunca.

Input: propozitii_cls0_raw.parquet (6.047 propozitii brute)
Output: propozitii_cls0_no_cookies.parquet

Principiu v2: in loc sa tratam fiecare varianta de banner separat, definim
UN singur pattern care prinde orice secventa de banner cookie (indiferent
de pozitie in text), apoi decidem tratamentul in functie de pozitia
match-ului si de cat continut real ramane dupa extragere.

Scenarii tratate:
    1. Banner ocupa tot textul (continut real dupa eliminare < 6 cuvinte)
       → propozitie aruncata
    2. Banner la inceputul propozitiei
       → prefix eliminat, restul pastrat
    3. Banner la mijloc/final
       → se pastreaza bucata INAINTE de banner, banner-ul + tot ce e dupa
         se arunca (de obicei e doar banner-ul, segmentul post-banner e rar)

Rulare:
    python curatare_cookies_cls0_v2.py \\
        --input data/processed/propozitii_cls0_raw.parquet \\
        --output data/processed/propozitii_cls0_no_cookies.parquet
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Pattern unic care prinde orice varianta de banner cookie Digi24
# ---------------------------------------------------------------------------
# Combinam cei doi indicatori de banner observati in corpus:
#   A. „Setarile tale privind cookie-urile nu permit afisarea continutul
#       din aceasta sectiune." (banner classic, 11 cuvinte)
#   B. „Poti actualiza setarile modulelor coookie direct din browser sau
#       de aici – e nevoie sa accepti cookie-urile social media" (banner
#       de 19 cuvinte, de obicei lipit cu continut real)
#
# Pattern-urile sunt LUNGI intentionat: vrem sa matcheze tot banner-ul,
# nu doar un fragment, ca sa il putem elimina curat din text.
#
# `co+kie` prinde `cookie`, `coookie`, `cooookie` (typo scraping Digi24).
# `[ss]` si `[tt]` prind diacriticele cu/fara cedila.
# `\s+` tolerant la spatii multiple / non-breaking space (dupa normalizare NFKC).

PATTERN_BANNER_A = re.compile(
    r"set[aă]rile?\s+tale?\s+privind\s+co+kie[-\s]?urile?"
    r"\s+nu\s+permit\s+afi[sș]are[a]?\s+con[tț]inutul(ui)?"
    r"\s+din\s+aceast[aă]\s+sec[tț]iune\s*\.?",
    re.IGNORECASE | re.DOTALL,
)

PATTERN_BANNER_B = re.compile(
    r"poti\s+actualiza\s+set[aă]rile?\s+modulelor?\s+co+kie"
    r".{0,150}?"  # banner-ul variaza putin in mijloc, toleram pana la 150 car
    r"co+kie[-\s]?urile?\s+social\s+media",
    re.IGNORECASE | re.DOTALL,
)

# Pattern de ultim plan — fragment partial de banner care poate aparea
# cand Stanza taie propozitia in mijlocul banner-ului (rar, dar posibil).
# Folosit doar ca safety net pentru detectie, nu pentru extragere.
PATTERN_BANNER_PARTIAL = re.compile(
    r"(set[aă]rile?\s+tale?\s+privind\s+co+kie)"
    r"|(afi[sș]are[a]?\s+con[tț]inutul(ui)?\s+din\s+aceast[aă]\s+sec[tț]iune)"
    r"|(poti\s+actualiza\s+set[aă]rile?\s+modulelor?\s+co+kie)"
    r"|(e\s+nevoie\s+s[aă]\s+accep[tț]i\s+co+kie[-\s]?urile?\s+social\s+media)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Normalizare pentru ghilimele orfane
# ---------------------------------------------------------------------------

def curata_ghilimele_orfane(text: str) -> str:
    """Elimina ghilimele ramase agatate la marginile textului dupa taiere banner.

    Dupa ce eliminam banner-ul din mijlocul propozitiei, pot ramane la finalul
    bucatii pastrate ghilimele inchise fara pereche (ex. `... existe"` cand
    ghilimeaua deschisa era in banner-ul taiat) sau invers.

    Aplicam o euristica simpla: daca textul se termina cu ghilimele inchise
    dar numarul lor e impar, eliminam ultima. Analog la inceput.
    """
    text = text.strip()

    # Ghilimele de inchidere la sfarsit fara deschidere in text
    for ghil_inchidere, ghil_deschidere in [('"', '"'), ('”', '„'), ('»', '«')]:
        n_inchise = text.count(ghil_inchidere)
        n_deschise = text.count(ghil_deschidere)
        # Daca textul se termina cu ghilimea de inchidere si numarul lor e
        # mai mare decat cel de deschidere, o scoatem
        while text.endswith(ghil_inchidere) and n_inchise > n_deschise:
            text = text[:-1].rstrip()
            n_inchise -= 1

    return text.strip()


# ---------------------------------------------------------------------------
# Clasificare propozitii
# ---------------------------------------------------------------------------

def gaseste_banner(text: str) -> tuple[int, int] | None:
    """Gaseste primul banner in text si returneaza (start, end) al match-ului.

    Incercam pattern-urile in ordinea: B (cel mai lung, mai specific) → A → partial.
    Returnam None daca nu exista niciun banner.
    """
    # Pattern B: banner lung „Poti actualiza..." (prioritar — e mai specific)
    m = PATTERN_BANNER_B.search(text)
    if m:
        return (m.start(), m.end())

    # Pattern A: banner classic „Setarile tale..."
    m = PATTERN_BANNER_A.search(text)
    if m:
        return (m.start(), m.end())

    # Pattern partial: fragmente (cand banner-ul e taiat de Stanza)
    m = PATTERN_BANNER_PARTIAL.search(text)
    if m:
        return (m.start(), m.end())

    return None


def trateaza_propozitie_o_data(
    text: str,
    prag_cuvinte_minim: int = 6,
) -> tuple[str, str]:
    """Aplica UN pas de tratament (non-iterativ).

    Identica logic cu v2. Folosita intern de trateaza_propozitie (iterativa).

    Returns:
        (text_rezultat, actiune)
    """
    rezultat = gaseste_banner(text)
    if rezultat is None:
        return text, "nemodificat"

    start, end = rezultat

    # Calculam cat e banner vs continut
    inainte = text[:start].strip()
    dupa = text[end:].strip()
    inainte = curata_ghilimele_orfane(inainte)
    dupa = curata_ghilimele_orfane(dupa)

    cuvinte_inainte = len(inainte.split()) if inainte else 0
    cuvinte_dupa = len(dupa.split()) if dupa else 0

    # Scenariul 1: banner dominant → aruncam
    if cuvinte_inainte < prag_cuvinte_minim and cuvinte_dupa < prag_cuvinte_minim:
        return "", "aruncat_banner_pur"

    # Scenariul 2: banner la inceput, continut dupa
    if cuvinte_inainte < prag_cuvinte_minim and cuvinte_dupa >= prag_cuvinte_minim:
        return dupa, "curatat_prefix"

    # Scenariul 3: banner la final/mijloc, continut inainte
    if cuvinte_inainte >= prag_cuvinte_minim:
        return inainte, "curatat_sufix"

    return "", "aruncat_prea_scurt_dupa_curatare"


def trateaza_propozitie(
    text: str,
    prag_cuvinte_minim: int = 6,
    max_iteratii: int = 5,
) -> tuple[str, str]:
    """Tratament iterativ — curata banner-ul pana nu mai exista in text.

    Corecteaza bug-ul v2 unde o propozitie cu DOUA banner-e consecutive
    era doar partial curatata (primul banner taiat, al doilea ramas in
    output ca banner pur nou).

    Algoritm:
        1. Aplica un pas de curatare (trateaza_propozitie_o_data).
        2. Daca rezultatul contine inca banner, ruleaza iar — pana nu mai
           gaseste sau pana atingem max_iteratii (safety).
        3. Daca dupa iteratii finale rezultatul e banner pur (cuvinte <
           prag), il marcheaza aruncat_banner_pur chiar daca incepuse ca
           curatat_prefix/sufix.

    Returns:
        (text_rezultat, actiune_finala)
        actiune_finala reflecta ce s-a intamplat IN TOTAL, nu doar la
        ultimul pas. Prioritate: aruncat > curatat_prefix > curatat_sufix
        > nemodificat.
    """
    text_curent = text
    actiuni_aplicate = []

    for _ in range(max_iteratii):
        text_nou, actiune = trateaza_propozitie_o_data(text_curent, prag_cuvinte_minim)
        actiuni_aplicate.append(actiune)

        if actiune == "nemodificat":
            break

        if actiune in ("aruncat_banner_pur", "aruncat_prea_scurt_dupa_curatare"):
            # Aruncat definitiv — ne oprim aici
            return "", actiune

        # Rezultat ne-gol — continuam cu el in iteratia urmatoare
        text_curent = text_nou

    # Actiuni concrete (ne-nemodificat) care s-au aplicat de-a lungul iteratiilor
    actiuni_concrete = [a for a in actiuni_aplicate if a != "nemodificat"]

    # Caz 1: nicio curatare efectiva — returnam textul original nemodificat
    # IMPORTANT: nu aplicam verificarea de lungime minima aici!
    # Propozitiile scurte fara banner NU sunt problema noastra — pe ele le
    # trateaza filtrul de lungime din pasul ulterior (p5/p95).
    if not actiuni_concrete:
        return text_curent, "nemodificat"

    # Caz 2: am facut cel putin o curatare — verificam daca rezultatul ramas
    # e substantial. Daca nu, a fost de fapt banner pur in toata propozitia.
    cuvinte_finale = len(text_curent.split())
    if cuvinte_finale < prag_cuvinte_minim:
        return "", "aruncat_prea_scurt_dupa_curatare"

    # Returnam textul curatat cu prima actiune informativa
    return text_curent, actiuni_concrete[0]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def aplica_curatare(df: pd.DataFrame, prag_cuvinte_minim: int = 6) -> tuple[pd.DataFrame, dict]:
    """Aplica tratamentul pe tot corpusul cu tracking detaliat."""
    stats = {
        "input_total": len(df),
        "nemodificat": 0,
        "aruncat_banner_pur": 0,
        "curatat_prefix": 0,
        "curatat_sufix": 0,
        "aruncat_prea_scurt_dupa_curatare": 0,
    }

    df = df.copy()
    texte_noi = []
    actiuni = []
    mask_pastreaza = []

    for text in df["propozitie"]:
        text_nou, actiune = trateaza_propozitie(text, prag_cuvinte_minim)
        stats[actiune] = stats.get(actiune, 0) + 1
        texte_noi.append(text_nou)
        actiuni.append(actiune)
        mask_pastreaza.append(actiune not in ("aruncat_banner_pur",
                                              "aruncat_prea_scurt_dupa_curatare"))

    df["propozitie"] = texte_noi
    df["_actiune_curatare"] = actiuni
    df_out = df[mask_pastreaza].copy()

    # Recalculam lungimile (importante pentru pasii ulteriori)
    df_out["nr_cuvinte"] = df_out["propozitie"].str.split().str.len()
    df_out["nr_caractere"] = df_out["propozitie"].str.len()

    stats["output_total"] = len(df_out)
    stats["retentie_pct"] = round(100 * len(df_out) / stats["input_total"], 2)

    return df_out, stats


# ---------------------------------------------------------------------------
# Extragere exemple pentru raport
# ---------------------------------------------------------------------------

def extrage_exemple(df_input: pd.DataFrame, prag: int = 6) -> dict:
    """Exemple reprezentative pentru fiecare scenariu, pentru raport."""
    exemple = {
        "banner_pur_aruncat": [],
        "curatat_prefix": [],
        "curatat_sufix": [],
    }

    for _, rand in df_input.iterrows():
        text_orig = rand["propozitie"]
        text_nou, actiune = trateaza_propozitie(text_orig, prag)

        if actiune == "aruncat_banner_pur" and len(exemple["banner_pur_aruncat"]) < 5:
            exemple["banner_pur_aruncat"].append({
                "sursa": rand["sursa_site"],
                "nr_cuvinte": len(text_orig.split()),
                "text": text_orig[:200],
            })
        elif actiune == "curatat_prefix" and len(exemple["curatat_prefix"]) < 5:
            exemple["curatat_prefix"].append({
                "sursa": rand["sursa_site"],
                "inainte": text_orig[:250],
                "dupa": text_nou[:250],
                "cuvinte_inainte": len(text_orig.split()),
                "cuvinte_dupa": len(text_nou.split()),
            })
        elif actiune == "curatat_sufix" and len(exemple["curatat_sufix"]) < 5:
            exemple["curatat_sufix"].append({
                "sursa": rand["sursa_site"],
                "inainte": text_orig[:300],
                "dupa": text_nou[:300],
                "cuvinte_inainte": len(text_orig.split()),
                "cuvinte_dupa": len(text_nou.split()),
            })

        if all(len(v) >= 5 for v in exemple.values()):
            break

    return exemple


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------

def scrie_raport(stats: dict, df_input: pd.DataFrame, df_output: pd.DataFrame,
                 exemple: dict, output_path: Path) -> None:
    lines = [
        "# Curățare cookies cls0 v3 — raport",
        "",
        "**Versiune:** v3 (rezolvă bug-ul banner dublu prin curățare iterativă)",
        "",
        "**Scope:** DOAR tratament boilerplate cookie, cu detecție robustă",
        "indiferent de poziția banner-ului, și aplicare iterativă a curățării",
        "pentru cazurile cu banner-e consecutive în aceeași propoziție.",
        "",
        "## 1. Rezumat operații",
        "",
        "| Operație | Număr propoziții |",
        "|---|---|",
        f"| Input brut | {stats['input_total']:,} |",
        f"| Nemodificate | {stats['nemodificat']:,} |",
        f"| Banner pur aruncat (scenariul 1) | {stats.get('aruncat_banner_pur', 0)} |",
        f"| Curățat prefix banner (scenariul 2) | {stats.get('curatat_prefix', 0)} |",
        f"| Curățat sufix banner (scenariul 3, NOU) | {stats.get('curatat_sufix', 0)} |",
        f"| Aruncat — prea scurt după curățare | {stats.get('aruncat_prea_scurt_dupa_curatare', 0)} |",
        f"| **Output** | **{stats['output_total']:,}** |",
        "",
        f"**Retenție:** {stats['retentie_pct']}%",
        "",
        "## 2. Comparație v1 → v2",
        "",
        "| Scenariu | v1 | v2 |",
        "|---|---|---|",
        "| Banner la început (scenariul 1+2) | Tratat | Tratat |",
        "| Banner la mijloc/final (scenariul 3) | **NETRATAT** | **TRATAT** |",
        "| Banner pur variații minore | Parțial | Complet |",
        "",
        "## 3. Breakdown pe sursă",
        "",
        "| Sursă | Input | Output | Retenție |",
        "|---|---|---|---|",
    ]
    for sursa in sorted(df_input["sursa_site"].unique()):
        n_in = (df_input["sursa_site"] == sursa).sum()
        n_out = (df_output["sursa_site"] == sursa).sum()
        ret = 100 * n_out / n_in if n_in else 0
        lines.append(f"| {sursa} | {n_in:,} | {n_out:,} | {ret:.2f}% |")

    lines += [
        "",
        "> Toate modificările ar trebui concentrate pe digi24.ro. G4Media = 100%.",
        "",
        "## 4. Exemple — banner pur aruncat",
        "",
    ]
    for ex in exemple.get("banner_pur_aruncat", []):
        lines.append(f"- [{ex['sursa']}] *{ex['nr_cuvinte']}w*: {ex['text']}")
    lines.append("")

    lines += [
        "## 5. Exemple — prefix curățat (banner la început)",
        "",
    ]
    for ex in exemple.get("curatat_prefix", []):
        lines.append(f"**Sursă:** {ex['sursa']} "
                     f"({ex['cuvinte_inainte']}w → {ex['cuvinte_dupa']}w)")
        lines.append(f"- ÎNAINTE: {ex['inainte']}")
        lines.append(f"- DUPĂ:   {ex['dupa']}")
        lines.append("")

    lines += [
        "## 6. Exemple — sufix curățat (banner la mijloc/final, NOU în v2)",
        "",
        "> Cazul critic: propoziții lungi cu conținut real + banner agățat la coadă.",
        "> Verifică că bucata păstrată e propoziție jurnalistică validă.",
        "",
    ]
    for ex in exemple.get("curatat_sufix", []):
        lines.append(f"**Sursă:** {ex['sursa']} "
                     f"({ex['cuvinte_inainte']}w → {ex['cuvinte_dupa']}w)")
        lines.append(f"- ÎNAINTE: {ex['inainte']}")
        lines.append(f"- DUPĂ:   {ex['dupa']}")
        lines.append("")

    lines += [
        "## 7. Verificare finală",
        "",
        "După rularea acestui script, rulează din nou `investigare_cookies_ramase.py`.",
        "Rezultat așteptat: **0 matches** pe toate pattern-urile (A, B, C, D).",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Curățare cookies v2 (robustă)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-dir", type=Path, default=Path("findings"))
    parser.add_argument("--prag-cuvinte-minim", type=int, default=6,
                        help="Prag sub care conținutul după curățare e considerat prea scurt")
    args = parser.parse_args()

    args.findings_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Încarc {args.input}")
    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"      {len(df):,} propoziții brute")

    print("[2/4] Extrag exemple pentru raport (înainte de modificare)")
    exemple = extrage_exemple(df, prag=args.prag_cuvinte_minim)

    print("[3/4] Aplic tratamentul robust (3 scenarii)")
    df_curat, stats = aplica_curatare(df, prag_cuvinte_minim=args.prag_cuvinte_minim)

    print(f"      Output: {len(df_curat):,} propoziții "
          f"({stats['retentie_pct']}%)")
    print(f"      Banner pur aruncat:       {stats.get('aruncat_banner_pur', 0)}")
    print(f"      Curățat prefix:           {stats.get('curatat_prefix', 0)}")
    print(f"      Curățat sufix (NOU):      {stats.get('curatat_sufix', 0)}")
    print(f"      Prea scurt după curățare: {stats.get('aruncat_prea_scurt_dupa_curatare', 0)}")

    print("[4/4] Salvez outputs")
    df_pt_salvare = df_curat.drop(columns=["_actiune_curatare"], errors="ignore")
    try:
        df_pt_salvare.to_parquet(args.output, index=False)
        print(f"      → {args.output}")
    except ImportError:
        csv_path = args.output.with_suffix(".csv")
        df_pt_salvare.to_csv(csv_path, index=False)
        print(f"      → {csv_path}")

    md_path = args.findings_dir / "curatare_cookies_cls0_v3.md"
    scrie_raport(stats, df, df_curat, exemple, md_path)
    print(f"      → {md_path}")

    json_path = args.findings_dir / "curatare_cookies_cls0_v3.json"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"      → {json_path}")


if __name__ == "__main__":
    main()