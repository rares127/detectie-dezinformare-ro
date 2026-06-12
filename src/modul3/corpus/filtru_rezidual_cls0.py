"""
Modul 3 — Pasul 3: Filtru rezidual.

Trateaza categoriile minore de zgomot ramase dupa curatarea cookies (v3).
Toate regulile sunt calibrate pe exemplele validate in audit-urile anterioare.

Input: propozitii_cls0_no_cookies.parquet (5.976 propozitii)
Output: propozitii_cls0_filtrat.parquet

Reguli aplicate:
    1. Prefix „Foto: <domeniu>" — eliminat (nu arunca propozitia, doar curata)
       Exemple: „Foto: highlandsystems.me Submarinul Kronos..."
    2. Prefix „Citeste si[:]" / „Vezi si[:]" — aruncate (link-uri navigatie)
    3. Etichete vorbitor (≤6 cuvinte + `:` la final) — aruncate
       Exemple: „Zelenski:", „Ursula von der Leyen:"
    4. Degenerate alfanumerice (<4 cuvinte + <60% litere) — aruncate
       Exemple: „ro.", „Romania 🇷🇴 ❤️"

NU se aplica (lasate pentru pasii ulteriori):
    - Deduplicare pe hash normalizat (pasul 4)
    - Filtrare lungime (pasul 5)
    - Titluri concatenate (4+ ghilimele, separator slash) — 80% FP in audit

Rulare:
    python filtru_rezidual_cls0.py \\
        --input data/processed/propozitii_cls0_no_cookies.parquet \\
        --output data/processed/propozitii_cls0_filtrat.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Pattern-uri rezidual — calibrate pe audit
# ---------------------------------------------------------------------------

# Prefix „Foto: <nume-site>" care precede continutul real. Uneori urmat de
# „Deschide galeria foto" (pattern CMS Digi24). Curatam doar prefix-ul,
# pastram continutul jurnalistic care urmeaza.
PATTERN_FOTO_PREFIX = re.compile(
    r"^\s*foto\s*:\s*[\w.-]+(\s+deschide\s+galeria\s+foto)?\s+",
    re.IGNORECASE,
)

# „Citeste si:", „Citeste si ...", „Vezi si:" — link-uri navigatie
# Propozitia intreaga e link → aruncam toata propozitia.
PATTERN_CITESTE_SI = re.compile(
    r"^\s*(cite[sș]te\s+[sș]i|vezi\s+[sș]i|vezi\s+aici|vezi\s+mai\s+(multe|jos))[\s:]",
    re.IGNORECASE,
)


def este_eticheta_vorbitor(text: str) -> bool:
    """Termina cu `:` si are ≤6 cuvinte = eticheta vorbitor fara citat.

    Calibrat pe audit: „Zelenski:", „Ursula von der Leyen:", „Kremlin:",
    „Institutul pentru Studiul Razboiului:".
    """
    text = text.strip()
    if not text.endswith(":"):
        return False
    return len(text.split()) <= 6


def este_degenerat(text: str) -> bool:
    """Propozitie fara suficient continut alfabetic — artefact scraping.

    Doua reguli (OR):
      1. Fragment foarte scurt: ≤1 cuvant SI ≤5 caractere totale.
         Prinde: „ro." (fragment URL/extensie).
      2. <4 cuvinte SI raport [litere / caractere non-spatiu] < 0.6.
         Prinde: „Romania 🇷🇴 ❤️", „- - -".

    Nu prinde: citate scurte reale gen „Ne vaneaza!" (4w),
    „O rusine." (2w, raport litere >60%).
    """
    cuvinte = text.split()

    # Regula 1: fragment foarte scurt (artefact URL/scraping)
    if len(cuvinte) <= 1 and len(text.strip()) <= 5:
        return True

    # Regula 2: scurt + raport litere mic (emoji, separatoare grafice)
    if len(cuvinte) >= 4:
        return False
    fara_spatii = re.sub(r"\s+", "", text)
    if len(fara_spatii) == 0:
        return True
    litere = sum(1 for c in fara_spatii if c.isalpha())
    return litere / len(fara_spatii) < 0.6


def curata_prefix_foto(text: str) -> tuple[str, bool]:
    """Elimina prefix „Foto: <domeniu>", pastreaza restul.

    Returns:
        (text_curatat, a_fost_modificat)
    """
    match = PATTERN_FOTO_PREFIX.match(text)
    if not match:
        return text, False
    restul = text[match.end():].strip()
    # Daca dupa eliminarea prefix-ului nu ramane nimic substantial, returnam gol
    # (caller-ul decide daca arunca sau pastreaza)
    return restul, True


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def trateaza_propozitie(text: str, prag_cuvinte_minim: int = 4) -> tuple[str, str]:
    """Aplica regulile filtrului rezidual pe o propozitie.

    Returns:
        (text_rezultat, actiune)
        actiune ∈ {"nemodificat", "aruncat_citeste_si", "aruncat_eticheta_vorbitor",
                   "aruncat_degenerat", "curatat_foto_prefix",
                   "aruncat_foto_prea_scurt"}
    """
    # 1. „Citeste si:" — arunca toata propozitia
    if PATTERN_CITESTE_SI.match(text):
        return "", "aruncat_citeste_si"

    # 2. Eticheta vorbitor — arunca
    if este_eticheta_vorbitor(text):
        return "", "aruncat_eticheta_vorbitor"

    # 3. Degenerat alfanumeric — arunca
    if este_degenerat(text):
        return "", "aruncat_degenerat"

    # 4. Prefix foto — curata (nu arunca)
    text_nou, modificat = curata_prefix_foto(text)
    if modificat:
        if len(text_nou.split()) < prag_cuvinte_minim:
            # Dupa eliminarea „Foto:" nu ramane continut substantial
            return "", "aruncat_foto_prea_scurt"
        return text_nou, "curatat_foto_prefix"

    return text, "nemodificat"


def aplica_filtrare(df: pd.DataFrame, prag_cuvinte_minim: int = 4) -> tuple[pd.DataFrame, dict]:
    """Aplica filtrul rezidual pe tot corpusul cu tracking detaliat."""
    stats = {
        "input_total": len(df),
        "nemodificat": 0,
        "aruncat_citeste_si": 0,
        "aruncat_eticheta_vorbitor": 0,
        "aruncat_degenerat": 0,
        "curatat_foto_prefix": 0,
        "aruncat_foto_prea_scurt": 0,
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

        arunca = actiune in (
            "aruncat_citeste_si",
            "aruncat_eticheta_vorbitor",
            "aruncat_degenerat",
            "aruncat_foto_prea_scurt",
        )
        mask_pastreaza.append(not arunca)

    df["propozitie"] = texte_noi
    df["_actiune_rezidual"] = actiuni
    df_out = df[mask_pastreaza].copy()

    # Recalculam lungimile (pentru pasii ulteriori)
    df_out["nr_cuvinte"] = df_out["propozitie"].str.split().str.len()
    df_out["nr_caractere"] = df_out["propozitie"].str.len()

    stats["output_total"] = len(df_out)
    stats["retentie_pct"] = round(100 * len(df_out) / stats["input_total"], 2)

    return df_out, stats


# ---------------------------------------------------------------------------
# Extragere exemple pentru raport
# ---------------------------------------------------------------------------

def extrage_exemple(df_input: pd.DataFrame, prag: int = 4) -> dict:
    """Colecteaza exemple per categorie pentru verificare vizuala."""
    exemple = {
        "aruncat_citeste_si": [],
        "aruncat_eticheta_vorbitor": [],
        "aruncat_degenerat": [],
        "curatat_foto_prefix": [],
    }

    for _, rand in df_input.iterrows():
        text_orig = rand["propozitie"]
        text_nou, actiune = trateaza_propozitie(text_orig, prag)

        if actiune in exemple and len(exemple[actiune]) < 5:
            item = {
                "sursa": rand["sursa_site"],
                "nr_cuvinte": len(text_orig.split()),
                "text": text_orig[:200],
            }
            if actiune == "curatat_foto_prefix":
                item["dupa"] = text_nou[:200]
                item["cuvinte_dupa"] = len(text_nou.split())
            exemple[actiune].append(item)

        if all(len(v) >= 5 for v in exemple.values()):
            break

    return exemple


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------

def scrie_raport(stats: dict, df_input: pd.DataFrame, df_output: pd.DataFrame,
                 exemple: dict, output_path: Path) -> None:
    lines = [
        "# Filtru rezidual cls0 — raport",
        "",
        "**Scope:** elimină categorii minore de zgomot rămase după curățarea",
        "cookies v3. Pașii de deduplicare și filtrare lungime vin separat.",
        "",
        "## 1. Rezumat operații",
        "",
        "| Operație | Număr propoziții |",
        "|---|---|",
        f"| Input (din no_cookies v3) | {stats['input_total']:,} |",
        f"| Nemodificate | {stats['nemodificat']:,} |",
        f"| Aruncat — link Citește și | {stats.get('aruncat_citeste_si', 0)} |",
        f"| Aruncat — etichetă vorbitor | {stats.get('aruncat_eticheta_vorbitor', 0)} |",
        f"| Aruncat — degenerat alfanumeric | {stats.get('aruncat_degenerat', 0)} |",
        f"| Curățat — prefix Foto: | {stats.get('curatat_foto_prefix', 0)} |",
        f"| Aruncat — rest prea scurt după Foto: | {stats.get('aruncat_foto_prea_scurt', 0)} |",
        f"| **Output** | **{stats['output_total']:,}** |",
        "",
        f"**Retenție:** {stats['retentie_pct']}%",
        "",
        "## 2. Breakdown pe sursă",
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
        "## 3. Exemple — link-uri Citește și",
        "",
    ]
    for ex in exemple.get("aruncat_citeste_si", []):
        lines.append(f"- [{ex['sursa']}] *{ex['nr_cuvinte']}w*: {ex['text']}")
    lines.append("")

    lines += [
        "## 4. Exemple — etichete vorbitor",
        "",
    ]
    for ex in exemple.get("aruncat_eticheta_vorbitor", []):
        lines.append(f"- [{ex['sursa']}] *{ex['nr_cuvinte']}w*: {ex['text']}")
    lines.append("")

    lines += [
        "## 5. Exemple — degenerate alfanumerice",
        "",
    ]
    for ex in exemple.get("aruncat_degenerat", []):
        lines.append(f"- [{ex['sursa']}] *{ex['nr_cuvinte']}w*: {ex['text']!r}")
    lines.append("")

    lines += [
        "## 6. Exemple — prefix Foto: curățat (ÎNAINTE / DUPĂ)",
        "",
        "> Verifică vizual că bucata păstrată e conținut jurnalistic real.",
        "",
    ]
    for ex in exemple.get("curatat_foto_prefix", []):
        lines.append(f"**Sursă:** {ex['sursa']} "
                     f"({ex['nr_cuvinte']}w → {ex.get('cuvinte_dupa', '?')}w)")
        lines.append(f"- ÎNAINTE: {ex['text']}")
        lines.append(f"- DUPĂ:   {ex.get('dupa', '?')}")
        lines.append("")

    lines += [
        "## 7. Pași următori",
        "",
        "1. **Deduplicare** pe hash normalizat — va elimina ~150 propoziții duplicate",
        "   (inclusiv cele 5 identice 'Foto: highlandsystems.me Submarinul Kronos...').",
        "2. **Filtrare lungime** pe percentilele [p5, p95] recalculate pe corpusul deduplicat.",
        "3. **Embeddings** pe corpusul final.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filtru rezidual cls0")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-dir", type=Path, default=Path("findings"))
    parser.add_argument("--prag-cuvinte-minim", type=int, default=4)
    args = parser.parse_args()

    args.findings_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Încarc {args.input}")
    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"      {len(df):,} propoziții")

    print("[2/4] Extrag exemple (înainte de modificare)")
    exemple = extrage_exemple(df, prag=args.prag_cuvinte_minim)

    print("[3/4] Aplic filtrul rezidual")
    df_curat, stats = aplica_filtrare(df, prag_cuvinte_minim=args.prag_cuvinte_minim)
    print(f"      Output: {len(df_curat):,} ({stats['retentie_pct']}%)")
    print(f"      Aruncat 'Citește și':     {stats.get('aruncat_citeste_si', 0)}")
    print(f"      Aruncat etichetă vorbitor: {stats.get('aruncat_eticheta_vorbitor', 0)}")
    print(f"      Aruncat degenerat:         {stats.get('aruncat_degenerat', 0)}")
    print(f"      Curățat prefix Foto:       {stats.get('curatat_foto_prefix', 0)}")
    print(f"      Aruncat Foto prea scurt:   {stats.get('aruncat_foto_prea_scurt', 0)}")

    print("[4/4] Salvez outputs")
    df_pt_salvare = df_curat.drop(columns=["_actiune_rezidual"], errors="ignore")
    try:
        df_pt_salvare.to_parquet(args.output, index=False)
        print(f"      → {args.output}")
    except ImportError:
        csv_path = args.output.with_suffix(".csv")
        df_pt_salvare.to_csv(csv_path, index=False)
        print(f"      → {csv_path}")

    md_path = args.findings_dir / "filtru_rezidual_cls0.md"
    scrie_raport(stats, df, df_curat, exemple, md_path)
    print(f"      → {md_path}")

    json_path = args.findings_dir / "filtru_rezidual_cls0.json"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"      → {json_path}")


if __name__ == "__main__":
    main()
