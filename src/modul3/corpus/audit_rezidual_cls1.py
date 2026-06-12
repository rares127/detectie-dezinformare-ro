"""
Audit rezidual pe corpus cls1 (propagandistic: Veridica + Stopfals).

Scop: identificarea de pattern-uri de zgomot rezidual in cele 6.082 propozitii
din `propozitii_cls1_corpus.parquet`, similar cu auditul efectuat pe cls0.

Decizie metodologica:
- daca zgomot detectat > 5%  → aplicam curatare similar cu cls0
- daca zgomot detectat < 2%  → mergem direct la benchmark v4
- daca 2–5%                  → decizie informata in functie de tip zgomot

Pattern-uri cautate (adaptate pentru Veridica/Stopfals, NU Digi24):
  1. Cookie banners si meta-elemente web (improbabil, dar verificam)
  2. Etichete vorbitor in citate ("X a declarat:", "Putin a spus:")
  3. Ghilimele desfacute / ghilimele orfane (citate taiate la segmentare)
  4. Meta-elemente Veridica ("Vezi si", "Citeste si", "Context:", "Sursa:")
  5. Etichete temporale orfane ("Astazi", "Saptamana trecuta,")
  6. Link-uri reziduale / URL-uri
  7. Prefix media ("Foto:", "Video:", "Imagine:")
  8. Propozitii cu fragmente de traducere/paranteze redactionale ("(n.red.)")
  9. Propozitii degenerate (majoritar non-alfabetic, numere, punctuatie)
 10. Propozitii cu continut pre-articol (autor, data, categorie)

Output:
  - findings/audit_rezidual_cls1.json  (statistici agregate)
  - findings/audit_rezidual_cls1.md    (raport lizibil cu exemple concrete)

Utilizare:
  python scripts/audit_rezidual_cls1.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configurare cai
# ---------------------------------------------------------------------------

CORPUS_PATH = Path("data/processed/propozitii_cls1_corpus.parquet")
OUT_JSON = Path("findings/audit_rezidual_cls1.json")
OUT_MD = Path("findings/audit_rezidual_cls1.md")

# Cat de multe exemple concrete afisam per pattern (pentru inspectie vizuala)
MAX_EXEMPLE = 10

# Prag "zgomot grav" (propozitie de aruncat complet)
# vs "zgomot minor" (prefix/sufix de curatat)
# Documentat in raport.


# ---------------------------------------------------------------------------
# Pattern-uri — expresii regulate compilate
# ---------------------------------------------------------------------------

# 1. Cookie banners / GDPR / meta-elemente web
#    Putin probabil pe Veridica (scraping mai curat), dar verificam pentru siguranta.
#    Folosim (?i) pentru case-insensitive si co+kie pentru a prinde typo-uri ('coookie', 'coooookie').
PATTERN_COOKIES = re.compile(
    r"(?i)\b("
    r"co+kie|gdpr|politica de confiden[tț]ialitate|accept[aă] (toate |ne)cookies?|"
    r"utiliz[aă]m cookies|setări cookies|acest site folose[sș]te cookies"
    r")\b"
)

# 2. Etichete vorbitor — „cuvinte putine" + „:" la inceputul propozitiei
#    Acelasi pattern ca pe cls0: ≤6 cuvinte inainte de ":"
#    Ex: "Vladimir Putin a declarat:", "Purtatorul de cuvant Peskov a spus:"
PATTERN_ETICHETA_VORBITOR = re.compile(
    r"^([A-ZĂÎÂȘȚ][\w\-\.\s]{0,80})\s*:\s*$"
)

# 3. Ghilimele desfacute / orfane — citate taiate neuniform la segmentare
#    Propozitii care incep cu " sau „ si NU se inchid, sau invers.
#    Numaram deschideri vs inchideri; daca diferenta e > 1 = ghilimele orfane.
#    Ghilimele considerate: " " " „ " « » ‟ ‟
GHILIMELE_DESCHIDERE = set('"„«‟❝"\u201C\u201E\u00AB')
GHILIMELE_INCHIDERE = set('"»"\u201D\u00BB')

# 4. Meta-elemente Veridica / navigatie redactionala
#    Pattern-uri specifice platformei Veridica.
PATTERN_META_VERIDICA = re.compile(
    r"(?i)^\s*("
    r"vezi [sș]i|cite[sș]te [sș]i|citi[tţ]i [sș]i|"
    r"context[:\s]|sursa[:\s]|sursele[:\s]|"
    r"articole [ai]semenea|articole conexe|materiale conexe|"
    r"cuvinte cheie|tag(uri|-uri)?[:\s]|"
    r"publicat (la |pe |în )|data public[aă]rii"
    r")"
)

# 5. Etichete temporale orfane la inceput de propozitie
#    Ex: "Astazi,", "Saptamana trecuta,", "Ieri," - cand apar ca propozitii scurte (<5 cuv)
PATTERN_TEMPORAL_ORFAN = re.compile(
    r"(?i)^\s*("
    r"ast[aă]zi|ieri|m[aâ]ine|aseară|săpt[aă]m[aâ]na (trecut[aă]|aceasta|viitoare)|"
    r"luna (trecut[aă]|aceasta|viitoare)|anul (trecut|acesta|viitor)|"
    r"recent|în ultima vreme|în ultimele zile"
    r")\s*[,\.]?\s*$"
)

# 6. URL-uri / link-uri reziduale
PATTERN_URL = re.compile(
    r"(?i)(https?://\S+|www\.\S+\.(ro|com|org|net|eu|md|ru)\b)"
)

# 7. Prefix media
PATTERN_PREFIX_MEDIA = re.compile(
    r"(?i)^\s*(foto|video|imagine|sursa foto|captur[aă])[\s:\-–]"
)

# 8. Paranteze redactionale / traducere
#    Permit text intre marker-ul redactional si inchiderea parantezei
#    Ex: "(n.red. afirmatie falsa documentata de Veridica)"
PATTERN_PARANTEZA_REDACTIONALA = re.compile(
    r"\((?:n\.\s?red\.|n\.\s?trad\.|nota redac[tț]iei|notă traducător)[^)]*\)"
)

# 9. Propozitii degenerate: raport alfabetic < 50%
#    (sau: lungime in caractere/cuvinte excesiv de mica pentru continut util)
def raport_alfabetic(text: str) -> float:
    """Calculeaza proportia de caractere alfabetice (litere) din text."""
    if not text:
        return 0.0
    total = len(text)
    alfa = sum(1 for c in text if c.isalpha())
    return alfa / total if total > 0 else 0.0


# 10. Header pre-articol (autor + data + categorie)
#     Ex: "De Ionel Popescu | Politica | 15 martie 2024"
#     Pipe-uri + cuvinte scurte = pattern suspect.
PATTERN_HEADER_PREARTICOL = re.compile(
    r"^(De\s+[A-ZĂÎÂȘȚ]|Autor[:\s]).{0,60}\|"
)


# ---------------------------------------------------------------------------
# Functii de detectie per pattern
# ---------------------------------------------------------------------------

def detect_cookies(text: str) -> bool:
    """Returneaza True daca textul contine elemente de cookie banner / GDPR."""
    return bool(PATTERN_COOKIES.search(text))


def detect_eticheta_vorbitor(text: str) -> bool:
    """Returneaza True daca propozitia pare o eticheta de vorbitor (X a declarat:)."""
    # Conditie: pattern regex + maxim 6 cuvinte reale (fara ":")
    if not PATTERN_ETICHETA_VORBITOR.match(text.strip()):
        return False
    fara_punct = text.replace(":", "").strip()
    return len(fara_punct.split()) <= 6


def detect_ghilimele_orfane(text: str) -> bool:
    """Returneaza True daca ghilimelele deschise nu se inchid (citate rupte)."""
    deschideri = sum(1 for c in text if c in GHILIMELE_DESCHIDERE)
    inchideri = sum(1 for c in text if c in GHILIMELE_INCHIDERE)
    # Consideram orfan daca diferenta absoluta e >= 1 SI exista macar un ghilimel
    diff = abs(deschideri - inchideri)
    return diff >= 1 and (deschideri + inchideri) >= 1


def detect_meta_veridica(text: str) -> bool:
    """Returneaza True daca propozitia pare meta-element Veridica ('Vezi si', 'Context:')."""
    return bool(PATTERN_META_VERIDICA.match(text))


def detect_temporal_orfan(text: str) -> bool:
    """Returneaza True daca propozitia e doar o eticheta temporala orfana."""
    return bool(PATTERN_TEMPORAL_ORFAN.match(text))


def detect_url(text: str) -> bool:
    """Returneaza True daca textul contine URL-uri reziduale."""
    return bool(PATTERN_URL.search(text))


def detect_prefix_media(text: str) -> bool:
    """Returneaza True daca propozitia incepe cu 'Foto:', 'Video:' etc."""
    return bool(PATTERN_PREFIX_MEDIA.match(text))


def detect_paranteza_redactionala(text: str) -> bool:
    """Returneaza True daca textul contine (n.red.) sau (n.trad.)."""
    return bool(PATTERN_PARANTEZA_REDACTIONALA.search(text))


def detect_degenerat(text: str) -> bool:
    """Returneaza True daca propozitia e majoritar non-alfabetica (< 50% litere)."""
    return raport_alfabetic(text) < 0.50


def detect_header_prearticol(text: str) -> bool:
    """Returneaza True daca propozitia pare header de articol (autor | data | categorie)."""
    return bool(PATTERN_HEADER_PREARTICOL.match(text))


# ---------------------------------------------------------------------------
# Structura detectoare: lista ordonata pentru iterare uniforma
# ---------------------------------------------------------------------------

DETECTOARE = [
    ("cookies_gdpr", detect_cookies,
     "Cookie banners / GDPR / meta-elemente web"),
    ("eticheta_vorbitor", detect_eticheta_vorbitor,
     "Etichetă vorbitor (X a declarat:) - propoziție ≤6 cuv + ':'"),
    ("ghilimele_orfane", detect_ghilimele_orfane,
     "Ghilimele desfăcute / orfane (citate rupte la segmentare)"),
    ("meta_veridica", detect_meta_veridica,
     "Meta-element Veridica ('Vezi și', 'Context:', 'Sursa:')"),
    ("temporal_orfan", detect_temporal_orfan,
     "Etichetă temporală orfană ('Astăzi,', 'Săptămâna trecută,')"),
    ("url_rezidual", detect_url,
     "URL / link rezidual în text"),
    ("prefix_media", detect_prefix_media,
     "Prefix media ('Foto:', 'Video:', 'Imagine:')"),
    ("paranteza_redactionala", detect_paranteza_redactionala,
     "Paranteză redacțională ('(n.red.)', '(n.trad.)')"),
    ("degenerat_alfanumeric", detect_degenerat,
     "Propoziție degenerată (< 50% caractere alfabetice)"),
    ("header_prearticol", detect_header_prearticol,
     "Header pre-articol (autor | categorie | data)"),
]


# ---------------------------------------------------------------------------
# Rulare audit
# ---------------------------------------------------------------------------

def ruleaza_audit(df: pd.DataFrame) -> dict:
    """
    Aplica toate detectoarele pe coloana `propozitie` si construieste raport.

    Returneaza dict cu:
      - totaluri per pattern
      - procent pattern din total
      - breakdown per sursa (Veridica vs Stopfals)
      - exemple concrete (max MAX_EXEMPLE per pattern)
    """
    total = len(df)
    rezultate = {
        "total_propozitii": total,
        "pattern_uri": {},
        "breakdown_surse": {},
        "rezumat": {},
    }

    # Identificam coloana de sursa (poate fi 'sursa_site' ca la cls0)
    col_sursa = "sursa_site" if "sursa_site" in df.columns else None
    if col_sursa is None:
        # Cautam fallback-uri plauzibile
        for candidat in ("sursa", "site", "source"):
            if candidat in df.columns:
                col_sursa = candidat
                break

    # Contor general: cate propozitii au CEL PUTIN un flag
    propozitii_cu_zgomot = set()

    for cheie, fn, descriere in DETECTOARE:
        # Aplicam detectorul — vector boolean pe toate propozitiile
        mask = df["propozitie"].astype(str).apply(fn)
        count = int(mask.sum())
        procent = (count / total * 100) if total > 0 else 0.0

        # Exemple concrete (pana la MAX_EXEMPLE)
        exemple = df.loc[mask, "propozitie"].astype(str).head(MAX_EXEMPLE).tolist()

        # Breakdown per sursa (daca avem coloana)
        breakdown = {}
        if col_sursa is not None:
            for sursa, sub in df.loc[mask].groupby(col_sursa):
                breakdown[str(sursa)] = int(len(sub))

        rezultate["pattern_uri"][cheie] = {
            "descriere": descriere,
            "count": count,
            "procent": round(procent, 3),
            "exemple": exemple,
            "breakdown_surse": breakdown,
        }

        # Acumulam indecsii pentru contor global
        propozitii_cu_zgomot.update(df.index[mask].tolist())

    # Breakdown general per sursa (cate propozitii totale per sursa)
    if col_sursa is not None:
        for sursa, sub in df.groupby(col_sursa):
            rezultate["breakdown_surse"][str(sursa)] = int(len(sub))

    # Rezumat global
    total_cu_zgomot = len(propozitii_cu_zgomot)
    procent_global = (total_cu_zgomot / total * 100) if total > 0 else 0.0
    rezultate["rezumat"] = {
        "propozitii_cu_macar_un_flag": total_cu_zgomot,
        "procent_global_zgomot": round(procent_global, 3),
        "decizie_sugerata": _sugestie_decizie(procent_global),
    }

    return rezultate


def _sugestie_decizie(procent_global: float) -> str:
    """Sugereaza decizia metodologica pe baza procentului de zgomot global."""
    if procent_global < 2.0:
        return (
            "MERGEM DIRECT LA BENCHMARK V4 — zgomot rezidual sub pragul de 2%, "
            "acceptabil fără curățare suplimentară."
        )
    if procent_global <= 5.0:
        return (
            "DECIZIE INFORMATĂ — zgomot între 2-5%. Analizează tipul de pattern: "
            "dacă e concentrat pe pattern-uri ușor de curățat (meta-Veridica, "
            "prefix media, header-e), aplicăm curățare punctuală. Altfel, "
            "poate fi tolerat pentru benchmark."
        )
    return (
        "APLICĂM CURĂȚARE — zgomot peste 5%. Scriem pipeline similar cu cls0 "
        "(curatare_cookies, filtru_rezidual, dedup) înainte de benchmark v4."
    )


# ---------------------------------------------------------------------------
# Generare rapoarte
# ---------------------------------------------------------------------------

def scrie_json(rezultate: dict, out: Path) -> None:
    """Scrie rezultatele complete in format JSON."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(rezultate, f, ensure_ascii=False, indent=2)


def scrie_md(rezultate: dict, out: Path) -> None:
    """Scrie raport lizibil in Markdown cu exemple concrete per pattern."""
    out.parent.mkdir(parents=True, exist_ok=True)

    total = rezultate["total_propozitii"]
    rezumat = rezultate["rezumat"]

    linii: list[str] = []
    linii.append("# Audit rezidual corpus cls1 (propagandistic)")
    linii.append("")
    linii.append(f"**Total propoziții analizate:** {total:,}")
    linii.append("")

    # Breakdown per sursa
    if rezultate["breakdown_surse"]:
        linii.append("## Distribuție per sursă")
        linii.append("")
        for sursa, n in sorted(
            rezultate["breakdown_surse"].items(), key=lambda x: -x[1]
        ):
            pct = n / total * 100
            linii.append(f"- **{sursa}**: {n:,} propoziții ({pct:.1f}%)")
        linii.append("")

    # Rezumat global
    linii.append("## Rezumat global")
    linii.append("")
    linii.append(
        f"- **Propoziții cu cel puțin un flag de zgomot:** "
        f"{rezumat['propozitii_cu_macar_un_flag']:,} "
        f"({rezumat['procent_global_zgomot']}%)"
    )
    linii.append("")
    linii.append(f"**Decizie sugerată:** {rezumat['decizie_sugerata']}")
    linii.append("")

    # Tabel sumar per pattern
    linii.append("## Tabel sumar per pattern")
    linii.append("")
    linii.append("| Pattern | Count | Procent | Descriere |")
    linii.append("|---|---:|---:|---|")
    for cheie, info in rezultate["pattern_uri"].items():
        linii.append(
            f"| `{cheie}` | {info['count']:,} | {info['procent']}% | "
            f"{info['descriere']} |"
        )
    linii.append("")

    # Detalii per pattern cu exemple
    linii.append("## Detalii și exemple concrete per pattern")
    linii.append("")

    for cheie, info in rezultate["pattern_uri"].items():
        linii.append(f"### `{cheie}` — {info['descriere']}")
        linii.append("")
        linii.append(
            f"- **Count:** {info['count']:,} ({info['procent']}% din total)"
        )
        if info["breakdown_surse"]:
            linii.append("- **Breakdown per sursă:**")
            for sursa, n in sorted(
                info["breakdown_surse"].items(), key=lambda x: -x[1]
            ):
                linii.append(f"  - {sursa}: {n}")
        linii.append("")

        if info["exemple"]:
            linii.append(f"**Exemple (max {MAX_EXEMPLE}):**")
            linii.append("")
            for i, ex in enumerate(info["exemple"], 1):
                # Trunchiem exemplele foarte lungi pentru lizibilitate
                ex_afisat = ex[:200] + "…" if len(ex) > 200 else ex
                # Scapam | din tabele accidentale
                ex_afisat = ex_afisat.replace("|", "\\|").replace("\n", " ")
                linii.append(f"{i}. `{ex_afisat}`")
            linii.append("")
        else:
            linii.append("_Fără exemple — pattern-ul nu a fost detectat._")
            linii.append("")

    # Note metodologice
    linii.append("## Note metodologice")
    linii.append("")
    linii.append(
        "- Pattern-urile sunt adaptate pentru Veridica + Stopfals "
        "(NU Digi24 de pe cls0). Cookie banners improbabile, dar verificate."
    )
    linii.append(
        "- Un procent mic de **ghilimele orfane** e așteptat pe cls1 — "
        "Stanza segmentează uneori citate multi-propoziționale la punct, "
        "separând ghilimelele de deschidere și închidere. Nu toate cazurile "
        "sunt propoziții de aruncat; unele sunt propoziții legitime cu "
        "ghilimel unilateral."
    )
    linii.append(
        "- Pattern-urile se pot suprapune (o propoziție poate fi prinsă "
        "de mai multe detectoare). Rezumatul global numără propoziții "
        "unice cu **cel puțin un flag**, nu suma tuturor flag-urilor."
    )
    linii.append("")
    linii.append(
        "*Audit generat pentru decizia pre-benchmark v4 · "
        "Modulul 3, pasul A1.3*"
    )

    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(linii))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Punct de intrare: incarca corpus, ruleaza audit, scrie rapoarte."""
    print(f"[audit_rezidual_cls1] Încarcă corpus: {CORPUS_PATH}")
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus cls1 nu a fost găsit la {CORPUS_PATH}. "
            f"Verifică dacă construieste_corpus_cls1.py a rulat cu succes."
        )

    df = pd.read_parquet(CORPUS_PATH)
    print(f"[audit_rezidual_cls1] Coloane disponibile: {list(df.columns)}")
    print(f"[audit_rezidual_cls1] Total propoziții: {len(df):,}")

    # Sanity check: coloana 'propozitie' trebuie sa existe
    if "propozitie" not in df.columns:
        raise KeyError(
            "Coloana 'propozitie' lipsește din corpus. "
            "Schema așteptată conform handoff: articol_id, sursa_site, an, "
            "pozitie_in_articol, propozitie, nr_cuvinte, nr_caractere, "
            "hash_exact, hash_normalizat."
        )

    print("[audit_rezidual_cls1] Rulează detectoare...")
    rezultate = ruleaza_audit(df)

    print(f"[audit_rezidual_cls1] Scrie JSON: {OUT_JSON}")
    scrie_json(rezultate, OUT_JSON)

    print(f"[audit_rezidual_cls1] Scrie Markdown: {OUT_MD}")
    scrie_md(rezultate, OUT_MD)

    # Rezumat in consola
    print()
    print("=" * 70)
    print("REZUMAT AUDIT")
    print("=" * 70)
    print(f"Total propoziții: {rezultate['total_propozitii']:,}")
    print(
        f"Propoziții cu zgomot: "
        f"{rezultate['rezumat']['propozitii_cu_macar_un_flag']:,} "
        f"({rezultate['rezumat']['procent_global_zgomot']}%)"
    )
    print()
    print("Per pattern:")
    for cheie, info in rezultate["pattern_uri"].items():
        marker = "⚠️ " if info["procent"] > 1.0 else "   "
        print(
            f"  {marker}{cheie:30s} {info['count']:5d} "
            f"({info['procent']:5.2f}%)"
        )
    print()
    print(f"DECIZIE SUGERATĂ: {rezultate['rezumat']['decizie_sugerata']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
