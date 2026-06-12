"""
common/thematic_filters.py
──────────────────────────
Modul partajat de filtre tematice pentru scraperele proiectului de licenta
„Sistem de Detectie Automata si Explicabila a Dezinformarii Pro-Ruse in Presa
Romaneasca".

Scop
────
Garanteaza simetrie tematica absoluta intre clasa 1 (dezinformare pro-rusa,
sursa: Veridica.ro) si clasa 0 (stiri credibile, surse: G4Media, Digi24,
Agerpres). Orice asimetrie intre filtrele tematice ale celor doua clase
devine imediat un feature exploitabil de clasificator — ceea ce ar anula
validitatea intregii evaluari.

Pattern-urile sunt o copie BYTE-EXACTA a listei `UKRAINE_PATTERNS` din
`scraper_veridica_v4_2.py` (liniile 181-247). Orice modificare aici trebuie
propagata si in scraperul Veridica si invers, altfel cele doua clase vor
avea universuri tematice diferite.

Structura
─────────
Pattern-urile sunt grupate in doua sub-liste accesibile separat:
    - `UKRAINE_CORE_PATTERNS`  → entitati direct legate de razboi kinetic
    - `MOLDOVA_HYBRID_PATTERNS` → front hibrid Kremlin (cluster FIX K din v4.1)

`UKRAINE_PATTERNS` este uniunea lor, folosita de functia publica principala.
Scraperele clasa 0 pot folosi `topic_match_details()` pentru a raporta
separat cate articole au fost prinse pe fiecare cluster — util pentru a
detecta asimetrii intre clase la etapa de validare anti source-bias.

API public
──────────
    is_ukraine_related(text: str) -> bool
        Echivalent byte cu `is_ukraine_related` din scraper_veridica_v4_2.py.

    topic_match_details(text: str) -> dict
        Returneaza informatii granulare despre ce pattern-uri au matched.

Self-test
─────────
La primul import, modulul ruleaza `_verify_integrity()` pe un set mic de
cazuri canonice. Daca ceva nu se mai potriveste, import-ul ESUEAZA LOUD
(ImportError) — preferat fata de a continua cu un filtru corupt care ar
contamina silentios dataset-ul.

Autor: Andrei, licenta Informatica 2025-2026
Versiune: 1.0.0 — extras din scraper_veridica_v4_2.py (2026-04-09)
"""

from __future__ import annotations

import re
from typing import NamedTuple

__version__ = "1.1.0"
__source_reference__ = (
    "scraper_veridica_v4_2.py, liniile 181-247 "
    "+ FIX P (Moldova standalone) adăugat la Pasul 1 clasa 0"
)


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN-URI — cluster Ucraina/Rusia (front kinetic)
# ══════════════════════════════════════════════════════════════════════════════
#
# Entitati geografice, persoane si termeni specifici conflictului direct
# Ucraina-Rusia. Eliminate meta-termenii „dezinformare", „propaganda",
# „fake news", „naratiune" (apar in orice articol Veridica si faceau filtrul
# inutil pentru discriminarea tematica reala).

UKRAINE_CORE_PATTERNS: list[re.Pattern] = [
    # Tari si gentilice — radical + word boundary
    re.compile(r"\bucrain\w*",                            re.IGNORECASE),  # ucraina, ucrainean, ucrainieni
    re.compile(r"\brus(ia|ă|ești|ească|esc|ilor|ești)\b", re.IGNORECASE),  # rusia, ruseasca, rusi...
    re.compile(r"\bruse\w+",                              re.IGNORECASE),  # ruse, ruseasca, rusesti
    re.compile(r"\bpro[- ]?rus\w*",                       re.IGNORECASE),  # pro-rus, pro rus, pro-rusa
    re.compile(r"\bsovietic\w*",                          re.IGNORECASE),  # uneori folosit in contextul razboiului

    # Persoane cheie
    re.compile(r"\b(putin|zelenski|zelensky|lavrov|medvedev|șoigu|soigu|prigojin|prigozhin)\b",
               re.IGNORECASE),

    # Institutii si termeni specifici propagandei Kremlin
    re.compile(r"\bkremlin\w*",                           re.IGNORECASE),
    re.compile(r"\bpro[- ]?kremlin\w*",                   re.IGNORECASE),
    re.compile(r"\bwagner\b",                             re.IGNORECASE),
    re.compile(r"\bduma\b",                               re.IGNORECASE),
    re.compile(r"\brosatom\b",                            re.IGNORECASE),
    re.compile(r"\bgazprom\b",                            re.IGNORECASE),
    re.compile(r"\brt\s*(news)?\b",                       re.IGNORECASE),  # Russia Today
    re.compile(r"\bsputnik\b",                            re.IGNORECASE),

    # Regiuni si orase — zona de conflict
    re.compile(r"\b(donbas|donbass|donețk|donetk|donetsk|lugansk|luhansk)\b",
               re.IGNORECASE),
    re.compile(r"\b(crimeea|crimea|sevastopol)\b",        re.IGNORECASE),
    re.compile(r"\b(mariupol|herson|kherson|zaporijia|zaporizhzhia|bahmut|bakhmut|avdiivka)\b",
               re.IGNORECASE),
    re.compile(r"\b(kiev|kyiv|harkov|harkiv|kharkiv|odesa|odessa|lvov|lviv|cernobil|chernobyl)\b",
               re.IGNORECASE),
    re.compile(r"\bmoscova\b",                            re.IGNORECASE),

    # Termeni specifici evenimentului
    re.compile(r"\binvazi\w*",                            re.IGNORECASE),  # invazie, invazia
    re.compile(r"\b(război|razboi|războiul|razboiul)\b",  re.IGNORECASE),
    re.compile(r"\boperați\w*\s+special\w*",              re.IGNORECASE),  # "operatiune speciala"
    re.compile(r"\bdenazifica\w*",                        re.IGNORECASE),  # termen propagandistic Kremlin
    re.compile(r"\bnazi[sș]ti?\s+ucrain\w*",              re.IGNORECASE),
    re.compile(r"\bazov\b",                               re.IGNORECASE),  # batalionul Azov — tema recurenta
    re.compile(r"\bbiolaborator\w*",                      re.IGNORECASE),  # naratiune clasica pro-Kremlin
]


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN-URI — cluster Moldova / front hibrid Kremlin (FIX K din v4.1)
# ══════════════════════════════════════════════════════════════════════════════
#
# Justificare: aceeasi infrastructura propagandistica Kremlin produce
# naratiunile pro-ruse din R. Moldova ca si pe cele direct anti-Ucraina.
# Temele recurente: apararea Transnistriei, blocarea aderarii la UE,
# discreditarea guvernarii pro-occidentale, slabirea NATO pe flancul estic.

MOLDOVA_HYBRID_PATTERNS: list[re.Pattern] = [
    # Transnistria — punct fierbinte direct legat de logistica razboiului
    re.compile(r"\btransnistr\w*",                        re.IGNORECASE),

    # Persoane cheie ale spatiului politic Moldova relevante propagandei Kremlin
    re.compile(r"\b(maia\s+sandu|igor\s+dodon|ilan\s+șor|ilan\s+sor|plahotniuc)\b",
               re.IGNORECASE),

    # Localitati Moldova cu rol in naratiunile separatiste / pro-ruse
    re.compile(r"\b(chișinău|chisinau|comrat|tiraspol|găgăuzia|gagauzia|gagauz\w*)\b",
               re.IGNORECASE),

    # Republica Moldova ca entitate (forma explicita)
    re.compile(r"\b(republica\s+moldova|r\.\s*moldova)\b",
               re.IGNORECASE),

    # ── FIX P (v1.1): Moldova standalone contemporana ───────────────────────
    # Decizie luata la Pasul 1 al clasei 0: testul de integritate pe
    # veridica_clean_v4.csv a aratat ca 7 articole Moldova-flank legitime
    # NU treceau filtrul vechi pentru ca citatul propagandistic contine
    # doar „Moldova" standalone (fara „Republica"), iar acceptul lor in
    # dataset-ul v3 se baza pe corpul jurnalistic care mentiona entitati
    # Ucraina/Rusia.
    #
    # Justificare academica: in presa romaneasca 2023-2026, „Moldova" ca
    # nume propriu in articole de actualitate geopolitica se refera in
    # covarsitoare majoritate la Republica Moldova contemporana.
    # Ambiguitatea cu Principatele istorice (secolul XIX) sau cu regiunea
    # Moldova din Romania este teoretica, nu practica — acestea nu apar
    # in corpusul nostru de propaganda pro-Kremlin contemporana.
    #
    # Stil consistent cu `\bucrain\w*` si `\brus\w+` — prinde toate formele:
    # Moldova, Moldovei, moldovean, moldoveni, moldovenilor, moldoveanca, etc.
    re.compile(r"\bmoldov\w*",                            re.IGNORECASE),

    # Termeni propagandistici recurenti in naratiunile Kremlin pe flancul Moldova
    re.compile(r"\bantirus\w*",                           re.IGNORECASE),  # antirusesc, antirusesti, sanctiuni antiruse
    re.compile(r"\brusofob\w*",                           re.IGNORECASE),  # rusofobie, rusofob, rusofoba
]


# ══════════════════════════════════════════════════════════════════════════════
# Lista unificata — echivalent byte-exact cu UKRAINE_PATTERNS din scraper_veridica_v4_2.py
# ══════════════════════════════════════════════════════════════════════════════

UKRAINE_PATTERNS: list[re.Pattern] = UKRAINE_CORE_PATTERNS + MOLDOVA_HYBRID_PATTERNS


# ══════════════════════════════════════════════════════════════════════════════
# API public
# ══════════════════════════════════════════════════════════════════════════════


class TopicMatch(NamedTuple):
    """
    Structura de return pentru `topic_match_details`.

    Campuri:
        matched:          True daca cel putin un pattern a prins
        matched_core:     True daca a prins cel putin un pattern Ucraina/Rusia
        matched_hybrid:   True daca a prins cel putin un pattern Moldova-flank
        matched_patterns: lista pattern-urilor care au prins (sursa regex ca string)
    """
    matched: bool
    matched_core: bool
    matched_hybrid: bool
    matched_patterns: list[str]


def is_ukraine_related(text: str) -> bool:
    """
    Verifica daca textul contine cel putin un marker Ucraina/Rusia sau
    Moldova-front-hibrid.

    Echivalent byte-exact cu `is_ukraine_related` din scraper_veridica_v4_2.py
    (linia 402) — garanteaza simetrie tematica intre clasele 0 si 1.

    Args:
        text: textul de verificat (titlu + lead + corp articol, de obicei)

    Returns:
        True daca exista cel putin un match, False altfel.
    """
    if not text:
        return False
    return any(p.search(text) for p in UKRAINE_PATTERNS)


def topic_match_details(text: str) -> TopicMatch:
    """
    Varianta granulara a `is_ukraine_related` care raporteaza care cluster
    a fost prins (core vs. hibrid) si ce pattern-uri au matched.

    Util pentru:
        - statistici per-sursa (cate articole pe front kinetic vs hibrid)
        - detectarea asimetriilor tematice intre clasa 0 si clasa 1 la validare
        - debugging scrapere (verifici ca un articol e retinut pentru motivul corect)

    Args:
        text: textul de verificat

    Returns:
        TopicMatch cu matched/matched_core/matched_hybrid/matched_patterns
    """
    if not text:
        return TopicMatch(False, False, False, [])

    matched_patterns: list[str] = []
    matched_core = False
    matched_hybrid = False

    for p in UKRAINE_CORE_PATTERNS:
        if p.search(text):
            matched_core = True
            matched_patterns.append(p.pattern)

    for p in MOLDOVA_HYBRID_PATTERNS:
        if p.search(text):
            matched_hybrid = True
            matched_patterns.append(p.pattern)

    return TopicMatch(
        matched=matched_core or matched_hybrid,
        matched_core=matched_core,
        matched_hybrid=matched_hybrid,
        matched_patterns=matched_patterns,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Self-test la import — fail loud daca ceva s-a rupt la refactorizare
# ══════════════════════════════════════════════════════════════════════════════

# Cazuri canonice: (text, matched_asteptat, cluster_asteptat)
#   cluster ∈ {"core", "hybrid", "neither"}
_CANONICAL_TESTS: list[tuple[str, bool, str]] = [
    # Front kinetic — trebuie sa prinda core
    ("Zelenski a declarat că Ucraina va primi noi rachete",     True,  "core"),
    ("Rusia a bombardat infrastructura energetică a Kievului",  True,  "core"),
    ("Forțele ruse au atacat Bahmut",                            True,  "core"),
    ("Putin a anunțat mobilizarea parțială",                     True,  "core"),
    ("Narațiunea Kremlinului despre denazificarea Ucrainei",     True,  "core"),

    # Front hibrid Moldova — trebuie sa prinda hybrid
    ("Maia Sandu a semnat acordul cu UE",                        True,  "hybrid"),
    ("Situația din Transnistria rămâne tensionată",              True,  "hybrid"),
    ("Chișinău respinge acuzațiile de la Comrat",                True,  "hybrid"),
    ("Republica Moldova candidează pentru aderarea la UE",       True,  "hybrid"),
    ("Rusofobia este un pretext pentru propagandă",              True,  "hybrid"),
    # FIX P v1.1: Moldova standalone contemporana
    ("Moldova se pregătește pentru alegerile parlamentare",      True,  "hybrid"),
    ("Agricultorii moldoveni cer sprijin de la guvern",          True,  "hybrid"),
    ("Moldovenii au votat masiv în diaspora",                    True,  "hybrid"),

    # Neutre — NU trebuie sa prinda nimic
    ("Guvernul României a aprobat un nou program social",        False, "neither"),
    ("Vremea va fi însorită în toată țara",                       False, "neither"),
    ("Echipa națională a câștigat meciul amical",                 False, "neither"),
    ("Prețul benzinei a scăzut cu 5 bani",                        False, "neither"),
    ("Principatele dunărene în secolul XIX",                      False, "neither"),  # edge: „moldova" istoric, nu prinde
]


def _verify_integrity() -> None:
    """
    Ruleaza setul canonic de test-cases si ridica ImportError daca vreunul
    esueaza. Apelat automat la primul import al modulului.

    De ce fail loud: daca cineva modifica pattern-urile fara sa inteleaga
    implicatiile, cel mai rau scenariu este ca modulul sa continue sa
    functioneze „aparent" si sa contamineze silentios dataset-ul clasei 0.
    Preferabil ca intreg proiectul sa refuze sa porneasca pana ce
    dezvoltatorul revede modificarea.
    """
    errors: list[str] = []

    for text, expected_matched, expected_cluster in _CANONICAL_TESTS:
        result = topic_match_details(text)

        if result.matched != expected_matched:
            errors.append(
                f"  FAIL: '{text[:60]}' → matched={result.matched}, "
                f"așteptat {expected_matched}"
            )
            continue

        if expected_cluster == "core" and not result.matched_core:
            errors.append(
                f"  FAIL: '{text[:60]}' → nu a prins core (așteptat)"
            )
        elif expected_cluster == "hybrid" and not result.matched_hybrid:
            errors.append(
                f"  FAIL: '{text[:60]}' → nu a prins hybrid (așteptat)"
            )
        elif expected_cluster == "neither" and (result.matched_core or result.matched_hybrid):
            errors.append(
                f"  FAIL: '{text[:60]}' → a prins {result.matched_patterns} "
                "(așteptat: nimic)"
            )

    if errors:
        msg = (
            "thematic_filters.py: self-test INTEGRITY FAIL\n"
            + "\n".join(errors)
            + "\n\nModulul refuză să se încarce pentru a preveni contaminarea "
            "silențioasă a dataset-ului. Verifică modificările aduse "
            "UKRAINE_CORE_PATTERNS sau MOLDOVA_HYBRID_PATTERNS."
        )
        raise ImportError(msg)


# Ruleaza self-test la import
_verify_integrity()


# ══════════════════════════════════════════════════════════════════════════════
# Numerotare pentru debug / raportare
# ══════════════════════════════════════════════════════════════════════════════

def describe() -> str:
    """
    Returneaza o descriere human-readable a modulului. Util pentru logging
    la startul scraperelor.
    """
    return (
        f"thematic_filters v{__version__}\n"
        f"  Source: {__source_reference__}\n"
        f"  UKRAINE_CORE_PATTERNS:    {len(UKRAINE_CORE_PATTERNS)} regex-uri\n"
        f"  MOLDOVA_HYBRID_PATTERNS:  {len(MOLDOVA_HYBRID_PATTERNS)} regex-uri\n"
        f"  UKRAINE_PATTERNS (total): {len(UKRAINE_PATTERNS)} regex-uri\n"
        f"  Self-test: ✓ OK (verificat la import)"
    )


if __name__ == "__main__":
    # Permite rularea directa pentru sanity check: `python thematic_filters.py`
    print(describe())
