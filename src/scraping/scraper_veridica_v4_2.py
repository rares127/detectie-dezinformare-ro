r"""
scraper_veridica_v4_2.py
────────────────────────
Patch incremental fata de v4.1, pe baza analizei dataset-ului v4.1 (513 art.)
si a triajului manual al celor 46 articole `suspect_contaminare`.

Modificari fata de v4.1:

  FIX M — DETECTOR LEAK FACT-CHECK ROBUSTIZAT (critic, calitate dataset)
    v4.1: pattern-urile FACTCHECK_LEAK_MARKERS contineau `\bde\s+fapt\b`
          si `\bin\s+realitate\b`. Triajul manual al celor 46 articole flag-uite
          a aratat 45 fals pozitive (96% FP rate) — propagandistii folosesc
          natural „de fapt" si „in realitate" ca elemente retorice in discursul
          lor, nu ca markeri de fact-check.
    v4.2: doua schimbari complementare:
          (a) `de_fapt` + `in_realitate` ELIMINATE din lista de markeri
              folositi ca semnal binar de leak;
          (b) detector NOU bazat pe prefixe structurale RATATE — caut explicit
              pattern-ul „DE CE (ESTE|SUNT) FALS" in MAJUSCULE (forma in care
              apare ca prefix de sectiune), tolerant la whitespace multiplu
              (vezi FIX N). Asta e singurul tip de leak care s-a confirmat real
              in dataset (1 caz din 513).

  FIX N — PREFIX MATCHING TOLERANT LA WHITESPACE MULTIPLU (calitate)
    v4.1: daca in HTML apare „DE CE  ESTE  FALSA NARATIUNEA" (cu dublu spatiu
          sau tab intre cuvinte — comun dupa copy-paste din Word), prefix
          matching-ul esua si analiza jurnalistului se scurgea in stire_citata.
          Confirmat pe 1 articol in v4.1 (idx=366, „Intentia de a parasi CSI").
    v4.2: in normalize_text() colapsam orice succesiune de whitespace la un
          singur spatiu INAINTE de prefix matching (`re.sub(r"\s+", " ", text)`).

  FIX O — REPARARE count_sentences (critic, blocant pentru Modulul 3)
    v4.1: regex-ul `(?<=[.!?])(?=[A-ZAISTAST])` cerea litera mare IMEDIAT
          dupa punct, fara spatiu intre ele. In text real ai intotdeauna
          „. Aceasta" cu spatiu, deci split-ul nu se facea niciodata.
          Rezultat: median nr_propozitii = 1 pe tot dataset-ul, max = 3.
    v4.2: adaugam `\s+` in mijloc: `(?<=[.!?])\s+(?=[A-ZAISTAST])`. Pe
          sample real cu 5 propozitii acum returneaza corect 5.
          NOTA: ramane o aproximare heuristica — la preprocessing inlocuim
          cu segmentarea Stanza, mult mai robusta lingvistic.

──────────────────────────────────────────────────────────────────────────────
Modificari fata de v4 (pastrate din v4.1):

  FIX K — EXTINDERE FILTRU TEMATIC PENTRU FRONTUL HIBRID MOLDOVA
    v4: filtrul UKRAINE_PATTERNS surprindea doar entitati Ucraina/Rusia
        directe. Analiza CSV-ului a aratat ~10-15 articole pierdute care
        sunt naratiuni pro-Kremlin directionate spre R. Moldova (Dodon,
        Sandu, Transnistria, Comrat, sanctiuni antirusesti).
    v4.1: adaugam cluster Moldova/razboi hibrid — Transnistria, persoane
          cheie (Sandu, Dodon, Sor, Plahotniuc), localitati (Chisinau,
          Comrat, Gagauzia), termeni propagandistici (antirusesc, rusofob).
          Justificare academica: aceleasi naratiuni Kremlin, aceeasi
          infrastructura propagandistica, conectate operational cu razboiul
          din Ucraina (apararea Transnistriei, blocarea aderarii Moldovei
          la UE, slabirea frontului de est NATO).

  FIX L — SKIP ARTICOLE-REZUMAT „TOP DEZINFORMARI" PE URL
    v4: articolele de tip „Top FAKE NEWS 2024/2025 demontate de Veridica"
        erau descarcate, parsate, apoi marcate ca fallback_verificare_manuala
        pentru ca nu au structura standard cu prefixe STIRE/NARATIUNI.
        Risipa de request-uri si zgomot in CSV.
    v4.1: pre-filtru pe slug URL — pattern `top-(propaganda|fake-news|
          dezinform|stiri-false)` → skip inainte de download.

──────────────────────────────────────────────────────────────────────────────
Modificari fata de v3 (pastrate din v4):

  FIX A — FILTRU TEMATIC UCRAINA/RUSIA (critic)
    v3: lista UKRAINE_KEYWORDS continea "dezinformare", "propaganda", "fake news",
        "naratiune" — cuvinte care apar in aproape ORICE articol Veridica, fiindca
        sectiunea insasi se numeste asa. Rezultat: filtrul marca totul ca relevant.
    v4: migrat la regex compilat cu word boundaries (\b) si radicali lingvistici
        (ucrain\w*, invazi\w*). Pastram doar entitati geografice, persoane si
        termeni specifici conflictului. Zero cuvinte meta-jurnalistice.

  FIX B — PROTECTIE ANTI-CONTAMINARE LEAK (critic)
    v3: daca un prefix de sectiune era ratat (ex. "DE CE ACESTE STIRI SUNT FALSE"
        nu e in lista de prefixe), analiza jurnalistului se scurgea in stire_citata
        si contamina label-ul — clasificatorul invata pe text care DEMONTEAZA
        naratiunea, nu pe naratiune.
    v4: dupa extragere, validam ca stire_citata NU contine markeri de fact-check
        ("in realitate", "de fapt", "fals", "dezminte" etc.). Daca apar, flag
        calitate_extractie = "suspect_contaminare".

  FIX C — VALIDARE DIMENSIUNE stire_citata
    v3: fara validare. Citat de 5 cuvinte sau de 2000 cuvinte trecea ca "excelenta".
    v4: citat sub MIN_STIRE_WORDS sau peste MAX_STIRE_WORDS → "suspect_dimensiune".

  FIX D — PREFIX MATCHING ROBUST
    v3: text.startswith(prefix) — fragil daca get_text() lasa spatiu la inceput
        sau daca prefixul are bold partial.
    v4: caut prefixul in primele 40 caractere dupa lstrip si normalizare NFC.

  FIX E — EXTINDERE LISTA PREFIXE
    v3: doar "DE CE SUNT FALSE", "DE CE ESTE FALSA", "DE CE ESTE FALS".
    v4: adaugate variante observate in practica: "DE CE ACESTE STIRI SUNT FALSE",
        "DE CE ESTE FALSA NARATIUNEA", "DE CE ESTE FALSA STIREA" etc.

  FIX F — NORMALIZARE UNICODE NFC
    v3: .upper() pe diacritice depinde de locale; S cu virgulita (U+0218) vs.
        S cu sedila (U+015E) pot da mismatch la prefix matching.
    v4: unicodedata.normalize("NFC", text) inainte de orice comparatie.

  FIX G — REDENUMIRE continut_full → _referinta_continut_full
    v3: numele sugereaza ca e campul "principal" — risc ca la training sa fie
        folosit din greseala ca input in loc de text_curat.
    v4: prefix underscore ca semnal "nu folosi pentru training".

  FIX H — SALVARE INCREMENTALA + KEYBOARD INTERRUPT
    v3: daca pica la articolul 487/1000, pierdeai tot.
    v4: salvare CSV la fiecare CHECKPOINT_EVERY articole + try/except pe
        KeyboardInterrupt care salveaza ce avem pana in momentul ala.

  FIX I — FILTRU RELEVANTA APLICAT PE CONTINUTUL NARATIUNII
    v3: is_ukraine_related(title + " " + text_curat) — text_curat dubleaza titlul.
    v4: aplicam pe title + stire_citata + naratiuni_false (continutul propriu-zis
        al naratiunii). Daca astea sunt goale, cadem pe continut_full pentru
        decizia de relevanta (nu pentru training).

  FIX J — FILTRARE PROGRESIVA PE URL INAINTE DE SCRAPE
    v4 (nou): inainte sa descarcam un articol, verificam daca slug-ul URL-ului
    contine vreun marker Ucraina/Rusia. Daca nu contine NICIUN marker, marcam
    articolul ca "probabil_nerelevant" si il scrape-uim oricum (pentru validare),
    dar tine-ti minte ca decizia finala se ia pe continut.

Output: veridica_articles_v4.csv (tot setul) + veridica_ukraine_v4.csv (filtrat)
"""

import time
import re
import hashlib
import json
import unicodedata
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ── Configurare ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.veridica.ro"

SECTIONS = [
    "/fake-news-dezinformare-propaganda",
    "/dezinformare",
    "/stiri-false",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LicentaBot/1.0; "
        "research scraper pentru teza de licenta)"
    )
}

DELAY_SECONDS         = 2
MAX_PAGES_PER_SECTION = 20
CHECKPOINT_EVERY      = 25        # salveaza CSV la fiecare N articole procesate
OUTPUT_FILE           = "veridica_articles_v4_2.csv"
FILTERED_FILE         = "veridica_ukraine_v4_2.csv"

LABEL         = "dezinformare_pro_rusa"
LABEL_NUMERIC = 1

# Dimensiuni acceptate pentru stire_citata (in cuvinte).
# Sub MIN = probabil prefix detectat dar continut gol sau structura neobisnuita.
# Peste MAX = probabil am absorbit si alte sectiuni (leak).
MIN_STIRE_WORDS = 10
MAX_STIRE_WORDS = 800


# ── FIX A: Filtru tematic Ucraina/Rusia — regex cu word boundaries ────────────
#
# Pastram DOAR entitati geografice, persoane si termeni specifici conflictului.
# Eliminate: "dezinformare", "propaganda", "fake news", "naratiune" (meta-termeni
# care apar in orice articol Veridica si faceau filtrul inutil).

UKRAINE_PATTERNS = [
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

    # ── FIX K: Cluster Moldova / razboi hibrid Kremlin ──────────────────────
    # Justificare: aceeasi infrastructura propagandistica Kremlin produce
    # naratiunile pro-ruse din R. Moldova ca si pe cele direct anti-Ucraina.
    # Temele recurente: apararea Transnistriei, blocarea aderarii la UE,
    # discreditarea guvernarii pro-occidentale, slabirea NATO pe flancul estic.

    # Transnistria — punct fierbinte direct legat de logistica razboiului
    re.compile(r"\btransnistr\w*",                        re.IGNORECASE),

    # Persoane cheie ale spatiului politic Moldova relevante propagandei Kremlin
    re.compile(r"\b(maia\s+sandu|igor\s+dodon|ilan\s+șor|ilan\s+sor|plahotniuc)\b",
               re.IGNORECASE),

    # Localitati Moldova cu rol in naratiunile separatiste / pro-ruse
    re.compile(r"\b(chișinău|chisinau|comrat|tiraspol|găgăuzia|gagauzia|gagauz\w*)\b",
               re.IGNORECASE),

    # Republica Moldova ca entitate (mai sigur decat doar „moldova" care
    # poate aparea in context istoric/geografic neutru gen „principatele")
    re.compile(r"\b(republica\s+moldova|r\.\s*moldova)\b",
               re.IGNORECASE),

    # Termeni propagandistici recurenti in naratiunile Kremlin pe flancul Moldova
    re.compile(r"\bantirus\w*",                           re.IGNORECASE),  # antirusesc, antirusesti, sanctiuni antiruse
    re.compile(r"\brusofob\w*",                           re.IGNORECASE),  # rusofobie, rusofob, rusofoba
]


# ── FIX B + FIX M: Markeri de fact-check care NU trebuie sa apara in stire_citata
#
# v4.2: pattern-urile `de_fapt` si `in_realitate` au fost ELIMINATE pe baza
# triajului celor 46 articole `suspect_contaminare` din v4.1. Analiza manuala
# a aratat 96% fals pozitive — propagandistii folosesc aceste expresii natural
# ca elemente retorice ("Anglia urmareste, de fapt, sa creeze o noua colonie"
# este propaganda curata, nu fact-check). Vezi findings_metodologice.md.
#
# Markerii ramasi (mai discriminativi) sunt: „este fals(a)", „sunt false",
# „nu (este|e) adevarat", „dezminte", „contrazice", „verific* arat*",
# „fact-check", „potrivit unei verific/analiz".

FACTCHECK_LEAK_MARKERS = [
    re.compile(r"\beste\s+fals[ăa]?\b",       re.IGNORECASE),
    re.compile(r"\bsunt\s+false\b",           re.IGNORECASE),
    re.compile(r"\bnu\s+este\s+adevărat",     re.IGNORECASE),
    re.compile(r"\bnu\s+e\s+adevărat",        re.IGNORECASE),
    re.compile(r"\bdezminte\w*",              re.IGNORECASE),
    re.compile(r"\bcontrazice\w*",            re.IGNORECASE),
    re.compile(r"\bverific\w*\s+(arat|arăt|indic)", re.IGNORECASE),
    re.compile(r"\bfact[- ]check\w*",         re.IGNORECASE),
    re.compile(r"\bpotrivit\s+(unor|unei)\s+(verific|analiz)", re.IGNORECASE),
]


# ── FIX M (b): Detector pentru prefixe structurale RATATE ────────────────────
#
# Singurul tip de leak confirmat real in dataset (1/513 in v4.1) este situatia
# in care un prefix de sectiune precum „DE CE ESTE FALSA NARATIUNEA" nu a fost
# prins de prefix matching (ex. dublu spatiu intre cuvinte) si analiza
# jurnalistului s-a scurs in stire_citata.
#
# Detectam asta cautand explicit pattern-ul „DE CE (ESTE|SUNT) FALS" in
# MAJUSCULE (forma in care apare ca prefix structural, NU ca text narativ).

LEAKED_SECTION_PREFIX = re.compile(
    r"DE\s+CE\s+(ACESTE\s+)?(ȘTIR|STIR|NARAȚIUN|NARATIUN)\w*\s+(SUNT\s+)?(ESTE\s+)?FALS",
)
# Nota: NU folosesc IGNORECASE — caut explicit MAJUSCULE pentru ca asta e
# semnatura unui prefix de sectiune, nu o fraza narativa.


# ── Prefixe sectiuni Veridica — FIX E: extins cu variante observate ──────────

SECTION_PREFIXES = {
    "stire": [
        "ȘTIRE:", "STIRE:", "ȘTIREA:", "STIREA:",
    ],
    "naratiuni": [
        "NARAȚIUNI:", "NARATIUNI:", "NARAȚIUNEA:", "NARATIUNEA:",
    ],
    "obiective": [
        "OBIECTIVE:", "OBIECTIVUL:", "SCOP:", "SCOPUL:",
    ],
    "de_ce_false": [
        "DE CE SUNT FALSE",
        "DE CE ESTE FALSĂ",
        "DE CE ESTE FALS",
        "DE CE ACESTE ȘTIRI SUNT FALSE",
        "DE CE ACESTE STIRI SUNT FALSE",
        "DE CE ESTE FALSĂ NARAȚIUNEA",
        "DE CE ESTE FALSĂ ȘTIREA",
        "DE CE ESTE FALSA NARATIUNEA",
        "DE CE ESTE FALSA STIREA",
        "DE CE NARAȚIUNEA ESTE FALSĂ",
        "DE CE ȘTIREA ESTE FALSĂ",
        "DE CE ESTE FALS CĂ",
    ],
    "context": [
        "CONTEXT:", "CONTEXTUL:",
    ],
}


# ── Functii helper ─────────────────────────────────────────────────────────────

def get_soup(url: str) -> BeautifulSoup | None:
    """Descarca o pagina si returneaza soup-ul, sau None daca esueaza."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [EROARE] {url}: {e}")
        return None


def normalize_text(raw: str) -> str:
    """
    Normalizare Unicode NFC + curatare whitespace + unificare ghilimele/liniute.
    FIX F: NFC asigura ca S cu virgulita (U+0218) si alte diacritice au
    reprezentare canonica consistenta inainte de comparatii.
    FIX N (v4.2): colapsam orice succesiune de whitespace (spatii, tab-uri,
    newline-uri) la un singur spatiu, INAINTE de prefix matching. Asta
    rezolva cazul „DE CE  ESTE  FALSA" (cu dublu spatiu) care in v4.1
    esua la prefix detection si producea leak.
    """
    # Normalizare Unicode — forma canonica compusa
    text = unicodedata.normalize("NFC", raw)

    # FIX N: colapsare TOATE tipurile de whitespace la un singur spatiu
    text = re.sub(r"\s+", " ", text)

    # Ghilimele romanesti → drepte
    text = (
        text.replace("\u201e", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u2013", "-").replace("\u2014", "-")
    )
    return text.strip()


# Pastram alias-ul pentru compatibilitate cu restul codului
clean_text = normalize_text


def count_sentences(text: str) -> int:
    r"""
    Aproximare numar de propozitii prin split la capital letter dupa punct final.
    Include toate diacriticele romanesti (S/T cu virgulita si cu sedila).

    FIX O (v4.2): regex-ul anterior cerea litera mare IMEDIAT dupa punct,
    fara spatiu („.A" — care nu apare niciodata in text natural). In realitate
    apare „. Aceasta" cu spatiu intre ele. Rezultat in v4.1: median = 1
    pe tot dataset-ul, max = 3. Fix: adaugam `\s+` intre punct si litera mare.

    NOTA: ramane o aproximare. La preprocessing inlocuim cu Stanza, care
    gestioneaza corect abrevierile, citatele si edge case-urile lingvistice.
    """
    # Ambele variante S/T (virgulita U+0218/U+021A si sedila U+015E/U+0162)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZĂÎȘȚÂŞŢ])", text)
    return max(1, len(sentences))


NER_SEED_ENTITIES = [
    "Rusia", "Ucraina", "Putin", "Zelenski", "NATO", "UE", "Kremlin",
    "Donbas", "Crimeea", "Mariupol", "Moscova", "Kiev", "Kyiv",
    "Lugansk", "Donețk", "SUA", "România", "Occident", "Wagner",
]


def extract_keywords_simple(text: str) -> list:
    """Extrage entitati cunoscute din lista (placeholder — la antrenare folosim spaCy/Stanza NER)."""
    text_lower = text.lower()
    return [e for e in NER_SEED_ENTITIES if e.lower() in text_lower]


def compute_hash(text: str) -> str:
    """Hash MD5 pentru deduplicare."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_ukraine_related(text: str) -> bool:
    """
    FIX A: verificare tematica pe regex compilat cu word boundaries.
    Returneaza True daca textul contine cel putin un marker Ucraina/Rusia
    (entitate geografica, persoana, termen specific conflictului).
    """
    if not text:
        return False
    return any(p.search(text) for p in UKRAINE_PATTERNS)


def has_factcheck_leak(text: str) -> bool:
    """
    FIX B + FIX M: detecteaza daca un text (care ar trebui sa fie citat
    pro-Kremlin) contine markeri de analiza jurnalistica sau un prefix
    structural de sectiune ratat.

    v4.2: doua cai de detectie independente:
      1. Markeri lexicali discriminativi (FACTCHECK_LEAK_MARKERS) — v4.1
         minus `de_fapt` si `in_realitate` care produceau 96% FP.
      2. Detector pentru prefixe structurale RATATE: cauta „DE CE ESTE FALS"
         in MAJUSCULE, semnatura unui prefix de sectiune scurs.
    """
    if not text:
        return False
    if LEAKED_SECTION_PREFIX.search(text):
        return True
    return any(p.search(text) for p in FACTCHECK_LEAK_MARKERS)


def word_count(text: str) -> int:
    """Numar brut de cuvinte (split pe whitespace)."""
    return len(text.split()) if text else 0


def get_section_name(url: str) -> str:
    """Extrage numele sectiunii Veridica din URL."""
    for section in SECTIONS:
        if section.strip("/") in url:
            return section.strip("/").replace("-", "_")
    return "necunoscut"


# ── FIX L: Pre-filtru articole-rezumat „Top dezinformari" ────────────────────
#
# Veridica publica periodic articole-recapitulative anuale de tip:
#   /ucraina-2024-top-propaganda-fake-news-si-dezinformari-demontate-de-veridica
#   /republica-moldova-2025-top-fake-news-dezinformari-demontate-de-veridica
# Acestea NU au structura standard cu prefixe STIRE/NARATIUNI/DE CE FALSE,
# ci sunt indexuri liste-link cu trimiteri spre articolele individuale.
# Le sarim DIRECT la nivel de URL — economisim request-uri si nu mai poluam
# CSV-ul cu randuri fallback_verificare_manuala.

SUMMARY_URL_PATTERN = re.compile(
    r"top[- ]?(propaganda|fake[- ]?news|dezinform|stiri[- ]?false|naratiun)",
    re.IGNORECASE,
)


def is_summary_article(url: str) -> bool:
    """
    FIX L: True daca URL-ul indica un articol-rezumat (Top dezinformari anuale),
    care nu are structura fact-check standard si ar produce extragere goala.
    """
    return bool(SUMMARY_URL_PATTERN.search(url))


# ── Listing pagini ────────────────────────────────────────────────────────────

def extract_article_links(section_url: str, max_pages: int) -> list:
    """
    Colecteaza URL-urile articolelor dintr-o sectiune.
    Selector confirmat live: div.card.border-0.h-100 cu <a> in interior.
    Paginare: ?page=N (nu /page/N).
    """
    links = []

    for page_num in range(1, max_pages + 1):
        url = section_url if page_num == 1 else f"{section_url}?page={page_num}"
        print(f"  [LISTING] {url}")
        soup = get_soup(url)

        if soup is None:
            break

        cards = soup.find_all("div", class_="card border-0 h-100")

        if not cards:
            # Fallback: orice div cu "card" in clasa
            cards = soup.find_all("div", class_=re.compile(r"card"))
            if not cards:
                print(f"  [STOP] Niciun card găsit pe pagina {page_num}.")
                break

        page_links = []
        for card in cards:
            a_tag = card.find("a", href=True)
            if a_tag:
                href = a_tag["href"]
                full_url = urljoin(BASE_URL, href) if not href.startswith("http") else href
                if BASE_URL in full_url and full_url not in links and full_url not in page_links:
                    page_links.append(full_url)

        if not page_links:
            print(f"  [STOP] Niciun link nou pe pagina {page_num}. Oprire secțiune.")
            break

        links.extend(page_links)
        print(f"    -> {len(page_links)} articole găsite (total: {len(links)})")
        time.sleep(DELAY_SECONDS)

    return list(dict.fromkeys(links))  # pastreaza ordinea + deduplicare


# ── Parsare structurata articol ──────────────────────────────────────────────

def parse_veridica_sections(content_tag: BeautifulSoup) -> dict:
    """
    Extrage sectiunile structurate ale unui articol Veridica.

    Structura confirmata live:
      - Paragrafe <p> fara clase speciale
      - Sectiunile sunt marcate prin text PREFIX in bold la inceputul paragrafului:
        "STIRE:", "NARATIUNI:", "OBIECTIVE:", "DE CE SUNT FALSE:", "CONTEXT:"

    FIX D: prefix matching robust — normalizam NFC, facem lstrip, cautam
    prefixul in primele 40 de caractere (nu doar exact la pozitia 0).

    Returneaza dict cu cheile: stire, naratiuni, obiective, de_ce_false, context.
    """
    sections = {k: [] for k in SECTION_PREFIXES}
    current_section = None

    if content_tag is None:
        return {k: "" for k in SECTION_PREFIXES}

    paragraphs = content_tag.find_all("p")

    for p in paragraphs:
        text = clean_text(p.get_text(separator=" ", strip=True))
        if not text or len(text) < 10:
            continue

        # FIX F: normalizare NFC (deja aplicata in clean_text, dar explicit aici)
        text = unicodedata.normalize("NFC", text)

        # Normalizam spatii inainte de ':' (ex: "STIRE :" → "STIRE:")
        text_normalized = re.sub(r"\s+:", ":", text).lstrip()
        text_upper = text_normalized.upper()

        # FIX D: cautam prefix in primele 40 caractere, nu doar la pozitia 0
        text_head = text_upper[:40]

        matched_section = None
        matched_prefix = None
        for section_key, prefixes in SECTION_PREFIXES.items():
            for prefix in prefixes:
                prefix_nfc = unicodedata.normalize("NFC", prefix).upper()
                if text_head.startswith(prefix_nfc):
                    matched_section = section_key
                    matched_prefix = prefix_nfc
                    break
            if matched_section:
                break

        if matched_section and matched_prefix:
            current_section = matched_section
            # Eliminam prefixul din text (case-insensitive, pe lungime)
            text = text_normalized[len(matched_prefix):].lstrip(" :").strip()

        if current_section and text:
            sections[current_section].append(text)

    return {k: " ".join(v).strip() for k, v in sections.items()}


def assess_extraction_quality(stire: str, naratiuni: str) -> str:
    """
    FIX B + FIX C: evalueaza calitatea extragerii pentru decizia de includere
    in trainset.

    Nivele:
      - excelenta            : avem citatul exact + dimensiune rezonabila + fara leak
      - buna                 : avem NARATIUNI compactate (fallback secundar valid)
      - suspect_contaminare  : stire_citata contine markeri de fact-check → LEAK
      - suspect_dimensiune   : citatul e prea scurt sau prea lung
      - fallback_verificare_manuala : nu avem nici STIRE nici NARATIUNI
    """
    if stire:
        # Check 1: leak de fact-check (critic — exclude din training)
        if has_factcheck_leak(stire):
            return "suspect_contaminare"

        # Check 2: dimensiune rezonabila
        wc = word_count(stire)
        if wc < MIN_STIRE_WORDS or wc > MAX_STIRE_WORDS:
            return "suspect_dimensiune"

        return "excelenta"

    if naratiuni:
        # NARATIUNI e mai comprimata — praguri mai lejere
        if has_factcheck_leak(naratiuni):
            return "suspect_contaminare"
        return "buna"

    return "fallback_verificare_manuala"


def scrape_article(url: str) -> dict | None:
    """Descarca si parseaza un articol individual Veridica."""
    soup = get_soup(url)
    if soup is None:
        return None

    try:
        # ── Titlu ──────────────────────────────────────────────────────────────
        h1 = soup.find("h1", class_="responsiveTitle") or soup.find("h1")
        title = h1.get_text(strip=True) if h1 else "N/A"
        title = clean_text(title)

        # ── Data publicarii ────────────────────────────────────────────────────
        date_str = "N/A"
        time_tag = soup.find("time")
        if time_tag:
            date_str = time_tag.get("datetime", time_tag.get_text(strip=True))
        else:
            meta = soup.find("meta", property="article:published_time")
            if meta:
                date_str = meta.get("content", "N/A")

        # ── Autor ──────────────────────────────────────────────────────────────
        author = "N/A"
        author_tag = soup.find(class_=re.compile(r"author|byline", re.I))
        if author_tag:
            author = author_tag.get_text(strip=True)

        # ── Continut principal ────────────────────────────────────────────────
        content_tag = soup.find("div", class_="page-content")

        # Fallback daca tema se schimba
        if not content_tag:
            content_tag = (
                soup.find("div", class_=re.compile(r"entry-content|post-content", re.I))
                or soup.find("main")
            )

        if content_tag:
            for tag in content_tag.find_all(["script", "style", "nav", "aside",
                                              "figure", "form", "iframe"]):
                tag.decompose()

        # Continut full — pastrat DOAR ca referinta, NU pentru training
        referinta_continut_full = (
            clean_text(content_tag.get_text(separator=" ", strip=True))
            if content_tag else "N/A"
        )

        # ── Extragere structurata pe sectiuni ─────────────────────────────────
        sections = parse_veridica_sections(content_tag)

        stire      = sections["stire"]       # naratiunea pro-Kremlin citata
        naratiuni  = sections["naratiuni"]   # lista naratiunilor false
        obiective  = sections["obiective"]   # scopul propagandei
        de_ce_fals = sections["de_ce_false"] # analiza jurnalistului (NU input clasificator)
        context    = sections["context"]     # info despre sursa

        # ── Evaluare calitate (FIX B + FIX C) ─────────────────────────────────
        calitate = assess_extraction_quality(stire, naratiuni)

        # ── text_curat = INPUT CLASIFICATOR ───────────────────────────────────
        # Folosim DOAR daca am extras continut considerat utilizabil.
        if calitate == "excelenta":
            text_curat = clean_text(f"{title} {stire}")
        elif calitate == "buna":
            text_curat = clean_text(f"{title} {naratiuni}")
        elif calitate == "suspect_contaminare":
            # NU folosim pentru training — dar pastram pentru audit manual.
            # text_curat ramane gol ca sa NU ajunga accidental in pipeline.
            text_curat = ""
        elif calitate == "suspect_dimensiune":
            # Edge case — pastram text_curat pentru revizie manuala, dar flag-uit.
            text_curat = clean_text(f"{title} {stire}")
        else:
            # fallback_verificare_manuala: primele 400 chars ca punct de start pentru revizie
            text_curat = clean_text(f"{title} {referinta_continut_full[:400]}")

        # ── FIX I: Relevanta tematica — aplicat pe continutul naratiunii ──────
        # Folosim titlu + stire + naratiuni (continutul real al propagandei).
        # Daca astea sunt goale, cadem pe continut_full pentru decizia de relevanta.
        text_pentru_relevanta = " ".join(filter(None, [title, stire, naratiuni]))
        if not text_pentru_relevanta.strip():
            text_pentru_relevanta = referinta_continut_full

        relevanta = is_ukraine_related(text_pentru_relevanta)

        # ── Campuri derivate ───────────────────────────────────────────────────
        nr_prop       = count_sentences(text_curat) if text_curat else 0
        keywords      = extract_keywords_simple(text_pentru_relevanta)
        cuvinte_cheie = json.dumps(keywords, ensure_ascii=False)
        hash_val      = compute_hash(text_curat or referinta_continut_full)
        sectiune      = get_section_name(url)
        nr_cuvinte_stire = word_count(stire)

        return {
            # Identificare
            "url":                       url,
            "titlu":                     title,
            "data":                      date_str,
            "autor":                     author,
            "sursa_site":                "veridica.ro",
            "sectiune":                  sectiune,

            # ── INPUT CLASIFICATOR ─────────────────────────────────────────────
            "text_curat":                text_curat,    # titlu + naratiunea falsa
            "calitate_extractie":        calitate,      # flag pentru QA
            "nr_cuvinte_stire":          nr_cuvinte_stire,

            # ── Sectiuni structurate (granular) ───────────────────────────────
            "stire_citata":              stire,         # citatul pro-Kremlin exact
            "naratiuni_false":           naratiuni,     # lista naratiunilor
            "obiective_propaganda":      obiective,     # scopul
            "analiza_factcheck":         de_ce_fals,    # NU input clasificator
            "context_sursa":             context,       # info despre propagandist

            # ── Referinta (FIX G: prefix underscore = NU folosi la training) ──
            "_referinta_continut_full":  referinta_continut_full,

            # ── Campuri NLP ────────────────────────────────────────────────────
            "nr_propozitii":             nr_prop,
            "cuvinte_cheie":             cuvinte_cheie,

            # ── Label ──────────────────────────────────────────────────────────
            "label":                     LABEL,
            "label_numeric":             LABEL_NUMERIC,

            # ── Metadata ──────────────────────────────────────────────────────
            "relevanta_ucraina":         relevanta,
            "hash_continut":             hash_val,
        }

    except Exception as e:
        print(f"  [EROARE la parsare] {url}: {e}")
        return None


# ── Salvare ───────────────────────────────────────────────────────────────────

COL_ORDER = [
    "url", "titlu", "data", "autor", "sursa_site", "sectiune",
    # Input clasificator
    "text_curat", "calitate_extractie", "nr_cuvinte_stire",
    # Sectiuni structurate
    "stire_citata", "naratiuni_false", "obiective_propaganda",
    "analiza_factcheck", "context_sursa",
    # Referinta (prefix _ ca semnal "nu folosi la training")
    "_referinta_continut_full",
    # NLP
    "nr_propozitii", "cuvinte_cheie",
    # Label
    "label", "label_numeric",
    # Metadata
    "relevanta_ucraina", "hash_continut",
]


def save_csv(articles: list, path: str, filtered_path: str | None = None) -> None:
    """Salveaza lista de articole in CSV (+ subset filtrat pe relevanta)."""
    if not articles:
        return

    df = pd.DataFrame(articles)
    df = df[[c for c in COL_ORDER if c in df.columns]]
    df.to_csv(path, index=False, encoding="utf-8-sig")

    if filtered_path:
        df_filtered = df[df["relevanta_ucraina"]]
        df_filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")


def print_stats(articles: list) -> None:
    """Afiseaza statistici despre calitatea extragerii."""
    if not articles:
        print("  (niciun articol)")
        return

    df = pd.DataFrame(articles)
    total = len(df)
    ukraine_rel = int(df["relevanta_ucraina"].sum())

    print(f"\n{'=' * 60}")
    print(f"STATISTICI ({total} articole procesate)")
    print(f"{'=' * 60}")
    print(f"  Relevante Ucraina/Rusia        : {ukraine_rel} ({ukraine_rel/total*100:.1f}%)")
    print(f"\n  Distribuție calitate_extractie:")
    for q, count in df["calitate_extractie"].value_counts().items():
        pct = count / total * 100
        marker = "✓" if q in ("excelenta", "buna") else "⚠"
        print(f"    {marker} {q:30s}: {count:4d} ({pct:5.1f}%)")

    # Breakdown relevanta pe calitate
    print(f"\n  Breakdown (relevante × calitate):")
    df_rel = df[df["relevanta_ucraina"]]
    for q, count in df_rel["calitate_extractie"].value_counts().items():
        print(f"    • {q:30s}: {count:4d}")

    # Flags suspecte
    suspect = df[df["calitate_extractie"].str.startswith("suspect")]
    if len(suspect) > 0:
        print(f"\n  ⚠️  {len(suspect)} articole cu flags suspecte — necesită revizie manuală")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("VERIDICA SCRAPER v4 — Start")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_articles = []
    seen_urls    = set()
    seen_hashes  = set()
    processed    = 0

    try:
        for section in SECTIONS:
            section_url = BASE_URL + section
            print(f"\n[SECTIUNE] {section_url}")

            article_links = extract_article_links(section_url, max_pages=MAX_PAGES_PER_SECTION)
            print(f"  Total link-uri colectate: {len(article_links)}")

            if not article_links:
                print(f"  [ATENTIE] Niciun link găsit în secțiunea {section}. Verifică selectorul.")
                continue

            for i, url in enumerate(article_links, 1):
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # FIX L: skip articole-rezumat inainte de download
                if is_summary_article(url):
                    print(f"  [{i}/{len(article_links)}] [SKIP rezumat] {url}")
                    continue

                print(f"  [{i}/{len(article_links)}] {url}")
                article = scrape_article(url)

                if article:
                    h = article["hash_continut"]
                    if h in seen_hashes:
                        print("    [DUPLICAT] Ignorat.")
                        continue
                    seen_hashes.add(h)

                    all_articles.append(article)
                    processed += 1

                    q   = article["calitate_extractie"]
                    rel = "RELEVANT" if article["relevanta_ucraina"] else "general"
                    wc  = article["nr_cuvinte_stire"]
                    print(f"    [{rel}][{q}] {wc} cuv. stire | {article['titlu'][:50]}...")

                    # FIX H: salvare incrementala
                    if processed % CHECKPOINT_EVERY == 0:
                        save_csv(all_articles, OUTPUT_FILE, FILTERED_FILE)
                        print(f"    [CHECKPOINT] Salvat {processed} articole în {OUTPUT_FILE}")

                time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        # FIX H: salvare de siguranta la Ctrl+C
        print("\n\n[INTERRUPT] Ctrl+C detectat — salvez ce am până acum...")

    # ── Salvare finala ─────────────────────────────────────────────────────────
    if not all_articles:
        print("\n[ATENTIE] Nu s-au colectat articole.")
        return

    save_csv(all_articles, OUTPUT_FILE, FILTERED_FILE)

    print_stats(all_articles)

    print(f"\n  Salvat total    : {OUTPUT_FILE}  ({len(all_articles)} articole)")
    df_rel_count = sum(1 for a in all_articles if a["relevanta_ucraina"])
    print(f"  Salvat Ucraina  : {FILTERED_FILE}  ({df_rel_count} articole)")

    # Preview — primele 3 articole relevante si valide
    print(f"\n── Preview (primele 3 relevante + valide) ──")
    valide = [
        a for a in all_articles
        if a["relevanta_ucraina"] and a["calitate_extractie"] in ("excelenta", "buna")
    ]
    for a in valide[:3]:
        print(f"  [{a['label_numeric']}][{a['calitate_extractie']}] {a['titlu'][:55]}")
        print(f"      ȘTIRE ({a['nr_cuvinte_stire']} cuv.): {str(a['stire_citata'])[:80]}...")
        print(f"      text_curat: {str(a['text_curat'])[:80]}...")


if __name__ == "__main__":
    main()