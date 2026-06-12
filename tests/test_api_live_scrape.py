"""
test_api_live_scrape.py
───────────────────────
Script de testare end-to-end al endpoint-ului POST /api/predict.

Pipeline:
  1. Scrape-uieste 5 articole din sectiunea fake-news de pe veridica.ro
     → extrage DOAR stire_citata (citatul pro-Kremlin), NU demontarea
  2. Scrape-uieste 5 articole despre Ucraina de pe surse credibile
     (g4media.ro cu fallback pe libertatea.ro, biziday.ro)
  3. Trimite fiecare text la POST /api/predict
  4. Compara verdictul primit cu eticheta asteptata
  5. Afiseaza tabel + metrici: accuracy, FP, FN

Comentariile si variabilele: romana.
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Configurare ────────────────────────────────────────────────────────────────

API_URL        = "http://localhost:8000/api/predict"
HEALTH_URL     = "http://localhost:8000/api/health"
DELAY_SECONDS  = 1.5      # pauza politicoasa intre request-uri de scraping
TIMEOUT_HTTP   = 15       # timeout HTTP in secunde
TIMEOUT_API    = 60       # timeout API (inferenta poate dura ~2-3s)

HEADERS_SCRAPER = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LicentaTestBot/1.0; "
        "test automatizat pentru teza de licenta)"
    )
}

# Etichete canonical — identice cu cele din API
LABEL_DEZINFO   = "dezinformare_pro_rusa"
LABEL_CREDIBIL  = "stire_credibila"

# Prefixe sectiuni Veridica — identice cu scraper_veridica_v4_2.py
SECTION_PREFIXES_STIRE = [
    "ȘTIRE:", "STIRE:", "ȘTIREA:", "STIREA:",
]
SECTION_PREFIXES_NARATIUNI = [
    "NARAȚIUNI:", "NARATIUNI:", "NARAȚIUNEA:", "NARATIUNEA:",
]
SECTION_PREFIXES_DE_CE_FALSE = [
    "DE CE SUNT FALSE", "DE CE ESTE FALSĂ", "DE CE ESTE FALS",
    "DE CE ACESTE ȘTIRI SUNT FALSE", "DE CE ACESTE STIRI SUNT FALSE",
    "DE CE ESTE FALSĂ NARAȚIUNEA", "DE CE ESTE FALSĂ ȘTIREA",
    "DE CE ESTE FALSA NARATIUNEA", "DE CE NARAȚIUNEA ESTE FALSĂ",
    "DE CE ȘTIREA ESTE FALSĂ", "DE CE ESTE FALS CĂ",
]


# ── Structura articol de test ──────────────────────────────────────────────────

@dataclass
class ArticolTest:
    """Un articol cu eticheta asteptata si rezultatul clasificatorului."""
    sursa:           str
    url:             str
    titlu:           str
    text_input:      str   # textul trimis la /api/predict
    label_asteptat:  str   # dezinformare_pro_rusa / stire_credibila

    # Campuri completate dupa apelul API
    verdict_primit:  Optional[str]   = None
    incredere:       Optional[float] = None
    scor_baseline:   Optional[float] = None
    scor_modul3:     Optional[float] = None
    timp_inferenta:  Optional[float] = None
    eroare_api:      Optional[str]   = None

    @property
    def corect(self) -> Optional[bool]:
        if self.verdict_primit is None:
            return None
        # Tratam 'incert' ca incorect — decizia nu e clara
        return self.verdict_primit == self.label_asteptat


# ── Utilitare scraping ─────────────────────────────────────────────────────────

def get_soup(url: str) -> Optional[BeautifulSoup]:
    """Descarca o pagina HTML si returneaza soup-ul BeautifulSoup."""
    try:
        r = requests.get(url, headers=HEADERS_SCRAPER, timeout=TIMEOUT_HTTP)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [EROARE HTTP] {url}: {e}")
        return None


def normalizeaza_text(raw: str) -> str:
    """
    NFC + colapsare whitespace + unificare ghilimele.
    Identic cu normalize_text() din scraper_veridica_v4_2.py.
    """
    text = unicodedata.normalize("NFC", raw)
    text = re.sub(r"\s+", " ", text)
    text = (
        text.replace("„", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
            .replace("–", "-").replace("—", "-")
    )
    return text.strip()


# ── Scraping Veridica ──────────────────────────────────────────────────────────

def extrage_stire_citata_veridica(content_tag: BeautifulSoup) -> tuple[str, str]:
    """
    Extrage stire_citata (citatul propagandistic) dintr-un articol Veridica.

    Logica: iteram paragrafele <p>. Cand gasim un prefix de sectiune (STIRE:,
    NARATIUNI: etc.), inregistram continutul pana la primul prefix de DEMONTARE
    (DE CE SUNT FALSE / DE CE ESTE FALSA etc.). Returnam (stire, naratiuni).

    Returneaza: (stire_citata, naratiuni_false)
    Daca nu gasim prefixe structurate, returnam ("", "").
    """
    stire_chunks: list[str]     = []
    naratiuni_chunks: list[str] = []
    sectiune_curenta: Optional[str] = None   # "stire", "naratiuni", "stop"

    if content_tag is None:
        return "", ""

    for p in content_tag.find_all("p"):
        text = normalizeaza_text(p.get_text(separator=" ", strip=True))
        if not text or len(text) < 5:
            continue

        text_upper = text.upper()
        text_head  = text_upper[:50]  # cautam prefixul in primele 50 caractere

        # ── Detectare prefix DEMONTARE → oprim colectarea ─────────────────────
        e_demontare = any(
            text_head.startswith(unicodedata.normalize("NFC", p_).upper())
            for p_ in SECTION_PREFIXES_DE_CE_FALSE
        )
        if e_demontare:
            sectiune_curenta = "stop"
            break

        # ── Detectare prefix STIRE: ────────────────────────────────────────────
        e_stire = False
        for prefix in SECTION_PREFIXES_STIRE:
            prefix_nfc = unicodedata.normalize("NFC", prefix).upper()
            if text_head.startswith(prefix_nfc):
                sectiune_curenta = "stire"
                # Eliminam prefixul din text
                text = text[len(prefix):].lstrip(" :").strip()
                e_stire = True
                break

        # ── Detectare prefix NARATIUNI: ────────────────────────────────────────
        if not e_stire:
            for prefix in SECTION_PREFIXES_NARATIUNI:
                prefix_nfc = unicodedata.normalize("NFC", prefix).upper()
                if text_head.startswith(prefix_nfc):
                    sectiune_curenta = "naratiuni"
                    text = text[len(prefix):].lstrip(" :").strip()
                    break

        # ── Colectare text in sectiunea curenta ───────────────────────────────
        if sectiune_curenta == "stire" and text:
            stire_chunks.append(text)
        elif sectiune_curenta == "naratiuni" and text:
            naratiuni_chunks.append(text)

    stire     = " ".join(stire_chunks).strip()
    naratiuni = " ".join(naratiuni_chunks).strip()
    return stire, naratiuni


def scrape_articol_veridica(url: str) -> Optional[ArticolTest]:
    """Descarca si parseaza un articol individual de pe Veridica."""
    soup = get_soup(url)
    if soup is None:
        return None

    # Titlu
    h1    = soup.find("h1", class_="responsiveTitle") or soup.find("h1")
    titlu = normalizeaza_text(h1.get_text(strip=True)) if h1 else "N/A"

    # Continut
    content_tag = soup.find("div", class_="page-content")
    if not content_tag:
        content_tag = soup.find("div", class_=re.compile(r"entry-content|post-content", re.I)) or soup.find("main")

    if content_tag:
        for tag in content_tag.find_all(["script", "style", "nav", "aside", "figure", "form", "iframe"]):
            tag.decompose()

    stire, naratiuni = extrage_stire_citata_veridica(content_tag)

    # Input pentru clasificator: preferam stire_citata, fallback pe naratiuni
    if stire and len(stire.split()) >= 10:
        text_input = normalizeaza_text(f"{titlu} {stire}")
    elif naratiuni and len(naratiuni.split()) >= 10:
        text_input = normalizeaza_text(f"{titlu} {naratiuni}")
    else:
        # Fallback: primele 400 caractere din continut (pentru articole fara structura)
        continut_brut = normalizeaza_text(
            content_tag.get_text(separator=" ", strip=True) if content_tag else ""
        )
        text_input = normalizeaza_text(f"{titlu} {continut_brut[:400]}")

    if not text_input.strip() or len(text_input.split()) < 5:
        return None

    return ArticolTest(
        sursa="veridica.ro",
        url=url,
        titlu=titlu[:80],
        text_input=text_input,
        label_asteptat=LABEL_DEZINFO,
    )


def colecteaza_url_veridica(nr_dorit: int = 5) -> list[str]:
    """
    Colecteaza URL-uri de articole din sectiunile fake-news Veridica.
    Returneaza primele `nr_dorit` URL-uri unice, filtrand articolele-rezumat.
    """
    sectiuni = [
        "https://www.veridica.ro/fake-news-dezinformare-propaganda",
        "https://www.veridica.ro/dezinformare",
    ]
    # Pattern articole-rezumat anuale (de tip "Top fake-news 2024") — le sarim
    PATTERN_REZUMAT = re.compile(
        r"top[- ]?(propaganda|fake[- ]?news|dezinform|stiri[- ]?false|naratiun)",
        re.IGNORECASE,
    )

    urls_colectate: list[str] = []
    vazute: set[str] = set()

    for sectiune_url in sectiuni:
        if len(urls_colectate) >= nr_dorit:
            break

        for page_num in range(1, 5):
            url_pagina = sectiune_url if page_num == 1 else f"{sectiune_url}?page={page_num}"
            soup = get_soup(url_pagina)
            if soup is None:
                break

            carduri = soup.find_all("div", class_="card border-0 h-100")
            if not carduri:
                carduri = soup.find_all("div", class_=re.compile(r"\bcard\b"))

            for card in carduri:
                a_tag = card.find("a", href=True)
                if not a_tag:
                    continue
                href = a_tag["href"]
                full_url = urljoin("https://www.veridica.ro", href) if not href.startswith("http") else href

                # Filtram articolele-rezumat si duplicatele
                if "veridica.ro" not in full_url:
                    continue
                if full_url in vazute:
                    continue
                if PATTERN_REZUMAT.search(full_url):
                    continue

                vazute.add(full_url)
                urls_colectate.append(full_url)

                if len(urls_colectate) >= nr_dorit * 3:  # colectam mai multe pentru rezerva
                    break

            if len(urls_colectate) >= nr_dorit * 3:
                break

            time.sleep(DELAY_SECONDS)

    return urls_colectate[:nr_dorit * 3]


# ── Scraping surse credibile ───────────────────────────────────────────────────

def scrape_articol_g4media(url: str) -> Optional[ArticolTest]:
    """Descarca si parseaza un articol de stire de pe G4Media."""
    soup = get_soup(url)
    if soup is None:
        return None

    # Titlu
    h1    = soup.find("h1", class_=re.compile(r"title|titlu", re.I)) or soup.find("h1")
    titlu = normalizeaza_text(h1.get_text(strip=True)) if h1 else "N/A"

    # Continut articol — selector validat pe G4Media
    content_tag = (
        soup.find("div", class_="article-content-area")
        or soup.find("div", class_=re.compile(r"entry-content|article.?content|post.?body", re.I))
        or soup.find("article")
    )

    if content_tag:
        for tag in content_tag.find_all(["script", "style", "aside", "figure", "nav", "iframe", "form"]):
            tag.decompose()
        text_articol = normalizeaza_text(content_tag.get_text(separator=" ", strip=True))
    else:
        text_articol = ""

    # Limitam la primele ~800 cuvinte pentru a nu depasi limitele modelului
    cuvinte = text_articol.split()
    if len(cuvinte) > 800:
        text_articol = " ".join(cuvinte[:800])

    text_input = normalizeaza_text(f"{titlu} {text_articol}")

    if not text_input.strip() or len(text_input.split()) < 10:
        return None

    return ArticolTest(
        sursa="g4media.ro",
        url=url,
        titlu=titlu[:80],
        text_input=text_input,
        label_asteptat=LABEL_CREDIBIL,
    )


def scrape_articol_libertatea(url: str) -> Optional[ArticolTest]:
    """Descarca si parseaza un articol de pe Libertatea."""
    soup = get_soup(url)
    if soup is None:
        return None

    h1    = soup.find("h1")
    titlu = normalizeaza_text(h1.get_text(strip=True)) if h1 else "N/A"

    content_tag = (
        soup.find("div", class_=re.compile(r"article.?body|story.?body|entry.?content|article.?content", re.I))
        or soup.find("section", class_=re.compile(r"article", re.I))
        or soup.find("article")
    )

    if content_tag:
        for tag in content_tag.find_all(["script", "style", "aside", "figure", "nav", "iframe", "form"]):
            tag.decompose()
        text_articol = normalizeaza_text(content_tag.get_text(separator=" ", strip=True))
    else:
        text_articol = ""

    cuvinte = text_articol.split()
    if len(cuvinte) > 800:
        text_articol = " ".join(cuvinte[:800])

    text_input = normalizeaza_text(f"{titlu} {text_articol}")

    if not text_input.strip() or len(text_input.split()) < 10:
        return None

    return ArticolTest(
        sursa="libertatea.ro",
        url=url,
        titlu=titlu[:80],
        text_input=text_input,
        label_asteptat=LABEL_CREDIBIL,
    )


def colecteaza_articole_g4media(nr_dorit: int = 5) -> list[ArticolTest]:
    """
    Colecteaza articole despre Ucraina de pe G4Media via tag-ul editorial.
    Returneaza o lista de ArticolTest cu label_asteptat = stire_credibila.
    """
    TAG_URL = "https://www.g4media.ro/tag/razboi-ucraina"
    articole: list[ArticolTest] = []
    vazute_urls: set[str] = set()    # deduplicare globala dupa URL intre pagini
    vazute_titluri: set[str] = set() # deduplicare secundara dupa titlu (G4Media featured vs card)
    pagina = 1

    while len(articole) < nr_dorit and pagina <= 3:
        url_pagina = TAG_URL if pagina == 1 else f"{TAG_URL}/page/{pagina}/"
        soup = get_soup(url_pagina)
        if soup is None:
            break

        # Selector articole G4Media confirmat empiric
        linkuri = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (
                "g4media.ro" in href
                and "/tag/" not in href
                and "/categoria/" not in href
                and "/page/" not in href
                and href not in vazute_urls
                and len(href) > 30
            ):
                linkuri.append(href)
                vazute_urls.add(href)

        for url_art in linkuri:
            if len(articole) >= nr_dorit:
                break
            time.sleep(DELAY_SECONDS)
            articol = scrape_articol_g4media(url_art)
            if articol and len(articol.text_input.split()) >= 30:
                # Deduplicare dupa titlu — G4Media pune acelasi articol in slot
                # featured si in cardul normal pe aceeasi pagina de tag
                titlu_norm = articol.titlu.lower().strip()
                if titlu_norm in vazute_titluri:
                    continue
                vazute_titluri.add(titlu_norm)
                articole.append(articol)
                print(f"  [G4Media] ✓ {articol.titlu[:60]}...")

        pagina += 1
        time.sleep(DELAY_SECONDS)

    return articole


def colecteaza_articole_libertatea(nr_dorit: int = 5) -> list[ArticolTest]:
    """
    Colecteaza articole despre Ucraina de pe Libertatea.ro.
    Folosim pagina de cautare sau tag-ul /razboi-rusia-ucraina/.
    """
    URLS_CANDIDAT = [
        "https://www.libertatea.ro/stiri/externe/razboi-rusia-ucraina",
        "https://www.libertatea.ro/tag/razboi-ucraina",
        "https://www.libertatea.ro/tag/ucraina",
    ]

    articole: list[ArticolTest] = []
    vazute: set[str] = set()

    for url_tag in URLS_CANDIDAT:
        if len(articole) >= nr_dorit:
            break

        soup = get_soup(url_tag)
        if soup is None:
            continue

        # Extrage link-uri de articole
        for a in soup.find_all("a", href=True):
            if len(articole) >= nr_dorit:
                break
            href = a["href"]

            # Normalizam URL-ul relativ
            if href.startswith("/"):
                href = "https://www.libertatea.ro" + href

            if (
                "libertatea.ro" in href
                and "/stiri/" in href
                and href not in vazute
                and not href.endswith("/stiri/externe/razboi-rusia-ucraina")
                and len(href) > 40
            ):
                vazute.add(href)
                time.sleep(DELAY_SECONDS)
                articol = scrape_articol_libertatea(href)
                if articol and len(articol.text_input.split()) >= 30:
                    articole.append(articol)
                    print(f"  [Libertatea] ✓ {articol.titlu[:60]}...")

        time.sleep(DELAY_SECONDS)

    return articole


def colecteaza_articole_biziday(nr_dorit: int = 5) -> list[ArticolTest]:
    """
    Colecteaza articole despre Ucraina de pe Biziday.ro — agregator de stiri.
    """
    URL_TAG = "https://biziday.ro/tag/ucraina/"
    articole: list[ArticolTest] = []
    vazute: set[str] = set()

    soup = get_soup(URL_TAG)
    if soup is None:
        return articole

    for a in soup.find_all("a", href=True):
        if len(articole) >= nr_dorit:
            break
        href = a["href"]
        if href.startswith("/"):
            href = "https://biziday.ro" + href

        if (
            "biziday.ro" in href
            and "/tag/" not in href
            and href not in vazute
            and len(href) > 30
        ):
            vazute.add(href)
            time.sleep(DELAY_SECONDS)

            sub_soup = get_soup(href)
            if sub_soup is None:
                continue

            h1    = sub_soup.find("h1")
            titlu = normalizeaza_text(h1.get_text(strip=True)) if h1 else "N/A"

            content_tag = (
                sub_soup.find("div", class_=re.compile(r"article.?content|entry.?content|post.?content", re.I))
                or sub_soup.find("article")
            )

            if content_tag:
                for tag in content_tag.find_all(["script", "style", "aside", "figure", "nav"]):
                    tag.decompose()
                text_articol = normalizeaza_text(content_tag.get_text(separator=" ", strip=True))
            else:
                continue

            cuvinte = text_articol.split()
            if len(cuvinte) > 800:
                text_articol = " ".join(cuvinte[:800])

            text_input = normalizeaza_text(f"{titlu} {text_articol}")

            if len(text_input.split()) >= 30:
                articole.append(ArticolTest(
                    sursa="biziday.ro",
                    url=href,
                    titlu=titlu[:80],
                    text_input=text_input,
                    label_asteptat=LABEL_CREDIBIL,
                ))
                print(f"  [Biziday] ✓ {titlu[:60]}...")

    return articole


# ── Apel API ───────────────────────────────────────────────────────────────────

def apeleaza_api(articol: ArticolTest) -> None:
    """
    Trimite text_input la POST /api/predict si completeaza campurile de rezultat
    in articolul dat (in-place).
    """
    try:
        start = time.time()
        r = requests.post(
            API_URL,
            json={"text": articol.text_input},
            timeout=TIMEOUT_API,
        )
        articol.timp_inferenta = round(time.time() - start, 2)

        if r.status_code != 200:
            articol.eroare_api = f"HTTP {r.status_code}: {r.text[:100]}"
            return

        data = r.json()
        articol.verdict_primit = data.get("decizie")
        articol.incredere      = data.get("incredere")
        articol.scor_baseline  = data.get("scor_baseline_prob_cls1")
        articol.scor_modul3    = data.get("scor_modul3_diff_mean")

    except requests.RequestException as e:
        articol.eroare_api     = str(e)
        articol.timp_inferenta = None


# ── Afisare tabel + metrici ────────────────────────────────────────────────────

def afiseaza_tabel(articole: list[ArticolTest]) -> None:
    """Afiseaza tabelul de rezultate in terminal, cu latimi fixe."""

    # Antet
    print()
    print("=" * 130)
    print(f"{'SURSĂ':<15} {'TITLU':<38} {'AȘTEPTAT':<26} {'PRIMIT':<26} {'OK':<4} {'ÎNCRED.':<8} {'T(s)':<6}")
    print("-" * 130)

    for a in articole:
        sursa   = a.sursa[:14]
        titlu   = a.titlu[:37]
        astept  = a.label_asteptat[:25]
        primit  = (a.verdict_primit or f"EROARE: {a.eroare_api or '?'}")[:25]
        ok      = "DA" if a.corect else ("ERR" if a.verdict_primit is None else "NU")
        incred  = f"{a.incredere:.2f}" if a.incredere is not None else "N/A"
        timp    = f"{a.timp_inferenta:.1f}" if a.timp_inferenta is not None else "N/A"

        # Marcam randurile gresite cu *
        marker = " " if a.corect else "*"
        print(f"{marker}{sursa:<14} {titlu:<38} {astept:<26} {primit:<26} {ok:<4} {incred:<8} {timp:<6}")

    print("=" * 130)
    print("  * = predicție greșită sau eroare")


def afiseaza_metrici(articole: list[ArticolTest]) -> None:
    """Calculeaza si afiseaza accuracy, FP, FN."""
    total     = len(articole)
    cu_rasp   = [a for a in articole if a.verdict_primit is not None]
    corecte   = [a for a in cu_rasp if a.corect]
    erori_api = [a for a in articole if a.eroare_api]

    # FP: label real = credibil, verdict = dezinformare
    fp = [
        a for a in cu_rasp
        if a.label_asteptat == LABEL_CREDIBIL and a.verdict_primit == LABEL_DEZINFO
    ]
    # FN: label real = dezinformare, verdict = credibil
    fn = [
        a for a in cu_rasp
        if a.label_asteptat == LABEL_DEZINFO and a.verdict_primit == LABEL_CREDIBIL
    ]
    # Incerte (decizie_incerta)
    incerte = [a for a in cu_rasp if a.verdict_primit == "incert"]

    accuracy = len(corecte) / len(cu_rasp) * 100 if cu_rasp else 0.0

    print()
    print("═" * 50)
    print("  SUMAR METRICI")
    print("═" * 50)
    print(f"  Articole testate    : {total}")
    print(f"  Cu răspuns API      : {len(cu_rasp)}")
    print(f"  Erori API           : {len(erori_api)}")
    print(f"  Corecte             : {len(corecte)} / {len(cu_rasp)}")
    print(f"  Accuracy            : {accuracy:.1f}%")
    print(f"  False Positive (FP) : {len(fp)}  (știre credibilă clasificată ca dezinfo)")
    print(f"  False Negative (FN) : {len(fn)}  (dezinfo clasificată ca știre credibilă)")
    print(f"  Incerte             : {len(incerte)}")
    print("═" * 50)

    if fp:
        print("\n  FP detalii:")
        for a in fp:
            print(f"    - [{a.sursa}] {a.titlu[:60]}")
    if fn:
        print("\n  FN detalii:")
        for a in fn:
            print(f"    - [{a.sursa}] {a.titlu[:60]}")
    if erori_api:
        print("\n  Erori API detalii:")
        for a in erori_api:
            print(f"    - [{a.sursa}] {a.titlu[:50]}: {a.eroare_api}")


# ── Main ────────────────────────────────────────────────────────────────────────

def verifica_api() -> bool:
    """Verifica ca API-ul este online si toate modulele sunt incarcate."""
    try:
        r = requests.get(HEALTH_URL, timeout=10)
        data = r.json()
        status = data.get("status", "unknown")
        modele = data.get("models_loaded", {})
        print(f"  API status     : {status}")
        print(f"  Modele încărcate:")
        for modul, val in modele.items():
            print(f"    {modul:<35}: {val}")
        return status == "ok"
    except requests.RequestException as e:
        print(f"  [EROARE] API indisponibil: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("TEST API LIVE — Detector Dezinformare Pro-Rusă")
    print("=" * 60)

    # ── Pasul 1: Verificare sanatate API ─────────────────────────────────────
    print("\n[1/4] Verificare API...")
    if not verifica_api():
        print("  API nu este disponibil. Oprire test.")
        sys.exit(1)

    # ── Pasul 2: Colectare articole dezinformare (Veridica) ───────────────────
    print("\n[2/4] Colectare articole dezinformare (veridica.ro)...")
    url_veridica = colecteaza_url_veridica(nr_dorit=5)
    print(f"  {len(url_veridica)} URL-uri candidate descoperite")

    articole_dezinfo: list[ArticolTest] = []
    for url in url_veridica:
        if len(articole_dezinfo) >= 5:
            break
        time.sleep(DELAY_SECONDS)
        articol = scrape_articol_veridica(url)
        if articol and len(articol.text_input.split()) >= 10:
            articole_dezinfo.append(articol)
            print(f"  [Veridica] ✓ {articol.titlu[:60]}...")
        else:
            print(f"  [Veridica] ✗ articol fără text valid: {url[:70]}")

    if len(articole_dezinfo) < 5:
        print(f"  ⚠ Colectate doar {len(articole_dezinfo)}/5 articole dezinfo — continuăm cu ce avem")

    # ── Pasul 3: Colectare articole credibile (G4Media → Libertatea → Biziday) ─
    print("\n[3/4] Colectare articole credibile...")
    articole_credibile: list[ArticolTest] = []

    # Incercam G4Media primul (cel mai structurat)
    print("  Încerc G4Media...")
    articole_credibile = colecteaza_articole_g4media(nr_dorit=5)

    # Fallback pe Libertatea daca G4Media nu returneaza suficiente
    if len(articole_credibile) < 5:
        lipsa = 5 - len(articole_credibile)
        print(f"  G4Media: {len(articole_credibile)}/5. Încerc Libertatea pentru {lipsa} articole...")
        supliment = colecteaza_articole_libertatea(nr_dorit=lipsa)
        articole_credibile.extend(supliment)

    # Ultimul fallback: Biziday
    if len(articole_credibile) < 5:
        lipsa = 5 - len(articole_credibile)
        print(f"  Libertatea: total {len(articole_credibile)}/5. Încerc Biziday pentru {lipsa}...")
        supliment = colecteaza_articole_biziday(nr_dorit=lipsa)
        articole_credibile.extend(supliment)

    articole_credibile = articole_credibile[:5]

    if len(articole_credibile) < 5:
        print(f"  ⚠ Colectate doar {len(articole_credibile)}/5 articole credibile — continuăm cu ce avem")

    # ── Pasul 4: Clasificare prin API ─────────────────────────────────────────
    toate_articolele = articole_dezinfo + articole_credibile
    print(f"\n[4/4] Clasificare {len(toate_articolele)} articole prin API...")
    print(f"  Endpoint: {API_URL}")

    for i, articol in enumerate(toate_articolele, 1):
        print(f"  [{i}/{len(toate_articolele)}] {articol.sursa}: {articol.titlu[:50]}...")
        apeleaza_api(articol)
        verdict_scurt = articol.verdict_primit or f"EROARE({articol.eroare_api or '?'})"
        incred_str    = f"{articol.incredere:.2f}" if articol.incredere else "N/A"
        print(f"    → {verdict_scurt} | încredere={incred_str} | {articol.timp_inferenta}s")

    # ── Afisare rezultate ─────────────────────────────────────────────────────
    afiseaza_tabel(toate_articolele)
    afiseaza_metrici(toate_articolele)


if __name__ == "__main__":
    main()
