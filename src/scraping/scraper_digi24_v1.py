"""
scraper_digi24_v1.py — Scraper pentru Digi24.ro, clasa 0 (stiri credibile)

Parte din proiectul de licenta „Sistem de Detectie Automata si Explicabila a
Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din Ucraina".

Scopul acestui script:
    Colecteaza articole Digi24 din cele doua tag-uri editoriale dedicate
    razboiului din Ucraina (`razboi-ucraina` + `razboi-in-ucraina`), le salveaza
    in raw CSV si pregateste terenul pentru cleaning + sampling stratificat.

Subcomenzi:
    discovery  — parcurge paginarea tag-urilor si scrie URL-urile intr-un JSONL
    fetch      — citeste discovery JSONL si descarca articolele, salvand raw CSV

Flag-uri globale:
    --limit N  — opreste dupa N intrari (pentru audit rapid pe 50 URL-uri)
    --resume   — reia de unde a ramas (default pornit; --no-resume pentru restart)

Conventii:
    - Toate comentariile si docstring-urile sunt in romana.
    - Dedup dupa ID numeric extras din URL (ultimul grup numeric din slug).
    - Data folosita pentru distributia temporala: Data publicarii (NU Data actualizarii).
    - Oprire cronologica la discovery: cand data listing < 01.01.2023.
    - Throttling: random.uniform(1.5, 2.2) s intre requesturi.

Usage:
    python scraper_digi24_v1.py discovery --limit 50
    python scraper_digi24_v1.py discovery
    python scraper_digi24_v1.py fetch --limit 50
    python scraper_digi24_v1.py fetch
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# Configuratie
# ─────────────────────────────────────────────────────────────────────────────

# Directoare (relative la radacina proiectului, NU la locatia scriptului)
# Scriptul e presupus a fi rulat din radacina proiectului: `python src/scraping/scraper_digi24_v1.py ...`
ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "cache" / "digi24_v1"
LOG_FILE = RAW_DIR / "scraper_digi24_v1.log"
DISCOVERY_FILE = RAW_DIR / "discovery_digi24_v1.jsonl"
RAW_CSV = RAW_DIR / "digi24_v1_raw.csv"

# Tag-uri sursa (ambele contribuie la discovery, dedup dupa ID numeric)
TAG_URLS = [
    "https://www.digi24.ro/eticheta/razboi-ucraina",
    "https://www.digi24.ro/eticheta/razboi-in-ucraina",
]

DATE_CUTOFF = datetime(2023, 1, 1)

# Throttling
SLEEP_MIN = 1.5
SLEEP_MAX = 2.2

# HTTP
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# Marker sursa pentru ID-urile din dataset
ID_PREFIX = "d24_v1_"

# Regex pentru extragerea ID-ului numeric din URL (ultimul grup numeric din slug)
# Ex: /stiri/externe/ucraina-rusia/slug-descriptiv-3719909 → 3719909
RE_ARTICLE_ID = re.compile(r"-(\d{5,})/?$")

# Regex pentru datele articolului (Data actualizarii / Data publicarii)
# Format observat: „Data actualizarii: 25.03.2026 14:32 Data publicarii: 25.03.2026 09:15"
RE_DATA_PUBLICARII = re.compile(
    r"Data\s+public[aă]rii\s*:\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})",
    re.IGNORECASE,
)
RE_DATA_ACTUALIZARII = re.compile(
    r"Data\s+actualiz[aă]rii\s*:\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass-uri
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiscoveryItem:
    """Un URL gasit in faza de discovery (listing parsing)."""

    id_articol: str  # ID numeric extras din URL, ex. "3719909"
    url: str
    titlu_listing: str  # titlul asa cum apare pe pagina de listing
    sectiune_listing: str  # sectiunea extrasa din URL, ex. "externe/ucraina-rusia"
    data_listing: str  # data afisata pe listing (poate fi relativa sau formatata)
    tag_sursa: str  # care tag l-a descoperit: "razboi-ucraina" / "razboi-in-ucraina"
    pagina: int  # numarul paginii in paginare


@dataclass
class ArticleRaw:
    """Un articol fetched si parsat in faza de fetch."""

    id_dataset: str  # ID-ul nostru cu prefix, ex. "d24_v1_3719909"
    id_articol: str  # ID-ul numeric Digi24
    url: str
    titlu: str
    data_publicarii: str  # format ISO "YYYY-MM-DD HH:MM" sau ""
    data_actualizarii: str  # format ISO "YYYY-MM-DD HH:MM" sau ""
    sectiune: str
    corp_articol: str
    nr_cuvinte: int
    tag_sursa: str
    sursa: str  # constant "digi24"
    hash_continut: str  # sha1 pe titlu+corp, pentru dedup post-hoc
    fetch_ok: bool
    fetch_error: str  # mesaj eroare daca fetch_ok=False


# ─────────────────────────────────────────────────────────────────────────────
# Utilitare generale
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:
    """Configureaza logging dual: consola + fisier."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("digi24")
    logger.setLevel(logging.INFO)
    # Evitam dublarea handler-elor daca functia e apelata de mai multe ori
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def polite_sleep() -> None:
    """Pauza aleatoare intre requesturi, ca sa nu stresam serverul."""
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def http_get(url: str, logger: logging.Logger) -> requests.Response | None:
    """
    GET cu retry exponential pe 5xx/timeout.

    Returneaza None daca dupa MAX_RETRIES incercari tot esueaza sau daca raspunsul
    e 404 (nu are sens sa insistam). 404 → log si skip.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            wait = 2**attempt
            logger.warning(
                "GET %s → %s (încercarea %d/%d), retry în %ds",
                url,
                type(exc).__name__,
                attempt,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp
        if resp.status_code == 404:
            logger.warning("GET %s → 404, skip", url)
            return None
        if 500 <= resp.status_code < 600:
            wait = 2**attempt
            logger.warning(
                "GET %s → %d (încercarea %d/%d), retry în %ds",
                url,
                resp.status_code,
                attempt,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
            continue
        # Alte coduri (403, 429, etc.) — logam si renuntam
        logger.error("GET %s → %d, renunț", url, resp.status_code)
        return None

    logger.error("GET %s → epuizat retries", url)
    return None


def extract_article_id(url: str) -> str | None:
    """Extrage ID-ul numeric din finalul URL-ului articolului."""
    m = RE_ARTICLE_ID.search(url.rstrip("/"))
    return m.group(1) if m else None


def extract_sectiune_from_url(url: str) -> str:
    """
    Extrage sectiunea din URL-ul articolului.

    Ex: /stiri/externe/ucraina-rusia/slug-3719909 → "externe/ucraina-rusia"
    """
    # Scoatem https://www.digi24.ro si /stiri/ prefix
    m = re.match(r"https?://www\.digi24\.ro/stiri/([^/]+(?:/[^/]+)?)/", url)
    if m:
        return m.group(1)
    return ""


def parse_data_ro(date_str: str) -> datetime | None:
    """
    Parseaza o data Digi24 format „DD.MM.YYYY HH:MM" → datetime.

    Returneaza None daca parsarea esueaza.
    """
    try:
        return datetime.strptime(date_str.strip(), "%d.%m.%Y %H:%M")
    except (ValueError, AttributeError):
        return None


def fmt_data_iso(dt: datetime | None) -> str:
    """Formateaza datetime → string ISO-ish „YYYY-MM-DD HH:MM", sau "" daca None."""
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────


def load_existing_discovery() -> dict[str, DiscoveryItem]:
    """
    Citeste JSONL-ul existent de discovery si returneaza dict id_articol → item.

    Permite resume: la o noua rulare, stim ce ID-uri avem deja.
    """
    existing: dict[str, DiscoveryItem] = {}
    if not DISCOVERY_FILE.exists():
        return existing
    with DISCOVERY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                item = DiscoveryItem(**d)
                existing[item.id_articol] = item
            except (json.JSONDecodeError, TypeError) as exc:
                logging.getLogger("digi24").warning(
                    "Linie coruptă în discovery JSONL: %s", exc
                )
    return existing


def append_discovery(item: DiscoveryItem) -> None:
    """Scrie un DiscoveryItem in JSONL (append)."""
    with DISCOVERY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def parse_listing_page(
    html: str, tag_slug: str, pagina: int, logger: logging.Logger
) -> list[DiscoveryItem]:
    """
    Parseaza o pagina de listing tag si extrage URL-urile articolelor.

    Returneaza lista de DiscoveryItem-uri gasite. Data articolului NU se extrage
    din listing — Digi24 nu expune `<time>` in carduri, confirmat empiric la
    audit-ul pe 50 URL-uri (toate au rezultat „? " la parsarea datei pe listing).
    Data se obtine separat prin `probe_article_date` pe ultimul articol al
    paginii (vezi run_discovery).

    Strategie: cautam toate linkurile <a href="/stiri/..."> cu ID numeric in slug.
    Filtram duplicatele din aceeasi pagina (cardul + titlul pot duce la acelasi
    articol). Pastram ordinea din HTML (BeautifulSoup respecta DOM order), astfel
    incat `items[-1]` = ultimul articol de pe pagina = cel mai vechi cronologic,
    presupunand ca Digi24 ordoneaza descrescator (standard pentru listing-uri).
    """
    soup = BeautifulSoup(html, "lxml")
    items: list[DiscoveryItem] = []
    seen_in_page: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/stiri/") and not href.startswith(
            "https://www.digi24.ro/stiri/"
        ):
            continue
        full_url = urljoin("https://www.digi24.ro", href)
        art_id = extract_article_id(full_url)
        if not art_id:
            continue
        if art_id in seen_in_page:
            continue
        seen_in_page.add(art_id)

        # Titlul — luam textul linkului sau al unui h-tag parinte
        titlu = a.get_text(strip=True)
        if not titlu:
            parent = a.find_parent(["article", "div"])
            if parent:
                h = parent.find(["h1", "h2", "h3", "h4"])
                if h:
                    titlu = h.get_text(strip=True)

        sectiune = extract_sectiune_from_url(full_url)

        items.append(
            DiscoveryItem(
                id_articol=art_id,
                url=full_url,
                titlu_listing=titlu[:500],
                sectiune_listing=sectiune,
                data_listing="",  # populat ulterior doar pentru articolul-sonda
                tag_sursa=tag_slug,
                pagina=pagina,
            )
        )

    logger.info(
        "Listing %s p=%d → %d articole extrase",
        tag_slug,
        pagina,
        len(items),
    )
    return items


def probe_article_date(
    url: str, id_articol: str, logger: logging.Logger
) -> datetime | None:
    """
    Face un request pe pagina unui articol si extrage `data_publicarii`.

    Folosit pentru cutoff cronologic la discovery: pe ultimul articol al fiecarei
    pagini de listing (cel mai vechi, presupunand ordonare descrescatoare), ca
    sa decidem daca oprim paginarea.

    Reutilizeaza cache-ul fetch daca articolul e deja descarcat — la rulari
    repetate evitam requesturi suplimentare. Altfel face un GET si aplica apoi
    throttling, fara sa scrie in cache (cache-ul e strict pentru faza fetch).

    Returneaza datetime sau None daca parsarea esueaza.
    """
    html = load_html_from_cache(id_articol)
    if html is None:
        resp = http_get(url, logger)
        if resp is None:
            return None
        html = resp.text
        polite_sleep()

    soup = BeautifulSoup(html, "lxml")

    # Meta tag article:published_time — confirmat 50/50 la audit
    meta_pub = soup.find("meta", property="article:published_time")
    if meta_pub:
        val = meta_pub.get("content", "")
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt.replace(tzinfo=None)
        except ValueError:
            pass

    # Fallback: regex pe textul paginii
    text_page = soup.get_text(" ", strip=True)
    m = RE_DATA_PUBLICARII.search(text_page)
    if m:
        return parse_data_ro(m.group(1))

    return None


def run_discovery(limit: int | None, logger: logging.Logger) -> None:
    """
    Parcurge ambele tag-uri cu paginare ?p=N, extrage articole, dedup dupa ID,
    opreste cand data articolului-sonda (ultimul de pe pagina) < DATE_CUTOFF
    sau la --limit atins.

    Probe: pentru fiecare pagina listing, dupa extragerea URL-urilor, facem un
    request suplimentar pe ULTIMUL articol al paginii (cel mai vechi cronologic,
    presupunand ordonare descrescatoare) ca sa-i luam `data_publicarii`. Folosim
    aceasta data pentru cutoff. Cost: +1 request / pagina listing (~+150 pentru
    full run), adica +5 min. Acceptabil vs. riscul de a descoperi 1000+ articole
    sub cutoff.
    """
    existing = load_existing_discovery()
    logger.info("Discovery start — %d ID-uri preexistente în JSONL", len(existing))
    total_new = 0

    for tag_url in TAG_URLS:
        tag_slug = tag_url.rstrip("/").rsplit("/", 1)[-1]
        logger.info("=== Tag: %s ===", tag_slug)
        pagina = 1
        consecutive_old_pages = 0

        while True:
            if limit is not None and total_new >= limit:
                logger.info("Limit %d atins, opresc discovery", limit)
                return

            url = f"{tag_url}?p={pagina}"
            resp = http_get(url, logger)
            if resp is None:
                logger.warning("Listing %s eșuat, trec la tag-ul următor", url)
                break

            items = parse_listing_page(resp.text, tag_slug, pagina, logger)
            if not items:
                logger.info(
                    "Listing %s p=%d → 0 articole, probabil sfârșitul paginării",
                    tag_slug,
                    pagina,
                )
                break

            # Scriem items noi in JSONL inainte de probe (nu vrem sa pierdem
            # munca daca probe-ul esueaza sau daca user-ul apasa Ctrl+C)
            new_on_page = 0
            for item in items:
                if item.id_articol in existing:
                    continue
                append_discovery(item)
                existing[item.id_articol] = item
                total_new += 1
                new_on_page += 1

            # Probe pe ultimul articol al paginii — cutoff cronologic
            # IMPORTANT: rulam probe-ul INAINTE de check-ul de limit, ca sa
            # validam cutoff-ul chiar si la audit pe --limit 50 (o pagina = 50
            # articole = fix pragul, altfel nu am atinge probe-ul niciodata).
            last_item = items[-1]
            polite_sleep()
            probe_dt = probe_article_date(
                last_item.url, last_item.id_articol, logger
            )
            if probe_dt is None:
                logger.warning(
                    "Probe eșuat pe p=%d (url=%s) — continuu fără cutoff",
                    pagina,
                    last_item.url,
                )
                consecutive_old_pages = 0
            elif probe_dt < DATE_CUTOFF:
                consecutive_old_pages += 1
                logger.info(
                    "Pagina %d sub cutoff (%s < %s), consecutive=%d",
                    pagina,
                    probe_dt.strftime("%Y-%m-%d"),
                    DATE_CUTOFF.strftime("%Y-%m-%d"),
                    consecutive_old_pages,
                )
                if consecutive_old_pages >= 2:
                    logger.info("2 pagini consecutive sub cutoff, opresc tag-ul")
                    break
            else:
                logger.info(
                    "Pagina %d probe OK: %s (%d noi, total=%d)",
                    pagina,
                    probe_dt.strftime("%Y-%m-%d"),
                    new_on_page,
                    total_new,
                )
                consecutive_old_pages = 0

            # Check limit dupa probe, ca sa avem validare probe chiar la --limit 50
            if limit is not None and total_new >= limit:
                logger.info("Limit %d atins, opresc discovery", limit)
                return

            pagina += 1
            polite_sleep()

    logger.info(
        "Discovery TERMINAT — %d ID-uri noi adăugate, total în JSONL: %d",
        total_new,
        len(existing),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────────────────────────────────────


CSV_FIELDS = [
    "id_dataset",
    "id_articol",
    "url",
    "titlu",
    "data_publicarii",
    "data_actualizarii",
    "sectiune",
    "corp_articol",
    "nr_cuvinte",
    "tag_sursa",
    "sursa",
    "hash_continut",
    "fetch_ok",
    "fetch_error",
]


def load_fetched_ids() -> set[str]:
    """Returneaza set-ul de id_articol deja prezente in CSV-ul raw (pentru resume)."""
    fetched: set[str] = set()
    if not RAW_CSV.exists():
        return fetched
    with RAW_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fetched.add(row["id_articol"])
    return fetched


def cache_path(id_articol: str) -> Path:
    """Calea catre HTML-ul cache-uit pentru un articol."""
    return CACHE_DIR / f"{id_articol}.html"


def load_html_from_cache(id_articol: str) -> str | None:
    p = cache_path(id_articol)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def save_html_to_cache(id_articol: str, html: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(id_articol).write_text(html, encoding="utf-8")


def parse_article(html: str, item: DiscoveryItem) -> ArticleRaw:
    """
    Parseaza HTML-ul unui articol Digi24 si extrage campurile.

    Strategia pentru corp_articol:
        - Luam containerul principal al articolului (selector best-effort)
        - Stripuim scripturile, stilurile, elementele de navigare
        - Extragem textul cu separator newline
        - La cleaning vom taia de la `Editor : ` pana la EOF; aici pastram brut

    La audit-ul pe 50 URL-uri vom verifica daca selectorii functioneaza bine.
    """
    soup = BeautifulSoup(html, "lxml")

    # Titlu: din <h1> sau og:title
    titlu = ""
    h1 = soup.find("h1")
    if h1:
        titlu = h1.get_text(strip=True)
    if not titlu:
        og = soup.find("meta", property="og:title")
        if og:
            titlu = og.get("content", "").strip()
    if not titlu:
        titlu = item.titlu_listing

    # Data publicarii si actualizarii — cautam in TEXTUL paginii, pentru ca pot
    # fi randate in formate variate. Folosim primul 2000 caractere dupa h1 ca
    # zona de cautare (reduce zgomotul).
    text_page = soup.get_text(" ", strip=True)
    m_pub = RE_DATA_PUBLICARII.search(text_page)
    m_act = RE_DATA_ACTUALIZARII.search(text_page)
    dt_pub = parse_data_ro(m_pub.group(1)) if m_pub else None
    dt_act = parse_data_ro(m_act.group(1)) if m_act else None

    # Fallback pentru data publicarii: meta tag article:published_time
    if dt_pub is None:
        meta_pub = soup.find("meta", property="article:published_time")
        if meta_pub:
            val = meta_pub.get("content", "")
            try:
                dt_pub = datetime.fromisoformat(val.replace("Z", "+00:00"))
                dt_pub = dt_pub.replace(tzinfo=None)
            except ValueError:
                pass
    if dt_act is None:
        meta_mod = soup.find("meta", property="article:modified_time")
        if meta_mod:
            val = meta_mod.get("content", "")
            try:
                dt_act = datetime.fromisoformat(val.replace("Z", "+00:00"))
                dt_act = dt_act.replace(tzinfo=None)
            except ValueError:
                pass

    # Corp articol: incercam selectori tipici Digi24 (best-effort, de validat empiric)
    corp = ""
    for selector in [
        "div.entry-content",
        "article .article-content",
        "div.article-body",
        "div.data-app-meta-article",
        "article",
    ]:
        container = soup.select_one(selector)
        if container:
            # Eliminam script/style/nav/aside/footer
            for tag in container.find_all(
                ["script", "style", "nav", "aside", "footer", "iframe"]
            ):
                tag.decompose()
            corp = container.get_text("\n", strip=True)
            if len(corp) > 200:  # heuristic: daca e prea scurt, incercam alt selector
                break

    # Fallback extrem: tot body-ul minus nav/footer
    if len(corp) < 200:
        body = soup.find("body")
        if body:
            for tag in body.find_all(
                ["script", "style", "nav", "aside", "footer", "header", "iframe"]
            ):
                tag.decompose()
            corp = body.get_text("\n", strip=True)

    sectiune = extract_sectiune_from_url(item.url) or item.sectiune_listing
    nr_cuvinte = len(corp.split())
    hash_continut = hashlib.sha1(
        (titlu + "||" + corp).encode("utf-8")
    ).hexdigest()

    return ArticleRaw(
        id_dataset=f"{ID_PREFIX}{item.id_articol}",
        id_articol=item.id_articol,
        url=item.url,
        titlu=titlu,
        data_publicarii=fmt_data_iso(dt_pub),
        data_actualizarii=fmt_data_iso(dt_act),
        sectiune=sectiune,
        corp_articol=corp,
        nr_cuvinte=nr_cuvinte,
        tag_sursa=item.tag_sursa,
        sursa="digi24",
        hash_continut=hash_continut,
        fetch_ok=True,
        fetch_error="",
    )


def write_csv_header_if_needed() -> None:
    """Scrie header-ul CSV-ului daca fisierul nu exista inca."""
    if RAW_CSV.exists():
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_article(article: ArticleRaw) -> None:
    """
    Append un articol in CSV-ul raw.

    Flush + fsync explicit, pentru ca scriptul ruleaza 4h peste noapte — daca
    pica laptop-ul sau apare un crash, vrem garantat ca toate articolele deja
    parsate sa fie pe disc, nu in buffer-ul Python.
    """
    with RAW_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(asdict(article))
        f.flush()
        import os
        os.fsync(f.fileno())


def iter_discovery_items() -> Iterator[DiscoveryItem]:
    """Itereaza peste toate intrarile din discovery JSONL."""
    if not DISCOVERY_FILE.exists():
        return
    with DISCOVERY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                yield DiscoveryItem(**d)
            except (json.JSONDecodeError, TypeError):
                continue


def count_discovery_items() -> int:
    """Numara liniile din JSONL-ul de discovery, pentru calcul ETA la fetch."""
    if not DISCOVERY_FILE.exists():
        return 0
    with DISCOVERY_FILE.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_fetch(limit: int | None, logger: logging.Logger) -> None:
    """
    Fetch articolele din discovery JSONL. Resume automat (skip ID-urile deja in CSV).

    Afiseaza progres cu % si ETA la fiecare 50 articole fetched. ETA e estimat
    pe baza ratei medii de la inceputul rularii curente (nu pe toata istoria
    CSV-ului, pentru ca resume-urile pot avea viteze foarte diferite).
    """
    if not DISCOVERY_FILE.exists():
        logger.error("Discovery JSONL inexistent: %s", DISCOVERY_FILE)
        logger.error("Rulează întâi: python scraper_digi24_v1.py discovery")
        return

    fetched_ids = load_fetched_ids()
    total_in_discovery = count_discovery_items()
    remaining_at_start = total_in_discovery - len(fetched_ids)
    logger.info(
        "Fetch start — %d ID-uri deja în CSV, %d de fetch-uit (total discovery: %d)",
        len(fetched_ids),
        remaining_at_start,
        total_in_discovery,
    )
    if limit is not None:
        logger.info("Limit activ: %d articole maxim în această rulare", limit)

    write_csv_header_if_needed()
    total_new = 0
    total_ok = 0
    total_err = 0
    start_time = time.monotonic()

    for item in iter_discovery_items():
        if item.id_articol in fetched_ids:
            continue
        if limit is not None and total_new >= limit:
            logger.info("Limit %d atins, opresc fetch", limit)
            break

        # Cache lookup
        html = load_html_from_cache(item.id_articol)
        if html is None:
            resp = http_get(item.url, logger)
            if resp is None:
                article = ArticleRaw(
                    id_dataset=f"{ID_PREFIX}{item.id_articol}",
                    id_articol=item.id_articol,
                    url=item.url,
                    titlu=item.titlu_listing,
                    data_publicarii="",
                    data_actualizarii="",
                    sectiune=item.sectiune_listing,
                    corp_articol="",
                    nr_cuvinte=0,
                    tag_sursa=item.tag_sursa,
                    sursa="digi24",
                    hash_continut="",
                    fetch_ok=False,
                    fetch_error="http_get_failed",
                )
                append_article(article)
                total_new += 1
                total_err += 1
                polite_sleep()
                continue
            html = resp.text
            save_html_to_cache(item.id_articol, html)
            polite_sleep()  # doar cand am facut efectiv un request

        try:
            article = parse_article(html, item)
            append_article(article)
            total_new += 1
            total_ok += 1
            if total_new % 50 == 0:
                elapsed = time.monotonic() - start_time
                rate = total_new / elapsed if elapsed > 0 else 0
                # Tinta: cate articole vrem total in rularea curenta
                target = (
                    min(limit, remaining_at_start)
                    if limit is not None
                    else remaining_at_start
                )
                remaining = target - total_new
                eta_sec = remaining / rate if rate > 0 else 0
                eta_h = int(eta_sec // 3600)
                eta_m = int((eta_sec % 3600) // 60)
                pct = 100 * total_new / target if target > 0 else 0
                logger.info(
                    "Progres: %d/%d (%.1f%%) | ok=%d err=%d | rate=%.2f art/s | ETA=%dh%02dm | ultim: %s",
                    total_new,
                    target,
                    pct,
                    total_ok,
                    total_err,
                    rate,
                    eta_h,
                    eta_m,
                    article.titlu[:50],
                )
        except Exception as exc:  # noqa: BLE001 — vrem sa prindem tot la parse
            logger.error("Parse failed pentru %s: %s", item.url, exc)
            article = ArticleRaw(
                id_dataset=f"{ID_PREFIX}{item.id_articol}",
                id_articol=item.id_articol,
                url=item.url,
                titlu=item.titlu_listing,
                data_publicarii="",
                data_actualizarii="",
                sectiune=item.sectiune_listing,
                corp_articol="",
                nr_cuvinte=0,
                tag_sursa=item.tag_sursa,
                sursa="digi24",
                hash_continut="",
                fetch_ok=False,
                fetch_error=f"parse_error: {exc}",
            )
            append_article(article)
            total_new += 1
            total_err += 1

    logger.info(
        "Fetch TERMINAT — %d articole noi (ok=%d, err=%d)",
        total_new,
        total_ok,
        total_err,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scraper Digi24 — discovery + fetch pentru clasa 0."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_disc = sub.add_parser("discovery", help="Colectează URL-uri din listing-uri")
    p_disc.add_argument("--limit", type=int, default=None, help="Max URL-uri noi")

    p_fetch = sub.add_parser("fetch", help="Descarcă și parsează articolele")
    p_fetch.add_argument(
        "--limit", type=int, default=None, help="Max articole de fetch"
    )

    args = parser.parse_args()
    logger = setup_logging()

    random.seed(42)  # seed consistent cu G4Media v2

    if args.cmd == "discovery":
        run_discovery(limit=args.limit, logger=logger)
    elif args.cmd == "fetch":
        run_fetch(limit=args.limit, logger=logger)


if __name__ == "__main__":
    main()