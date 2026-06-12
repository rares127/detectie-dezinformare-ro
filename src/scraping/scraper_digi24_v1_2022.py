"""
scraper_digi24_v1_2022.py — Scraper pentru Digi24.ro pe fereastra 2022 exclusiv

Parte din proiectul de licenta „Sistem de Detectie Automata si Explicabila a
Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din Ucraina".

CONTEXT:
    Varianta v1 a acoperit doar 2023-2026 (6348 articole). Pentru a completa
    clasa 0 pe anul 2022 (213 articole tinta, distributie identica cu clasa 1),
    este nevoie de o rulare separata cu fereastra temporala [2022-01-01, 2022-12-31].

DIFERENTE FATA DE v1:
    - Fereastra temporala inchisa: [2022-01-01, 2022-12-31] inclusiv.
    - Cutoff inferior: opreste paginarea cand probe < 2022-01-01 (2 pagini consecutive).
    - Cutoff superior: paginile cu probe_dt >= 2023-01-01 sunt COMPLET IGNORATE
      la discovery (NU se salveaza in JSONL). Economisim astfel ~4000 URL-uri
      inutile din 2023-2026 si toate requesturile de fetch aferente.
    - Articolele out-of-range la FETCH (probe-ul poate esua uneori la listing,
      articole strecurate prin filtrul de listing ajung la fetch) sunt marcate
      fetch_ok=False, fetch_error="out_of_range_year".
    - --max-pages N (doar la discovery): opreste dupa N pagini per tag. Util
      pentru audit rapid, fiindca --limit numara articole in-range (care apar
      abia dupa pagina ~80), deci nu declanseaza stopping pentru audit scurt.
    - Fisiere de output separate, cu sufix `_2022`, pentru a nu suprascrie v1.
    - Cache HTML separat: data/raw/cache/digi24_v1_2022/

FISIERE OUTPUT:
    data/raw/discovery_digi24_v1_2022.jsonl
    data/raw/digi24_v1_2022_raw.csv
    data/raw/scraper_digi24_v1_2022.log
    data/raw/cache/digi24_v1_2022/*.html

USAGE:
    # Audit rapid (2-3 minute): parcurge 3 pagini listing per tag, verifica doar
    # ca scraperul nu crapa pe HTML-uri noi. Nu va gasi articole 2022 pe numai
    # 3 pagini (paginile 1-3 sunt din 2026), scopul e sa testezi selectorii.
    python scraper_digi24_v1_2022.py discovery --max-pages 3
    python scraper_digi24_v1_2022.py fetch --limit 50

    # Full run (25-30 minute):
    python scraper_digi24_v1_2022.py discovery
    python scraper_digi24_v1_2022.py fetch

CONVENTII:
    - Toate comentariile si docstring-urile sunt in romana.
    - Dedup dupa ID numeric extras din URL.
    - Seed random.seed(42) pentru reproducibilitate.
    - Throttling: random.uniform(1.5, 2.2) s intre requesturi.

ESTIMARE COST:
    - Paginile Digi24 pentru tag-uri razboi-ucraina/razboi-in-ucraina cu ordonare
      descrescatoare cronologic → paginile 1-127 aprox. contin 2023-2026, paginile
      128+ ajung in 2022.
    - Discovery total: ~160-180 pagini × (1 listing + 1 probe) × 2s = ~12-15 min
    - Fetch estimat: ~250-300 articole 2022 × (1 request + 2s) = ~10-12 min
    - Total estimat: ~25-30 minute pentru full run.
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

# Directoare (relative la radacina proiectului)
ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "cache" / "digi24_v1_2022"
LOG_FILE = RAW_DIR / "scraper_digi24_v1_2022.log"
DISCOVERY_FILE = RAW_DIR / "discovery_digi24_v1_2022_sampled.jsonl"
RAW_CSV = RAW_DIR / "digi24_v1_2022_raw.csv" 
# Tag-uri sursa — identice cu v1
TAG_URLS = [
    "https://www.digi24.ro/eticheta/razboi-ucraina",
    "https://www.digi24.ro/eticheta/razboi-in-ucraina",
]

# ─────────────────────────────────────────────────────────────────────────────
# CUTOFF-uri temporale
# ─────────────────────────────────────────────────────────────────────────────
# Fereastra tinta: [2022-01-01 00:00, 2023-01-01 00:00)
# - LOWER_CUTOFF: sub aceasta data, paginarea se opreste (2 pagini consecutive).
# - UPPER_CUTOFF: la/peste aceasta data, articolul e descarcat dar marcat ca
#   „out_of_range_year" (nu intra in clasa 0 2022).
LOWER_CUTOFF = datetime(2022, 1, 1)
UPPER_CUTOFF = datetime(2023, 1, 1)

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

# Marker sursa — acelasi prefix ca v1, asa ID-urile raman compatibile intre rulari
ID_PREFIX = "d24_v1_"

# Regex-uri identice cu v1
RE_ARTICLE_ID = re.compile(r"-(\d{5,})/?$")
RE_DATA_PUBLICARII = re.compile(
    r"Data\s+public[aă]rii\s*:\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})",
    re.IGNORECASE,
)
RE_DATA_ACTUALIZARII = re.compile(
    r"Data\s+actualiz[aă]rii\s*:\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass-uri (identice cu v1)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiscoveryItem:
    """Un URL gasit in faza de discovery."""

    id_articol: str
    url: str
    titlu_listing: str
    sectiune_listing: str
    data_listing: str
    tag_sursa: str
    pagina: int


@dataclass
class ArticleRaw:
    """Un articol fetched si parsat."""

    id_dataset: str
    id_articol: str
    url: str
    titlu: str
    data_publicarii: str
    data_actualizarii: str
    sectiune: str
    corp_articol: str
    nr_cuvinte: int
    tag_sursa: str
    sursa: str
    hash_continut: str
    fetch_ok: bool
    fetch_error: str


# ─────────────────────────────────────────────────────────────────────────────
# Utilitare (identice cu v1)
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:
    """Configureaza logging dual: consola + fisier."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("digi24_2022")
    logger.setLevel(logging.INFO)
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
    """Pauza aleatoare intre requesturi."""
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def http_get(url: str, logger: logging.Logger) -> requests.Response | None:
    """GET cu retry exponential pe 5xx/timeout. Identic cu v1."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            wait = 2**attempt
            logger.warning(
                "GET %s → %s (încercarea %d/%d), retry în %ds",
                url, type(exc).__name__, attempt, MAX_RETRIES, wait,
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
                url, resp.status_code, attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue
        logger.error("GET %s → %d, renunț", url, resp.status_code)
        return None

    logger.error("GET %s → epuizat retries", url)
    return None


def extract_article_id(url: str) -> str | None:
    """Extrage ID-ul numeric din finalul URL-ului."""
    m = RE_ARTICLE_ID.search(url.rstrip("/"))
    return m.group(1) if m else None


def extract_sectiune_from_url(url: str) -> str:
    """Extrage sectiunea din URL-ul articolului."""
    m = re.match(r"https?://www\.digi24\.ro/stiri/([^/]+(?:/[^/]+)?)/", url)
    if m:
        return m.group(1)
    return ""


def parse_data_ro(date_str: str) -> datetime | None:
    """Parseaza „DD.MM.YYYY HH:MM" → datetime."""
    try:
        return datetime.strptime(date_str.strip(), "%d.%m.%Y %H:%M")
    except (ValueError, AttributeError):
        return None


def fmt_data_iso(dt: datetime | None) -> str:
    """datetime → „YYYY-MM-DD HH:MM", sau "" daca None."""
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY (adaptat pentru fereastra 2022)
# ─────────────────────────────────────────────────────────────────────────────


def load_existing_discovery() -> dict[str, DiscoveryItem]:
    """Citeste JSONL-ul de discovery si returneaza dict id_articol → item."""
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
                logging.getLogger("digi24_2022").warning(
                    "Linie coruptă în discovery JSONL: %s", exc
                )
    return existing


def append_discovery(item: DiscoveryItem) -> None:
    """Append un DiscoveryItem in JSONL."""
    with DISCOVERY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def parse_listing_page(
    html: str, tag_slug: str, pagina: int, logger: logging.Logger
) -> list[DiscoveryItem]:
    """Parseaza pagina de listing si extrage URL-urile. Identic cu v1."""
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
                data_listing="",
                tag_sursa=tag_slug,
                pagina=pagina,
            )
        )

    logger.info(
        "Listing %s p=%d → %d articole extrase", tag_slug, pagina, len(items),
    )
    return items


def probe_article_date(
    url: str, id_articol: str, logger: logging.Logger
) -> datetime | None:
    """
    Face un request pe pagina unui articol si extrage data_publicarii.

    Reutilizeaza cache-ul fetch daca articolul e deja descarcat. Altfel face
    un GET si aplica throttling, fara sa scrie in cache (cache-ul e strict
    pentru faza fetch).
    """
    html = load_html_from_cache(id_articol)
    if html is None:
        resp = http_get(url, logger)
        if resp is None:
            return None
        html = resp.text
        polite_sleep()

    soup = BeautifulSoup(html, "lxml")

    # Meta tag article:published_time
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


def run_discovery(
    limit: int | None, max_pages: int | None, logger: logging.Logger
) -> None:
    """
    Parcurge ambele tag-uri cu paginare ?p=N.

    DIFERENTA MAJORA FATA DE v1:
    - Cutoff inferior: opreste cand probe < LOWER_CUTOFF (2022-01-01) — 2 pagini
      consecutive.
    - Cutoff superior: paginile cu probe_dt >= UPPER_CUTOFF (2023-01-01) sunt
      COMPLET IGNORATE — nu salvam nimic in JSONL. Economisim astfel ~4000 URL-uri
      inutile si toate requesturile de fetch aferente.
    - --max-pages N: opreste dupa N pagini per tag (pentru audit rapid, ca sa
      poti verifica ca scraperul ajunge in fereastra 2022 fara sa astepti
      discovery-ul complet).

    Paginile „unknown" (probe esuat) sunt salvate preventiv — sunt rare si
    preferam sa avem articole in plus decat sa pierdem date.

    PENTRU FULL RUN: foloseste `discovery` fara argumente. Paginile 1-~80 contin
    2023-2026 si sunt skippuite. Paginile ~80-~180 contin 2022 si sunt salvate.
    Cutoff-ul de jos opreste automat la inceputul lui 2021.
    """
    existing = load_existing_discovery()
    logger.info("Discovery 2022 start — %d ID-uri preexistente în JSONL", len(existing))
    logger.info("Fereastra țintă: [%s, %s)",
                LOWER_CUTOFF.strftime("%Y-%m-%d"),
                UPPER_CUTOFF.strftime("%Y-%m-%d"))
    if max_pages is not None:
        logger.info("Max pages activ: %d pagini listing (per tag)", max_pages)
    total_new = 0
    total_in_range = 0  # ID-uri noi in fereastra 2022
    total_skipped_above = 0  # ID-uri din paginile „above range" (nesalvate)

    for tag_url in TAG_URLS:
        tag_slug = tag_url.rstrip("/").rsplit("/", 1)[-1]
        logger.info("=== Tag: %s ===", tag_slug)
        pagina = 1
        consecutive_below_lower = 0

        while True:
            if limit is not None and total_in_range >= limit:
                logger.info(
                    "Limit %d atins (articole IN RANGE), opresc discovery",
                    limit,
                )
                return
            if max_pages is not None and pagina > max_pages:
                logger.info(
                    "Max pages %d atins pe tag %s, trec la următorul tag",
                    max_pages, tag_slug,
                )
                break

            url = f"{tag_url}?p={pagina}"
            resp = http_get(url, logger)
            if resp is None:
                logger.warning("Listing %s eșuat, trec la tag-ul următor", url)
                break

            items = parse_listing_page(resp.text, tag_slug, pagina, logger)
            if not items:
                logger.info(
                    "Listing %s p=%d → 0 articole, probabil sfârșitul paginării",
                    tag_slug, pagina,
                )
                break

            # Probe pe ultimul articol al paginii (cel mai vechi cronologic)
            # IMPORTANT: rulam probe INAINTE de a scrie in JSONL, pentru a
            # putea decide daca pagina merita salvata. Evitam astfel sa
            # poluam JSONL-ul cu ~4000 URL-uri din 2023-2026, pe care nu le
            # vom fetch-ui niciodata.
            last_item = items[-1]
            polite_sleep()
            probe_dt = probe_article_date(
                last_item.url, last_item.id_articol, logger
            )

            # Determinam zona paginii pe baza probe_dt
            zone = "unknown"
            if probe_dt is None:
                zone = "unknown"
            elif probe_dt < LOWER_CUTOFF:
                zone = "below"
            elif probe_dt >= UPPER_CUTOFF:
                zone = "above"
            else:
                zone = "in_range"

            # Salvam in JSONL DOAR paginile din zona tinta sau edge cases.
            # Paginile „above" sunt skippuite complet — nu ne intereseaza 2023-2026.
            # Paginile „unknown" (probe esuat) sunt salvate ca precautie: e rar
            # si preferam sa avem date extra decat sa pierdem articole.
            new_on_page = 0
            if zone in ("in_range", "below", "unknown"):
                for item in items:
                    if item.id_articol in existing:
                        continue
                    append_discovery(item)
                    existing[item.id_articol] = item
                    total_new += 1
                    new_on_page += 1
            else:
                # zone == "above" → sarim peste (contam doar pentru log)
                total_skipped_above += len(items)

            # Logging & cutoff handling
            if zone == "unknown":
                logger.warning(
                    "Probe eșuat pe p=%d (url=%s) — salvez preventiv, fără cutoff",
                    pagina, last_item.url,
                )
                consecutive_below_lower = 0
            elif zone == "below":
                consecutive_below_lower += 1
                logger.info(
                    "Pagina %d SUB ținta (%s < %s), consecutive=%d (salvat: %d)",
                    pagina, probe_dt.strftime("%Y-%m-%d"),
                    LOWER_CUTOFF.strftime("%Y-%m-%d"),
                    consecutive_below_lower, new_on_page,
                )
                if consecutive_below_lower >= 2:
                    logger.info(
                        "2 pagini consecutive sub cutoff inferior, opresc tag-ul"
                    )
                    break
            elif zone == "above":
                logger.info(
                    "Pagina %d DEASUPRA ferestrei (%s >= %s), %d articole SKIPPED "
                    "(nu le salvăm, total skipped=%d)",
                    pagina, probe_dt.strftime("%Y-%m-%d"),
                    UPPER_CUTOFF.strftime("%Y-%m-%d"),
                    len(items), total_skipped_above,
                )
                consecutive_below_lower = 0
            else:  # in_range
                total_in_range += new_on_page
                logger.info(
                    "Pagina %d ÎN FEREASTRA (%s), %d noi in-range (total_in_range=%d)",
                    pagina, probe_dt.strftime("%Y-%m-%d"),
                    new_on_page, total_in_range,
                )
                consecutive_below_lower = 0

            if limit is not None and total_in_range >= limit:
                logger.info(
                    "Limit %d atins (in range), opresc discovery", limit
                )
                return

            pagina += 1
            polite_sleep()

    logger.info(
        "Discovery 2022 TERMINAT — %d ID-uri noi în JSONL | in_range=%d | "
        "skipped_above (2023-2026)=%d",
        total_new, total_in_range, total_skipped_above,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FETCH (adaptat pentru fereastra 2022)
# ─────────────────────────────────────────────────────────────────────────────


CSV_FIELDS = [
    "id_dataset", "id_articol", "url", "titlu",
    "data_publicarii", "data_actualizarii", "sectiune",
    "corp_articol", "nr_cuvinte", "tag_sursa", "sursa",
    "hash_continut", "fetch_ok", "fetch_error",
]


def load_fetched_ids() -> set[str]:
    """Set de id_articol deja prezente in CSV-ul raw (pentru resume)."""
    fetched: set[str] = set()
    if not RAW_CSV.exists():
        return fetched
    with RAW_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fetched.add(row["id_articol"])
    return fetched


def cache_path(id_articol: str) -> Path:
    """Calea catre HTML-ul cache-uit."""
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

    DIFERENTA FATA DE v1: dupa parsarea date_publicarii, verificam daca
    articolul se incadreaza in fereastra [LOWER_CUTOFF, UPPER_CUTOFF).
    Daca e in afara, pastram toate datele extrase dar setam fetch_ok=False
    si fetch_error="out_of_range_year:<YYYY>" pentru a fi usor filtrat ulterior.

    IMPORTANT: articolele fara data extrasa (data_publicarii="") NU se filtreaza
    aici — le trecem cu fetch_ok=True si fetch_error="missing_pub_date" pentru
    audit manual (foarte rare, dar vrem sa le vedem explicit).
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

    # Date — regex pe text + fallback meta
    text_page = soup.get_text(" ", strip=True)
    m_pub = RE_DATA_PUBLICARII.search(text_page)
    m_act = RE_DATA_ACTUALIZARII.search(text_page)
    dt_pub = parse_data_ro(m_pub.group(1)) if m_pub else None
    dt_act = parse_data_ro(m_act.group(1)) if m_act else None

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

    # Corp articol: selectori tipici Digi24
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
            for tag in container.find_all(
                ["script", "style", "nav", "aside", "footer", "iframe"]
            ):
                tag.decompose()
            corp = container.get_text("\n", strip=True)
            if len(corp) > 200:
                break

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

    # ─── Filtrare in fereastra temporala ───
    fetch_ok = True
    fetch_error = ""
    if dt_pub is None:
        # Nu filtram, dar marcam pentru audit
        fetch_error = "missing_pub_date"
    elif dt_pub < LOWER_CUTOFF or dt_pub >= UPPER_CUTOFF:
        fetch_ok = False
        fetch_error = f"out_of_range_year:{dt_pub.year}"

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
        fetch_ok=fetch_ok,
        fetch_error=fetch_error,
    )


def write_csv_header_if_needed() -> None:
    """Scrie header-ul CSV-ului daca fisierul nu exista."""
    if RAW_CSV.exists():
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_article(article: ArticleRaw) -> None:
    """Append un articol in CSV-ul raw, cu flush + fsync."""
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
    """Numara liniile din JSONL-ul de discovery."""
    if not DISCOVERY_FILE.exists():
        return 0
    with DISCOVERY_FILE.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_fetch(limit: int | None, logger: logging.Logger) -> None:
    """
    Fetch articolele din discovery JSONL. Resume automat.

    DIFERENTA FATA DE v1: la fetch inregistram separat:
    - total_ok_in_range: articole valide in fereastra 2022
    - total_out_of_range: articole valide dar in afara ferestrei 2022
    - total_err: articole cu parse_error sau http fail

    Logam progres cu toate cele 3 metrici pentru transparenta.
    """
    if not DISCOVERY_FILE.exists():
        logger.error("Discovery JSONL inexistent: %s", DISCOVERY_FILE)
        logger.error(
            "Rulează întâi: python scraper_digi24_v1_2022.py discovery"
        )
        return

    fetched_ids = load_fetched_ids()
    total_in_discovery = count_discovery_items()
    remaining_at_start = total_in_discovery - len(fetched_ids)
    logger.info(
        "Fetch 2022 start — %d ID-uri deja în CSV, %d de fetch-uit (total discovery: %d)",
        len(fetched_ids), remaining_at_start, total_in_discovery,
    )
    logger.info(
        "Fereastra țintă pentru fetch_ok=True: [%s, %s)",
        LOWER_CUTOFF.strftime("%Y-%m-%d"),
        UPPER_CUTOFF.strftime("%Y-%m-%d"),
    )
    if limit is not None:
        logger.info("Limit activ: %d articole maxim în această rulare", limit)

    write_csv_header_if_needed()
    total_new = 0
    total_ok_in_range = 0
    total_out_of_range = 0
    total_err = 0
    start_time = time.monotonic()

    for item in iter_discovery_items():
        if item.id_articol in fetched_ids:
            continue
        if limit is not None and total_new >= limit:
            logger.info("Limit %d atins, opresc fetch", limit)
            break

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
            polite_sleep()

        try:
            article = parse_article(html, item)
            append_article(article)
            total_new += 1
            if article.fetch_error.startswith("out_of_range_year"):
                total_out_of_range += 1
            elif article.fetch_ok:
                total_ok_in_range += 1
            else:
                total_err += 1

            if total_new % 1 == 0:
                elapsed = time.monotonic() - start_time
                rate = total_new / elapsed if elapsed > 0 else 0
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
                    "Progres: %d/%d (%.1f%%) | in_range=%d out_of_range=%d err=%d | "
                    "rate=%.2f art/s | ETA=%dh%02dm | ultim: %s",
                    total_new, target, pct,
                    total_ok_in_range, total_out_of_range, total_err,
                    rate, eta_h, eta_m, article.titlu[:50],
                )
        except Exception as exc:  # noqa: BLE001
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
        "Fetch 2022 TERMINAT — %d articole noi (in_range=%d, out_of_range=%d, err=%d)",
        total_new, total_ok_in_range, total_out_of_range, total_err,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scraper Digi24 — fereastra 2022 exclusiv, pentru completarea clasei 0."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_disc = sub.add_parser("discovery", help="Colectează URL-uri din listing-uri")
    p_disc.add_argument("--limit", type=int, default=None,
                        help="Max URL-uri NOI ÎN FEREASTRA 2022 (nu total)")
    p_disc.add_argument("--max-pages", type=int, default=None,
                        help="Max pagini listing de parcurs per tag "
                             "(util pentru audit rapid: --max-pages 3)")

    p_fetch = sub.add_parser("fetch", help="Descarcă și parsează articolele")
    p_fetch.add_argument("--limit", type=int, default=None,
                         help="Max articole de fetch")

    args = parser.parse_args()
    logger = setup_logging()

    random.seed(42)  # seed consistent cu v1

    if args.cmd == "discovery":
        run_discovery(limit=args.limit, max_pages=args.max_pages, logger=logger)
    elif args.cmd == "fetch":
        run_fetch(limit=args.limit, logger=logger)


if __name__ == "__main__":
    main()