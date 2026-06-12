"""
Scraper G4Media v2 — discovery prin tag editorial /tag/razboi-ucraina/.

Pivot fata de v1: in loc sa facem discovery prin paginarea categoriei /articole/
si sa filtram tematic cu reguli lexicale, folosim direct tag-area editoriala
G4Media. Vezi findings_recon_g4media_v2.md pentru justificare completa.

Principii cheie:
1. Discovery exhaustiv prin /tag/razboi-ucraina/page/{1..103}/ — paginare
   declarata a tag-ului. Probarea a confirmat ca pagina 103 contine articole
   din 28 februarie 2022 (a treia zi dupa invazie), iar pagina 1 contine
   articole curente. Distributie temporala uniforma (~21 articole/pagina,
   ~25 pagini/an).
2. Fetch exhaustiv pe toate articolele descoperite — volum mic (~2160 total),
   poate fi intreg downloadat in ~1h cu rate limit 2s.
3. Sampling stratificat pe ANI se face POST-FETCH (in script separat sau
   sectiune dedicata) cu seed reproducibil. Tinta: ~1000 articole distribuite
   pe 2022–2026.
4. `thematic_filters.is_ukraine_related` ruleaza ca AUDIT PASIV — flag in CSV,
   nu drop. Tag-area editoriala e suverana pentru includere.

Stack:
- Resume capability simpla: discovery JSONL + cache HTML local + skip URL-uri
  deja in CSV.
- Refoloseste parser-ul de articol din v1 (selectorii sunt validati empiric).
- Nu mai aplica filtru iran_dominant (nu mai e nevoie — tag-area editoriala
  e mai precisa).

Usage:
    python scraper_g4media_v2.py discovery   # faza 1: descoperire URL-uri
    python scraper_g4media_v2.py fetch       # faza 2: fetch + parse
    python scraper_g4media_v2.py all         # ambele
    python scraper_g4media_v2.py stats       # raport stadiul curent
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Importuri din modulul partajat ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from common.thematic_filters import is_ukraine_related, topic_match_details

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATIE
# ══════════════════════════════════════════════════════════════════════════════

BASE_URL = "https://www.g4media.ro"
TAG_PATH = "/tag/razboi-ucraina"

# Paginarea declarata a tag-ului — verificata empiric:
#   page 1   → ~2025-11-19
#   page 25  → ~2024-03-19
#   page 50  → ~2023-07-18
#   page 75  → ~2022-11-19
#   page 103 → ~2022-02-28 (a treia zi dupa invazie)
#   page 104+ → 404
# Lasam un mic buffer (page 105) ca sa detectam capatul natural prin 404.
TAG_PAGE_START = 1
TAG_PAGE_END = 105

# Rate limit conservator — politicos cu G4Media. Volumul total e mic.
RATE_LIMIT_SECONDS = 2.0

# Validare articol
MIN_TEXT_WORDS = 64
MAX_TEXT_WORDS = 2000

# User-Agent realist (scraperul v1 a functionat fara blocaje cu acesta)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
    # IMPORTANT: NU brotli — bug cunoscut din v1 (requests primeste binary nedecomprimat)
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# ── Path-uri (relative la radacina proiectului Licenta/) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = WORK_DIR / "cache" / "g4media_v2"
DISCOVERY_FILE = WORK_DIR / "discovery_g4media_v2.jsonl"
OUTPUT_CSV = WORK_DIR / "g4media_v2_raw.csv"
LOG_FILE = WORK_DIR / "scraper_g4media_v2.log"

WORK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """Configurare logging cu output si pe ecran si in fisier."""
    logger = logging.getLogger("g4media_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = setup_logging()


# ══════════════════════════════════════════════════════════════════════════════
# HTTP (cu rate limit + retry + cache pe disk)
# ══════════════════════════════════════════════════════════════════════════════

_last_request_time = 0.0


def _rate_limit() -> None:
    """Asigura pauza minima intre request-uri consecutive."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def http_get(url: str, max_retries: int = 3) -> Optional[str]:
    """
    Fetch HTTP cu retry exponential. Returneaza HTML decodat sau None la esec.
    """
    for attempt in range(1, max_retries + 1):
        _rate_limit()
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                # 404 e legitim (pagina peste plafon) — nu retry
                return None
            log.warning(f"HTTP {resp.status_code} pentru {url} (atempt {attempt})")
        except requests.RequestException as e:
            log.warning(f"Eroare request {url} (atempt {attempt}): {e}")

        if attempt < max_retries:
            time.sleep(2 ** attempt)  # backoff exponential

    log.error(f"Eșec definitiv pentru {url} după {max_retries} încercări")
    return None


def url_to_cache_path(url: str) -> Path:
    """Genereaza path local pentru cache-ul HTML al unui URL."""
    h = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{h}.html"


def fetch_with_cache(url: str) -> Optional[str]:
    """Fetch HTML — foloseste cache local daca exista, altfel descarca si salveaza."""
    cache_path = url_to_cache_path(url)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    html = http_get(url)
    if html is not None:
        cache_path.write_text(html, encoding="utf-8")
    return html


# ══════════════════════════════════════════════════════════════════════════════
# FAZA 1 — DISCOVERY (parcurgere /tag/razboi-ucraina/page/N/)
# ══════════════════════════════════════════════════════════════════════════════

# Pattern URL articol G4Media: https://www.g4media.ro/{slug}.html
# Slug-ul nu poate contine slash (e direct sub root)
ARTICLE_URL_RE = re.compile(r"^https?://(?:www\.)?g4media\.ro/([^/]+)\.html$")


@dataclass
class DiscoveryEntry:
    """Inregistrare discovery: URL + slug + pagina pe care a fost gasit."""
    url: str
    slug: str
    found_on_page: int


def extract_article_urls_from_listing(html: str) -> list[tuple[str, str]]:
    """
    Extrage toate link-urile catre articole individuale dintr-o pagina de tag.
    Returneaza lista deduplicata de (url_canonic, slug).
    """
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    results = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = ARTICLE_URL_RE.match(href)
        if not m:
            continue

        # Normalizare — totdeauna cu www
        canonical_url = href.replace("https://g4media.ro", "https://www.g4media.ro")
        if canonical_url in seen:
            continue
        seen.add(canonical_url)

        slug = m.group(1)
        results.append((canonical_url, slug))

    return results


def load_existing_discovery() -> set[str]:
    """Incarca URL-urile deja descoperite (pentru resume)."""
    if not DISCOVERY_FILE.exists():
        return set()
    seen = set()
    with open(DISCOVERY_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                seen.add(entry["url"])
            except json.JSONDecodeError:
                continue
    return seen


def append_discovery(entry: DiscoveryEntry) -> None:
    """Append incremental la JSONL — siguranta la crash."""
    with open(DISCOVERY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def run_discovery_phase() -> None:
    """
    Faza 1: parcurge /tag/razboi-ucraina/page/{1..105}/, extrage URL-uri
    de articole, le scrie incremental in discovery_g4media_v2.jsonl.

    Se opreste natural cand doua pagini consecutive returneaza 404
    (capatul real al paginarii tag-ului).
    """
    log.info("=" * 70)
    log.info("FAZA 1 — DISCOVERY (tag /razboi-ucraina)")
    log.info("=" * 70)

    seen = load_existing_discovery()
    log.info(f"Discovery existent: {len(seen)} URL-uri deja descoperite")

    consecutive_404 = 0
    new_count = 0

    for page in range(TAG_PAGE_START, TAG_PAGE_END + 1):
        if page == 1:
            url = f"{BASE_URL}{TAG_PATH}/"
        else:
            url = f"{BASE_URL}{TAG_PATH}/page/{page}/"

        log.info(f"[page {page:3}/{TAG_PAGE_END}] {url}")
        html = http_get(url)

        if html is None:
            consecutive_404 += 1
            log.warning(f"  Pagina {page} indisponibilă (404 sau eșec)")
            if consecutive_404 >= 2:
                log.info(f"  Două pagini consecutive eșuate → oprire discovery")
                break
            continue

        consecutive_404 = 0
        articles = extract_article_urls_from_listing(html)
        log.info(f"  Găsite {len(articles)} articole pe pagină")

        page_new = 0
        for url_art, slug in articles:
            if url_art in seen:
                continue
            entry = DiscoveryEntry(url=url_art, slug=slug, found_on_page=page)
            append_discovery(entry)
            seen.add(url_art)
            new_count += 1
            page_new += 1

        log.info(f"  Noi adăugate: {page_new} (total cumulat: {len(seen)})")

    log.info(f"Discovery COMPLETĂ — total {len(seen)} URL-uri unice "
             f"(+{new_count} noi în această rulare)")


# ══════════════════════════════════════════════════════════════════════════════
# FAZA 2 — FETCH + PARSE ARTICOLE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedArticle:
    """Reprezentarea unui articol G4Media dupa parse."""
    url: str
    titlu: str
    data: str  # ISO format (din meta article:published_time)
    autor: str
    sectiune: str
    text_curat: str
    nr_cuvinte: int
    tags: list[str] = field(default_factory=list)


# Surse internationale frecvent citate de G4Media (preluat din v1)
# Justificare strip: vezi v1 — sursele Reuters/AFP/BBC etc. apar doar in clasa 0
# si ar deveni shortcut stilistic exploit-abil.
ATTRIBUTION_SOURCES = (
    r"AFP|Reuters|BBC|CNN|Bloomberg|Associated\s+Press|AP|"
    r"Agerpres|Mediafax|EFE|DPA|TASS|Interfax|Sky\s+News|"
    r"Financial\s+Times|FT|Wall\s+Street\s+Journal|WSJ|"
    r"New\s+York\s+Times|NYT|Washington\s+Post|Guardian|"
    r"Politico|Deutsche\s+Welle|DW|Euronews|Der\s+Spiegel"
)
ATTRIBUTION_VERBS = (
    r"relatează|transmite|potrivit|conform|preluat\s+de|citat\s+de|"
    r"informează|notează|scrie|menționează|anunță"
)
ATTRIBUTION_PATTERN = re.compile(
    rf",?\s*({ATTRIBUTION_VERBS})\s+({ATTRIBUTION_SOURCES})\s*\.?\s*$",
    re.IGNORECASE,
)
ATTRIBUTION_INLINE_PATTERN = re.compile(
    rf",\s*({ATTRIBUTION_VERBS})\s+({ATTRIBUTION_SOURCES})\.\s*",
    re.IGNORECASE,
)


def strip_attribution_signatures(text: str) -> str:
    """
    Elimina boilerplate-ul de atribuire de tip "..., transmite Agerpres."
    de la sfarsitul textului. Aplicat repetitiv (uneori sunt incadrate).
    """
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = ATTRIBUTION_PATTERN.sub("", cur).strip()
    # Inline (in mijlocul textului) — varianta mai conservativa, doar daca e
    # urmata de o propozitie clar noua
    cur = ATTRIBUTION_INLINE_PATTERN.sub(". ", cur)
    return cur


def parse_article_html(html: str, url: str) -> Optional[ParsedArticle]:
    """
    Extrage titlu, data, autor, text curat din HTML-ul unui articol G4Media.
    Returneaza None daca structura nu corespunde.

    Selectori validati empiric pe articole din 2022, 2023, 2024, 2026
    (vezi findings_recon_g4media.md si findings_recon_g4media_v2.md).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Titlu — h1 principal
    h1 = soup.find("h1")
    if not h1:
        return None
    titlu = h1.get_text(strip=True)
    if not titlu:
        return None

    # Data — meta property article:published_time (cel mai fiabil)
    meta_date = soup.find("meta", property="article:published_time")
    data = meta_date["content"] if meta_date and meta_date.get("content") else ""

    # Autor — span.single__authors
    autor = ""
    authors_span = soup.find("span", class_="single__authors")
    if authors_span:
        first_link = authors_span.find("a")
        if first_link:
            autor = first_link.get_text(strip=True)

    # Sectiune — meta article:section
    sectiune = ""
    meta_section = soup.find("meta", property="article:section")
    if meta_section and meta_section.get("content"):
        sectiune = meta_section["content"]

    # Tags — div.single__tags (vital pentru audit — verificam ca tag-ul
    # razboi-ucraina e efectiv prezent in articol)
    tags = []
    tags_div = soup.find("div", class_="single__tags")
    if tags_div:
        tags = [a.get_text(strip=True) for a in tags_div.find_all("a", rel="tag")]

    # Continut — div.single__text > p[id^="p-"]
    single_text = soup.find("div", class_="single__text")
    if not single_text:
        return None

    paragrafe = single_text.find_all("p", id=lambda x: x and x.startswith("p-"))
    if not paragrafe:
        return None

    text_parts = []
    for p in paragrafe:
        t = p.get_text(separator=" ", strip=True)
        if t:
            text_parts.append(t)

    text_raw = " ".join(text_parts)
    text_curat = strip_attribution_signatures(text_raw)

    # Concatenare titlu + text (oglinda cu structura clasei 1 Veridica)
    text_complet = f"{titlu} {text_curat}".strip()
    text_complet = re.sub(r"\s+", " ", text_complet)
    nr_cuvinte = len(text_complet.split())

    return ParsedArticle(
        url=url,
        titlu=titlu,
        data=data,
        autor=autor,
        sectiune=sectiune,
        text_curat=text_complet,
        nr_cuvinte=nr_cuvinte,
        tags=tags,
    )


def is_valid_article(art: ParsedArticle) -> tuple[bool, str]:
    """
    Validare articol parsat. Nota importanta fata de v1: NU mai filtram
    tematic aici (filtrul era `is_ukraine_related`). Tag-area editoriala
    G4Media e suverana pentru includere; filtrul lexical devine audit pasiv.
    """
    if not art.titlu or len(art.titlu) < 10:
        return False, "titlu lipsă sau prea scurt"

    if not art.data:
        return False, "data lipsă (meta article:published_time absent)"

    if not art.text_curat or art.nr_cuvinte < MIN_TEXT_WORDS:
        return False, f"text prea scurt ({art.nr_cuvinte} cuvinte, min {MIN_TEXT_WORDS})"

    if art.nr_cuvinte > MAX_TEXT_WORDS:
        return False, f"text prea lung ({art.nr_cuvinte} cuvinte, max {MAX_TEXT_WORDS})"

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# CSV WRITER
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "id", "url", "titlu", "data", "sursa_site", "sectiune",
    "text_curat", "nr_cuvinte", "tags", "autor",
    "matched_core", "matched_hybrid",
    "audit_thematic_pass",  # NOU in v2 — audit pasiv, nu drop
    "found_on_page",        # NOU in v2 — pagina de discovery (pentru sampling)
    "calitate_extractie", "label", "label_numeric", "hash_continut",
]


def article_to_csv_row(
    art: ParsedArticle,
    art_id: int,
    found_on_page: int,
) -> dict:
    """
    Converteste ParsedArticle in rand CSV. Schema e compatibila cu Veridica
    + doua campuri noi:
      - audit_thematic_pass: True daca filtrul lexical confirma tematica
      - found_on_page: pagina tag-ului unde a fost descoperit (pentru sampling)
    """
    text_hash = hashlib.md5(art.text_curat.encode()).hexdigest()[:16]
    det = topic_match_details(art.text_curat)

    # AUDIT PASIV — ruleaza filtrul lexical, dar nu drop-uie
    audit_pass = is_ukraine_related(art.text_curat)

    return {
        "id": f"g4m_v2_{art_id:05d}",
        "url": art.url,
        "titlu": art.titlu,
        "data": art.data,
        "sursa_site": "g4media.ro",
        "sectiune": art.sectiune,
        "text_curat": art.text_curat,
        "nr_cuvinte": art.nr_cuvinte,
        "tags": "|".join(art.tags),
        "autor": art.autor,
        "matched_core": int(det.matched_core),
        "matched_hybrid": int(det.matched_hybrid),
        "audit_thematic_pass": int(audit_pass),
        "found_on_page": found_on_page,
        "calitate_extractie": "excelenta",
        "label": "stire_credibila",
        "label_numeric": 0,
        "hash_continut": text_hash,
    }


def load_existing_csv_urls() -> set[str]:
    """URL-urile deja procesate in CSV (pentru resume faza 2)."""
    if not OUTPUT_CSV.exists():
        return set()
    seen = set()
    with open(OUTPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(row["url"])
    return seen


def append_csv_row(row: dict) -> None:
    """Append incremental la CSV — header scris doar la prima rulare."""
    write_header = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_fetch_phase() -> None:
    """
    Faza 2: pentru fiecare URL din discovery, fetch + parse + validare + write.
    """
    log.info("=" * 70)
    log.info("FAZA 2 — FETCH + PARSE")
    log.info("=" * 70)

    if not DISCOVERY_FILE.exists():
        log.error(f"Fișier discovery lipsă: {DISCOVERY_FILE}")
        log.error("Rulează mai întâi: python scraper_g4media_v2.py discovery")
        return

    # Incarca toate URL-urile descoperite
    candidates: list[DiscoveryEntry] = []
    with open(DISCOVERY_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                candidates.append(DiscoveryEntry(**d))
            except (json.JSONDecodeError, TypeError):
                continue

    log.info(f"Total candidate în discovery: {len(candidates)}")

    # Resume: skip URL-uri deja in CSV
    already_done = load_existing_csv_urls()
    log.info(f"Deja procesate (resume): {len(already_done)}")

    pending = [c for c in candidates if c.url not in already_done]
    log.info(f"De procesat: {len(pending)}")

    # Determina ID-ul de start (pentru numerotare consistenta)
    next_id = len(already_done) + 1

    stats = {
        "ok": 0,
        "parse_fail": 0,
        "invalid": {},
        "fetch_fail": 0,
    }

    for i, entry in enumerate(pending, 1):
        if i % 25 == 0:
            log.info(f"  Progres: {i}/{len(pending)} "
                     f"(ok={stats['ok']}, "
                     f"parse_fail={stats['parse_fail']}, "
                     f"fetch_fail={stats['fetch_fail']})")

        html = fetch_with_cache(entry.url)
        if html is None:
            stats["fetch_fail"] += 1
            continue

        art = parse_article_html(html, entry.url)
        if art is None:
            stats["parse_fail"] += 1
            log.debug(f"  parse_fail: {entry.url}")
            continue

        valid, reason = is_valid_article(art)
        if not valid:
            stats["invalid"][reason] = stats["invalid"].get(reason, 0) + 1
            log.debug(f"  invalid ({reason}): {entry.url}")
            continue

        row = article_to_csv_row(art, next_id, entry.found_on_page)
        append_csv_row(row)
        next_id += 1
        stats["ok"] += 1

    # Raport final
    log.info("=" * 70)
    log.info("FETCH COMPLET")
    log.info("=" * 70)
    log.info(f"Total candidate procesate:  {len(pending)}")
    log.info(f"Articole valide salvate:    {stats['ok']}")
    log.info(f"Eșec fetch:                 {stats['fetch_fail']}")
    log.info(f"Eșec parse:                 {stats['parse_fail']}")
    if stats["invalid"]:
        log.info(f"Invalide după parse:")
        for reason, count in sorted(
            stats["invalid"].items(), key=lambda x: -x[1]
        ):
            log.info(f"  {count:4} × {reason}")
    log.info(f"CSV final: {OUTPUT_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
# RAPORT STADIU
# ══════════════════════════════════════════════════════════════════════════════

def show_stats() -> None:
    """Raport rapid asupra stadiului discovery + CSV."""
    log.info("=" * 70)
    log.info("STADIU SCRAPER G4MEDIA v2")
    log.info("=" * 70)

    # Discovery
    if DISCOVERY_FILE.exists():
        with open(DISCOVERY_FILE, encoding="utf-8") as f:
            n_disc = sum(1 for _ in f)
        log.info(f"Discovery: {n_disc} URL-uri în {DISCOVERY_FILE.name}")
    else:
        log.info("Discovery: ÎNCĂ NEFĂCUT")

    # CSV
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        log.info(f"CSV: {len(rows)} articole în {OUTPUT_CSV.name}")

        if rows:
            # Distributie temporala pe ani
            from collections import Counter
            ani = Counter()
            audit_pass = 0
            for r in rows:
                an = r["data"][:4] if r["data"] else "?"
                ani[an] += 1
                if r.get("audit_thematic_pass") == "1":
                    audit_pass += 1

            log.info("Distribuție pe ani:")
            for an in sorted(ani.keys()):
                log.info(f"  {an}: {ani[an]:4}")
            log.info(f"Audit thematic pass: {audit_pass}/{len(rows)} "
                     f"({100*audit_pass/len(rows):.1f}%)")
    else:
        log.info("CSV: ÎNCĂ NEFĂCUT")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "discovery":
        run_discovery_phase()
    elif cmd == "fetch":
        run_fetch_phase()
    elif cmd == "all":
        run_discovery_phase()
        run_fetch_phase()
    elif cmd == "stats":
        show_stats()
    else:
        print(f"Comandă necunoscută: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()