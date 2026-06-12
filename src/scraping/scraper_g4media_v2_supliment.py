"""
Scraper G4Media v2 SUPLIMENT — discovery prin tag editorial /tag/razboi-rusia/.

Context: scraperul principal `scraper_g4media_v2.py` a descoperit ca tag-ul
umbrella `/tag/razboi-ucraina/` se opreste la 2025-11-19 (G4Media nu il mai
aplica editorial dupa aceasta data). Asta a lasat fereastra dec 2025 → apr 2026
goala in CSV-ul principal v2.

Proba tag-uri alternative (vezi RECAPITULARE_pentru_chat_nou_v2.md, sectiunea
"Solutie: tag suplimentar"):
  page  1  → 2026-02-25 (aniversarea 4 ani de invazie)
  page  8  → 2023-06-21
  page 16  → 2022-02-23 (CU O ZI INAINTEA invaziei)
  page 17+ → 404

`razboi-rusia` e tag PARALEL cu `razboi-ucraina`, nu succesor — acopera
intreaga perioada a razboiului dar e populat ~6× mai parcimonios (~336
articole in 4 ani vs 2160 pentru `razboi-ucraina`). Important: pagina 1 are
2026-02-25, deci tag-ul ramane activ in 2026 cand `razboi-ucraina` nu mai e
folosit. Asteptare pentru fereastra dec 2025 → apr 2026: ~30-50 articole.

Strategia suplimentului:
1. Discovery exhaustiv pe pages 1..18 (16 confirmat + buffer pentru 404 detect).
2. Fetch toate cele ~336 URL-uri candidate.
3. FILTRU TEMPORAL post-parse: pastram doar art.data >= 2025-11-20.
   Toate articolele cu data anterioara sunt deja in v2 principal — filtrul
   temporal joaca rolul de deduplicare implicita (nu mai e nevoie de cross-check
   pe URL-uri intre cele doua CSV-uri in aceasta faza; deduplicarea finala
   pe URL se face in merge_g4media_v2.py).
4. Reutilizeaza 100% logica din scraperul principal: HTTP+cache, parser HTML,
   strip atribuiri, validare, audit pasiv. Singurele diferente sunt constantele
   de path si filtrul temporal.

ID-uri: prefixul `g4m_v2s_` (s = supliment) ca sa nu se ciocneasca cu `g4m_v2_`
si sa fie filtrabil la merge.

Usage:
    python scraper_g4media_v2_supliment.py discovery
    python scraper_g4media_v2_supliment.py fetch
    python scraper_g4media_v2_supliment.py all
    python scraper_g4media_v2_supliment.py stats
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import requests

# ── Importuri din scraperul principal v2 (refolosire 100%) ────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from scraper_g4media_v2 import (
    BASE_URL,
    REQUEST_HEADERS,
    RATE_LIMIT_SECONDS,
    DiscoveryEntry,
    ParsedArticle,
    extract_article_urls_from_listing,
    parse_article_html,
    is_valid_article,
)
from common.thematic_filters import is_ukraine_related, topic_match_details


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATIE SUPLIMENT
# ══════════════════════════════════════════════════════════════════════════════

TAG_PATH = "/tag/razboi-rusia"

# Plafon empiric: pagina 16 e ultima valida (2022-02-23), pagina 17 → 404.
# Lasam buffer la 18 ca sa confirmam capatul prin 2× consecutive 404.
TAG_PAGE_START = 1
TAG_PAGE_END = 18

# FILTRU TEMPORAL — articolele dinaintea acestei date sunt deja in v2 principal
# (tag-ul `razboi-ucraina` le acopera pana pe 2025-11-19 inclusiv).
# Folosim 2025-11-20 ca prag inclusiv (>=).
DATE_CUTOFF = "2025-11-20"

# Path-uri (relative la radacina proiectului Licenta/) — paralel cu v2 principal
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = WORK_DIR / "cache" / "g4media_v2_supliment"
DISCOVERY_FILE = WORK_DIR / "discovery_g4media_v2_supliment.jsonl"
OUTPUT_CSV = WORK_DIR / "g4media_v2_supliment_raw.csv"
LOG_FILE = WORK_DIR / "scraper_g4media_v2_supliment.log"

WORK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING (independent de scraperul principal — log file separat)
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """Logger dedicat suplimentului — separat de cel al scraperului principal."""
    logger = logging.getLogger("g4media_v2_supliment")
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
# HTTP — copie locala a stratului de cache (path-uri diferite fata de principal)
# ══════════════════════════════════════════════════════════════════════════════
#
# Reimplementez `_rate_limit`, `http_get` si `fetch_with_cache` aici (in loc sa
# le import) din doua motive:
#  1. `CACHE_DIR` e diferit (cache/g4media_v2_supliment/), iar functiile din
#     scraperul principal au acea constanta hard-codata in closure.
#  2. Variabila globala `_last_request_time` din modulul principal e partajata,
#     dar daca rulam doar suplimentul nu vrem sa atingem starea acelui modul.
# Logica e identica — orice modificare in principal trebuie reflectata aici.

_last_request_time = 0.0


def _rate_limit() -> None:
    """Pauza minima intre request-uri consecutive (politicos cu G4Media)."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def http_get(url: str, max_retries: int = 3) -> str | None:
    """Fetch HTTP cu retry exponential. Returneaza HTML decodat sau None."""
    for attempt in range(1, max_retries + 1):
        _rate_limit()
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None
            log.warning(f"HTTP {resp.status_code} pentru {url} (atempt {attempt})")
        except requests.RequestException as e:
            log.warning(f"Eroare request {url} (atempt {attempt}): {e}")

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    log.error(f"Eșec definitiv pentru {url} după {max_retries} încercări")
    return None


def url_to_cache_path(url: str) -> Path:
    """Genereaza path local pentru cache-ul HTML al unui URL."""
    h = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{h}.html"


def fetch_with_cache(url: str) -> str | None:
    """Fetch HTML — foloseste cache local daca exista, altfel descarca si salveaza."""
    cache_path = url_to_cache_path(url)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    html = http_get(url)
    if html is not None:
        cache_path.write_text(html, encoding="utf-8")
    return html


# ══════════════════════════════════════════════════════════════════════════════
# FAZA 1 — DISCOVERY (parcurgere /tag/razboi-rusia/page/N/)
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_discovery() -> set[str]:
    """Incarca URL-urile deja descoperite in suplimentul curent (resume)."""
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
    Faza 1: parcurge /tag/razboi-rusia/page/{1..18}/, extrage URL-uri,
    le scrie incremental in discovery_g4media_v2_supliment.jsonl.

    Nota: NU facem cross-deduplicare aici cu discovery-ul v2 principal.
    Multe URL-uri vor fi tagat-uite cu AMBELE tag-uri (razboi-ucraina SI
    razboi-rusia), deci se vor regasi in ambele discovery-uri. Filtrul
    temporal de la fetch + deduplicarea pe URL la merge se ocupa de asta.
    """
    log.info("=" * 70)
    log.info("FAZA 1 — DISCOVERY (tag /razboi-rusia)")
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
# FAZA 2 — FETCH + PARSE + FILTRU TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════

# Schema CSV — IDENTICA cu cea din v2 principal pentru a permite concatenare
# directa fara reshape. Singura diferenta e prefixul ID-ului.
CSV_FIELDS = [
    "id", "url", "titlu", "data", "sursa_site", "sectiune",
    "text_curat", "nr_cuvinte", "tags", "autor",
    "matched_core", "matched_hybrid",
    "audit_thematic_pass",
    "found_on_page",
    "calitate_extractie", "label", "label_numeric", "hash_continut",
]


def article_to_csv_row(
    art: ParsedArticle,
    art_id: int,
    found_on_page: int,
) -> dict:
    """Converteste ParsedArticle in rand CSV. Schema IDENTICA cu v2 principal."""
    text_hash = hashlib.md5(art.text_curat.encode()).hexdigest()[:16]
    det = topic_match_details(art.text_curat)
    audit_pass = is_ukraine_related(art.text_curat)

    return {
        "id": f"g4m_v2s_{art_id:05d}",  # prefix `s` = supliment
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
    """URL-urile deja procesate in CSV-ul suplimentului (pentru resume faza 2)."""
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


def passes_date_filter(art: ParsedArticle) -> bool:
    """
    Filtru temporal: pastram doar articolele cu data >= DATE_CUTOFF.
    Articolele anterioare sunt deja in v2 principal (acoperite de
    `razboi-ucraina` pana pe 2025-11-19).

    Comparatia se face lexicografic pe primele 10 caractere ISO ('YYYY-MM-DD').
    G4Media foloseste meta `article:published_time` in format ISO 8601 cu
    timezone (ex: '2026-02-25T14:32:18+02:00'), deci primele 10 caractere
    sunt mereu data calendaristica in forma comparabila lexicografic.
    """
    if not art.data or len(art.data) < 10:
        return False
    return art.data[:10] >= DATE_CUTOFF


def run_fetch_phase() -> None:
    """
    Faza 2: pentru fiecare URL din discovery, fetch + parse + validare +
    filtru temporal + write.
    """
    log.info("=" * 70)
    log.info("FAZA 2 — FETCH + PARSE + FILTRU TEMPORAL")
    log.info("=" * 70)
    log.info(f"Filtru temporal: păstrăm doar data >= {DATE_CUTOFF}")

    if not DISCOVERY_FILE.exists():
        log.error(f"Fișier discovery lipsă: {DISCOVERY_FILE}")
        log.error("Rulează mai întâi: python scraper_g4media_v2_supliment.py discovery")
        return

    candidates: list[DiscoveryEntry] = []
    with open(DISCOVERY_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                candidates.append(DiscoveryEntry(**d))
            except (json.JSONDecodeError, TypeError):
                continue

    log.info(f"Total candidate în discovery: {len(candidates)}")

    already_done = load_existing_csv_urls()
    log.info(f"Deja procesate (resume): {len(already_done)}")

    pending = [c for c in candidates if c.url not in already_done]
    log.info(f"De procesat: {len(pending)}")

    next_id = len(already_done) + 1

    stats = {
        "ok": 0,
        "parse_fail": 0,
        "fetch_fail": 0,
        "filtered_out_date": 0,  # SKIP din motiv temporal — asteptat sa fie majoritar
        "invalid": {},
    }

    for i, entry in enumerate(pending, 1):
        if i % 25 == 0:
            log.info(f"  Progres: {i}/{len(pending)} "
                     f"(ok={stats['ok']}, "
                     f"date_skip={stats['filtered_out_date']}, "
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

        # FILTRU TEMPORAL — inainte de validarea generala pentru a economisi
        # log-uri si pentru raportare clara
        if not passes_date_filter(art):
            stats["filtered_out_date"] += 1
            log.debug(f"  filtru_data ({art.data[:10]}): {entry.url}")
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

    log.info("=" * 70)
    log.info("FETCH SUPLIMENT COMPLET")
    log.info("=" * 70)
    log.info(f"Total candidate procesate:       {len(pending)}")
    log.info(f"Articole valide salvate:         {stats['ok']}")
    log.info(f"Skip filtru temporal (<{DATE_CUTOFF}): {stats['filtered_out_date']}")
    log.info(f"Eșec fetch:                      {stats['fetch_fail']}")
    log.info(f"Eșec parse:                      {stats['parse_fail']}")
    if stats["invalid"]:
        log.info(f"Invalide după parse:")
        for reason, count in sorted(stats["invalid"].items(), key=lambda x: -x[1]):
            log.info(f"  {count:4} × {reason}")
    log.info(f"CSV final: {OUTPUT_CSV}")
    log.info(f"")
    log.info(f"Așteptare per recapitulare: ~30-50 articole în fereastra")
    log.info(f"dec 2025 → apr 2026. Rezultat efectiv: {stats['ok']}")


# ══════════════════════════════════════════════════════════════════════════════
# RAPORT STADIU
# ══════════════════════════════════════════════════════════════════════════════

def show_stats() -> None:
    """Raport rapid asupra stadiului discovery + CSV pentru SUPLIMENT."""
    log.info("=" * 70)
    log.info("STADIU SCRAPER G4MEDIA v2 SUPLIMENT (razboi-rusia)")
    log.info("=" * 70)

    if DISCOVERY_FILE.exists():
        with open(DISCOVERY_FILE, encoding="utf-8") as f:
            n_disc = sum(1 for _ in f)
        log.info(f"Discovery: {n_disc} URL-uri în {DISCOVERY_FILE.name}")
    else:
        log.info("Discovery: ÎNCĂ NEFĂCUT")

    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        log.info(f"CSV: {len(rows)} articole în {OUTPUT_CSV.name}")

        if rows:
            from collections import Counter
            ani = Counter()
            luni = Counter()
            audit_pass = 0
            for r in rows:
                an = r["data"][:4] if r["data"] else "?"
                an_luna = r["data"][:7] if r["data"] else "?"
                ani[an] += 1
                luni[an_luna] += 1
                if r.get("audit_thematic_pass") == "1":
                    audit_pass += 1

            log.info("Distribuție pe ani:")
            for an in sorted(ani.keys()):
                log.info(f"  {an}: {ani[an]:4}")
            log.info("Distribuție pe luni (post-cutoff):")
            for luna in sorted(luni.keys()):
                log.info(f"  {luna}: {luni[luna]:4}")
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
