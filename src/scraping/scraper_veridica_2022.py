r"""
scraper_veridica_2022.py
────────────────────────
Script focusat exclusiv pe colectarea articolelor din 2022 de pe Veridica.

Parte din proiectul de licenta „Sistem de Detectie Automata si Explicabila
a Dezinformarii Pro-Ruse in Presa Romaneasca — Studiu de Caz: Razboiul din
Ucraina".

DE CE ACEST SCRIPT:
  Scraper-ul v5 pornea listing-ul de la pagina 1 si parcurgea inutil
  paginile 1–19 (2023–2025, deja colectate in v4.2). Acest script sare
  direct la paginile 20–38, unde se afla articolele din 2022.

CE FACE:
  - Listing DOAR paginile 20–38 din /fake-news-dezinformare-propaganda
  - Filtreaza URL-urile deja existente in veridica_ukraine_v4_2.csv
    (dedup inainte de scraping, fara request-uri inutile)
  - Parsare identica cu v4.2 (STIRE/NARATIUNI/OBIECTIVE, detectie leak,
    filtru tematic Ucraina/Rusia)
  - Oprire anticipata daca detecteaza articole din 2021 (pre-invazie)

OUTPUT:
  veridica_2022_raw.csv      → toate articolele scrape-uite (audit)
  veridica_2022_ukraine.csv  → subset filtrat tematic (input pentru cleaning)

PASUL URMATOR:
  Concateneaza veridica_ukraine_v4_2.csv + veridica_2022_ukraine.csv
  si ruleaza clean_veridica_v5.py pe dataset-ul combinat.
"""

import time
import re
import hashlib
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ── Configurare ────────────────────────────────────────────────────────────────

BASE_URL    = "https://www.veridica.ro"
SECTION     = "/fake-news-dezinformare-propaganda"

START_PAGE  = 21   # prima pagina cu 2022 (cea mai recenta din 2022)
END_PAGE    = 28   # ultima pagina cu 2022 (cea mai veche din 2022)

DELAY_SECONDS    = 2
CHECKPOINT_EVERY = 25

OUTPUT_FILE   = "veridica_2022_raw.csv"
FILTERED_FILE = "veridica_2022_ukraine.csv"

# CSV-ul existent — URL-urile din el sunt sarite fara request HTTP
EXISTING_CSV = "veridica_ukraine_v4_2.csv"

# Daca detectam articole exclusiv din acest an sau mai vechi, oprim paginarea
# STOP_BEFORE_YEAR = 2022

LABEL         = "dezinformare_pro_rusa"
LABEL_NUMERIC = 1

MIN_STIRE_WORDS = 10
MAX_STIRE_WORDS = 800

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LicentaBot/1.0; "
        "research scraper pentru teza de licenta)"
    )
}


# ── Filtre tematice Ucraina/Rusia (identice cu v4.2) ─────────────────────────

UKRAINE_PATTERNS = [
    re.compile(r"\bucrain\w*",                            re.IGNORECASE),
    re.compile(r"\brus(ia|ă|ești|ească|esc|ilor|ești)\b", re.IGNORECASE),
    re.compile(r"\bruse\w+",                              re.IGNORECASE),
    re.compile(r"\bpro[- ]?rus\w*",                       re.IGNORECASE),
    re.compile(r"\bsovietic\w*",                          re.IGNORECASE),
    re.compile(r"\b(putin|zelenski|zelensky|lavrov|medvedev|șoigu|soigu|prigojin|prigozhin)\b",
               re.IGNORECASE),
    re.compile(r"\bkremlin\w*",                           re.IGNORECASE),
    re.compile(r"\bpro[- ]?kremlin\w*",                   re.IGNORECASE),
    re.compile(r"\bwagner\b",                             re.IGNORECASE),
    re.compile(r"\bduma\b",                               re.IGNORECASE),
    re.compile(r"\brosatom\b",                            re.IGNORECASE),
    re.compile(r"\bgazprom\b",                            re.IGNORECASE),
    re.compile(r"\brt\s*(news)?\b",                       re.IGNORECASE),
    re.compile(r"\bsputnik\b",                            re.IGNORECASE),
    re.compile(r"\b(donbas|donbass|donețk|donetk|donetsk|lugansk|luhansk)\b",
               re.IGNORECASE),
    re.compile(r"\b(crimeea|crimea|sevastopol)\b",        re.IGNORECASE),
    re.compile(r"\b(mariupol|herson|kherson|zaporijia|zaporizhzhia|bahmut|bakhmut|avdiivka)\b",
               re.IGNORECASE),
    re.compile(r"\b(kiev|kyiv|harkov|harkiv|kharkiv|odesa|odessa|lvov|lviv|cernobil|chernobyl)\b",
               re.IGNORECASE),
    re.compile(r"\bmoscova\b",                            re.IGNORECASE),
    re.compile(r"\binvazi\w*",                            re.IGNORECASE),
    re.compile(r"\b(război|razboi|războiul|razboiul)\b",  re.IGNORECASE),
    re.compile(r"\boperați\w*\s+special\w*",              re.IGNORECASE),
    re.compile(r"\bdenazifica\w*",                        re.IGNORECASE),
    re.compile(r"\bnazi[sș]ti?\s+ucrain\w*",              re.IGNORECASE),
    re.compile(r"\bazov\b",                               re.IGNORECASE),
    re.compile(r"\bbiolaborator\w*",                      re.IGNORECASE),
    # Cluster Moldova / razboi hibrid
    re.compile(r"\btransnistr\w*",                        re.IGNORECASE),
    re.compile(r"\b(maia\s+sandu|igor\s+dodon|ilan\s+șor|ilan\s+sor|plahotniuc)\b",
               re.IGNORECASE),
    re.compile(r"\b(chișinău|chisinau|comrat|tiraspol|găgăuzia|gagauzia|gagauz\w*)\b",
               re.IGNORECASE),
    re.compile(r"\b(republica\s+moldova|r\.\s*moldova)\b", re.IGNORECASE),
    re.compile(r"\bantirus\w*",                           re.IGNORECASE),
    re.compile(r"\brusofob\w*",                           re.IGNORECASE),
]

FACTCHECK_LEAK_MARKERS = [
    re.compile(r"\beste\s+fals[ăa]?\b",             re.IGNORECASE),
    re.compile(r"\bsunt\s+false\b",                 re.IGNORECASE),
    re.compile(r"\bnu\s+este\s+adevărat",           re.IGNORECASE),
    re.compile(r"\bnu\s+e\s+adevărat",              re.IGNORECASE),
    re.compile(r"\bdezminte\w*",                    re.IGNORECASE),
    re.compile(r"\bcontrazice\w*",                  re.IGNORECASE),
    re.compile(r"\bverific\w*\s+(arat|arăt|indic)", re.IGNORECASE),
    re.compile(r"\bfact[- ]check\w*",               re.IGNORECASE),
    re.compile(r"\bpotrivit\s+(unor|unei)\s+(verific|analiz)", re.IGNORECASE),
]

LEAKED_SECTION_PREFIX = re.compile(
    r"DE\s+CE\s+(ACESTE\s+)?(ȘTIR|STIR|NARAȚIUN|NARATIUN)\w*\s+(SUNT\s+)?(ESTE\s+)?FALS",
)

SECTION_PREFIXES = {
    "stire":       ["ȘTIRE:", "STIRE:", "ȘTIREA:", "STIREA:"],
    "naratiuni":   ["NARAȚIUNI:", "NARATIUNI:", "NARAȚIUNEA:", "NARATIUNEA:"],
    "obiective":   ["OBIECTIVE:", "OBIECTIVUL:", "SCOP:", "SCOPUL:"],
    "de_ce_false": [
        "DE CE SUNT FALSE", "DE CE ESTE FALSĂ", "DE CE ESTE FALS",
        "DE CE ACESTE ȘTIRI SUNT FALSE", "DE CE ACESTE STIRI SUNT FALSE",
        "DE CE ESTE FALSĂ NARAȚIUNEA", "DE CE ESTE FALSĂ ȘTIREA",
        "DE CE ESTE FALSA NARATIUNEA", "DE CE ESTE FALSA STIREA",
        "DE CE NARAȚIUNEA ESTE FALSĂ", "DE CE ȘTIREA ESTE FALSĂ",
        "DE CE ESTE FALS CĂ",
    ],
    "context":     ["CONTEXT:", "CONTEXTUL:"],
}

SUMMARY_URL_PATTERN = re.compile(
    r"top[- ]?(propaganda|fake[- ]?news|dezinform|stiri[- ]?false|naratiun)",
    re.IGNORECASE,
)

# Regex pentru extragerea anului din URL (ex. /2022/03/... sau -2022-)
_YEAR_IN_URL_RE = re.compile(r"/(20\d{2})/|-(20\d{2})-")

NER_SEED_ENTITIES = [
    "Rusia", "Ucraina", "Putin", "Zelenski", "NATO", "UE", "Kremlin",
    "Donbas", "Crimeea", "Mariupol", "Moscova", "Kiev", "Kyiv",
    "Lugansk", "Donețk", "SUA", "România", "Occident", "Wagner",
]


# ── Functii helper ─────────────────────────────────────────────────────────────

def get_soup(url: str) -> BeautifulSoup | None:
    """Descarca pagina si returneaza soup-ul, sau None la eroare."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [EROARE] {url}: {e}")
        return None


def normalize_text(raw: str) -> str:
    """NFC + colapsare whitespace + unificare ghilimele/liniute."""
    text = unicodedata.normalize("NFC", raw)
    text = re.sub(r"\s+", " ", text)
    text = (
        text.replace("\u201e", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u2013", "-").replace("\u2014", "-")
    )
    return text.strip()


clean_text = normalize_text


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def count_sentences(text: str) -> int:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZĂÎȘȚÂŞŢ])", text)
    return max(1, len(sentences))


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_ukraine_related(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in UKRAINE_PATTERNS)


def has_factcheck_leak(text: str) -> bool:
    if not text:
        return False
    if LEAKED_SECTION_PREFIX.search(text):
        return True
    return any(p.search(text) for p in FACTCHECK_LEAK_MARKERS)


def is_summary_article(url: str) -> bool:
    return bool(SUMMARY_URL_PATTERN.search(url))


def extract_keywords_simple(text: str) -> list:
    text_lower = text.lower()
    return [e for e in NER_SEED_ENTITIES if e.lower() in text_lower]


def year_from_url(url: str) -> int | None:
    """Extrage anul din URL, daca e prezent explicit."""
    m = _YEAR_IN_URL_RE.search(url)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def load_existing_urls(csv_path: str) -> set:
    """Incarca URL-urile deja colectate pentru a le sari fara request HTTP."""
    path = Path(csv_path)
    if not path.exists():
        print(f"  [INFO] {csv_path} nu există — niciun URL de sărit.")
        return set()
    try:
        df   = pd.read_csv(path, usecols=["url"])
        urls = set(df["url"].dropna().tolist())
        print(f"  [INFO] {len(urls)} URL-uri existente încărcate din {csv_path}")
        return urls
    except Exception as e:
        print(f"  [WARN] Nu am putut citi {csv_path}: {e}")
        return set()


# ── Parsare structurata articol ───────────────────────────────────────────────

def parse_veridica_sections(content_tag) -> dict:
    """Extrage sectiunile STIRE/NARATIUNI/OBIECTIVE/DE CE FALSE/CONTEXT."""
    result = {"stire": "", "naratiuni": "", "obiective": "",
              "de_ce_false": "", "context": ""}
    if not content_tag:
        return result

    current_section = None
    current_text    = []

    def flush(sec):
        if sec and current_text:
            result[sec] = normalize_text(" ".join(current_text))

    for p in content_tag.find_all("p"):
        raw = p.get_text(separator=" ", strip=True)
        if not raw:
            continue
        norm       = normalize_text(raw)
        norm_upper = norm.upper()

        matched_section = None
        matched_prefix  = None
        for sec_key, prefixes in SECTION_PREFIXES.items():
            for prefix in prefixes:
                if norm_upper[:len(prefix) + 5].startswith(prefix):
                    matched_section = sec_key
                    matched_prefix  = prefix
                    break
            if matched_section:
                break

        if matched_section:
            flush(current_section)
            current_section = matched_section
            current_text    = []
            after = norm[len(matched_prefix):].strip().lstrip(":").strip()
            if after:
                current_text.append(after)
        elif current_section:
            current_text.append(norm)

    flush(current_section)
    return result


def assess_quality(stire: str, naratiuni: str) -> str:
    """Evalueaza calitatea extragerii (identic cu v4.2)."""
    if stire:
        if has_factcheck_leak(stire):
            return "suspect_contaminare"
        wc = word_count(stire)
        if wc < MIN_STIRE_WORDS or wc > MAX_STIRE_WORDS:
            return "suspect_dimensiune"
        return "excelenta"
    if naratiuni:
        if has_factcheck_leak(naratiuni):
            return "suspect_contaminare"
        return "buna"
    return "fallback_verificare_manuala"


def scrape_article(url: str) -> dict | None:
    """Descarca si parseaza un articol Veridica."""
    soup = get_soup(url)
    if soup is None:
        return None
    try:
        h1    = soup.find("h1", class_="responsiveTitle") or soup.find("h1")
        title = clean_text(h1.get_text(strip=True) if h1 else "N/A")

        date_str = "N/A"
        time_tag = soup.find("time")
        if time_tag:
            date_str = time_tag.get("datetime", time_tag.get_text(strip=True))
        else:
            meta = soup.find("meta", property="article:published_time")
            if meta:
                date_str = meta.get("content", "N/A")

        author     = "N/A"
        author_tag = soup.find(class_=re.compile(r"author|byline", re.I))
        if author_tag:
            author = author_tag.get_text(strip=True)

        content_tag = soup.find("div", class_="page-content")
        if not content_tag:
            content_tag = (
                soup.find("div", class_=re.compile(r"entry-content|post-content", re.I))
                or soup.find("main")
            )
        if content_tag:
            for tag in content_tag.find_all(
                ["script", "style", "nav", "aside", "figure", "form", "iframe"]
            ):
                tag.decompose()

        referinta = (
            clean_text(content_tag.get_text(separator=" ", strip=True))
            if content_tag else "N/A"
        )

        sections   = parse_veridica_sections(content_tag)
        stire      = sections["stire"]
        naratiuni  = sections["naratiuni"]
        obiective  = sections["obiective"]
        de_ce_fals = sections["de_ce_false"]
        context    = sections["context"]

        calitate = assess_quality(stire, naratiuni)

        if calitate == "excelenta":
            text_curat = clean_text(f"{title} {stire}")
        elif calitate == "buna":
            text_curat = clean_text(f"{title} {naratiuni}")
        elif calitate == "suspect_contaminare":
            text_curat = ""
        elif calitate == "suspect_dimensiune":
            text_curat = clean_text(f"{title} {stire}")
        else:
            text_curat = clean_text(f"{title} {referinta[:400]}")

        text_rel  = " ".join(filter(None, [title, stire, naratiuni])) or referinta
        relevanta = is_ukraine_related(text_rel)

        return {
            "url":                      url,
            "titlu":                    title,
            "data":                     date_str,
            "autor":                    author,
            "sursa_site":               "veridica.ro",
            "sectiune":                 "fake_news_dezinformare_propaganda",
            "text_curat":               text_curat,
            "calitate_extractie":       calitate,
            "nr_cuvinte_stire":         word_count(stire),
            "stire_citata":             stire,
            "naratiuni_false":          naratiuni,
            "obiective_propaganda":     obiective,
            "analiza_factcheck":        de_ce_fals,
            "context_sursa":            context,
            "_referinta_continut_full": referinta,
            "nr_propozitii":            count_sentences(text_curat) if text_curat else 0,
            "cuvinte_cheie":            json.dumps(
                                            extract_keywords_simple(text_rel),
                                            ensure_ascii=False
                                        ),
            "label":                    LABEL,
            "label_numeric":            LABEL_NUMERIC,
            "relevanta_ucraina":        relevanta,
            "hash_continut":            compute_hash(text_curat or referinta),
        }
    except Exception as e:
        print(f"  [EROARE parsare] {url}: {e}")
        return None


# ── Listing pagini 20–38 ───────────────────────────────────────────────────────

def collect_links_pages_20_38(skip_urls: set) -> list:
    """
    Parcurge DOAR paginile START_PAGE–END_PAGE din sectiunea principala.
    Oprire anticipata daca toate URL-urile de pe o pagina sunt din 2021
    sau mai vechi (pre-invazie, irelevante pentru proiect).
    """
    section_url = BASE_URL + SECTION
    all_links   = []

    for page_num in range(START_PAGE, END_PAGE + 1):
        url  = f"{section_url}?page={page_num}"
        print(f"  [LISTING pagina {page_num}] {url}")
        soup = get_soup(url)

        if soup is None:
            print(f"  [STOP] Eroare la pagina {page_num}.")
            break

        cards = soup.find_all("div", class_="card border-0 h-100")
        if not cards:
            cards = soup.find_all("div", class_=re.compile(r"card"))
        if not cards:
            print(f"  [STOP] Niciun card pe pagina {page_num} — probabil ultima pagină.")
            break

        page_links = []
        for card in cards:
            a_tag = card.find("a", href=True)
            if a_tag:
                href     = a_tag["href"]
                full_url = urljoin(BASE_URL, href) if not href.startswith("http") else href
                if BASE_URL in full_url and full_url not in all_links:
                    page_links.append(full_url)

        if not page_links:
            print(f"  [STOP] Niciun link pe pagina {page_num}.")
            break

        # Oprire anticipata: daca toti anii detectati sunt < STOP_BEFORE_YEAR
        years = [year_from_url(u) for u in page_links]
        years_known = [y for y in years if y is not None]
        # if years_known and all(y < STOP_BEFORE_YEAR for y in years_known):
            #print(
             #   f"  [STOP EARLY] Pagina {page_num} contine exclusiv "
              #  f"articole din {min(years_known)} — inainte de invazie. Oprire."
            #)
            #break

        # Raportam cate sunt noi vs. deja existente
        noi      = [u for u in page_links if u not in skip_urls]
        existente = len(page_links) - len(noi)
        print(f"    -> {len(page_links)} link-uri | {len(noi)} noi | {existente} deja existente")

        all_links.extend(page_links)
        time.sleep(DELAY_SECONDS)

    return list(dict.fromkeys(all_links))


# ── Salvare ───────────────────────────────────────────────────────────────────

COL_ORDER = [
    "url", "titlu", "data", "autor", "sursa_site", "sectiune",
    "text_curat", "calitate_extractie", "nr_cuvinte_stire",
    "stire_citata", "naratiuni_false", "obiective_propaganda",
    "analiza_factcheck", "context_sursa", "_referinta_continut_full",
    "nr_propozitii", "cuvinte_cheie",
    "label", "label_numeric", "relevanta_ucraina", "hash_continut",
]


def save_csv(articles: list) -> None:
    if not articles:
        return
    df          = pd.DataFrame(articles)
    df          = df[[c for c in COL_ORDER if c in df.columns]]
    df_ukraine  = df[df["relevanta_ucraina"]]
    df.to_csv(OUTPUT_FILE,   index=False, encoding="utf-8-sig")
    df_ukraine.to_csv(FILTERED_FILE, index=False, encoding="utf-8-sig")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("VERIDICA 2022 SCRAPER — Start")
    print(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pagini     : {START_PAGE}–{END_PAGE}")
    print(f"Secțiune   : {SECTION}")
    print(f"Existing   : {EXISTING_CSV}")
    print("=" * 60)

    # Incarca URL-urile deja colectate — sarite fara request HTTP
    seen_urls   = load_existing_urls(EXISTING_CSV)
    seen_hashes : set = set()
    all_articles      = []
    processed         = 0

    # Listing pagini 20–38
    print(f"\n[1/2] Colectare link-uri din paginile {START_PAGE}–{END_PAGE}...")
    all_links = collect_links_pages_20_38(skip_urls=seen_urls)
    noi       = [u for u in all_links if u not in seen_urls]

    print(f"\n  Total link-uri găsite pe paginile {START_PAGE}–{END_PAGE}: {len(all_links)}")
    print(f"  Deja în dataset (sărite): {len(all_links) - len(noi)}")
    print(f"  De scrape-uit acum      : {len(noi)}")

    if not noi:
        print("\n[INFO] Toate articolele din paginile 20–38 sunt deja colectate.")
        return

    # Scraping articole noi
    print(f"\n[2/2] Scraping {len(noi)} articole noi...")
    try:
        for i, url in enumerate(noi, 1):
            if is_summary_article(url):
                print(f"  [{i}/{len(noi)}] [SKIP rezumat] {url}")
                continue

            print(f"  [{i}/{len(noi)}] {url}")
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
                print(f"    [{rel}][{q}] {wc} cuv. | {article['titlu'][:55]}...")

                if processed % CHECKPOINT_EVERY == 0:
                    save_csv(all_articles)
                    print(f"    [CHECKPOINT] {processed} articole salvate")

            time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C — salvez ce am...")

    # Salvare finala
    if not all_articles:
        print("\n[ATENȚIE] Niciun articol nou colectat.")
        return

    save_csv(all_articles)

    # Statistici finale
    df      = pd.DataFrame(all_articles)
    total   = len(df)
    ukraine = int(df["relevanta_ucraina"].sum())

    print(f"\n{'=' * 60}")
    print(f"REZULTAT FINAL")
    print(f"{'=' * 60}")
    print(f"  Articole scrape-uite  : {total}")
    print(f"  Relevante Ucraina     : {ukraine} ({ukraine/total*100:.1f}%)")
    print(f"\n  Distribuție calitate:")
    for q, n in df["calitate_extractie"].value_counts().items():
        marker = "✓" if q in ("excelenta", "buna") else "⚠"
        print(f"    {marker} {q:30s}: {n:4d} ({n/total*100:5.1f}%)")

    # Distributie pe an — verificare ca am colectat doar 2022
    df["_an"] = pd.to_datetime(df["data"], errors="coerce").dt.year
    print(f"\n  Distribuție pe an (verificare):")
    for an, n in df["_an"].value_counts().sort_index().items():
        if pd.notna(an):
            flag = "✓" if int(an) == 2022 else "⚠ VERIFICĂ"
            print(f"    {int(an)}: {n} articole  {flag}")

    print(f"\n  Salvat: {OUTPUT_FILE}  ({total} articole)")
    print(f"  Salvat: {FILTERED_FILE}  ({ukraine} articole relevante)")
    print(f"\nPASUL URMĂTOR:")
    print(f"  Concatenează {EXISTING_CSV} + {FILTERED_FILE}")
    print(f"  și rulează clean_veridica_v5.py pe dataset-ul combinat.")


if __name__ == "__main__":
    main()
