"""
Scraping automatizat articole cls0 externe pentru benchmark v3.

Scop:
    Colecteaza ~75 articole cls0 din 3 surse (stirileprotv.ro, hotnews.ro,
    libertatea.ro), stratificate temporal pentru a se alinia cu distributia
    corpusului de referinta (28/23/21/23/5% pe 2022-2026).

Flux:
    1. Descoperire URL-uri: pentru fiecare sursa, parcurge pagini de tag
       „ucraina"/„razboi" si colecteaza URL-uri candidate.
    2. Validare an: extrage data publicarii din meta tag OpenGraph
       (`article:published_time`). URL-uri fara data detectabila sunt sarite.
    3. Stratificare: adauga URL la coada anului corespunzator DOAR daca
       cota pe anul respectiv nu e umpluta.
    4. Descarcare si parsare: cod identic cu cel validat manual anterior.
    5. Salvare CSV + raport.

Stratificare tinta (aliniata cu corpus):
    2022: 28% → 21 articole
    2023: 23% → 17 articole
    2024: 21% → 16 articole
    2025: 23% → 17 articole
    2026:  5% →  4 articole
    Total: 75 articole

Rate limiting: 1.5s intre request-uri (politicos, evita 429).

Input:
    - data/raw/test_cls0_external.csv (optional — pentru deduplicare cu
      cele 15 articole deja colectate manual)

Output:
    - data/raw/test_cls0_external_v2.csv (setul consolidat, 75+ articole)
    - findings/scraping_cls0_extern_raport.md

Rulare:
    python scripts/scraping_cls0_extern.py

Nota: scraping-ul e best-effort. Se asteapta esecuri partiale (20-30%).
      Daca nu atingem cota pe un an, raportul noteaza deficitul.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup


# -----------------------------------------------------------------------------
# Configuratie
# -----------------------------------------------------------------------------
CALE_EXISTENT = Path("data/raw/test_cls0_external.csv")
CALE_OUT_CSV = Path("data/raw/test_cls0_external_v2.csv")
CALE_OUT_RAPORT = Path("findings/scraping_cls0_extern_raport.md")

# Cote stratificare temporala (aliniate cu corpus)
COTE_PER_AN = {
    2022: 21,
    2023: 17,
    2024: 16,
    2025: 17,
    2026: 4,
}
TOTAL_TARGET = sum(COTE_PER_AN.values())  # 75

# Rate limiting
DELAY_REQUEST_SEC = 1.5
TIMEOUT_SEC = 12

# Limite parcurgere pagini de listare (evitam runaway)
MAX_PAGINI_LISTARE = 50
MAX_CANDIDATI_PER_SURSA = 500

# Filtrare minima lungime articol (cuvinte)
LUNG_MIN_CUVINTE = 150
LUNG_MAX_CUVINTE = 1500  # prea lung = probabil live-blog aglomerat

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ro-RO,ro;q=0.9"}

# URL-uri de pornire pentru descoperire candidati per sursa.
# NB: libertatea blocheaza cu HTTP 403 pe articole individuale (anti-bot).
# Il lasam in lista ca „best effort" — daca din 200 candidati pica 180,
# tot ce reusim la cele 20 e bonus. Daca esti sigur ca nu merge deloc,
# comenteaza linia.
URL_PAGINI_LISTARE = {
    "stirileprotv.ro": [
        "https://stirileprotv.ro/stiri-despre/ucraina/?page={}",
    ],
    "hotnews.ro": [
        "https://hotnews.ro/c/actualitate/razboi-in-ucraina/page/{}",
    ],
    "libertatea.ro": [
        "https://www.libertatea.ro/subiect/conflict-rusia-ucraina/page/{}",
    ],
}

# STRATEGIE PAGINARE INVERSATA: incepem de la pagini MARI (articole vechi)
# si mergem spre mici (recente). Motivul: la prima rulare cotele pentru
# 2025-2026 sunt usor de umplut din paginile 1-30, dar 2022-2023 sunt greu
# si necesita paginile 100-250. Inversam ca sa prioritizam anii cu deficit.
PAGINI_DE_PARCURS = [
    # mai intai pagini MARI pentru articole din 2022-2023
    250, 220, 200, 180, 160, 140, 120, 100,
    # apoi pagini medii pentru 2024
    90, 80, 70, 60, 50, 40, 30,
    # la final pagini mici pentru 2025-2026 (daca mai sunt cote deschise)
    20, 10, 5, 3, 2, 1,
    # pagini intermediare ca backup
    110, 130, 150, 170, 190, 210, 240,
    7, 15, 25, 35, 45, 55, 65, 75, 85, 95,
    105, 115, 125, 135, 145, 155, 165, 175, 185, 195,
]


# -----------------------------------------------------------------------------
# Utilitare HTTP
# -----------------------------------------------------------------------------
def fetch(url: str) -> tuple[str | None, int]:
    """GET cu headers standard, timeout si rate limiting politicos.

    Returneaza tuple (html_or_none, http_status).
    status=0 inseamna exceptie network.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SEC)
        time.sleep(DELAY_REQUEST_SEC)
        if resp.status_code == 200:
            return resp.text, 200
        print(f"  ⚠️  HTTP {resp.status_code}: {url}")
        return None, resp.status_code
    except Exception as e:
        print(f"  ⚠️  Eroare fetch {url}: {e}")
        return None, 0


# -----------------------------------------------------------------------------
# Descoperire URL-uri candidate
# -----------------------------------------------------------------------------
def descopera_urluri(sursa: str) -> list[str]:
    """Parcurge paginile de listare si colecteaza URL-uri candidate.

    Parcurge PAGINI_DE_PARCURS (amestec de pagini mici si mari) pentru a
    acoperi intreg spectrul temporal al site-ului. Detecteaza cand site-ul
    returneaza continut duplicat (signal ca paginarea nu functioneaza pentru
    acel numar) si sare la urmatoarea.
    """
    print(f"\n[Descoperire] {sursa}")
    colectate = set()
    pagini_duplicate_consecutive = 0

    for sablon in URL_PAGINI_LISTARE[sursa]:
        for nr_pagina in PAGINI_DE_PARCURS:
            url_listare = sablon.format(nr_pagina)
            html, _status = fetch(url_listare)
            if html is None:
                continue  # pagina nu exista sau eroare — trec la urmatoarea

            soup = BeautifulSoup(html, "html.parser")
            linkuri_pagina = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = urljoin(url_listare, href)
                if not href.startswith("http"):
                    continue
                if sursa not in href:
                    continue
                if is_url_articol(href, sursa):
                    linkuri_pagina.add(href.split("?")[0].rstrip("/"))

            if not linkuri_pagina:
                print(f"  pagina {nr_pagina}: 0 link-uri — sar peste")
                continue

            noi = linkuri_pagina - colectate
            colectate.update(linkuri_pagina)
            pct_noi = len(noi) / len(linkuri_pagina) * 100 if linkuri_pagina else 0
            print(f"  pagina {nr_pagina}: +{len(noi)} noi "
                  f"({pct_noi:.0f}% unic, total {len(colectate)})")

            # detecteaza paginare stricata: 3 pagini consecutive cu 0% noi
            if len(noi) == 0:
                pagini_duplicate_consecutive += 1
                if pagini_duplicate_consecutive >= 3:
                    print(f"  ⚠️  3 pagini consecutive fără link-uri noi — "
                          f"sar la următorul sablon/sursă")
                    break
            else:
                pagini_duplicate_consecutive = 0

            if len(colectate) >= MAX_CANDIDATI_PER_SURSA:
                print(f"  limită {MAX_CANDIDATI_PER_SURSA} atinsă")
                break

        if len(colectate) >= MAX_CANDIDATI_PER_SURSA:
            break
    return list(colectate)


def is_url_articol(href: str, sursa: str) -> bool:
    """Euristici per sursa pentru a distinge URL-uri de articole de altele."""
    href_l = href.lower()
    # excluderi generale
    if any(x in href_l for x in [
        "/tag/", "/etichete/", "/autor/", "/author/",
        "/categorie/", "/category/", "/pagina-", "/page/",
        "/stiri-despre/", "/subiect/", "/c/",  # pagini de listare
        ".jpg", ".png", ".pdf", "/video/", "/galerie/",
        "/abonament", "/contact", "/despre", "#"
    ]):
        return False

    if sursa == "stirileprotv.ro":
        # articolele au /stiri/SECTIUNE/SLUG.html (NU /stiri-despre/)
        return "/stiri/" in href_l and href_l.endswith(".html")
    if sursa == "hotnews.ro":
        # articolele au un ID numeric la final: -NNNNN
        # format: https://hotnews.ro/slug-articol-XXXXXX
        return bool(re.search(r"-\d{4,}/?$", href_l))
    if sursa == "libertatea.ro":
        # articolele au /stiri/...-NNNNNN
        return "/stiri/" in href_l and bool(re.search(r"-\d{6,}/?$", href_l))
    return False


# -----------------------------------------------------------------------------
# Extragere an fiabila din meta tag
# -----------------------------------------------------------------------------
def extrage_an(soup: BeautifulSoup, html_raw: str) -> int | None:
    """Extrage anul publicarii cu fallback multiplu.

    Ordine:
    1. meta property="article:published_time" (OpenGraph, cel mai fiabil)
    2. meta name="pubdate" / "publishdate" / "date"
    3. <time datetime="...">
    4. JSON-LD schema.org Article.datePublished
    5. Text romanesc: `15 martie 2022` / `15 mar. 2022` in primele 3000 chars
    6. URL-ul articolului contine data: /2022/03/15/ sau /2022-03-15/
    """
    # 1. OpenGraph
    meta = soup.find("meta", attrs={"property": "article:published_time"})
    if meta and meta.get("content"):
        an = _parse_an_din_iso(meta["content"])
        if an:
            return an

    # 2. alte meta tags
    for name_val in ["pubdate", "publishdate", "date", "DC.date.issued",
                     "article:published", "sailthru.date"]:
        meta = soup.find("meta", attrs={"name": name_val})
        if meta and meta.get("content"):
            an = _parse_an_din_iso(meta["content"])
            if an:
                return an

    # 3. <time datetime="...">
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        an = _parse_an_din_iso(time_tag["datetime"])
        if an:
            return an

    # 4. JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        txt = script.string or ""
        match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', txt)
        if match:
            an = _parse_an_din_iso(match.group(1))
            if an:
                return an

    # 5. Text romanesc din primele 3000 chars (deasupra articolului, unde
    #    de obicei e data de publicare)
    text_preview = soup.get_text()[:3000].lower()
    luni_ro = {
        "ianuarie": 1, "februarie": 2, "martie": 3, "aprilie": 4, "mai": 5,
        "iunie": 6, "iulie": 7, "august": 8, "septembrie": 9, "octombrie": 10,
        "noiembrie": 11, "decembrie": 12,
        "ian": 1, "feb": 2, "mar": 3, "apr": 4, "iun": 6, "iul": 7,
        "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    pattern_data_ro = re.compile(
        r"\b(\d{1,2})\s+("
        + "|".join(luni_ro.keys())
        + r")\.?\s+(\d{4})"
    )
    match = pattern_data_ro.search(text_preview)
    if match:
        an = int(match.group(3))
        if 2022 <= an <= 2026:
            return an

    # 6. URL-ul articolului contine data
    # (verific si canonical URL daca e setat)
    canonical = soup.find("link", attrs={"rel": "canonical"})
    url_check = canonical["href"] if canonical and canonical.get("href") else ""
    match = re.search(r"/(20\d{2})[-/](\d{1,2})[-/]", url_check)
    if match:
        an = int(match.group(1))
        if 2022 <= an <= 2026:
            return an

    # fara an detectabil
    return None


def _parse_an_din_iso(s: str) -> int | None:
    """Parse an din string ISO-like (2022-03-15T..., 2022/03/15, 15.03.2022)."""
    # pattern 1: YYYY-MM-DD sau YYYY/MM/DD
    m = re.match(r"(\d{4})[-/]\d{2}[-/]\d{2}", s)
    if m:
        an = int(m.group(1))
        if 2022 <= an <= 2026:
            return an
    # pattern 2: DD.MM.YYYY
    m = re.match(r"\d{2}\.\d{2}\.(\d{4})", s)
    if m:
        an = int(m.group(1))
        if 2022 <= an <= 2026:
            return an
    return None


# -----------------------------------------------------------------------------
# Descarcare si parsare articol (cod adaptat din scriptul existent)
# -----------------------------------------------------------------------------
def parseaza_articol(url: str, idx: int) -> tuple[dict | None, int]:
    """Descarca si parseaza un articol.

    Returneaza (dict_articol_sau_None, http_status).
    status=200 inseamna succes HTTP (dar poate esua la parsare/validare).
    status=403 inseamna blocat de server.
    status=0 inseamna exceptie network.
    """
    html, status = fetch(url)
    if html is None:
        return None, status

    soup = BeautifulSoup(html, "html.parser")

    # titlu
    titlu_tag = soup.find("h1")
    if titlu_tag:
        titlu = titlu_tag.text.strip()
    elif soup.title:
        titlu = soup.title.text.strip()
    else:
        return None, 200  # HTTP OK, dar parsare esuata

    # sursa
    if "stirileprotv.ro" in url:
        sursa = "stirileprotv.ro"
    elif "hotnews.ro" in url:
        sursa = "hotnews.ro"
    elif "libertatea.ro" in url:
        sursa = "libertatea.ro"
    else:
        return None, 200

    # an fiabil
    an = extrage_an(soup, html)
    if an is None:
        return None, 200

    # corp articol — pastram doar <p>-urile suficient de lungi
    paragrafe = soup.find_all("p")
    text_bucati = [p.text.strip() for p in paragrafe if len(p.text.strip()) > 60]
    text_complet = " ".join(text_bucati)

    # curatare pentru CSV
    text_complet = text_complet.replace("\n", " ").replace("\r", "").replace('"', "„")
    titlu = titlu.replace('"', "„").replace("\n", " ").replace("\r", "")

    # validare lungime
    nw = len(text_complet.split())
    if nw < LUNG_MIN_CUVINTE or nw > LUNG_MAX_CUVINTE:
        return None, 200

    return {
        "id": f"ext_{idx:03d}",
        "url": url,
        "titlu": titlu,
        "data": f"{an}-01-01",  # data exacta nu e necesara pentru similaritate
        "an": an,
        "sursa_site": sursa,
        "stire_citata": text_complet,
        "label_numeric": 0,
    }, 200


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    """Pipeline scraping complet cu stratificare temporala."""
    print("=" * 70)
    print("SCRAPING cls0 EXTERN — v2 cu stratificare temporală")
    print(f"Target: {TOTAL_TARGET} articole, distribuție aliniată cu corpus")
    print("=" * 70)

    # 1. incarc articolele existente pentru deduplicare
    urls_existente = set()
    articole_existente = []
    if CALE_EXISTENT.exists():
        df_ex = pd.read_csv(CALE_EXISTENT)
        urls_existente = set(df_ex["url"].values)
        articole_existente = df_ex.to_dict("records")
        print(f"\nArticole existente (din colectarea manuală): {len(df_ex)}")
        # ajustez cotele cu cele deja existente pe fiecare an
        for an, n in df_ex["an"].value_counts().items():
            if an in COTE_PER_AN:
                COTE_PER_AN[an] = max(0, COTE_PER_AN[an] - n)
        print("Cote rămase după scăderea existentelor:")
        for an, cota in COTE_PER_AN.items():
            print(f"  {an}: {cota}")

    # 2. descoperire URL-uri per sursa
    toate_urlurile = {}
    for sursa in URL_PAGINI_LISTARE:
        toate_urlurile[sursa] = descopera_urluri(sursa)
        print(f"[{sursa}] Total candidați: {len(toate_urlurile[sursa])}")

    # 3. parseaza si aplica stratificare
    umplere = dict(COTE_PER_AN)  # copie — decrementam pe masura ce umplem
    rezultate_noi = []
    esec = 0
    sarite_an_plin = 0
    sarite_an_nedetectat = 0

    # contor 403-uri consecutive per sursa — daca una e total blocata,
    # o excludem din restul rulajului ca sa nu pierdem timp
    blocate_403_consecutive = {s: 0 for s in URL_PAGINI_LISTARE}
    surse_blocate = set()
    PRAG_403_BLOCARE = 10

    # intercalam sursele pentru diversitate — luam round-robin
    max_per_sursa = max(len(v) for v in toate_urlurile.values()) if toate_urlurile else 0
    iterator_urluri = []
    for i in range(max_per_sursa):
        for sursa, urls in toate_urlurile.items():
            if i < len(urls):
                iterator_urluri.append((sursa, urls[i]))

    idx_articol = len(articole_existente)
    print(f"\n[Parsare + stratificare] ~{len(iterator_urluri)} candidați de procesat")

    for sursa_url, url in iterator_urluri:
        # toate cotele umplute? oprire anticipata
        if all(c <= 0 for c in umplere.values()):
            print("\n✓ Toate cotele umplute — ne oprim.")
            break

        # sursa e blocata complet? sar
        if sursa_url in surse_blocate:
            continue

        if url in urls_existente:
            continue

        idx_articol += 1
        rez, status = parseaza_articol(url, idx_articol)

        # detectare blocare sursa: 403 consecutive pe aceeasi sursa
        if status == 403:
            blocate_403_consecutive[sursa_url] += 1
            if blocate_403_consecutive[sursa_url] >= PRAG_403_BLOCARE:
                surse_blocate.add(sursa_url)
                print(f"\n🚫 Sursa '{sursa_url}' blocată complet "
                      f"({PRAG_403_BLOCARE} x HTTP 403). Nu mai încerc articolele ei.\n")
        else:
            blocate_403_consecutive[sursa_url] = 0

        if rez is None:
            esec += 1
            idx_articol -= 1
            continue

        # verificare cota pe anul articolului
        an = rez["an"]
        if an not in umplere or umplere[an] <= 0:
            sarite_an_plin += 1
            idx_articol -= 1
            continue

        rezultate_noi.append(rez)
        urls_existente.add(url)
        umplere[an] -= 1
        total_acum = len(articole_existente) + len(rezultate_noi)
        status_cota = ", ".join(
            f"{an_c}:{c}" for an_c, c in umplere.items() if c > 0
        )
        print(f"  ✅ [{total_acum}/{TOTAL_TARGET}] [{rez['sursa_site']}] "
              f"{an}: {rez['titlu'][:50]}... (cote rămase: {status_cota})")

    # 4. consolidare + salvare
    articole_finale = articole_existente + rezultate_noi
    df_final = pd.DataFrame(articole_finale)

    # resetez id-urile ext_NNN pentru consecutivitate curata
    df_final = df_final.sort_values(["an", "sursa_site"]).reset_index(drop=True)
    df_final["id"] = [f"ext_{i+1:03d}" for i in range(len(df_final))]

    CALE_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(CALE_OUT_CSV, index=False, encoding="utf-8")

    # 5. raport final
    print("\n" + "=" * 70)
    print("SUMAR SCRAPING")
    print("=" * 70)
    print(f"Articole totale în CSV final: {len(df_final)}")
    print(f"Articole noi adăugate: {len(rezultate_noi)}")
    print(f"Articole preluate din colectare manuală: {len(articole_existente)}")
    print(f"Eșecuri (parse/fetch/lungime): {esec}")
    print(f"Sărite (cota anului plină): {sarite_an_plin}")
    print(f"Sărite (an nedetectabil): {sarite_an_nedetectat}")
    print(f"\nDistribuție finală per an:")
    for an, n in df_final["an"].value_counts().sort_index().items():
        target = COTE_PER_AN.get(an, 0) + sum(
            1 for r in articole_existente if r.get("an") == an
        )
        status = "✓" if n >= target else "⚠️ deficit"
        print(f"  {an}: {n} {status}")
    print(f"\nDistribuție per sursă:")
    for sursa, n in df_final["sursa_site"].value_counts().items():
        print(f"  {sursa}: {n}")
    print(f"\n✅ Salvat: {CALE_OUT_CSV}")

    # raport markdown
    CALE_OUT_RAPORT.parent.mkdir(parents=True, exist_ok=True)
    linii = [
        "# Raport scraping cls0 extern v2",
        "",
        f"**Data rulării:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total articole în CSV final:** {len(df_final)}",
        f"  - Din colectarea manuală (v1): {len(articole_existente)}",
        f"  - Nou scraped automat: {len(rezultate_noi)}",
        "",
        "## Distribuție per an (vs target)",
        "",
        "| An | Target (corpus) | Realizat | Status |",
        "|---|---|---|---|",
    ]
    for an in sorted([2022, 2023, 2024, 2025, 2026]):
        target_initial = {2022: 21, 2023: 17, 2024: 16, 2025: 17, 2026: 4}[an]
        realizat = int((df_final["an"] == an).sum())
        status = "✓" if realizat >= target_initial else "⚠️ deficit"
        linii.append(f"| {an} | {target_initial} | {realizat} | {status} |")
    linii += [
        "",
        "## Distribuție per sursă",
        "",
        "| Sursă | Nr. articole |",
        "|---|---|",
    ]
    for sursa, n in df_final["sursa_site"].value_counts().items():
        linii.append(f"| {sursa} | {n} |")

    linii += [
        "",
        "## Statistici parsare",
        "",
        f"- Eșecuri (fetch/parse/lungime invalidă): **{esec}**",
        f"- Sărite (cota anului deja plină): **{sarite_an_plin}**",
        f"- Sărite (an nedetectabil din meta): **{sarite_an_nedetectat}**",
        "",
        "## Lungime articole (cuvinte)",
        "",
    ]
    nw = df_final["stire_citata"].fillna("").str.split().str.len()
    linii.append(f"- Min: {nw.min()}")
    linii.append(f"- Mediană: {nw.median():.0f}")
    linii.append(f"- Max: {nw.max()}")
    linii.append(f"- Medie: {nw.mean():.0f}")
    linii.append("")
    linii.append("*Generat automat.*")
    CALE_OUT_RAPORT.write_text("\n".join(linii), encoding="utf-8")
    print(f"✅ Raport: {CALE_OUT_RAPORT}")


if __name__ == "__main__":
    main()