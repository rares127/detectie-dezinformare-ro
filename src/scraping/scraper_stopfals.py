"""
Scraper pentru stopfals.md — colectare articole fact-checking pro-Kremlin
relevante pentru conflictul din Ucraina (2022–2024).

Strategie: scanare secventiala pe ID-uri numerice (180300–181100),
extragere din HTML static (SSR), filtrare pe cuvinte cheie,
output CSV compatibil cu schema proiectului (clasa 1 = dezinformare_pro_rusa).

Autor: licenta 2025-2026
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import hashlib
import logging
from pathlib import Path
from datetime import datetime

# ─── Configurare logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper_stopfals.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── Constante ──────────────────────────────────────────────────────────────

# Intervalul de ID-uri de scanat.
# ID-uri confirmate empiric:
#   ~180300 = articole din prima jumatate 2022
#   ~181100 = articole din 2024
# Adaugam marje pentru siguranta.
ID_START = 180560  # confirmat empiric: 180580=dec2021, deci 2022 incepe imediat dupa
ID_END   = 181200  # acopera pana in 2024 (181448=apr2026, dar 2025+ nu ne intereseaza)

# Delay intre request-uri (secunde) — politicos fata de server
DELAY_BETWEEN_REQUESTS = 1.5  # 1.5s → ~400 req/10min, rezonabil

# Delay suplimentar la erori 429/503
DELAY_ON_ERROR = 30

# Cuvinte cheie pentru filtrare articole relevante (Ucraina / pro-Kremlin)
# Cel putin unul trebuie sa apara in titlu SAU in primele 500 de caractere din text
KEYWORDS_UCRAINA = [
    "ucraina", "ucraine", "ucrainean", "ucraineni",
    "rusia", "rusă", "ruși", "kremlin", "kremlinului",
    "război", "razboiul", "invazia", "invaziei",
    "nato", "zelenski", "putin", "donbas", "mariupol",
    "odesa", "harkov", "kiev", "bucea",
    "propagandă rusă", "propaganda rusă", "pro-kremlin",
    "dezinformare", "dezinformarea",
    "operațiune specială", "operatiune speciala",
    "frontul", "ofensivă rusă",
]

# Categorii acceptate (badge-ul articolului) — excludem escrocherii, COVID pur, etc.
CATEGORII_ACCEPTATE = [
    "fact-checking", "gândește critic", "securitate",
    "profil de propagandist", "dezbateri",
]

# Limita de caractere text minim pentru a considera articolul valid
MIN_TEXT_LEN = 300

# Headers HTTP pentru a simula un browser normal
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Director output
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "stopfals_raw_v1.csv"

# ─── Functii utilitare ───────────────────────────────────────────────────────

def normalizeaza_data(data_str: str) -> str:
    """
    Converteste data din formatul romanesc afisat pe site
    (ex: '30 SEPTEMBRIE 2023') in formatul ISO 'YYYY-MM-DD'.
    Returneaza string gol daca nu poate parsa.
    """
    luni_ro = {
        "IANUARIE": "01", "FEBRUARIE": "02", "MARTIE": "03",
        "APRILIE": "04", "MAI": "05", "IUNIE": "06",
        "IULIE": "07", "AUGUST": "08", "SEPTEMBRIE": "09",
        "OCTOMBRIE": "10", "NOIEMBRIE": "11", "DECEMBRIE": "12",
    }
    data_str = data_str.strip().upper()
    parts = data_str.split()
    if len(parts) == 3:
        zi, luna, an = parts
        luna_nr = luni_ro.get(luna, "00")
        try:
            return f"{an}-{luna_nr}-{int(zi):02d}"
        except ValueError:
            pass
    return ""


def contine_keyword(text: str, titlu: str) -> bool:
    """
    Verifica daca articolul este relevant pentru Ucraina/Kremlin.
    Cauta in titlu (prioritate) si in primele 600 de caractere din text.
    """
    zona_verificare = (titlu + " " + text[:600]).lower()
    return any(kw.lower() in zona_verificare for kw in KEYWORDS_UCRAINA)


def categorie_acceptata(badge_text: str) -> bool:
    """
    Verifica daca categoria articolului e relevanta.
    Daca badge-ul e gol, acceptam oricum (unele articole nu au badge).
    """
    if not badge_text:
        return True
    badge_lower = badge_text.lower()
    return any(cat in badge_lower for cat in CATEGORII_ACCEPTATE)


def hash_text(text: str) -> str:
    """Calculeaza hash SHA-256 pentru deduplicare."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ─── Extragere date dintr-o pagina de articol ───────────────────────────────

def extrage_articol(soup: BeautifulSoup, url: str, id_numeric: int) -> dict | None:
    """
    Extrage campurile relevante din HTML-ul unui articol stopfals.md.

    Structura confirmata empiric (recon 18 apr 2026):
      - Titlu:    .wrap--two .title
      - Data:     .wrap--two .date  (primul element)
      - Views:    .wrap--two .views
      - Categorie: .wrap--two .badge
      - Text:     .content--article  (innerText complet)
      - Citat pk: <em> si <i> cu lungime > 50 chars din .content--article

    Returneaza dict cu campurile sau None daca articolul nu e valid/relevant.
    """
    try:
        wrap = soup.select_one(".wrap--two")
        if not wrap:
            return None

        # Titlu
        titlu_el = wrap.select_one(".title")
        titlu = titlu_el.get_text(strip=True) if titlu_el else ""
        if not titlu:
            return None

        # Data — luam primul .date din wrap--two (nu din sidebar)
        data_els = wrap.select(".date")
        data_str = data_els[0].get_text(strip=True) if data_els else ""
        data_iso = normalizeaza_data(data_str)

        # An si luna din data ISO
        an = int(data_iso[:4]) if data_iso else 0
        luna = int(data_iso[5:7]) if data_iso else 0

        # Filtrare temporala: dorim 2022–2024
        # Daca data nu poate fi parsata, lasam sa treaca (verificare manuala)
        if an and an not in (2022, 2023, 2024):
            return None

        # Categorie / badge
        badge_els = wrap.select(".badge")
        badge = ", ".join(b.get_text(strip=True) for b in badge_els)

        if not categorie_acceptata(badge):
            log.debug(f"ID {id_numeric}: categorie exclusă → {badge}")
            return None

        # Textul articolului
        continut_el = wrap.select_one(".content--article")
        if not continut_el:
            return None
        text_curat = continut_el.get_text(separator=" ", strip=True)
        text_curat = re.sub(r"\s+", " ", text_curat).strip()

        if len(text_curat) < MIN_TEXT_LEN:
            return None

        # Filtrare pe cuvinte cheie Ucraina/Kremlin
        if not contine_keyword(text_curat, titlu):
            log.debug(f"ID {id_numeric}: fără keyword Ucraina → '{titlu[:60]}'")
            return None

        # Citate pro-Kremlin: toate <em> si <i> cu lungime > 50 chars
        # Acestea contin de obicei textul sursa pe care il demonteaza articolul
        citate = []
        for tag in continut_el.find_all(["em", "i"]):
            t = tag.get_text(strip=True)
            if len(t) > 50:
                citate.append(t)
        stire_citata = " | ".join(citate) if citate else ""

        # Hash pentru deduplicare
        hash_val = hash_text(text_curat)

        return {
            "id": f"stopfals_{id_numeric}",
            "url": url,
            "titlu": titlu,
            "data": data_iso,
            "an": an,
            "luna": luna,
            "sursa": "stopfals",
            "badge": badge,
            "text_curat": text_curat,
            "stire_citata": stire_citata,
            "nr_cuvinte": len(text_curat.split()),
            "hash_continut": hash_val,
            "label": "dezinformare_pro_rusa",
            "label_numeric": 1,
        }

    except Exception as e:
        log.warning(f"ID {id_numeric}: eroare extragere → {e}")
        return None


# ─── Loop principal de scraping ─────────────────────────────────────────────

def scrape_stopfals(
    id_start: int = ID_START,
    id_end: int = ID_END,
    delay: float = DELAY_BETWEEN_REQUESTS,
) -> pd.DataFrame:
    """
    Scrapeaza articolele stopfals.md iterand prin ID-uri numerice.

    Logica:
    1. Construieste URL-ul din ID si slug generic (slug-ul nu conteaza —
       serverul ruteaza dupa ID-ul numeric din coada URL-ului).
    2. Face GET, verifica status code.
    3. Parseaza HTML cu BeautifulSoup.
    4. Extrage si filtreaza articolul.
    5. Salveaza progresiv in CSV (checkpoint dupa fiecare 50 articole valide).

    Returneaza DataFrame cu toate articolele colectate.
    """
    sesiune = requests.Session()
    sesiune.headers.update(HEADERS)

    articole = []
    erori_consecutive = 0
    MAX_ERORI_CONSECUTIVE = 10

    total_scanate = 0
    total_valide = 0
    total_filtrate_keyword = 0
    total_filtrate_data = 0
    total_404 = 0

    log.info(f"Start scraping stopfals.md: ID {id_start} → {id_end}")
    log.info(f"Total ID-uri de scanat: {id_end - id_start + 1}")

    for id_num in range(id_start, id_end + 1):
        # URL cu slug generic — serverul ignora slug-ul, ruteaza dupa ID
        url = f"https://stopfals.md/ro/article/article-{id_num}"

        try:
            resp = sesiune.get(url, timeout=15, allow_redirects=True)
            total_scanate += 1

            # 404 = ID inexistent, continuam
            if resp.status_code == 404:
                total_404 += 1
                erori_consecutive = 0
                log.debug(f"ID {id_num}: 404 (inexistent)")
                time.sleep(delay * 0.5)  # delay mai mic pentru 404
                continue

            # Rate limiting — asteptam mai mult
            if resp.status_code in (429, 503):
                log.warning(f"ID {id_num}: status {resp.status_code} — aștept {DELAY_ON_ERROR}s")
                time.sleep(DELAY_ON_ERROR)
                erori_consecutive += 1
                if erori_consecutive >= MAX_ERORI_CONSECUTIVE:
                    log.error("Prea multe erori consecutive — opresc scraping-ul.")
                    break
                continue

            if resp.status_code != 200:
                log.warning(f"ID {id_num}: status neașteptat {resp.status_code}")
                erori_consecutive += 1
                time.sleep(delay * 2)
                continue

            # Redirect spre URL real cu slug corect — e normal
            url_final = resp.url

            soup = BeautifulSoup(resp.text, "html.parser")

            # Verifica daca e pagina de articol (nu homepage sau 404 mascat)
            if not soup.select_one(".wrap--two .title"):
                log.debug(f"ID {id_num}: nu e pagina de articol valida")
                erori_consecutive = 0
                time.sleep(delay * 0.5)
                continue

            articol = extrage_articol(soup, url_final, id_num)

            if articol:
                articole.append(articol)
                total_valide += 1
                erori_consecutive = 0
                log.info(
                    f"✓ ID {id_num} | {articol['data']} | "
                    f"{articol['nr_cuvinte']} cuvinte | '{articol['titlu'][:60]}'"
                )

                # Checkpoint progresiv: salveaza la fiecare 50 articole valide
                if total_valide % 50 == 0:
                    df_temp = pd.DataFrame(articole)
                    df_temp.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
                    log.info(f"  → Checkpoint salvat: {total_valide} articole în {OUTPUT_FILE}")
            else:
                erori_consecutive = 0

        except requests.exceptions.Timeout:
            log.warning(f"ID {id_num}: timeout")
            erori_consecutive += 1
            time.sleep(delay * 3)
            continue

        except requests.exceptions.ConnectionError as e:
            log.warning(f"ID {id_num}: connection error → {e}")
            erori_consecutive += 1
            time.sleep(delay * 5)
            continue

        except Exception as e:
            log.error(f"ID {id_num}: eroare neașteptată → {e}")
            erori_consecutive += 1
            continue

        time.sleep(delay)

    # ─── Finalizare ────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info(f"SCRAPING FINALIZAT")
    log.info(f"  Total ID-uri scanate:    {total_scanate}")
    log.info(f"  Total 404:               {total_404}")
    log.info(f"  Articole valide găsite:  {total_valide}")
    log.info("─" * 60)

    if not articole:
        log.warning("Nu s-au găsit articole. Verifică intervalul de ID-uri.")
        return pd.DataFrame()

    df = pd.DataFrame(articole)

    # Deduplicare pe hash_continut
    inainte = len(df)
    df = df.drop_duplicates(subset=["hash_continut"])
    dupa = len(df)
    if inainte != dupa:
        log.info(f"Deduplicate: {inainte - dupa} duplicate eliminate")

    # Sortare cronologica
    df = df.sort_values("data").reset_index(drop=True)

    # Salvare finala
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    log.info(f"CSV final salvat: {OUTPUT_FILE} ({len(df)} articole)")

    # Statistici per an
    log.info("\nDistribuție per an:")
    for an, cnt in df["an"].value_counts().sort_index().items():
        log.info(f"  {an}: {cnt} articole")

    return df


# ─── Functie auxiliara: test rapid pe un subset mic ─────────────────────────

def test_rapid(n_sample: int = 20) -> None:
    """
    Testeaza scraper-ul pe un subset mic de ID-uri (180840–180860)
    pentru validare inainte de rulare completa.
    Articolul cu ID 180844 e cel pe care l-am analizat in recon.
    """
    log.info(f"=== TEST RAPID pe {n_sample} ID-uri ===")
    df = scrape_stopfals(id_start=180840, id_end=180840 + n_sample - 1, delay=1.0)
    if not df.empty:
        log.info(f"\nArticole găsite în test:")
        for _, row in df.iterrows():
            log.info(f"  [{row['data']}] {row['titlu'][:70]}")
    else:
        log.info("Niciun articol relevant găsit în intervalul de test.")


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Mod test: ruleaza pe 20 ID-uri ca sa verifici ca totul functioneaza
        test_rapid(n_sample=20)
    else:
        # Rulare completa
        df = scrape_stopfals()
        print(f"\nRezultat final: {len(df)} articole salvate în {OUTPUT_FILE}")