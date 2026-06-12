"""
Test de generalizare cross-source pentru sistemul de detectie dezinformare.

Modelul a fost antrenat pe: Digi24, G4Media, Veridica, Stopfals.
Acest test foloseste surse NOI (biziday, libertatea, hotnews) pentru CLS0
si stopfals.md pentru CLS1 (flagged — overlap partial cu training).

Pentru fiecare articol: extragere → POST /api/predict → comparatie verdict.
La final: tabel + accuracy globala/per sursa + analiza FP/FN.
"""

import re
import sys
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ─── Configurare ────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/api/predict"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
}
TIMEOUT = 15
N_PER_SURSA = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cross_source")


# ─── Modele de date ─────────────────────────────────────────────────────────
@dataclass
class Articol:
    sursa: str
    url: str
    titlu: str
    text: str
    verdict_asteptat: str  # 'stire_credibila' / 'dezinformare_pro_rusa'


@dataclass
class Rezultat:
    art: Articol
    verdict_primit: Optional[str] = None
    incredere: Optional[float] = None
    scor_modul3_diff: Optional[float] = None
    scor_baseline: Optional[float] = None
    timp_ms: Optional[int] = None
    eroare: Optional[str] = None


# ─── Helper generic fetch ───────────────────────────────────────────────────
def _get(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            log.warning("GET %s → %s", url, r.status_code)
            return None
        return r.text
    except Exception as e:
        log.warning("GET %s a eșuat: %s", url, e)
        return None


def _curat(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


# ─── Scraperi per sursa ─────────────────────────────────────────────────────

def scrape_biziday(n: int) -> list[Articol]:
    """biziday.ro — sectiunea cu tag razboi-ucraina."""
    listing = "https://www.biziday.ro/tag/razboi-ucraina/"
    html = _get(listing)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    EXCLUDE = ("/tag/", "/category/", "/contact", "/privacy", "/about", "/wp-")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("https://www.biziday.ro/"):
            continue
        if any(e in href for e in EXCLUDE):
            continue
        # Slug articol: cuvinte legate cu liniute; ignoram sluguri pur-numerice
        slug = href.rstrip("/").rsplit("/", 1)[-1]
        if not re.search(r"[a-z]{4}", slug):
            continue
        if not any(k in slug.lower() for k in ("ucraina", "rusia", "putin", "zelenski", "kiev", "moscova", "nato")):
            continue
        if href not in links:
            links.append(href)
    log.info("biziday: %d linkuri candidate", len(links))
    out = []
    for url in links[: n * 3]:
        if len(out) >= n:
            break
        h = _get(url)
        if not h:
            continue
        s = BeautifulSoup(h, "html.parser")
        titlu_tag = s.find("h1")
        titlu = _curat(titlu_tag.get_text()) if titlu_tag else ""
        body = s.find("article") or s.find("div", class_=re.compile("post|content|entry"))
        if not body:
            continue
        for tag in body.select("script,style,aside,.related,.share,figcaption,.tags"):
            tag.decompose()
        paragrafe = [_curat(p.get_text()) for p in body.find_all("p")]
        paragrafe = [p for p in paragrafe if len(p) > 30]
        text = " ".join(paragrafe[:8])
        if len(text) < 300 or not titlu:
            continue
        out.append(Articol("biziday.ro", url, titlu, f"{titlu}. {text}", "stire_credibila"))
        time.sleep(0.6)
    return out


def scrape_libertatea(n: int) -> list[Articol]:
    """libertatea.ro — sectiunea razboiul Rusia-Ucraina."""
    listing = "https://www.libertatea.ro/tag/razboi-ucraina-rusia"
    html = _get(listing)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    KW = ("ucraina", "rusia", "putin", "zelenski", "kiev", "moscova", "nato", "ucrainei", "rusiei")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("https://www.libertatea.ro/"):
            continue
        if "/tag/" in href or "/stiri/" not in href:
            continue
        slug = href.rstrip("/").rsplit("/", 1)[-1].lower()
        if not any(k in slug for k in KW):
            continue
        if href not in links:
            links.append(href)
    log.info("libertatea: %d linkuri candidate", len(links))
    out = []
    for url in links[: n * 3]:
        if len(out) >= n:
            break
        h = _get(url)
        if not h:
            continue
        s = BeautifulSoup(h, "html.parser")
        titlu_tag = s.find("h1")
        titlu = _curat(titlu_tag.get_text()) if titlu_tag else ""
        body = s.find("article") or s.find("div", class_=re.compile("article|content|body"))
        if not body:
            continue
        for tag in body.select("script,style,aside,figcaption,.related,.share"):
            tag.decompose()
        paragrafe = [_curat(p.get_text()) for p in body.find_all("p")]
        paragrafe = [p for p in paragrafe if len(p) > 40]
        text = " ".join(paragrafe[:8])
        if len(text) < 300 or not titlu:
            continue
        out.append(Articol("libertatea.ro", url, titlu, f"{titlu}. {text}", "stire_credibila"))
        time.sleep(0.6)
    return out


def scrape_hotnews(n: int) -> list[Articol]:
    """hotnews.ro — sectiunea razboi-in-ucraina."""
    listing = "https://hotnews.ro/c/actualitate/razboi-in-ucraina"
    html = _get(listing)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Articole hotnews: au sufix numeric (id) la final
        if href.startswith("https://hotnews.ro/") and re.search(r"-\d{6,}$", href):
            if href not in links:
                links.append(href)
    log.info("hotnews: %d linkuri candidate", len(links))
    out = []
    for url in links[: n * 3]:
        if len(out) >= n:
            break
        h = _get(url)
        if not h:
            continue
        s = BeautifulSoup(h, "html.parser")
        titlu_tag = s.find("h1")
        titlu = _curat(titlu_tag.get_text()) if titlu_tag else ""
        body = s.find("article") or s.find("div", class_=re.compile("article|content|entry|post"))
        if not body:
            continue
        for tag in body.select("script,style,aside,figcaption,.related,.share,.tags"):
            tag.decompose()
        paragrafe = [_curat(p.get_text()) for p in body.find_all("p")]
        paragrafe = [p for p in paragrafe if len(p) > 40]
        text = " ".join(paragrafe[:8])
        if len(text) < 300 or not titlu:
            continue
        out.append(Articol("hotnews.ro", url, titlu, f"{titlu}. {text}", "stire_credibila"))
        time.sleep(0.6)
    return out


def scrape_stopfals(n: int) -> list[Articol]:
    """stopfals.md — extragere citat propagandistic izolat (primul paragraf
    distinct inainte de demontare)."""
    home = _get("https://stopfals.md/")
    if not home:
        return []
    soup = BeautifulSoup(home, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/ro/article/"):
            full = "https://stopfals.md" + href
            if full not in urls:
                urls.append(full)
    log.info("stopfals: %d linkuri candidate", len(urls))
    out = []
    for url in urls[: n * 4]:
        if len(out) >= n:
            break
        h = _get(url)
        if not h:
            continue
        s = BeautifulSoup(h, "html.parser")
        # H1 real: ignoram „Cautare" si „Comments"
        titlu = ""
        for h1 in s.find_all("h1"):
            t = _curat(h1.get_text())
            if t and "Căutare" not in t and "Comments" not in t:
                titlu = t
                break

        # Body articol stopfals: div.wrap--two (validat empiric)
        body = s.find("div", class_=re.compile(r"wrap--two"))
        if not body:
            continue
        # Strategia 1: extragem textul dintre ghilimelele romanesti „...".
        # Acesta e citatul propagandist izolat inainte de demontare.
        text_full = body.get_text(" ", strip=True)
        # Ghilimele romanesti „..." (U+201E ... U+201D)
        citate = re.findall("„([^„”]{60,1200})”", text_full)
        citat = " ".join(_curat(c) for c in citate[:3]) if citate else None
        # Strategia 2 (fallback): primul paragraf > 200 caractere care nu e
        # intro/meta, inainte de „In realitate" / „fals".
        if not citat or len(citat) < 100:
            DEBUNK = ("în realitate", "in realitate", "fals", "verdict",
                      "stopfals", "am contactat", "afirmațiile", "afirmatiile",
                      "potrivit", "verificat", "nu corespunde")
            for p in body.find_all("p"):
                t = _curat(p.get_text())
                if len(t) < 150:
                    continue
                if any(k in t.lower() for k in DEBUNK):
                    continue
                citat = t
                break
        if not citat or len(citat) < 80:
            continue
        out.append(Articol("stopfals.md", url, titlu or "(fără titlu)", citat, "dezinformare_pro_rusa"))
        time.sleep(0.6)
    return out


# ─── Apel API ───────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    t0 = time.time()
    r = requests.post(API_URL, json={"text": text}, timeout=120)
    r.raise_for_status()
    j = r.json()
    j["_t_ms_total"] = int((time.time() - t0) * 1000)
    return j


# ─── Orchestrator ───────────────────────────────────────────────────────────
def ruleaza() -> list[Rezultat]:
    scraperi = [
        ("biziday.ro", lambda: scrape_biziday(N_PER_SURSA)),
        ("libertatea.ro", lambda: scrape_libertatea(N_PER_SURSA)),
        ("hotnews.ro", lambda: scrape_hotnews(N_PER_SURSA)),
        ("stopfals.md", lambda: scrape_stopfals(N_PER_SURSA)),
    ]
    rezultate: list[Rezultat] = []
    surse_skipped: list[str] = []
    for nume, fn in scraperi:
        log.info("=== Sursă: %s ===", nume)
        try:
            articole = fn()
        except Exception as e:
            log.error("Scraper %s a eșuat: %s", nume, e)
            surse_skipped.append(f"{nume} (scraper: {e})")
            continue
        if not articole:
            log.warning("Niciun articol scrapat pentru %s — se sare peste.", nume)
            surse_skipped.append(f"{nume} (zero articole)")
            continue
        log.info("%s: %d articole scrapate", nume, len(articole))
        for art in articole:
            r = Rezultat(art=art)
            try:
                resp = predict(art.text)
                r.verdict_primit = resp.get("decizie")
                r.incredere = resp.get("incredere")
                r.scor_modul3_diff = resp.get("scor_modul3_diff_mean")
                r.scor_baseline = resp.get("scor_baseline_prob_cls1")
                r.timp_ms = resp.get("metadata", {}).get("timp_inferenta_ms")
            except Exception as e:
                r.eroare = str(e)
                log.error("predict() a eșuat pentru %s: %s", art.url, e)
            rezultate.append(r)
            time.sleep(0.2)

    # Salvam rezultatele brute pentru inspectie ulterioara
    payload = {
        "rezultate": [
            {
                "sursa": r.art.sursa, "url": r.art.url, "titlu": r.art.titlu,
                "verdict_asteptat": r.art.verdict_asteptat,
                "verdict_primit": r.verdict_primit, "incredere": r.incredere,
                "scor_modul3_diff": r.scor_modul3_diff,
                "scor_baseline_prob_cls1": r.scor_baseline,
                "timp_ms": r.timp_ms, "eroare": r.eroare,
                "text_input_preview": r.art.text[:300],
            }
            for r in rezultate
        ],
        "surse_skipped": surse_skipped,
    }
    with open("/Users/rares/Documents/Licenta/tests/cross_source/rezultate.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return rezultate, surse_skipped


# ─── Raportare ──────────────────────────────────────────────────────────────
def trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def raport(rezultate: list[Rezultat], surse_skipped: list[str]) -> None:
    print("\n" + "=" * 130)
    print(" REZULTATE TEST CROSS-SOURCE ".center(130, "="))
    print("=" * 130)
    print(f"{'Sursă':<14} {'Titlu':<52} {'Așt.':<6} {'Primit':<6} {'OK':<3} "
          f"{'Încr.':>6} {'M3Δ':>8} {'P(1)':>6} {'ms':>5}")
    print("-" * 130)
    for r in rezultate:
        ast = "CLS1" if r.art.verdict_asteptat == "dezinformare_pro_rusa" else "CLS0"
        if r.eroare:
            primit, ok = "ERR", "?"
            incr = m3 = pb = ms_s = "-"
        else:
            primit = "CLS1" if r.verdict_primit == "dezinformare_pro_rusa" else "CLS0"
            ok = "✓" if r.verdict_primit == r.art.verdict_asteptat else "✗"
            incr = f"{r.incredere:.3f}" if r.incredere is not None else "-"
            m3 = f"{r.scor_modul3_diff:+.4f}" if r.scor_modul3_diff is not None else "-"
            pb = f"{r.scor_baseline:.3f}" if r.scor_baseline is not None else "-"
            ms_s = f"{r.timp_ms}" if r.timp_ms is not None else "-"
        print(f"{r.art.sursa:<14} {trunc(r.art.titlu, 52):<52} {ast:<6} {primit:<6} {ok:<3} "
              f"{incr:>6} {m3:>8} {pb:>6} {ms_s:>5}")

    # ─ Metrici ─
    total = sum(1 for r in rezultate if not r.eroare)
    corecte = sum(1 for r in rezultate if not r.eroare and r.verdict_primit == r.art.verdict_asteptat)
    print("-" * 130)
    print(f"Accuracy GLOBALĂ: {corecte}/{total} = {corecte/total*100:.1f}%" if total else "Niciun rezultat valid.")

    per_sursa: dict[str, list[Rezultat]] = {}
    for r in rezultate:
        per_sursa.setdefault(r.art.sursa, []).append(r)
    print("\nAccuracy per sursă:")
    for sursa, rs in per_sursa.items():
        valid = [r for r in rs if not r.eroare]
        oks = sum(1 for r in valid if r.verdict_primit == r.art.verdict_asteptat)
        print(f"  {sursa:<14} {oks}/{len(valid)}  ({(oks/len(valid)*100 if valid else 0):.1f}%)")

    # ─ FP / FN ─
    fp = [r for r in rezultate if not r.eroare
          and r.art.verdict_asteptat == "stire_credibila"
          and r.verdict_primit == "dezinformare_pro_rusa"]
    fn = [r for r in rezultate if not r.eroare
          and r.art.verdict_asteptat == "dezinformare_pro_rusa"
          and r.verdict_primit == "stire_credibila"]

    def _print_err(rs, eticheta):
        print(f"\n{eticheta} ({len(rs)}):")
        if not rs:
            print("  — niciuna")
            return
        for r in rs:
            print(f"  • [{r.art.sursa}] {trunc(r.art.titlu, 90)}")
            print(f"      P(cls1)={r.scor_baseline:.3f}  Δmodul3={r.scor_modul3_diff:+.4f}  "
                  f"încr={r.incredere:.3f}")
            print(f"      URL: {r.art.url}")

    _print_err(fp, "FALSE POSITIVES (CLS0 → CLS1)")
    _print_err(fn, "FALSE NEGATIVES (CLS1 → CLS0)")

    if surse_skipped:
        print("\nSurse omise:")
        for s in surse_skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    rezultate, skipped = ruleaza()
    raport(rezultate, skipped)
