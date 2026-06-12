"""
Curatare test set extern cls0 — eliminare cookie banners, comentarii si meta-elemente.

Context:
--------
Diagnosticul benchmark v4 a expus ca propozitiile „min" pe cls0 erau dominate
de cookie banners (HotNews.ro, Pro TV, Libertatea), nu de continut articol.
Asta a produs AUC 0.9774 artefactual pe Test A min.

Corpusul cls0 de referinta (Digi24 + G4Media) fusese curatat de cookie banners
in pipeline-ul original (curatare_cookies_cls0_v3). Dar test set-ul extern
(scraped de pe HotNews, Libertatea, Pro TV) NU a trecut prin aceeasi curatare,
creand asimetrie intre train si test.

Pattern-uri identificate (scanare exhaustiva pe subset_benchmark_v3):
  1. Cookie banners PURE (propozitie = doar banner):
     - HotNews: „HotNews.ro utilizeaza cookie-uri...", variante scurte/lungi (~25)
     - Pro TV: „Continuarea navigarii implica acceptarea..." (45 aparitii!)
     - Libertatea: „Logheaza-te in contul tau pentru a adauga comentarii..." (3)
  2. Cookie banners CONCATENATE (continut real + banner):
     - Ex: „Dmitri Peskov: ...pozitia fata de Ucraina HotNews.ro utilizeaza cookie-uri"
     - Solutie: taiere chirurgicala la inceputul pattern-ului cookie
  3. Comentarii Libertatea:
     - Pattern: „username     DD.MM.YYYY, HH:MM text" (6 cazuri)
  4. LIVETEXT lead-ins:
     - Ex: „Urmareste ultimele evolutii din a X-a zi..." (~5)

Strategie:
  - Curatare in 3 faze: taiere sufix cookie > aruncare banner pur > aruncare comentarii/LIVETEXT
  - Dupa curatare, re-aplicam filtru lungime [7, 54] cuvinte (consistent cu corpus)
  - Output: subset_benchmark_v3_curat.parquet cu aceeasi schema + stat raport

Output:
  - data/processed/subset_benchmark_v3_curat.parquet
  - findings/curatare_test_extern_cookies.md + .json

Utilizare:
  python scripts/curatare_test_extern_cookies.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configurare
# ---------------------------------------------------------------------------

SUBSET_IN = Path("data/processed/subset_benchmark_v3.parquet")
SUBSET_OUT = Path("data/processed/subset_benchmark_v3_curat.parquet")
RAPORT_MD = Path("findings/curatare_test_extern_cookies.md")
RAPORT_JSON = Path("findings/curatare_test_extern_cookies.json")

# Filtru lungime (consistent cu corpusurile cls0 si cls1)
MIN_CUVINTE = 7
MAX_CUVINTE = 54


# ---------------------------------------------------------------------------
# Pattern-uri cookie banners — taiere sufix (pentru cazurile concatenate)
# ---------------------------------------------------------------------------
# Ordinea conteaza: incercam match pe cele mai specifice intai.
# Strategia: gasim inceputul cookie-ului si taiem propozitia la acel punct.

PATTERNS_SUFIX_COOKIE = [
    # HotNews — cel mai frecvent
    re.compile(r"\s*HotNews\.ro utilizează cookie-uri.*$", re.IGNORECASE),
    # HotNews varianta: „Accesati Modifica Setarile..." singur
    re.compile(r"\s*Accesați Modifică Setările pentru preferințe.*$", re.IGNORECASE),
    # Pro TV / Stirile Pro TV
    re.compile(r"\s*Continuarea navigării implică acceptarea.*$", re.IGNORECASE),
    # Libertatea
    re.compile(r"\s*Loghează-te în contul tău pentru a adăuga comentarii.*$", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Pattern-uri pentru aruncare completa (propozitie = pur banner/meta)
# ---------------------------------------------------------------------------

PATTERNS_ARUNCARE = [
    # Cookie banners pure (match complet sau aproape complet)
    re.compile(r"^\s*HotNews\.ro utilizează cookie-uri", re.IGNORECASE),
    re.compile(r"^\s*Accesați Modifică Setările", re.IGNORECASE),
    re.compile(r"^\s*Continuarea navigării implică acceptarea", re.IGNORECASE),
    re.compile(r"^\s*Loghează-te în contul tău pentru a adăuga comentarii", re.IGNORECASE),

    # LIVETEXT lead-ins (HotNews pattern)
    re.compile(r"^\s*Urmărește ultimele evoluții din a \d+[-ae]?\s*[aă]?\s*zi", re.IGNORECASE),
    re.compile(r"^\s*Urmăriți cele mai recente evoluții ale războiului", re.IGNORECASE),
    re.compile(r"^\s*Urmărește pe Libertatea LIVETEXT", re.IGNORECASE),
    re.compile(r"^\s*Evenimentele de .+ ziua \d+ a războiului.*au fost LIVE", re.IGNORECASE),
    re.compile(r"^\s*Informațiile de .+ ziua \d+.*au fost LIVE", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Pattern pentru comentarii Libertatea (timestamp format)
# ---------------------------------------------------------------------------
# Format: „username    DD.MM.YYYY, HH:MM text" sau „Acest comentariu a fost moderat"

PATTERN_COMENTARIU = re.compile(
    r"(\d{2}\.\d{2}\.\d{4},\s*\d{2}:\d{2})|"
    r"(^Acest comentariu a fost moderat)",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Functii de curatare
# ---------------------------------------------------------------------------

def taie_sufix_cookie(text: str) -> tuple[str, bool]:
    """
    Incearca sa taie sufixul cookie banner dintr-o propozitie concatenata.

    Returneaza (text_curat, a_taiat).
    Daca nu gaseste niciun pattern, returneaza textul original si False.
    """
    for pattern in PATTERNS_SUFIX_COOKIE:
        match = pattern.search(text)
        if match:
            # Taiem de la inceputul match-ului
            taiat = text[:match.start()].strip()
            # Verificam ca ce-a ramas nu e prea scurt sau gol
            if len(taiat) > 0:
                return taiat, True
            # Daca nu ramane nimic, text-ul era integral cookie — returnam gol
            return "", True
    return text, False


def e_pentru_aruncare(text: str) -> bool:
    """Returneaza True daca propozitia pura match-uieste un pattern de aruncare."""
    for pattern in PATTERNS_ARUNCARE:
        if pattern.match(text):
            return True
    return False


def e_comentariu(text: str) -> bool:
    """Returneaza True daca propozitia pare comentariu Libertatea."""
    return bool(PATTERN_COMENTARIU.search(text))


# ---------------------------------------------------------------------------
# Pipeline de curatare
# ---------------------------------------------------------------------------

def curata_test_set(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aplica pipeline-ul de curatare pe test set.

    Faze:
      1. Taiere sufix cookie pentru propozitii concatenate (pastram continut real)
      2. Recalculare nr_cuvinte + nr_caractere dupa taiere
      3. Aruncare propozitii care match pattern aruncare (cookie pur, LIVETEXT)
      4. Aruncare comentarii Libertatea
      5. Re-aplicare filtru lungime [7, 54] cuvinte (consistenta cu corpus)

    Returneaza (df_curat, raport_stats).
    """
    stats: dict = {
        "total_inainte": len(df),
        "sufix_taiat": 0,
        "aruncate_banner_pur": 0,
        "aruncate_comentariu": 0,
        "aruncate_sub_7w": 0,
        "aruncate_peste_54w": 0,
        "exemple_sufix_taiat": [],
        "exemple_banner_pur": [],
        "exemple_comentariu": [],
    }

    df = df.copy()
    df["_propozitie_orig"] = df["propozitie"].copy()

    # --- FAZA 1: taiere sufix cookie pe propozitii concatenate ---
    taieri = df["propozitie"].astype(str).apply(taie_sufix_cookie)
    df["propozitie"] = taieri.apply(lambda x: x[0])
    df["_sufix_taiat"] = taieri.apply(lambda x: x[1])

    tăiate = df[df["_sufix_taiat"]]
    stats["sufix_taiat"] = len(tăiate)
    stats["exemple_sufix_taiat"] = [
        {
            "articol_id": str(r["articol_id"]),
            "sursa": str(r["sursa_site"]),
            "orig": str(r["_propozitie_orig"])[:300],
            "curat": str(r["propozitie"])[:300],
        }
        for _, r in tăiate.head(10).iterrows()
    ]

    # --- Recalculare nr_cuvinte / nr_caractere dupa taiere ---
    df["nr_cuvinte"] = df["propozitie"].astype(str).apply(lambda t: len(t.split()))
    df["nr_caractere"] = df["propozitie"].astype(str).apply(len)

    # --- FAZA 2: aruncare propozitii banner pur ---
    mask_banner = df["propozitie"].astype(str).apply(e_pentru_aruncare)
    stats["aruncate_banner_pur"] = int(mask_banner.sum())
    stats["exemple_banner_pur"] = [
        {
            "articol_id": str(r["articol_id"]),
            "sursa": str(r["sursa_site"]),
            "propozitie": str(r["propozitie"])[:300],
        }
        for _, r in df[mask_banner].head(10).iterrows()
    ]
    df = df[~mask_banner].copy()

    # --- FAZA 3: aruncare comentarii Libertatea ---
    mask_comentariu = df["propozitie"].astype(str).apply(e_comentariu)
    stats["aruncate_comentariu"] = int(mask_comentariu.sum())
    stats["exemple_comentariu"] = [
        {
            "articol_id": str(r["articol_id"]),
            "sursa": str(r["sursa_site"]),
            "propozitie": str(r["propozitie"])[:300],
        }
        for _, r in df[mask_comentariu].head(10).iterrows()
    ]
    df = df[~mask_comentariu].copy()

    # --- FAZA 4: re-aplicare filtru lungime ---
    # Propozitiile care au fost taiate pot sa fi scazut sub 7w acum
    mask_prea_scurte = df["nr_cuvinte"] < MIN_CUVINTE
    mask_prea_lungi = df["nr_cuvinte"] > MAX_CUVINTE
    stats["aruncate_sub_7w"] = int(mask_prea_scurte.sum())
    stats["aruncate_peste_54w"] = int(mask_prea_lungi.sum())
    df = df[~(mask_prea_scurte | mask_prea_lungi)].copy()

    stats["total_dupa"] = len(df)

    # Curatam coloanele helper
    df = df.drop(columns=["_propozitie_orig", "_sufix_taiat"]).reset_index(drop=True)

    return df, stats


# ---------------------------------------------------------------------------
# Raportare
# ---------------------------------------------------------------------------

def scrie_raport(df_inainte: pd.DataFrame, df_dupa: pd.DataFrame, stats: dict) -> None:
    """Scrie rapoarte JSON + Markdown cu statisticile curatarii."""
    # Articole afectate: cate au pierdut propozitii, cate au devenit goale
    art_inainte = df_inainte.groupby("articol_id").size()
    art_dupa = df_dupa.groupby("articol_id").size()

    # Articole care au pierdut toate propozitiile (drastic)
    ids_disparute = set(art_inainte.index) - set(art_dupa.index)

    # Articole care au pierdut cel putin 50% din propozitii
    art_merged = pd.concat([art_inainte, art_dupa], axis=1, keys=["inainte", "dupa"])
    art_merged = art_merged.fillna(0).astype(int)
    art_merged["delta"] = art_merged["dupa"] - art_merged["inainte"]
    art_merged["pct_pastrate"] = art_merged["dupa"] / art_merged["inainte"] * 100
    art_pierduri_mari = art_merged[art_merged["pct_pastrate"] < 50]

    # Breakdown per sursa
    breakdown_sursa = {}
    for sursa in df_inainte["sursa_site"].unique():
        nr_inainte = (df_inainte["sursa_site"] == sursa).sum()
        nr_dupa = (df_dupa["sursa_site"] == sursa).sum()
        breakdown_sursa[sursa] = {
            "prop_inainte": int(nr_inainte),
            "prop_dupa": int(nr_dupa),
            "delta": int(nr_dupa - nr_inainte),
            "pct_pastrat": round(nr_dupa / nr_inainte * 100, 2) if nr_inainte else 0,
        }

    stats["articole_disparute_complet"] = len(ids_disparute)
    stats["exemple_articole_disparute"] = sorted(list(ids_disparute))[:10]
    stats["articole_cu_pierderi_mari_pct50"] = int(len(art_pierduri_mari))
    stats["breakdown_sursa"] = breakdown_sursa

    # JSON
    RAPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RAPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nJSON scris: {RAPORT_JSON}")

    # Markdown
    linii: list[str] = []
    linii.append("# Curățare test set extern cls0 — cookie banners și meta-elemente")
    linii.append("")
    linii.append(
        "Diagnostic benchmark v4 a expus că AUC 0.98 pe Test A min era "
        "artefact: propozițiile 'min' pe cls0 erau cookie banners HotNews/"
        "Pro TV, nu conținut articol. Acest script elimină poluarea pentru "
        "re-rulare benchmark v4."
    )
    linii.append("")

    linii.append("## Rezumat numeric")
    linii.append("")
    linii.append(f"- Propoziții înainte: **{stats['total_inainte']:,}**")
    linii.append(f"- Sufix cookie tăiat (concatenate): **{stats['sufix_taiat']}**")
    linii.append(f"- Banner pur aruncat: **{stats['aruncate_banner_pur']}**")
    linii.append(f"- Comentarii aruncate: **{stats['aruncate_comentariu']}**")
    linii.append(f"- Sub 7 cuvinte (post-curățare): **{stats['aruncate_sub_7w']}**")
    linii.append(f"- Peste 54 cuvinte: **{stats['aruncate_peste_54w']}**")
    linii.append(f"- Propoziții după: **{stats['total_dupa']:,}** "
                 f"(retenție {stats['total_dupa']/stats['total_inainte']*100:.1f}%)")
    linii.append("")

    linii.append("## Breakdown per sursă")
    linii.append("")
    linii.append("| Sursă | Înainte | După | Δ | % păstrat |")
    linii.append("|---|---:|---:|---:|---:|")
    for sursa, s in sorted(breakdown_sursa.items()):
        linii.append(
            f"| {sursa} | {s['prop_inainte']} | {s['prop_dupa']} | "
            f"{s['delta']:+d} | {s['pct_pastrat']}% |"
        )
    linii.append("")

    linii.append("## Articole afectate")
    linii.append("")
    linii.append(
        f"- Articole dispărute complet: **{stats['articole_disparute_complet']}** "
        f"(toate propozițiile au fost eliminate — suspect)"
    )
    linii.append(
        f"- Articole cu pierderi mari (>50% propoziții): "
        f"**{stats['articole_cu_pierderi_mari_pct50']}**"
    )
    if stats["exemple_articole_disparute"]:
        linii.append("")
        linii.append("Articole dispărute:")
        for aid in stats["exemple_articole_disparute"]:
            linii.append(f"- `{aid}`")
    linii.append("")

    # Exemple taieri sufix
    if stats["exemple_sufix_taiat"]:
        linii.append("## Exemple: tăiere sufix cookie (concatenate)")
        linii.append("")
        for i, ex in enumerate(stats["exemple_sufix_taiat"], 1):
            linii.append(f"**{i}.** `{ex['articol_id']}` · {ex['sursa']}")
            linii.append("")
            linii.append(f"- Original: `{ex['orig']}`")
            linii.append(f"- Curățat: `{ex['curat']}`")
            linii.append("")

    # Exemple aruncate banner pur
    if stats["exemple_banner_pur"]:
        linii.append("## Exemple: banner pur aruncat")
        linii.append("")
        for i, ex in enumerate(stats["exemple_banner_pur"], 1):
            linii.append(
                f"**{i}.** `{ex['articol_id']}` · {ex['sursa']}: "
                f"*{ex['propozitie']}*"
            )
        linii.append("")

    # Exemple comentarii
    if stats["exemple_comentariu"]:
        linii.append("## Exemple: comentarii aruncate")
        linii.append("")
        for i, ex in enumerate(stats["exemple_comentariu"], 1):
            linii.append(
                f"**{i}.** `{ex['articol_id']}` · {ex['sursa']}: "
                f"*{ex['propozitie']}*"
            )
        linii.append("")

    linii.append("## Următorul pas")
    linii.append("")
    linii.append(
        "Re-rulare `benchmark_v4.py` folosind `subset_benchmark_v3_curat.parquet` "
        "în loc de `subset_benchmark_v3.parquet`. Embeddings-urile corpusurilor "
        "(cls0 + cls1) rămân valide — cache hit instant. Doar embeddings-urile "
        "test set se recalculează (<5 sec pe MPS)."
    )
    linii.append("")
    linii.append("---")
    linii.append("")
    linii.append("*Modul 3 · Pasul A2.6 · Curățare test set extern pre-re-rulare*")

    RAPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with RAPORT_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(linii))
    print(f"MD scris: {RAPORT_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestreaza incarcare → curatare → raport → salvare."""
    print("=" * 70)
    print("CURĂȚARE TEST SET EXTERN cls0")
    print("=" * 70)

    if not SUBSET_IN.exists():
        raise FileNotFoundError(f"Test set nu găsit: {SUBSET_IN}")

    df_inainte = pd.read_parquet(SUBSET_IN)
    print(f"\nTest set încărcat: {len(df_inainte):,} propoziții")
    print(f"Articole: {df_inainte['articol_id'].nunique()}")
    print(f"Surse: {df_inainte['sursa_site'].value_counts().to_dict()}")

    # Aplicam curatarea doar pe cls0 (cls1 nu are aceste pattern-uri —
    # scraping Veridica si Stopfals a fost diferit). Dar procesam tot
    # ca sa fim siguri ca nu scapam ceva pe cls1 si pentru simetrie.
    print("\nRulez pipeline de curățare...")
    df_dupa, stats = curata_test_set(df_inainte)

    # Raportare consola
    print(f"\n{'='*70}")
    print("REZULTATE CURĂȚARE")
    print(f"{'='*70}")
    print(f"  Total înainte:       {stats['total_inainte']:,}")
    print(f"  Sufix cookie tăiat:  {stats['sufix_taiat']}")
    print(f"  Banner pur aruncat:  {stats['aruncate_banner_pur']}")
    print(f"  Comentarii aruncate: {stats['aruncate_comentariu']}")
    print(f"  Sub 7w post-curățare: {stats['aruncate_sub_7w']}")
    print(f"  Peste 54w:           {stats['aruncate_peste_54w']}")
    print(f"  Total după:          {stats['total_dupa']:,}")
    print(f"  Retenție:            "
          f"{stats['total_dupa']/stats['total_inainte']*100:.1f}%")

    # Salvare
    SUBSET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_dupa.to_parquet(SUBSET_OUT, index=False)
    print(f"\nSubset curat salvat: {SUBSET_OUT}")

    # Raport
    scrie_raport(df_inainte, df_dupa, stats)

    # Breakdown articole cls0
    print(f"\nArticole cls0 — breakdown propoziții:")
    cls0_dupa = df_dupa[df_dupa["label_numeric"] == 0]
    per_art = cls0_dupa.groupby("articol_id").size()
    print(f"  min: {per_art.min()}")
    print(f"  mediana: {per_art.median():.1f}")
    print(f"  max: {per_art.max()}")
    print(f"  articole cls0 rămase: {len(per_art)} (din 55 inițiale)")

    print(f"\n{'='*70}")
    print("Rulează acum: python scripts/benchmark_v4.py")
    print("(modifică TEST_SET_PATH în benchmark_v4.py la subset_benchmark_v3_curat.parquet)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
