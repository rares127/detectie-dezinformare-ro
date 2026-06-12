"""
Merge G4Media v2 complet — concatenare 3 surse pentru clasa 0.

Context: dupa sesiunea cu suplimentul `razboi-rusia` care a livrat doar 8
articole in fereastra dec 2025 → apr 2026, am descoperit ca acea fereastra
e structural saraca pe G4Media (volum editorial redus dupa nov 2025). Solutia:
recuperam CSV-ul v1 care fusese drop-uit metodologic — dar care, ironic,
acopera exact fereastra feb-apr 2026 care lipseste acum (636 articole prin
discovery `/articole/` paginare cronologica).

Decizia metodologica:
- v1 a fost drop-uit din DOUA motive: bias temporal (toate in 2 luni) si
  asimetrie de lungime (median 328 vs Veridica 196).
- Bias temporal: NU mai e problema acum — il FOLOSIM exact pentru ca e
  concentrat in fereastra care ne lipseste, ca felie complementara.
- Asimetrie lungime: are deja solutie validata (truncate la 250 cuvinte la
  cleaning v2).

Cele trei surse combinate:

| Sursa            | Acoperire                  | Volum | Discovery               | Marker found_on_page |
|------------------|----------------------------|-------|-------------------------|----------------------|
| v2 principal     | 25 feb 2022 → 19 nov 2025  | 2081  | tag /razboi-ucraina/    | 1..103 (real)        |
| v2 supliment     | 21 nov 2025 → 25 feb 2026  | 8     | tag /razboi-rusia/      | 1..16 (real)         |
| v1 recuperat     | 4 feb 2026 → 9 apr 2026    | 636   | paginare /articole/     | -1 (sentinel)        |

Marker `found_on_page = -1` permite la validarea adversariala finala sa
testam daca modelul invata shortcut-ul "vine din v1" — daca da, cleaning
suplimentar.

ID-uri:
- v2 principal: g4m_v2_xxxxx (existent)
- v2 supliment: g4m_v2s_xxxxx (existent)
- v1 recuperat: g4m_v1r_xxxxx (NOU — `v1r` = v1 recovered)

Schema finala: schema v2 (cu audit_thematic_pass + found_on_page).
v1 e normalizat: `audit_thematic_pass` derivat din matched_core OR matched_hybrid
(coloane deja prezente in v1), `found_on_page = -1`, drop text_curat_raw +
nr_cuvinte_raw (artefacte din truncate v1 vechi).

Output:
- g4media_v2_complet_raw.csv — toate articolele combinate, deduplicate pe URL
- raport detaliat in consola (distributie pe ani+luni, audit pass rate per sursa,
  overlap intre surse, statistici lungime)

Usage:
    python merge_g4media_v2.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Path-uri ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = PROJECT_ROOT / "data" / "raw"

V2_PRINCIPAL_CSV = WORK_DIR / "g4media_v2_raw.csv"
V2_SUPLIMENT_CSV = WORK_DIR / "g4media_v2_supliment_raw.csv"
V1_RECOVERED_CSV = PROJECT_ROOT / "data" / "processed" / "g4media_clean_v1.csv"

OUTPUT_CSV = WORK_DIR / "g4media_v2_complet_raw.csv"

# Schema finala — IDENTICA cu v2 principal
SCHEMA_V2 = [
    "id", "url", "titlu", "data", "sursa_site", "sectiune",
    "text_curat", "nr_cuvinte", "tags", "autor",
    "matched_core", "matched_hybrid",
    "audit_thematic_pass",
    "found_on_page",
    "calitate_extractie", "label", "label_numeric", "hash_continut",
]


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZARE v1 → schema v2
# ══════════════════════════════════════════════════════════════════════════════

def normalize_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizeaza schema CSV-ului v1 la schema v2.

    Modificari:
    1. Re-prefixeaza id-urile cu `g4m_v1r_` (recovered) ca sa fie usor de
       filtrat post-merge si sa nu se ciocneasca cu `g4m_v2_` sau `g4m_v2s_`.
    2. Deriva `audit_thematic_pass` din coloanele existente `matched_core` si
       `matched_hybrid` (logica din thematic_filters: pass daca cel putin
       una e 1). Verificat empiric: 100% pass pentru cele 636 articole v1.
    3. Seteaza `found_on_page = -1` (sentinel pentru "discovery cronologic,
       nu prin tag editorial").
    4. Drop coloanele `text_curat_raw` si `nr_cuvinte_raw` (artefacte din
       truncate v1 — la cleaning v2 vom re-trunchia oricum).
    5. Reordoneaza coloanele exact ca in SCHEMA_V2.
    """
    df = df.copy()

    # 1. Re-prefixeaza id-urile (pastram sufixul numeric original)
    def reid(old_id: str) -> str:
        # extragem partea numerica finala (ultimul grup de cifre)
        # tipic v1: "g4m_00001" → "g4m_v1r_00001"
        parts = str(old_id).split("_")
        suffix = parts[-1] if parts[-1].isdigit() else "00000"
        return f"g4m_v1r_{suffix}"

    df["id"] = df["id"].apply(reid)

    # 2. Deriva audit_thematic_pass
    df["audit_thematic_pass"] = (
        (df["matched_core"].astype(int) == 1) |
        (df["matched_hybrid"].astype(int) == 1)
    ).astype(int)

    # 3. Marker sentinel pentru discovery cronologic
    df["found_on_page"] = -1

    # 4. Drop artefacte v1
    for col in ["text_curat_raw", "nr_cuvinte_raw"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 5. Reordonare schema
    missing = [c for c in SCHEMA_V2 if c not in df.columns]
    if missing:
        raise ValueError(f"Coloane lipsă în v1 după normalizare: {missing}")
    df = df[SCHEMA_V2]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# RAPORT
# ══════════════════════════════════════════════════════════════════════════════

def report_distribution(df: pd.DataFrame, label: str) -> None:
    """Raport rapid asupra unei surse: ani, luni, audit pass, lungime."""
    print(f"\n── {label} ─ {len(df)} articole ──")

    if len(df) == 0:
        return

    # Distributie pe ani
    ani = Counter(d[:4] for d in df["data"].astype(str) if len(d) >= 4)
    print(f"  Ani: {dict(sorted(ani.items()))}")

    # Audit pass rate
    if "audit_thematic_pass" in df.columns:
        pass_rate = df["audit_thematic_pass"].astype(int).mean() * 100
        print(f"  Audit thematic pass: {pass_rate:.1f}%")

    # Lungime
    if "nr_cuvinte" in df.columns:
        nw = df["nr_cuvinte"].astype(int)
        print(f"  Lungime (cuvinte): min={nw.min()}, "
              f"median={int(nw.median())}, "
              f"p75={int(nw.quantile(0.75))}, "
              f"max={nw.max()}")


def report_monthly_coverage(df: pd.DataFrame, start: str, end: str) -> None:
    """Distributie lunara pe o fereastra — pentru a vedea continuitatea acoperirii."""
    print(f"\n── Acoperire lunară fereastră {start} → {end} ──")
    luni = Counter()
    for d in df["data"].astype(str):
        if len(d) >= 7 and start <= d[:7] <= end:
            luni[d[:7]] += 1

    if not luni:
        print("  (gol)")
        return

    # Generam toate lunile intre start si end ca sa vedem gaurile
    from datetime import date
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    all_months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        all_months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    for luna in all_months:
        n = luni.get(luna, 0)
        bar = "█" * min(n, 50)
        marker = " ⚠ GAP" if n == 0 else ""
        print(f"  {luna}: {n:4} {bar}{marker}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MERGE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("MERGE G4MEDIA v2 COMPLET — v2 principal + supliment + v1 recuperat")
    print("=" * 70)

    # ── Incarcare surse ───────────────────────────────────────────────────────
    if not V2_PRINCIPAL_CSV.exists():
        sys.exit(f"EROARE: lipsește {V2_PRINCIPAL_CSV}")
    if not V2_SUPLIMENT_CSV.exists():
        sys.exit(f"EROARE: lipsește {V2_SUPLIMENT_CSV}")
    if not V1_RECOVERED_CSV.exists():
        sys.exit(f"EROARE: lipsește {V1_RECOVERED_CSV}")

    df_principal = pd.read_csv(V2_PRINCIPAL_CSV)
    df_supliment = pd.read_csv(V2_SUPLIMENT_CSV)
    df_v1_raw = pd.read_csv(V1_RECOVERED_CSV)

    print(f"\nÎncărcat:")
    print(f"  v2 principal:  {len(df_principal):5} articole")
    print(f"  v2 supliment:  {len(df_supliment):5} articole")
    print(f"  v1 recuperat:  {len(df_v1_raw):5} articole (nenormalizat)")

    # ── Normalizare v1 ────────────────────────────────────────────────────────
    print("\nNormalizare v1 → schema v2...")
    df_v1 = normalize_v1(df_v1_raw)
    print(f"  Schema v1 după normalizare: OK ({len(df_v1.columns)} coloane)")
    print(f"  ID prefix: {df_v1['id'].iloc[0]} ... {df_v1['id'].iloc[-1]}")

    # ── Verificare overlap pe URL intre cele trei surse ────────────────────
    print("\nVerificare overlap pe URL între surse...")
    urls_p = set(df_principal["url"])
    urls_s = set(df_supliment["url"])
    urls_1 = set(df_v1["url"])

    print(f"  principal ∩ supliment: {len(urls_p & urls_s)}")
    print(f"  principal ∩ v1:        {len(urls_p & urls_1)}")
    print(f"  supliment ∩ v1:        {len(urls_s & urls_1)}")
    print(f"  triple intersection:   {len(urls_p & urls_s & urls_1)}")

    # ── Concat + deduplicare pe URL ───────────────────────────────────────────
    # Ordinea conteaza: pastram prima ocurenta, deci punem v2 principal primul
    # (cea mai bine validata sursa), apoi supliment, apoi v1 recuperat.
    df_complet = pd.concat([df_principal, df_supliment, df_v1], ignore_index=True)
    print(f"\nConcat brut: {len(df_complet)} rânduri")

    n_before = len(df_complet)
    df_complet = df_complet.drop_duplicates(subset=["url"], keep="first")
    n_after = len(df_complet)
    print(f"După dedup pe URL: {n_after} rânduri ({n_before - n_after} duplicate eliminate)")

    # ── Rapoarte per sursa ─────────────────────────────────────────────────────
    report_distribution(df_principal, "v2 principal")
    report_distribution(df_supliment, "v2 supliment")
    report_distribution(df_v1, "v1 recuperat (normalizat)")
    report_distribution(df_complet, "TOTAL COMPLET (post-dedup)")

    # ── Acoperire lunara pe fereastra critica ─────────────────────────────────
    report_monthly_coverage(df_complet, "2025-09", "2026-04")

    # ── Distributie pe ani — finala ────────────────────────────────────────────
    print("\n── Distribuție finală pe ani (toate sursele combinate) ──")
    ani = Counter(d[:4] for d in df_complet["data"].astype(str) if len(d) >= 4)
    veridica_per_year = {"2022": 0, "2023": 135, "2024": 143, "2025": 167, "2026": 52}
    print(f"  {'An':6} {'G4Media':>10} {'Veridica':>10} {'Ratio':>8}")
    for an in sorted(set(ani.keys()) | set(veridica_per_year.keys())):
        g = ani.get(an, 0)
        v = veridica_per_year.get(an, 0)
        ratio = f"{g/v:.1f}x" if v > 0 else "—"
        print(f"  {an:6} {g:>10} {v:>10} {ratio:>8}")

    # ── Salvare output ─────────────────────────────────────────────────────────
    df_complet.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Salvat: {OUTPUT_CSV}")
    print(f"   {len(df_complet)} articole, schema {len(df_complet.columns)} coloane")


if __name__ == "__main__":
    main()
