"""
clean_veridica_v4_2.py
──────────────────────
Pipeline de cleaning final pentru dataset-ul Veridica v4.2 (clasa 1 —
dezinformare pro-rusa). Rezultatul este veridica_clean_v3.csv, dataset-ul
folosit pentru antrenarea clasificatorului XLM-RoBERTa.

Decizii de cleaning (consolidate din iteratiile anterioare):
  - `suspect_contaminare` (2 art. ramase dupa FIX M) → promovate la
    `excelenta` pe baza triajului manual documentat in findings_metodologice.md
    (ambele contin markerul `contrazice` natural in citat propagandistic,
    nu ca marker de fact-check).
  - `suspect_dimensiune` (3 art.) → promovate la `excelenta`
    (XLM-RoBERTa trunchiaza la 512 tokens, lungimea nu e problema).
  - `fallback_verificare_manuala` (2 art.) → DROP.
  - `buna` (8 art.) → pastrate.
  - Prag minim: 10 cuvinte.
  - Coloane eliminate: `autor` (100% null), `_referinta_continut_full`.

Output:
  - veridica_clean_v3.csv     → dataset principal
  - veridica_clean_v3_report.txt → raport cu statistici
"""

import pandas as pd

INPUT_CSV     = "/mnt/user-data/uploads/veridica_ukraine_v4_2.csv"
OUTPUT_CLEAN  = "veridica_clean_v3.csv"
OUTPUT_REPORT = "veridica_clean_v3_report.txt"

COLOANE_FINALE = [
    "id", "url", "titlu", "data", "sursa_site", "sectiune",
    "text_curat", "stire_citata", "naratiuni_false", "obiective_propaganda",
    "nr_cuvinte_stire", "nr_propozitii", "cuvinte_cheie",
    "calitate_extractie",
    "label", "label_numeric", "hash_continut",
]


def log(report, msg):
    print(msg)
    report.append(msg)


def main():
    report = []
    log(report, "=" * 70)
    log(report, "CLEANING FINAL: Veridica v4.2 → veridica_clean_v3 (clasa 1)")
    log(report, "=" * 70)

    df = pd.read_csv(INPUT_CSV)
    n_initial = len(df)
    log(report, f"\n[1] Încărcat: {n_initial} articole din v4.2")

    # ── 2. Promovari pe baza deciziilor anterioare ────────────────────────────
    # suspect_dimensiune → excelenta
    mask_dim = df["calitate_extractie"] == "suspect_dimensiune"
    n_dim = mask_dim.sum()
    df.loc[mask_dim, "calitate_extractie"] = "excelenta"
    log(report, f"[2] Promovare suspect_dimensiune → excelenta: {n_dim}")

    # suspect_contaminare (2 ramase) → excelenta pe baza triajului manual
    # Ambele contin verbul „contrazice" in context narativ, nu fact-check.
    mask_cont = df["calitate_extractie"] == "suspect_contaminare"
    n_cont = mask_cont.sum()
    if n_cont > 0:
        log(report, f"[3] Promovare suspect_contaminare → excelenta: {n_cont}")
        log(report, "    (verificate vizual — markerul `contrazice` apare natural)")
        for _, r in df[mask_cont].iterrows():
            log(report, f"    - {r['titlu'][:80]}")
        df.loc[mask_cont, "calitate_extractie"] = "excelenta"
        # Scraper-ul goleste text_curat pentru suspect_contaminare (linia 530
        # din v4.2). Pentru articolele promovate, il reconstruim ca
        # titlu + stire_citata, exact ca pentru calitate=excelenta.
        def rebuild(r):
            t = str(r["titlu"]) if pd.notna(r["titlu"]) else ""
            s = str(r["stire_citata"]) if pd.notna(r["stire_citata"]) else ""
            return f"{t} {s}".strip()
        df.loc[mask_cont, "text_curat"] = df[mask_cont].apply(rebuild, axis=1)
        # nr_propozitii a fost calculat de scraper pe text_curat gol → 0.
        # Marker -1 = „recalculeaza la preprocessing cu Stanza".
        df.loc[mask_cont, "nr_propozitii"] = -1
        log(report, "    (text_curat reconstruit, nr_propozitii=-1 până la Stanza)")

    # ── 3. Drop fallback ──────────────────────────────────────────────────────
    mask_fb = df["calitate_extractie"] == "fallback_verificare_manuala"
    n_fb = mask_fb.sum()
    df = df[~mask_fb].copy()
    log(report, f"[4] Drop fallback_verificare_manuala: -{n_fb}")

    # ── 4. Parsare data + sortare + ID stabil ────────────────────────────────
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values("data", na_position="last").reset_index(drop=True)
    df["id"] = [f"vrd_{i:04d}" for i in range(len(df))]
    log(report, f"[5] Sortat cronologic + ID stabil generat")

    # ── 5. Filtru ≥10 cuvinte ─────────────────────────────────────────────────
    df["wc"] = df["text_curat"].fillna("").str.split().str.len()
    n_inainte = len(df)
    df = df[df["wc"] >= 10].drop(columns=["wc"])
    log(report, f"[6] Filtru text_curat ≥10 cuvinte: -{n_inainte - len(df)}")

    # ── 6. Deduplicare ────────────────────────────────────────────────────────
    n_inainte = len(df)
    df = df.drop_duplicates(subset="hash_continut", keep="first")
    log(report, f"[7] Deduplicare pe hash_continut: -{n_inainte - len(df)}")

    # ── 7. Selectie coloane finale ────────────────────────────────────────────
    df = df[[c for c in COLOANE_FINALE if c in df.columns]]

    # ── 8. Salvare ────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8-sig")
    log(report, f"\n[8] Scris {OUTPUT_CLEAN}: {len(df)} articole")

    # ── 9. Statistici ─────────────────────────────────────────────────────────
    log(report, "\n" + "=" * 70)
    log(report, "STATISTICI FINALE — veridica_clean_v3")
    log(report, "=" * 70)

    log(report, f"\nDimensiune dataset: {len(df)} articole")
    log(report, f"\n── Distribuție calitate ──")
    for q, c in df["calitate_extractie"].value_counts().items():
        log(report, f"  {q:30s}: {c:4d} ({c/len(df)*100:5.1f}%)")

    log(report, f"\n── text_curat (cuvinte) ──")
    wc = df["text_curat"].str.split().str.len()
    log(report, f"  min/Q1/median/Q3/max: {wc.min()}/{int(wc.quantile(.25))}/{int(wc.median())}/{int(wc.quantile(.75))}/{wc.max()}")
    log(report, f"  mean: {wc.mean():.1f}")

    log(report, f"\n── nr_propozitii (FIX O validat) ──")
    log(report, f"  min/median/max: {df['nr_propozitii'].min()}/{int(df['nr_propozitii'].median())}/{df['nr_propozitii'].max()}")
    log(report, f"  mean: {df['nr_propozitii'].mean():.1f}")

    log(report, f"\n── Distribuție temporală ──")
    for an, c in df["data"].dt.year.value_counts().sort_index().items():
        if pd.notna(an):
            log(report, f"  {int(an)}: {c} articole")
    n_nat = df["data"].isna().sum()
    if n_nat > 0:
        log(report, f"  fără dată: {n_nat}")

    log(report, f"\n── Sanity checks ──")
    log(report, f"  Toate label=1?    {(df['label_numeric']==1).all()}")
    log(report, f"  text_curat OK?    {(df['text_curat'].str.len() > 0).all()}")
    log(report, f"  Hash unic?        {df['hash_continut'].is_unique}")
    log(report, f"  ID unic?          {df['id'].is_unique}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n[REPORT] {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
