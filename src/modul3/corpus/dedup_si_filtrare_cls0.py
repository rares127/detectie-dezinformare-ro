"""
Modul 3 — Pasii 4+5: Deduplicare + Filtrare lungime.

Combinem cele doua operatii intr-un singur script pentru ca:
    1. Ordinea conteaza: deduplicam INAINTE de a calcula percentilele,
       altfel pragurile [p5, p95] sunt calculate pe un corpus cu duplicate.
    2. Ambele sunt mecanice (fara reguli pattern, fara ambiguitate).

Input: propozitii_cls0_filtrat.parquet (5.915 propozitii)
Output: propozitii_cls0_corpus.parquet (corpus final gata de embeddings)

Pasul 4 — Deduplicare:
    - Recalculam hash normalizat pe textul CURATAt (dupa toti pasii anteriori).
      IMPORTANT: hashul din parquet-ul initial e calculat pe textul brut —
      propozitiile modificate (Foto: curatat, Pantir-S1 fara banner etc.)
      au alt hash acum.
    - Pastram prima aparitie, eliminam duplicatele ulterioare.
    - Raportam cate duplicate s-au eliminat per sursa.

Pasul 5 — Filtrare lungime:
    - Recalculam percentilele pe corpusul deduplicat.
    - Aplicam [p5, p95] ca praguri.
    - Raportam distributia inainte/dupa.

Rulare:
    python dedup_si_filtrare_cls0.py \\
        --input data/processed/propozitii_cls0_filtrat.parquet \\
        --output data/processed/propozitii_cls0_corpus.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42


# ---------------------------------------------------------------------------
# Hash normalizat (acelasi algoritm ca in pasii anteriori)
# ---------------------------------------------------------------------------

def normalizeaza_pentru_dedup(text: str) -> str:
    """NFKC → lowercase → fara punctuatie → whitespace colapsat."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pasul 4: Deduplicare
# ---------------------------------------------------------------------------

def aplica_deduplicare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicare pe hash normalizat recalculat pe textul curent.

    Recalculam hash-ul pentru ca textele au fost modificate fata de raw
    (cookie banners eliminate, prefix Foto: curatat etc.).
    """
    df = df.copy()

    # Recalculam hash-ul pe textul curent
    df["hash_normalizat_curent"] = df["propozitie"].apply(
        lambda t: hash_text(normalizeaza_pentru_dedup(t))
    )

    inainte = len(df)
    df_dedup = df.drop_duplicates(subset=["hash_normalizat_curent"], keep="first").copy()
    eliminate = inainte - len(df_dedup)

    # Statistici per sursa
    per_sursa = {}
    for sursa in df["sursa_site"].unique():
        n_inainte = (df["sursa_site"] == sursa).sum()
        n_dupa = (df_dedup["sursa_site"] == sursa).sum()
        per_sursa[sursa] = {
            "inainte": int(n_inainte),
            "dupa": int(n_dupa),
            "eliminate": int(n_inainte - n_dupa),
        }

    # Top duplicate — propozitiile care apareau de mai multe ori
    hash_counts = df["hash_normalizat_curent"].value_counts()
    top_dup = hash_counts[hash_counts > 1].head(10)
    exemple_duplicate = []
    for h, count in top_dup.items():
        exemplu = df[df["hash_normalizat_curent"] == h]["propozitie"].iloc[0]
        exemple_duplicate.append({
            "aparitii": int(count),
            "nr_cuvinte": len(exemplu.split()),
            "surse": df[df["hash_normalizat_curent"] == h]["sursa_site"].unique().tolist(),
            "text": exemplu[:200],
        })

    stats = {
        "input_total": inainte,
        "eliminate_duplicate": eliminate,
        "output_dupa_dedup": len(df_dedup),
        "per_sursa": per_sursa,
        "exemple_duplicate": exemple_duplicate,
    }

    return df_dedup, stats


# ---------------------------------------------------------------------------
# Pasul 5: Filtrare lungime
# ---------------------------------------------------------------------------

def aplica_filtrare_lungime(
    df: pd.DataFrame,
    p_min_percentila: int = 5,
    p_max_percentila: int = 95,
) -> tuple[pd.DataFrame, dict, int, int]:
    """Filtrare pe percentilele [p_min, p_max] calculate pe corpusul curent.

    Returns:
        (df_filtrat, stats, p_min_val, p_max_val)
    """
    wc = df["nr_cuvinte"]

    p_min = int(np.percentile(wc, p_min_percentila))
    p_max = int(np.percentile(wc, p_max_percentila))

    mask_scurt = wc < p_min
    mask_lung = wc > p_max

    df_filtrat = df[~mask_scurt & ~mask_lung].copy()

    stats = {
        "input_total": len(df),
        "p_min": p_min,
        "p_max": p_max,
        "p_min_percentila": p_min_percentila,
        "p_max_percentila": p_max_percentila,
        "eliminate_prea_scurt": int(mask_scurt.sum()),
        "eliminate_prea_lung": int(mask_lung.sum()),
        "output_total": len(df_filtrat),
        "retentie_pct": round(100 * len(df_filtrat) / len(df), 2),
        "distributie_output": {
            "min": int(df_filtrat["nr_cuvinte"].min()),
            "p25": float(df_filtrat["nr_cuvinte"].quantile(0.25)),
            "mediana": float(df_filtrat["nr_cuvinte"].median()),
            "medie": round(float(df_filtrat["nr_cuvinte"].mean()), 2),
            "p75": float(df_filtrat["nr_cuvinte"].quantile(0.75)),
            "max": int(df_filtrat["nr_cuvinte"].max()),
        },
    }

    return df_filtrat, stats, p_min, p_max


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------

def scrie_raport(
    stats_dedup: dict,
    stats_lungime: dict,
    n_input_original: int,
    n_output_final: int,
    output_path: Path,
) -> None:
    lines = [
        "# Deduplicare + Filtrare lungime cls0 — raport",
        "",
        "**Pașii 4+5 din pipeline-ul de preprocessing.**",
        "Aceștia produc `propozitii_cls0_corpus.parquet` —",
        "corpusul final gata de embeddings.",
        "",
        "## 1. Pasul 4 — Deduplicare",
        "",
        "| Metrică | Valoare |",
        "|---|---|",
        f"| Input (după filtru rezidual) | {stats_dedup['input_total']:,} |",
        f"| Duplicate eliminate | {stats_dedup['eliminate_duplicate']} |",
        f"| Output după deduplicare | {stats_dedup['output_dupa_dedup']:,} |",
        "",
        "### Breakdown per sursă",
        "",
        "| Sursă | Înainte | După | Eliminate |",
        "|---|---|---|---|",
    ]
    for sursa, s in stats_dedup["per_sursa"].items():
        lines.append(f"| {sursa} | {s['inainte']:,} | {s['dupa']:,} | {s['eliminate']} |")

    lines += [
        "",
        "### Top duplicate eliminate",
        "",
        "> Propoziții care apăreau de mai multe ori — confirmare că deduplicarea",
        "> a prins artefactele repetate (ex. aceeași fotografie de presă, aceleași",
        "> fraze standard preluate de la agenții).",
        "",
    ]
    for ex in stats_dedup["exemple_duplicate"]:
        surse_str = ", ".join(ex["surse"])
        lines.append(
            f"- **{ex['aparitii']} apariții** ({ex['nr_cuvinte']}w) "
            f"[{surse_str}]: {ex['text']}"
        )
    lines.append("")

    p_min = stats_lungime["p_min"]
    p_max = stats_lungime["p_max"]
    d = stats_lungime["distributie_output"]

    lines += [
        "## 2. Pasul 5 — Filtrare lungime",
        "",
        f"**Praguri calculate pe corpusul deduplicat:**",
        f"[p{stats_lungime['p_min_percentila']}={p_min}w, "
        f"p{stats_lungime['p_max_percentila']}={p_max}w]",
        "",
        "| Metrică | Valoare |",
        "|---|---|",
        f"| Input (după deduplicare) | {stats_lungime['input_total']:,} |",
        f"| Eliminate < {p_min}w (sub p5) | {stats_lungime['eliminate_prea_scurt']} |",
        f"| Eliminate > {p_max}w (peste p95) | {stats_lungime['eliminate_prea_lung']} |",
        f"| **Output final** | **{stats_lungime['output_total']:,}** |",
        f"| Retenție față de input pasul 5 | {stats_lungime['retentie_pct']}% |",
        "",
        "### Distribuție lungime după filtrare",
        "",
        "| min | p25 | mediană | medie | p75 | max |",
        "|---|---|---|---|---|---|",
        f"| {d['min']} | {d['p25']:.0f} | {d['mediana']:.0f} | "
        f"{d['medie']} | {d['p75']:.0f} | {d['max']} |",
        "",
        "## 3. Sumar final pipeline preprocessing",
        "",
        "| Pas | Fișier | Propoziții |",
        "|---|---|---|",
        f"| Segmentare (Stanza) | propozitii_cls0_raw.parquet | 6,047 |",
        f"| Curățare cookies v3 | propozitii_cls0_no_cookies.parquet | 5,976 |",
        f"| Filtru rezidual | propozitii_cls0_filtrat.parquet | 5,915 |",
        f"| Deduplicare | — | {stats_dedup['output_dupa_dedup']:,} |",
        f"| Filtrare lungime [{p_min}w, {p_max}w] | **propozitii_cls0_corpus.parquet** | "
        f"**{n_output_final:,}** |",
        "",
        f"**Retenție totală față de raw:** "
        f"{round(100 * n_output_final / 6047, 2)}%",
        "",
        "## 4. Pasul următor",
        "",
        "Corpusul `propozitii_cls0_corpus.parquet` e gata pentru:",
        "**Benchmark model embeddings** — comparație XLM-RoBERTa mean-pooled",
        "vs sentence-transformers multilingv pe acest corpus.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deduplicare + filtrare lungime cls0")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-dir", type=Path, default=Path("findings"))
    parser.add_argument("--p-min", type=int, default=5,
                        help="Percentila minimă pentru filtrare lungime (default: p5)")
    parser.add_argument("--p-max", type=int, default=95,
                        help="Percentila maximă pentru filtrare lungime (default: p95)")
    args = parser.parse_args()

    args.findings_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Încarc {args.input}")
    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"      {len(df):,} propoziții")
    n_input = len(df)

    print("[2/4] Pasul 4 — Deduplicare")
    df_dedup, stats_dedup = aplica_deduplicare(df)
    print(f"      Eliminate: {stats_dedup['eliminate_duplicate']} duplicate")
    print(f"      Rămase: {stats_dedup['output_dupa_dedup']:,}")

    print(f"[3/4] Pasul 5 — Filtrare lungime [p{args.p_min}, p{args.p_max}]")
    df_final, stats_lungime, p_min, p_max = aplica_filtrare_lungime(
        df_dedup, args.p_min, args.p_max
    )
    print(f"      Praguri calculate: [{p_min}w, {p_max}w]")
    print(f"      Eliminate scurte (<{p_min}w): {stats_lungime['eliminate_prea_scurt']}")
    print(f"      Eliminate lungi (>{p_max}w):  {stats_lungime['eliminate_prea_lung']}")
    print(f"      Output final: {len(df_final):,} propoziții")

    print("[4/4] Salvez outputs")
    # Eliminam coloane auxiliare de bookkeeping
    cols_de_sters = ["hash_normalizat_curent", "_actiune_curatare",
                     "_actiune_rezidual", "_trace"]
    df_pt_salvare = df_final.drop(
        columns=[c for c in cols_de_sters if c in df_final.columns],
        errors="ignore"
    )
    try:
        df_pt_salvare.to_parquet(args.output, index=False)
        print(f"      → {args.output}")
    except ImportError:
        csv_path = args.output.with_suffix(".csv")
        df_pt_salvare.to_csv(csv_path, index=False)
        print(f"      → {csv_path}")

    md_path = args.findings_dir / "dedup_si_filtrare_cls0.md"
    scrie_raport(stats_dedup, stats_lungime, n_input, len(df_final), md_path)
    print(f"      → {md_path}")

    json_path = args.findings_dir / "dedup_si_filtrare_cls0.json"
    json_combined = {"deduplicare": stats_dedup, "filtrare_lungime": stats_lungime}
    json_path.write_text(
        json.dumps(json_combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      → {json_path}")

    print()
    print("=" * 60)
    print("SUMAR FINAL")
    print("=" * 60)
    print(f"Input (filtru rezidual):  {n_input:,}")
    print(f"Duplicate eliminate:      {stats_dedup['eliminate_duplicate']}")
    print(f"Prea scurte eliminate:    {stats_lungime['eliminate_prea_scurt']}")
    print(f"Prea lungi eliminate:     {stats_lungime['eliminate_prea_lung']}")
    print(f"Corpus final:             {len(df_final):,}")
    print(f"Retenție față de raw:     {round(100 * len(df_final) / 6047, 2)}%")
    print(f"Praguri lungime:          [{p_min}w, {p_max}w]")


if __name__ == "__main__":
    main()
