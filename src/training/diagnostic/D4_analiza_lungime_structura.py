"""
Proba D4: Analiza lungime si structura citate pe LOSO-V.

Ipoteza H3: Veridica si Stopfals difera structural (lungime citat,
densitate ghilimele, stil introductiv fact-checking). Modelul LOSO-V
antrenat pe Stopfals (+ cls0) esueaza pe Veridica pentru ca Veridica
are pattern structural diferit — NU pentru ca continutul propagandistic
e diferit.

Ce masuram:
1. Distributii de features structurale per sursa (TOATE articolele):
   - nr_cuvinte in stire_citata
   - nr_ghilimele
   - densitate ghilimele (n/nr_cuvinte)
   - nr_paragrafe (split pe \\n\\n)
   - caractere NON-ASCII (proxy pt diacritice + chirilica)
2. Corelatia features ↔ predictia LOSO-V pe Veridica

Daca Veridica si Stopfals difera semnificativ pe aceste features
(test Kolmogorov-Smirnov), iar modelul LOSO-V esueaza pe articolele
Veridica cu features cele mai „diferite" de Stopfals → H3 confirmata.

Ruleaza fara antrenare (doar analiza statistica + citire predictions
LOSO-V existente, care sunt in findings_loso_v_v2_predictions.csv).

Usage:
    python src/training/D4_analiza_lungime_structura.py \\
        --dataset data/raw/dataset_licenta_complet.csv \\
        --loso_predictions findings/findings_loso_v_v2_predictions.csv \\
        --output_dir findings
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def count_features(text):
    """Calculeaza features structurale dintr-un text."""
    t = str(text)
    nr_cuvinte = len(t.split())
    nr_ghilimele_duble = t.count('"') + t.count('„') + t.count('”')
    nr_ghilimele_simple = t.count("'") + t.count("‘") + t.count("’")
    nr_ghilimele = nr_ghilimele_duble + nr_ghilimele_simple
    nr_paragrafe = len([p for p in t.split("\n\n") if p.strip()])
    # Caractere non-ASCII (exclude spatiile)
    nr_non_ascii = sum(1 for c in t if ord(c) > 127 and not c.isspace())
    # Densitate
    densitate_ghilimele = nr_ghilimele / max(nr_cuvinte, 1)
    pct_non_ascii = nr_non_ascii / max(len(t), 1) * 100
    return {
        "nr_cuvinte": nr_cuvinte,
        "nr_ghilimele": nr_ghilimele,
        "densitate_ghilimele": densitate_ghilimele,
        "nr_paragrafe": nr_paragrafe,
        "nr_non_ascii": nr_non_ascii,
        "pct_non_ascii": pct_non_ascii,
    }


def construieste_text(row):
    return f"{str(row['titlu']).strip()}\n\n{str(row['stire_citata']).strip()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--loso_predictions", required=True,
                        help="CSV cu predicții LOSO-V (findings/findings_loso_v_v2_predictions.csv)")
    parser.add_argument("--output_dir", default="findings")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === Load ===
    df = pd.read_csv(args.dataset)
    df["text"] = df.apply(construieste_text, axis=1)
    loso_preds = pd.read_csv(args.loso_predictions)

    # === Features per articol ===
    print("[INFO] Calculez features structurale...")
    features_list = df["stire_citata"].apply(count_features).tolist()
    features_df = pd.DataFrame(features_list)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # === Comparatie Veridica vs Stopfals (doar cls1) ===
    veridica = df[df["sursa_site"] == "veridica.ro"]
    stopfals = df[df["sursa_site"] == "stopfals.md"]
    digi24 = df[df["sursa_site"] == "digi24.ro"]
    g4media = df[df["sursa_site"] == "g4media.ro"]

    feature_names = ["nr_cuvinte", "nr_ghilimele", "densitate_ghilimele",
                     "nr_paragrafe", "nr_non_ascii", "pct_non_ascii"]

    print("\n=== Statistici features pe sursă ===")
    stats_table = {}
    for sursa, sub in [("veridica.ro", veridica), ("stopfals.md", stopfals),
                        ("digi24.ro", digi24), ("g4media.ro", g4media)]:
        stats_table[sursa] = {}
        for f in feature_names:
            stats_table[sursa][f] = {
                "mean": float(sub[f].mean()),
                "median": float(sub[f].median()),
                "std": float(sub[f].std()),
            }

    # Print compact
    print(f"{'feature':25s} | "
          f"{'veridica':>12s} | {'stopfals':>12s} | {'digi24':>12s} | {'g4media':>12s}")
    print("-" * 90)
    for f in feature_names:
        row_vals = " | ".join([f"{stats_table[s][f]['median']:>12.2f}"
                                for s in ["veridica.ro", "stopfals.md",
                                          "digi24.ro", "g4media.ro"]])
        print(f"{f:25s} (med) | {row_vals}")

    # === Test KS: Veridica vs Stopfals (sursele cls1) ===
    print("\n=== Test Kolmogorov-Smirnov: Veridica vs Stopfals ===")
    ks_vs = {}
    for f in feature_names:
        ks_stat, p_val = stats.ks_2samp(veridica[f], stopfals[f])
        ks_vs[f] = {"ks_stat": float(ks_stat), "p_value": float(p_val)}
        semn = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        print(f"  {f:25s} KS={ks_stat:.4f}  p={p_val:.4g}  {semn}")

    # === Corelatia features ↔ predictia LOSO-V pe Veridica ===
    print("\n=== Corelație features ↔ predicție LOSO-V pe Veridica ===")
    # Merge predictiile LOSO-V cu features
    # loso_preds are: id, sursa_site, an, titlu, label_numeric, pred, prob_cls1
    merged = loso_preds.merge(
        df[["id"] + feature_names],
        on="id",
        how="inner",
    )
    assert (merged["sursa_site"] == "veridica.ro").all(), "LOSO-V preds ar trebui doar Veridica"
    print(f"  Articole în analiză: {len(merged)}")

    # Corelatie feature ↔ prob_cls1 (continuu) SI feature ↔ pred (binar)
    corr_table = {}
    print(f"\n  Corelații (Pearson cu prob_cls1, Pearson cu pred=1, "
          f"mean pe pred=0 vs pred=1):")
    print(f"  {'feature':25s} | {'r(prob_cls1)':>14s} | {'r(pred)':>10s} | "
          f"{'mean@pred0':>12s} | {'mean@pred1':>12s} | {'delta':>10s}")
    print("  " + "-" * 100)
    for f in feature_names:
        r_prob = float(merged[[f, "prob_cls1"]].corr().iloc[0, 1])
        r_pred = float(merged[[f, "pred"]].corr().iloc[0, 1])
        mean_at_0 = float(merged[merged["pred"] == 0][f].mean()) if (merged["pred"] == 0).any() else float("nan")
        mean_at_1 = float(merged[merged["pred"] == 1][f].mean()) if (merged["pred"] == 1).any() else float("nan")
        delta = mean_at_1 - mean_at_0 if not (np.isnan(mean_at_0) or np.isnan(mean_at_1)) else float("nan")
        corr_table[f] = {
            "corr_prob_cls1": r_prob,
            "corr_pred": r_pred,
            "mean_at_pred0": mean_at_0,
            "mean_at_pred1": mean_at_1,
            "delta_pred1_minus_pred0": delta,
        }
        print(f"  {f:25s} | {r_prob:>+14.4f} | {r_pred:>+10.4f} | "
              f"{mean_at_0:>12.2f} | {mean_at_1:>12.2f} | {delta:>+10.2f}")

    # === Interpretare ===
    # Features cu (a) KS semnificativ intre Veridica si Stopfals + (b) corelatie
    # notabila cu predictia LOSO-V sunt candidatii pentru shortcut structural
    print("\n=== Features candidate pentru shortcut structural ===")
    candidati = []
    for f in feature_names:
        ks_p = ks_vs[f]["p_value"]
        r_pred = corr_table[f]["corr_pred"]
        if ks_p < 0.01 and abs(r_pred) > 0.15:
            candidati.append((f, ks_p, r_pred))
            print(f"  ✓ {f}: KS p={ks_p:.4g}, corr(pred)={r_pred:+.4f}")

    if len(candidati) == 0:
        interp = ("H3 RESPINSĂ: niciun feature structural nu îndeplinește AMBELE criterii "
                  "(diferă între Veridica/Stopfals + corelează cu predicția LOSO-V). "
                  "Shortcut-ul NU e structural. Cauza e probabil lexicală (H1/H2) sau "
                  "subtilă/distribuită (reprezentare contextuală la XLM-R).")
    elif len(candidati) <= 2:
        interp = (f"H3 CONFIRMATĂ PARȚIAL: {len(candidati)} feature(s) candidate "
                  f"({', '.join(c[0] for c in candidati)}). Explică O PARTE din shortcut, "
                  f"dar probabil se combină cu alte cauze.")
    else:
        interp = (f"H3 CONFIRMATĂ PUTERNIC: {len(candidati)} features structurale diferă "
                  f"între surse și corelează cu decizia LOSO-V. Shortcut-ul e în mare parte "
                  f"structural, nu conceptual.")
    print(f"\n=== Interpretare ===\n{interp}")

    # === Save ===
    result = {
        "feature_names": feature_names,
        "stats_per_sursa": stats_table,
        "ks_veridica_vs_stopfals": ks_vs,
        "corr_features_vs_loso_pred": corr_table,
        "candidati_shortcut_structural": [c[0] for c in candidati],
        "interpretare": interp,
    }
    with open(out / "d4_analiza_structura.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] JSON: {out / 'd4_analiza_structura.json'}")

    # CSV features per articol veridica (cu predictie)
    merged.to_csv(out / "d4_veridica_features_loso.csv", index=False)

    # === Markdown ===
    md = [
        "# Proba D4 — Analiză lungime și structură citate",
        "",
        "## Ipoteza H3",
        "",
        "Veridica și Stopfals diferă structural (lungime, ghilimele, non-ASCII). Modelul",
        "LOSO-V eșuează pe Veridica pentru că Veridica are pattern STRUCTURAL diferit",
        "de cls1 văzută la antrenare (Stopfals) — nu pentru că conținutul diferă.",
        "",
        "## Features măsurate",
        "",
        "Pe coloana `stire_citata`:",
        "- `nr_cuvinte` — lungimea citatului",
        "- `nr_ghilimele` — deschideri+închideri de citate",
        "- `densitate_ghilimele` — normalizat la lungime",
        "- `nr_paragrafe` — segmentare",
        "- `nr_non_ascii`, `pct_non_ascii` — proxy pentru diacritice + chirilică",
        "",
        "## Statistici (median) per sursă",
        "",
        "| Feature | Veridica | Stopfals | Digi24 | G4Media |",
        "|---------|----------|----------|--------|---------|",
    ]
    for f in feature_names:
        row = [f"{stats_table[s][f]['median']:.2f}"
               for s in ["veridica.ro", "stopfals.md", "digi24.ro", "g4media.ro"]]
        md.append(f"| {f} | " + " | ".join(row) + " |")

    md.extend([
        "",
        "## Test Kolmogorov-Smirnov: Veridica vs Stopfals",
        "",
        "| Feature | KS statistic | p-value | Semnificativ |",
        "|---------|--------------|---------|--------------|",
    ])
    for f in feature_names:
        ks = ks_vs[f]
        semn = "***" if ks["p_value"] < 0.001 else ("**" if ks["p_value"] < 0.01
                                                      else ("*" if ks["p_value"] < 0.05 else "n.s."))
        md.append(f"| {f} | {ks['ks_stat']:.4f} | {ks['p_value']:.4g} | {semn} |")

    md.extend([
        "",
        "## Corelații cu predicția LOSO-V pe Veridica",
        "",
        "| Feature | r(prob_cls1) | r(pred) | mean@pred=0 | mean@pred=1 | Delta |",
        "|---------|--------------|---------|-------------|-------------|-------|",
    ])
    for f in feature_names:
        c = corr_table[f]
        md.append(f"| {f} | {c['corr_prob_cls1']:+.4f} | {c['corr_pred']:+.4f} | "
                  f"{c['mean_at_pred0']:.2f} | {c['mean_at_pred1']:.2f} | "
                  f"{c['delta_pred1_minus_pred0']:+.2f} |")

    md.extend([
        "",
        "## Candidați pentru shortcut structural",
        "",
        "Criterii: KS p < 0.01 (Veridica ≠ Stopfals) ȘI |corr(pred)| > 0.15.",
        "",
    ])
    if candidati:
        for f, p, r in candidati:
            md.append(f"- **{f}** (KS p={p:.4g}, corr={r:+.4f})")
    else:
        md.append("_Niciun feature nu îndeplinește ambele criterii._")

    md.extend([
        "",
        "## Interpretare",
        "",
        interp,
    ])
    with open(out / "findings_d4_analiza_structura.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[OK] Findings: {out / 'findings_d4_analiza_structura.md'}")


if __name__ == "__main__":
    main()
