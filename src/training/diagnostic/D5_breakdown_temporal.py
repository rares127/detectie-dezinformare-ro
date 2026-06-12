"""
Proba D5: Breakdown LOSO-V per an.

Ipoteza H6: drop-ul de recall pe LOSO-V vine partial dintr-un shift
temporal. Stopfals (singura sursa cls1 in LOSO-V train) poate avea o
distributie anuala diferita de Veridica. Daca Veridica 2022 e dominata
de naratiuni specifice (invazie initiala, propaganda justificatoare) pe
care Stopfals nu le acopera, LOSO-V va esua mai mult pe 2022.

Ce masuram:
1. Distributie temporala Stopfals (train cls1 in LOSO-V)
2. Recall cls1 LOSO-V pe Veridica, stratificat per an
3. Confidence distribution per an

Daca recall scade dramatic doar pe anumiti ani → shift temporal.
Daca e uniform slab peste toti anii → shortcut generic, independent
de an.

Ruleaza pe predictiile LOSO-V deja salvate + distributia Stopfals
din dataset.

Usage:
    python src/training/D5_breakdown_temporal.py \\
        --loso_predictions findings/findings_loso_v_v2_predictions.csv \\
        --dataset data/raw/dataset_licenta_complet.csv \\
        --output_dir findings
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loso_predictions", required=True,
                        help="findings/findings_loso_v_v2_predictions.csv")
    parser.add_argument("--dataset", required=True,
                        help="Pentru distribuția Stopfals per an")
    parser.add_argument("--output_dir", default="findings")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === Load ===
    loso = pd.read_csv(args.loso_predictions)
    assert (loso["sursa_site"] == "veridica.ro").all(), "LOSO-V preds doar Veridica"
    assert (loso["label_numeric"] == 1).all(), "LOSO-V preds doar cls1"
    print(f"[INFO] N predicții LOSO-V: {len(loso)}")

    df = pd.read_csv(args.dataset)
    stopfals_dist = df[df["sursa_site"] == "stopfals.md"]["an"].value_counts().sort_index()
    veridica_dist = df[df["sursa_site"] == "veridica.ro"]["an"].value_counts().sort_index()

    print("\n=== Distribuție temporală Stopfals (train LOSO-V cls1) ===")
    for an, n in stopfals_dist.items():
        pct = 100 * n / stopfals_dist.sum()
        print(f"  {an}: {n:3d} ({pct:5.1f}%)")
    print(f"  TOTAL: {stopfals_dist.sum()}")

    print("\n=== Distribuție Veridica (test LOSO-V) ===")
    for an, n in veridica_dist.items():
        pct = 100 * n / veridica_dist.sum()
        print(f"  {an}: {n:3d} ({pct:5.1f}%)")

    # === Recall cls1 LOSO-V per an ===
    print("\n=== Recall cls1 LOSO-V pe Veridica, per an ===")
    per_an = []
    for an, sub in loso.groupby("an"):
        n = len(sub)
        recall = (sub["pred"] == 1).mean()
        mean_prob = sub["prob_cls1"].mean()
        median_prob = sub["prob_cls1"].median()
        n_corecte = int((sub["pred"] == 1).sum())

        # Proportia Stopfals pentru acest an (pentru context)
        stopfals_n_an = int(stopfals_dist.get(an, 0))

        per_an.append({
            "an": int(an),
            "n_veridica": int(n),
            "n_corecte": n_corecte,
            "recall_cls1": float(recall),
            "mean_prob_cls1": float(mean_prob),
            "median_prob_cls1": float(median_prob),
            "n_stopfals_train": stopfals_n_an,
        })
        print(f"  {an}: n={n:3d} | recall={recall:.4f} ({n_corecte:3d}/{n:3d}) | "
              f"mean_prob={mean_prob:.4f} | stopfals@an={stopfals_n_an}")

    # === Comparare cu recall baseline v2 (100% pe test) ===
    # Dar baseline include articolele in train — nu are sens per-an direct.
    # Comparam recall-urile intre ani LOSO-V:
    recall_min = min(p["recall_cls1"] for p in per_an)
    recall_max = max(p["recall_cls1"] for p in per_an)
    range_recall = recall_max - recall_min
    print(f"\n  Range recall per an: [{recall_min:.4f}, {recall_max:.4f}], "
          f"delta={range_recall:.4f}")

    # Subseturi critice: ani CU si FARA Stopfals in train
    ani_fara_stopfals = [p for p in per_an if p["n_stopfals_train"] == 0]
    ani_cu_stopfals = [p for p in per_an if p["n_stopfals_train"] > 0]
    if ani_fara_stopfals and ani_cu_stopfals:
        # Recall agregat ponderat pe n_veridica
        def recall_agregat(grup):
            total_n = sum(p["n_veridica"] for p in grup)
            total_correct = sum(p["n_corecte"] for p in grup)
            return total_correct / total_n if total_n else float("nan")

        r_fara = recall_agregat(ani_fara_stopfals)
        r_cu = recall_agregat(ani_cu_stopfals)
        n_fara = sum(p["n_veridica"] for p in ani_fara_stopfals)
        n_cu = sum(p["n_veridica"] for p in ani_cu_stopfals)
        print(f"\n  Ani FĂRĂ Stopfals@train (n={n_fara}): recall agregat = {r_fara:.4f}")
        print(f"  Ani CU Stopfals@train (n={n_cu}): recall agregat = {r_cu:.4f}")
        gap_cu_fara = r_cu - r_fara
        print(f"  Gap (cu − fără): {gap_cu_fara:+.4f}")
    else:
        r_fara = r_cu = gap_cu_fara = None

    # === Interpretare ===
    if range_recall < 0.10:
        interp = ("H6 RESPINSĂ: recall LOSO-V uniform slab peste toți anii "
                  f"(range <10pp). Shortcut-ul NU e temporal — e general. "
                  "Anul nu e cauza drop-ului.")
    elif range_recall < 0.25:
        interp = (f"H6 CONFIRMATĂ SLAB: variație recall {range_recall*100:.1f}pp între ani. "
                  "Shift temporal există dar e secundar față de alte cauze.")
    else:
        # Identifica anul cel mai problematic
        worst_an = min(per_an, key=lambda x: x["recall_cls1"])
        best_an = max(per_an, key=lambda x: x["recall_cls1"])
        interp = (f"H6 CONFIRMATĂ: recall variază dramatic între ani "
                  f"(range {range_recall*100:.1f}pp). "
                  f"Cel mai rău: {worst_an['an']} (recall {worst_an['recall_cls1']:.4f}). "
                  f"Cel mai bine: {best_an['an']} (recall {best_an['recall_cls1']:.4f}). "
                  "Atenuare: balansare temporală Stopfals sau augmentare cls1 "
                  "pentru anii sub-reprezentați în train.")

    # Extensie: daca gap cu/fara Stopfals e mare, il mentionam explicit
    if gap_cu_fara is not None and abs(gap_cu_fara) > 0.15:
        interp += (f" IMPORTANT: gap între ani cu vs fără Stopfals în train = "
                    f"{gap_cu_fara:+.4f} — modelul LOSO-V e dependent critic de "
                    f"acoperirea temporală Stopfals@an-ul testat.")

    print(f"\n=== Interpretare ===\n{interp}")

    # === Save ===
    result = {
        "stopfals_distributie_an": {int(k): int(v) for k, v in stopfals_dist.items()},
        "veridica_distributie_an": {int(k): int(v) for k, v in veridica_dist.items()},
        "loso_v_per_an": per_an,
        "recall_range": float(range_recall),
        "recall_min": float(recall_min),
        "recall_max": float(recall_max),
        "ani_fara_stopfals_train": [p["an"] for p in ani_fara_stopfals] if ani_fara_stopfals else [],
        "recall_agregat_ani_fara_stopfals": float(r_fara) if r_fara is not None else None,
        "recall_agregat_ani_cu_stopfals": float(r_cu) if r_cu is not None else None,
        "gap_cu_minus_fara": float(gap_cu_fara) if gap_cu_fara is not None else None,
        "interpretare": interp,
    }
    with open(out / "d5_breakdown_temporal.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] JSON: {out / 'd5_breakdown_temporal.json'}")

    # === Markdown ===
    md = [
        "# Proba D5 — Breakdown LOSO-V per an",
        "",
        "## Ipoteza H6",
        "",
        "Drop-ul recall cls1 pe LOSO-V are o componentă temporală: Stopfals (singura",
        "sursă cls1 în train LOSO-V) are distribuție temporală diferită de Veridica,",
        "iar modelul eșuează pe anii sub-reprezentați în train.",
        "",
        "## Distribuție Stopfals (train cls1 LOSO-V)",
        "",
        "| An | N | % |",
        "|----|---|---|",
    ]
    for an, n in stopfals_dist.items():
        pct = 100 * n / stopfals_dist.sum()
        md.append(f"| {an} | {n} | {pct:.1f}% |")

    md.extend([
        "",
        "## Distribuție Veridica (test LOSO-V)",
        "",
        "| An | N | % |",
        "|----|---|---|",
    ])
    for an, n in veridica_dist.items():
        pct = 100 * n / veridica_dist.sum()
        md.append(f"| {an} | {n} | {pct:.1f}% |")

    md.extend([
        "",
        "## Recall cls1 LOSO-V per an",
        "",
        "| An | N Veridica | Corecte | Recall | Mean P(cls1) | Median P(cls1) | Stopfals@an (train) |",
        "|----|-----------|---------|--------|--------------|-----------------|---------------------|",
    ])
    for p in per_an:
        md.append(f"| {p['an']} | {p['n_veridica']} | {p['n_corecte']} | "
                  f"{p['recall_cls1']:.4f} | {p['mean_prob_cls1']:.4f} | "
                  f"{p['median_prob_cls1']:.4f} | {p['n_stopfals_train']} |")

    md.extend([
        "",
        f"**Range recall**: [{recall_min:.4f}, {recall_max:.4f}] "
        f"(delta {range_recall*100:.1f}pp)",
        "",
    ])

    # Sectiune ani cu/fara Stopfals daca exista partitie
    if ani_fara_stopfals and ani_cu_stopfals:
        ani_fara_list = ", ".join(str(p["an"]) for p in ani_fara_stopfals)
        ani_cu_list = ", ".join(str(p["an"]) for p in ani_cu_stopfals)
        md.extend([
            "## Ani CU vs FĂRĂ Stopfals în training",
            "",
            f"- **Ani fără Stopfals@train** ({ani_fara_list}): "
            f"n={sum(p['n_veridica'] for p in ani_fara_stopfals)}, "
            f"recall agregat = **{r_fara:.4f}**",
            f"- **Ani cu Stopfals@train** ({ani_cu_list}): "
            f"n={sum(p['n_veridica'] for p in ani_cu_stopfals)}, "
            f"recall agregat = **{r_cu:.4f}**",
            f"- **Gap** (cu − fără): **{gap_cu_fara:+.4f}**",
            "",
        ])

    md.extend([
        "## Interpretare",
        "",
        interp,
    ])
    with open(out / "findings_d5_breakdown_temporal.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[OK] Findings: {out / 'findings_d5_breakdown_temporal.md'}")


if __name__ == "__main__":
    main()
