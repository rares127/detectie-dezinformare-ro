"""
Evaluare stratificata pe subsetul „Moldova" — verifica eficacitatea entity balancing.

Context: in v1, analiza DF (document frequency) a aratat ca termeni precum
„moldova", „chisinau", „maia sandu", „transnistria" apareau cu frecventa +30.8pp
mai mare in cls1 (Veridica) decat in cls0 (Digi24/G4Media). Entity balancing aplicat
in v2 a redus diferenta la +16.1pp. Acest script verifica daca modelul mai foloseste
„Moldova" ca proxy pentru cls1.

Metodologie:
1. Separa TEST in doua subseturi: articole-cu-moldova vs articole-fara-moldova
2. Calculeaza macro-F1 + recall pe fiecare subset
3. Interpretare:
   - F1(moldova) ≈ F1(non-moldova) → entity balancing functioneaza
   - F1(moldova) << F1(non-moldova) cu delta >5pp → bias rezidual

Usage:
    python src/training/04_eval_moldova_stratified.py \
        --predictions findings/test_predictions_v2.csv \
        --test_data data/processed/dataset_v2_test.csv \
        --output_dir findings
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Regex Moldova — acopera cele mai frecvente entitati din contextul propagandei pro-ruse
# adresate RO, conform DF analysis v1
MOLDOVA_REGEX = re.compile(
    r"\bmoldov[aă]\b|"           # Moldova, Moldova
    r"\bchi[sș]in[aă]u\b|"        # Chisinau (cu diacritice/fara)
    r"\bmaia\s+sandu\b|"
    r"\btransnistri[ae]\b|"       # Transnistria / Transnistrie
    r"\bg[aă]g[aă]uzi[ae]\b|"     # Gagauzia
    r"\bcomrat\b|"
    r"\bdodon\b|"                 # Igor Dodon
    r"\bsor\b|"                   # Ilan Sor (doar ca intreg cuvant)
    r"\bmd\b(?=\s|$|[,\.])",     # .md TLD mentionat
    flags=re.IGNORECASE,
)


def contine_moldova(text: str) -> bool:
    """True daca textul contine cel putin un termen-Moldova."""
    if not isinstance(text, str):
        return False
    return bool(MOLDOVA_REGEX.search(text))


def compute_subset_metrics(df: pd.DataFrame, nume: str) -> dict:
    """Metrici pe un subset."""
    if len(df) == 0:
        return {"n": 0, "note": "subset gol"}
    y_true = df["label_numeric"].values
    y_pred = df["pred"].values
    acc = accuracy_score(y_true, y_pred)
    # Macro doar daca ambele clase prezente
    if len(set(y_true)) == 2:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_per, r_per, f1_per, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )
    else:
        p = r = f1 = float("nan")
        p_per = r_per = f1_per = [float("nan"), float("nan")]

    n_cls0 = int((y_true == 0).sum())
    n_cls1 = int((y_true == 1).sum())

    print(f"\n--- {nume} ---")
    print(f"  n={len(df)}  (cls0={n_cls0}, cls1={n_cls1})")
    print(f"  Accuracy: {acc:.4f}")
    if not (isinstance(f1, float) and f1 != f1):  # not NaN
        print(f"  Macro-F1: {f1:.4f}")
        print(f"  Recall cls0: {r_per[0]:.4f}")
        print(f"  Recall cls1: {r_per[1]:.4f}")

    return {
        "n": len(df),
        "n_cls0": n_cls0,
        "n_cls1": n_cls1,
        "accuracy": float(acc),
        "macro_f1": float(f1) if not (isinstance(f1, float) and f1 != f1) else None,
        "recall_cls0": float(r_per[0]) if not (isinstance(r_per[0], float) and r_per[0] != r_per[0]) else None,
        "recall_cls1": float(r_per[1]) if not (isinstance(r_per[1], float) and r_per[1] != r_per[1]) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="CSV cu predicții (din 03_eval)")
    parser.add_argument("--test_data", required=True,
                        help="CSV test split (pt acces la `text`)")
    parser.add_argument("--output_dir", default="findings")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(args.predictions)
    test = pd.read_csv(args.test_data)

    # Join pentru a avea `text` langa predictii
    merged = preds.merge(
        test[["id", "text", "stire_citata", "nr_cuvinte_truncat"]],
        on="id",
        how="left",
    )
    assert merged["text"].notna().all(), "Merge incomplet"

    # Aplicare regex Moldova
    merged["are_moldova"] = merged["text"].apply(contine_moldova)

    print(f"[INFO] Total TEST: {len(merged)}")
    print(f"[INFO] Articole cu termeni Moldova: {merged['are_moldova'].sum()}")
    print(f"[INFO] Articole fără termeni Moldova: {(~merged['are_moldova']).sum()}")

    # Distributie clase pe subseturi
    print(f"\n[INFO] Distribuție clase — cu Moldova:")
    print(merged[merged["are_moldova"]]["label_numeric"].value_counts().to_string())
    print(f"\n[INFO] Distribuție clase — fără Moldova:")
    print(merged[~merged["are_moldova"]]["label_numeric"].value_counts().to_string())

    sub_mold = merged[merged["are_moldova"]]
    sub_nonmold = merged[~merged["are_moldova"]]

    m_mold = compute_subset_metrics(sub_mold, "Articole CU termeni Moldova")
    m_nonmold = compute_subset_metrics(sub_nonmold, "Articole FĂRĂ termeni Moldova")

    # === Interpretare ===
    print("\n=== Interpretare ===")
    if m_mold.get("macro_f1") and m_nonmold.get("macro_f1"):
        delta = m_nonmold["macro_f1"] - m_mold["macro_f1"]
        print(f"Delta macro_f1 (non-moldova - moldova): {delta:+.4f}")
        if abs(delta) < 0.05:
            interp = ("Entity balancing PARE să funcționeze — performanța e comparabilă "
                      "pe ambele subseturi (delta <5pp).")
        elif delta > 0.05:
            interp = (f"BIAS REZIDUAL posibil — modelul performează mai slab pe articolele "
                      f"cu termeni Moldova (delta {delta:+.4f}). Asta poate însemna că "
                      f"modelul încă exploatează parțial asocierea Moldova↔cls1, dar și "
                      f"că articolele cu Moldova sunt intrinsec mai grele.")
        else:
            interp = (f"Modelul e mai bun pe articolele cu Moldova (delta {delta:+.4f}) — "
                      f"surprinzător, posibil indică că subsetul Moldova are semnale "
                      f"mai clare sau e dominat de una din clase.")
        print(interp)
    else:
        interp = "Nu se poate calcula delta (un subset are o singură clasă)."
        print(interp)

    # === Save ===
    result = {
        "regex_moldova": MOLDOVA_REGEX.pattern,
        "metrics_cu_moldova": m_mold,
        "metrics_fara_moldova": m_nonmold,
        "interpretare": interp,
    }
    with open(out / "findings_moldova_stratified_v2.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # === Markdown findings ===
    md = [
        "# Findings — Evaluare stratificată Moldova v2",
        "",
        "Verifică eficacitatea entity balancing (D10) prin comparația performanței",
        "pe articole cu/fără termeni-Moldova.",
        "",
        "## 1. Regex termeni Moldova",
        "",
        f"```\n{MOLDOVA_REGEX.pattern}\n```",
        "",
        "## 2. Distribuție subseturi (TEST)",
        "",
        f"- Total TEST: {len(merged)}",
        f"- Cu Moldova: {m_mold['n']} (cls0={m_mold.get('n_cls0')}, cls1={m_mold.get('n_cls1')})",
        f"- Fără Moldova: {m_nonmold['n']} (cls0={m_nonmold.get('n_cls0')}, cls1={m_nonmold.get('n_cls1')})",
        "",
        "## 3. Metrici comparative",
        "",
        "| Subset | n | Accuracy | Macro-F1 | Recall cls0 | Recall cls1 |",
        "|--------|---|----------|----------|-------------|-------------|",
    ]
    for nume, m in [("Cu Moldova", m_mold), ("Fără Moldova", m_nonmold)]:
        f1 = f"{m['macro_f1']:.4f}" if m.get("macro_f1") else "N/A"
        r0 = f"{m['recall_cls0']:.4f}" if m.get("recall_cls0") is not None else "N/A"
        r1 = f"{m['recall_cls1']:.4f}" if m.get("recall_cls1") is not None else "N/A"
        md.append(f"| {nume} | {m['n']} | {m['accuracy']:.4f} | {f1} | {r0} | {r1} |")

    md.extend([
        "",
        "## 4. Interpretare",
        "",
        interp,
        "",
        "**Context v1:** în baseline v1, DF diff cls1-cls0 pentru termeni Moldova era +30.8pp.",
        "Entity balancing aplicat în v2 a redus-o la +16.1pp. Acest test măsoară efectul.",
    ])
    with open(out / "findings_moldova_stratified_v2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[OK] Salvat: {out / 'findings_moldova_stratified_v2.md'}")


if __name__ == "__main__":
    main()
