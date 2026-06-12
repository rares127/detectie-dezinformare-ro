"""
Proba D2: Ablatie caractere chirilice.

Ipoteza H1: modelul LOSO-V foloseste prezenta alfabetului chirilic ca
proxy pentru cls1. Articolele Veridica citeaza frecvent surse rusesti
in original (chirilica), articolele credibile si Stopfals (transliterat)
nu au la fel de multa chirilica.

Test: pentru fiecare articol Veridica din LOSO-V test set:
1. Masoara cate caractere chirilice sunt in text
2. Predict cu text original → prob_cls1_original
3. Sterge toata chirilica → predict ablated → prob_cls1_ablated
4. Delta = prob_cls1_original - prob_cls1_ablated

Interpretare:
- Delta > 0.20 pe multe articole → H1 confirmata, chirilica e drive major
- Delta ~ 0 → chirilica nu conteaza
- Delta < 0 → chirilica impinge spre cls0 (improbabil dar posibil)

Ruleaza pe modelul LOSO-V (unde shortcut-ul e activ).

Usage:
    python src/training/D2_ablatie_chirilica.py \\
        --model_dir models/loso_v/final \\
        --dataset data/raw/dataset_licenta_complet.csv \\
        --output_dir findings
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Regex alfabet chirilic (inclusiv caractere extinse: ucraineana, sarba, etc.)
CHIRILIC_REGEX = re.compile(r"[\u0400-\u04FF\u0500-\u052F]+")


def construieste_text(row):
    """Same logic ca in scripturile de training."""
    return f"{str(row['titlu']).strip()}\n\n{str(row['stire_citata']).strip()}"


def predict_probs(model, tokenizer, texts, device, max_length=256, batch_size=16):
    """Intoarce array P(cls1) pentru lista de texte."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs[:, 1].tolist())
    return np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="models/loso_v/final (modelul unde shortcut-ul e activ)")
    parser.add_argument("--dataset", required=True,
                        help="Dataset complet pentru filtrare Veridica")
    parser.add_argument("--output_dir", default="findings")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === Device ===
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[INFO] Device: {device}")

    # === Load ===
    print(f"[INFO] Model: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    # === Date: TOATE articolele Veridica (test set LOSO-V) ===
    df = pd.read_csv(args.dataset)
    df["text"] = df.apply(construieste_text, axis=1)
    veridica = df[df["sursa_site"] == "veridica.ro"].reset_index(drop=True)
    print(f"[INFO] Articole Veridica: {len(veridica)}")

    # === Contorizare chirilica per articol ===
    veridica["n_chirilic_chars"] = veridica["text"].apply(
        lambda t: sum(len(m) for m in CHIRILIC_REGEX.findall(str(t)))
    )
    veridica["pct_chirilic"] = veridica["n_chirilic_chars"] / veridica["text"].str.len() * 100

    print(f"\n[INFO] Statistici chirilică în text:")
    print(f"  Articole cu chirilică: {(veridica['n_chirilic_chars'] > 0).sum()}/{len(veridica)}")
    print(f"  Mediana caractere chirilice: {veridica['n_chirilic_chars'].median():.0f}")
    print(f"  Max caractere chirilice: {veridica['n_chirilic_chars'].max()}")
    print(f"  Mediana % chirilic: {veridica['pct_chirilic'].median():.2f}%")

    # === Predict text original ===
    print("\n[INFO] Predict text ORIGINAL...")
    texts_orig = veridica["text"].tolist()
    probs_orig = predict_probs(model, tokenizer, texts_orig, device, args.max_length)
    veridica["prob_cls1_orig"] = probs_orig
    veridica["pred_orig"] = (probs_orig > 0.5).astype(int)

    # === Predict text fara chirilica ===
    print("[INFO] Predict text ABLATED (fără chirilică)...")
    veridica["text_no_cyrillic"] = veridica["text"].apply(
        lambda t: CHIRILIC_REGEX.sub(" ", str(t))
    )
    texts_ablated = veridica["text_no_cyrillic"].tolist()
    probs_ablated = predict_probs(model, tokenizer, texts_ablated, device, args.max_length)
    veridica["prob_cls1_ablated"] = probs_ablated
    veridica["pred_ablated"] = (probs_ablated > 0.5).astype(int)

    # === Delta ===
    veridica["delta_prob_cls1"] = veridica["prob_cls1_orig"] - veridica["prob_cls1_ablated"]
    veridica["flip"] = (veridica["pred_orig"] != veridica["pred_ablated"]).astype(int)

    # === Analize ===
    # Pe TOATE articolele
    print("\n=== Rezultate pe TOATE articolele Veridica ===")
    print(f"  Prob cls1 (orig)    mean: {veridica['prob_cls1_orig'].mean():.4f}, "
          f"median: {veridica['prob_cls1_orig'].median():.4f}")
    print(f"  Prob cls1 (ablated) mean: {veridica['prob_cls1_ablated'].mean():.4f}, "
          f"median: {veridica['prob_cls1_ablated'].median():.4f}")
    print(f"  Delta               mean: {veridica['delta_prob_cls1'].mean():+.4f}, "
          f"median: {veridica['delta_prob_cls1'].median():+.4f}")
    print(f"  Flips: {veridica['flip'].sum()}/{len(veridica)} "
          f"({100*veridica['flip'].mean():.1f}%)")

    # Breakdown: articole cu chirilica vs fara
    cu_chir = veridica[veridica["n_chirilic_chars"] > 0]
    fara_chir = veridica[veridica["n_chirilic_chars"] == 0]

    print(f"\n=== Breakdown pe prezența chirilicei în original ===")
    print(f"  Articole CU chirilică ({len(cu_chir)}):")
    print(f"    Prob cls1 orig mean: {cu_chir['prob_cls1_orig'].mean():.4f}")
    print(f"    Prob cls1 ablated mean: {cu_chir['prob_cls1_ablated'].mean():.4f}")
    print(f"    Delta mean: {cu_chir['delta_prob_cls1'].mean():+.4f}")
    print(f"    Flips: {cu_chir['flip'].sum()}/{len(cu_chir)} "
          f"({100*cu_chir['flip'].mean():.1f}%)")
    print(f"    Recall cls1 orig: {cu_chir['pred_orig'].mean():.4f}")
    print(f"    Recall cls1 ablated: {cu_chir['pred_ablated'].mean():.4f}")
    if len(fara_chir) > 0:
        print(f"  Articole FĂRĂ chirilică ({len(fara_chir)}):")
        print(f"    Prob cls1 orig mean: {fara_chir['prob_cls1_orig'].mean():.4f}")
        print(f"    Prob cls1 ablated mean: {fara_chir['prob_cls1_ablated'].mean():.4f}")
        print(f"    Delta mean: {fara_chir['delta_prob_cls1'].mean():+.4f}")
        print(f"    (Ar trebui ~0, e control)")
        print(f"    Recall cls1 orig: {fara_chir['pred_orig'].mean():.4f}")

    # === Corelatie cantitativa chirilica ↔ delta ===
    if len(cu_chir) > 10:
        corr = cu_chir[["n_chirilic_chars", "delta_prob_cls1"]].corr().iloc[0, 1]
        print(f"\n  Corelația (cu chirilică): n_chirilic_chars vs delta_prob_cls1 = {corr:+.4f}")

    # === Interpretare ===
    delta_mean_cu_chir = cu_chir['delta_prob_cls1'].mean() if len(cu_chir) > 0 else 0
    flip_rate_cu_chir = cu_chir['flip'].mean() if len(cu_chir) > 0 else 0

    if delta_mean_cu_chir > 0.20 or flip_rate_cu_chir > 0.20:
        interp = ("H1 CONFIRMATĂ PARȚIAL: ștergerea chirilicei scade semnificativ P(cls1) "
                  "pe articolele care o conțin. Modelul folosește chirilica drept proxy "
                  "pentru cls1. Atenuare posibilă: transliterație chirilică→latină în "
                  "preprocessing, sau eliminare fragmente chirilice din stire_citata.")
    elif delta_mean_cu_chir > 0.10:
        interp = ("H1 CONFIRMATĂ SLAB: delta moderat, chirilica contribuie dar nu e "
                  "singurul factor. Combinat cu H2 (entități) explică probabil o parte "
                  "semnificativă din shortcut.")
    else:
        interp = ("H1 RESPINSĂ: ștergerea chirilicei nu modifică semnificativ predicțiile. "
                  "Shortcut-ul vine din altă parte (H2 entități, H3 stil, H4 sufixe).")

    print(f"\n=== Interpretare ===\n{interp}")

    # === Save ===
    # CSV detaliat pentru audit
    cols = ["id", "titlu", "an", "n_chirilic_chars", "pct_chirilic",
            "prob_cls1_orig", "prob_cls1_ablated", "delta_prob_cls1",
            "pred_orig", "pred_ablated", "flip"]
    veridica[cols].to_csv(out / "d2_ablatie_chirilica_per_articol.csv", index=False)

    result = {
        "model_used": args.model_dir,
        "regex_chirilic": CHIRILIC_REGEX.pattern,
        "n_articole": int(len(veridica)),
        "n_cu_chirilica": int((veridica["n_chirilic_chars"] > 0).sum()),
        "n_fara_chirilica": int((veridica["n_chirilic_chars"] == 0).sum()),
        "global": {
            "prob_cls1_orig_mean": float(veridica["prob_cls1_orig"].mean()),
            "prob_cls1_ablated_mean": float(veridica["prob_cls1_ablated"].mean()),
            "delta_mean": float(veridica["delta_prob_cls1"].mean()),
            "delta_median": float(veridica["delta_prob_cls1"].median()),
            "flip_rate": float(veridica["flip"].mean()),
            "recall_cls1_orig": float(veridica["pred_orig"].mean()),
            "recall_cls1_ablated": float(veridica["pred_ablated"].mean()),
        },
        "cu_chirilica": {
            "n": int(len(cu_chir)),
            "prob_cls1_orig_mean": float(cu_chir["prob_cls1_orig"].mean()) if len(cu_chir) > 0 else None,
            "prob_cls1_ablated_mean": float(cu_chir["prob_cls1_ablated"].mean()) if len(cu_chir) > 0 else None,
            "delta_mean": float(cu_chir["delta_prob_cls1"].mean()) if len(cu_chir) > 0 else None,
            "flip_rate": float(cu_chir["flip"].mean()) if len(cu_chir) > 0 else None,
        },
        "interpretare": interp,
    }
    with open(out / "d2_ablatie_chirilica.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Rezultate JSON: {out / 'd2_ablatie_chirilica.json'}")
    print(f"[OK] CSV per articol: {out / 'd2_ablatie_chirilica_per_articol.csv'}")

    # === Markdown ===
    md = [
        "# Proba D2 — Ablație caractere chirilice",
        "",
        "## Ipoteza H1",
        "",
        "Modelul LOSO-V folosește prezența alfabetului chirilic ca proxy pentru cls1.",
        "Veridica citează frecvent surse rusești în original; articolele credibile și",
        "Stopfals (transliterat) au mai puțină chirilică.",
        "",
        "## Metodologie",
        "",
        f"- **Model**: `{args.model_dir}` (modelul LOSO-V cu shortcut activ)",
        f"- **Test set**: {len(veridica)} articole Veridica",
        f"- **Ablație**: regex `{CHIRILIC_REGEX.pattern}` înlocuit cu spațiu",
        "- **Măsură**: Δ P(cls1) = orig − ablated",
        "",
        "## Distribuția chirilicei",
        "",
        f"- Articole cu chirilică: {int((veridica['n_chirilic_chars'] > 0).sum())}/{len(veridica)}",
        f"- Mediana caractere chirilice: {veridica['n_chirilic_chars'].median():.0f}",
        f"- Max: {veridica['n_chirilic_chars'].max()}",
        "",
        "## Rezultate globale",
        "",
        "| Măsură | Original | Ablated | Delta |",
        "|--------|----------|---------|-------|",
        f"| P(cls1) mean | {veridica['prob_cls1_orig'].mean():.4f} | "
        f"{veridica['prob_cls1_ablated'].mean():.4f} | "
        f"{veridica['delta_prob_cls1'].mean():+.4f} |",
        f"| P(cls1) median | {veridica['prob_cls1_orig'].median():.4f} | "
        f"{veridica['prob_cls1_ablated'].median():.4f} | "
        f"{veridica['delta_prob_cls1'].median():+.4f} |",
        f"| Recall cls1 | {veridica['pred_orig'].mean():.4f} | "
        f"{veridica['pred_ablated'].mean():.4f} | "
        f"{veridica['pred_orig'].mean() - veridica['pred_ablated'].mean():+.4f} |",
        "",
        f"**Flips de predicție**: {int(veridica['flip'].sum())}/{len(veridica)} "
        f"({100*veridica['flip'].mean():.1f}%)",
        "",
        "## Breakdown pe prezența chirilicei",
        "",
        "| Subset | N | P(cls1) orig | P(cls1) ablated | Delta | Flip rate |",
        "|--------|---|--------------|-----------------|-------|-----------|",
    ]
    for nume, sub in [("Cu chirilică", cu_chir), ("Fără chirilică (control)", fara_chir)]:
        if len(sub) == 0:
            continue
        md.append(f"| {nume} | {len(sub)} | {sub['prob_cls1_orig'].mean():.4f} | "
                  f"{sub['prob_cls1_ablated'].mean():.4f} | "
                  f"{sub['delta_prob_cls1'].mean():+.4f} | "
                  f"{100*sub['flip'].mean():.1f}% |")

    md.extend([
        "",
        "## Interpretare",
        "",
        interp,
        "",
        "## Audit",
        "",
        f"CSV cu date per-articol: `d2_ablatie_chirilica_per_articol.csv`",
    ])
    with open(out / "findings_d2_ablatie_chirilica.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[OK] Findings: {out / 'findings_d2_ablatie_chirilica.md'}")


if __name__ == "__main__":
    main()
