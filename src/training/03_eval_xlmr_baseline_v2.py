"""
Evaluare comprehensiva a modelului XLM-R baseline v2 pe VAL + TEST.

Outputs:
- Metrici globale (macro-F1, accuracy, precision, recall) pe VAL si TEST
- Confusion matrix pe TEST
- Breakdown per-source (digi24, g4media, veridica, stopfals — NOU v2)
- Analiza erorilor (FP + FN cu confidence + content)
- CSV cu predictii complete (id, sursa, label_true, label_pred, prob_cls1)

Usage:
    python src/training/03_eval_xlmr_baseline_v2.py \
        --model_dir models/xlmr_baseline_v2/final \
        --data_dir data/processed \
        --output_dir findings
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_dataframe(model, tokenizer, df: pd.DataFrame, device: str,
                      max_length: int = 256, batch_size: int = 16) -> pd.DataFrame:
    """Ruleaza predictii pe un DataFrame si intoarce DataFrame imbogatit cu preds + probs."""
    model.eval()
    all_preds = []
    all_probs = []
    texts = df["text"].tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].tolist())  # P(cls1)

    result = df.copy()
    result["pred"] = all_preds
    result["prob_cls1"] = all_probs
    result["correct"] = (result["pred"] == result["label_numeric"]).astype(int)
    return result


def compute_metrics_dict(y_true, y_pred) -> dict:
    """Calculeaza metrici complete."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "cls0": {"precision": float(p_per[0]), "recall": float(r_per[0]), "f1": float(f1_per[0])},
        "cls1": {"precision": float(p_per[1]), "recall": float(r_per[1]), "f1": float(f1_per[1])},
        "confusion_matrix": {
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="findings")
    parser.add_argument("--dataset_suffix", default="v2")
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
    print(f"[INFO] Încărcare model din: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    data_dir = Path(args.data_dir)
    val_df = pd.read_csv(data_dir / f"dataset_{args.dataset_suffix}_val.csv")
    test_df = pd.read_csv(data_dir / f"dataset_{args.dataset_suffix}_test.csv")

    # === Predict ===
    print(f"\n[INFO] Predicții pe VAL ({len(val_df)}) + TEST ({len(test_df)})...")
    val_preds = predict_dataframe(model, tokenizer, val_df, device, args.max_length)
    test_preds = predict_dataframe(model, tokenizer, test_df, device, args.max_length)

    # === Metrici globale ===
    val_metrics = compute_metrics_dict(val_preds["label_numeric"], val_preds["pred"])
    test_metrics = compute_metrics_dict(test_preds["label_numeric"], test_preds["pred"])

    print("\n=== VAL ===")
    print(f"  Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print("\n=== TEST ===")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    cm = test_metrics["confusion_matrix"]
    print(f"  Confusion: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")
    print(f"  Recall cls1: {test_metrics['cls1']['recall']:.4f}")

    # === Per-source breakdown (TEST) ===
    print("\n=== Per-source (TEST) ===")
    per_source = {}
    for sursa in sorted(test_preds["sursa_site"].unique()):
        sub = test_preds[test_preds["sursa_site"] == sursa]
        # Daca sursa are o singura clasa (cazul realist in v2), doar accuracy
        n = len(sub)
        acc = (sub["pred"] == sub["label_numeric"]).mean()
        majority_class = int(sub["label_numeric"].iloc[0]) if sub["label_numeric"].nunique() == 1 else -1
        per_source[sursa] = {
            "n": int(n),
            "accuracy": float(acc),
            "majority_class": majority_class,
        }
        print(f"  {sursa:15s} n={n:3d}  acc={acc:.4f}  (cls={majority_class})")

    # === Erori analiza ===
    print("\n=== Analiza erorilor TEST ===")
    errors = test_preds[test_preds["correct"] == 0].copy()
    errors["confidence"] = errors.apply(
        lambda r: r["prob_cls1"] if r["pred"] == 1 else 1 - r["prob_cls1"],
        axis=1,
    )
    print(f"Total erori: {len(errors)}/{len(test_preds)}")
    if len(errors) > 0:
        print("\nPrimele erori (sortate după confidence desc):")
        errors_sorted = errors.sort_values("confidence", ascending=False)
        for _, r in errors_sorted.head(10).iterrows():
            tip = "FP" if r["pred"] == 1 else "FN"
            titlu = str(r["titlu"])[:80]
            print(f"  [{tip}] conf={r['confidence']:.3f} | {r['sursa_site']} | {titlu}")

    # === Stabilitate val↔test ===
    delta_f1 = test_metrics["macro_f1"] - val_metrics["macro_f1"]
    print(f"\n=== Stabilitate ===")
    print(f"  VAL macro_f1: {val_metrics['macro_f1']:.4f}")
    print(f"  TEST macro_f1: {test_metrics['macro_f1']:.4f}")
    print(f"  Delta: {delta_f1:+.4f} pp " +
          ("(stabil)" if abs(delta_f1) < 0.02 else "(ATENȚIE: diferență notabilă)"))

    # === Save ===
    metrics_out = {
        "dataset_version": "v2",
        "model_dir": str(args.model_dir),
        "val": val_metrics,
        "test": test_metrics,
        "per_source_test": per_source,
        "n_errors_test": int(len(errors)),
        "delta_val_test_f1": float(delta_f1),
    }
    with open(out / "metrics_baseline_v2.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Metrici: {out / 'metrics_baseline_v2.json'}")

    # CSV predictii — coloane utile pentru LIME + audit ulterior
    cols_save = ["id", "sursa_site", "an", "titlu", "label_numeric",
                 "pred", "prob_cls1", "correct"]
    test_preds[cols_save].to_csv(out / "test_predictions_v2.csv", index=False)
    print(f"[OK] Predicții: {out / 'test_predictions_v2.csv'}")

    # === Markdown findings ===
    md_lines = [
        "# Findings — Baseline XLM-RoBERTa v2",
        "",
        f"**Dataset:** v2 (1483 articole, cu 2022 + Stopfals + entity balancing Moldova)",
        f"**Model:** {args.model_dir}",
        "",
        "## 1. Metrici globale",
        "",
        "### VAL (load_best_model_at_end=True)",
        f"- Macro-F1: **{val_metrics['macro_f1']:.4f}**",
        f"- Accuracy: {val_metrics['accuracy']:.4f}",
        f"- Recall cls1 (dezinformare): {val_metrics['cls1']['recall']:.4f}",
        "",
        "### TEST (gold, neatins la model selection)",
        f"- Macro-F1: **{test_metrics['macro_f1']:.4f}**",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Recall cls1: {test_metrics['cls1']['recall']:.4f}",
        "",
        "**Confusion matrix TEST:**",
        "",
        "|              | pred_0 | pred_1 |",
        "|--------------|--------|--------|",
        f"| true_0 (cred) | {cm['TN']} | {cm['FP']} |",
        f"| true_1 (dez)  | {cm['FN']} | {cm['TP']} |",
        "",
        f"**Stabilitate VAL → TEST:** delta macro_f1 = {delta_f1:+.4f}",
        "",
        "## 2. Breakdown per-source (TEST)",
        "",
        "| Sursa | n | Accuracy | Clasă |",
        "|-------|---|----------|-------|",
    ]
    for sursa, s in per_source.items():
        md_lines.append(f"| {sursa} | {s['n']} | {s['accuracy']:.4f} | cls{s['majority_class']} |")
    md_lines.extend([
        "",
        "## 3. Comparație cu baseline v1",
        "",
        "| Metric | v1 (1427 art) | v2 (1483 art) | Delta |",
        "|--------|---------------|---------------|-------|",
        f"| VAL macro-F1 | 0.9844 | {val_metrics['macro_f1']:.4f} | {val_metrics['macro_f1']-0.9844:+.4f} |",
        f"| TEST macro-F1 | 0.9897 | {test_metrics['macro_f1']:.4f} | {test_metrics['macro_f1']-0.9897:+.4f} |",
        f"| TEST recall cls1 | 1.0000 | {test_metrics['cls1']['recall']:.4f} | {test_metrics['cls1']['recall']-1.0000:+.4f} |",
        "",
        "**Interpretare:** o scădere ușoară a metricilor este de așteptat și DORITĂ dacă",
        "entity balancing Moldova a atenuat un shortcut. Scădere >3pp ar indica probleme.",
        "",
        f"## 4. Erori pe TEST (total: {len(errors)})",
        "",
    ])
    if len(errors) > 0:
        md_lines.append("| Tip | Confidence | Sursa | Titlu |")
        md_lines.append("|-----|------------|-------|-------|")
        for _, r in errors.sort_values("confidence", ascending=False).head(15).iterrows():
            tip = "FP" if r["pred"] == 1 else "FN"
            titlu = str(r["titlu"])[:90].replace("|", "\\|")
            md_lines.append(f"| {tip} | {r['confidence']:.3f} | {r['sursa_site']} | {titlu} |")
    else:
        md_lines.append("_Zero erori pe test set._")

    with open(out / "findings_baseline_xlmr_v2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] Findings: {out / 'findings_baseline_xlmr_v2.md'}")


if __name__ == "__main__":
    main()
