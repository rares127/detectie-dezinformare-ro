"""
LOSO (Leave-One-Source-Out) experiment — test critic pentru discriminare
shortcut-de-sursa vs detectare-genuina-de-dezinformare.

Context metodologic:
- In v1, recall cls1 a fost 100% pe Veridica, dar LIME fidelity 0.04-0.09.
- Ipoteza (a): modelul a invatat „voce fact-checking Veridica", NU dezinformarea.
- Ipoteza (b): LIME e intrinsec limitat pe transformere cu conf >0.99.

LOSO discrimina: daca (a), recall pe Veridica scade DRAMATIC cand Veridica
e exclusa din antrenare; daca (b), recall ramane ridicat.

Doua rulari:
- LOSO-V: train pe digi24 + g4media + stopfals → test DOAR pe veridica (661)
- LOSO-S: train pe digi24 + g4media + veridica → test DOAR pe stopfals (85)

Stopfals e mic (85 articole), deci LOSO-S da metrici zgomotoase dar e
simetric util.

Arhitectura:
- Re-folosim TOT datasetul v2 (nu split-urile), apoi refacem un split intern
  70/15/15 pe TRAIN_SOURCES doar pentru validare early stopping.
- TEST = TOATE articolele din held-out source.

Interpretare (prag decis in handoff):
- Recall cls1 drop < 10pp → model invata genuin dezinformarea
- Drop 10-30pp → mix dezinformare + stil sursei
- Drop > 30pp → predominant stil-de-sursa (shortcut confirmat)

Usage:
    python src/training/05_loso_experiment.py \
        --dataset data/raw/dataset_licenta_complet.csv \
        --held_out_source veridica.ro \
        --output_dir models/loso_v \
        --findings_path findings/findings_loso_v_v2.md

    python src/training/05_loso_experiment.py \
        --dataset data/raw/dataset_licenta_complet.csv \
        --held_out_source stopfals.md \
        --output_dir models/loso_s \
        --findings_path findings/findings_loso_s_v2.md
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "macro_f1": f1, "macro_precision": p, "macro_recall": r}


def construieste_text(row):
    return f"{str(row['titlu']).strip()}\n\n{str(row['stire_citata']).strip()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Path către dataset_licenta_complet.csv")
    parser.add_argument("--held_out_source", required=True,
                        choices=["veridica.ro", "stopfals.md"],
                        help="Sursa exclusă din train, folosită ca test")
    parser.add_argument("--output_dir", required=True,
                        help="Director salvare model LOSO")
    parser.add_argument("--findings_path", required=True,
                        help="Path markdown findings")
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Fracția din TRAIN_SOURCES folosită ca val intern")
    args = parser.parse_args()

    set_seed(args.seed)

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
    print(f"[INFO] LOSO — held_out: {args.held_out_source}")

    # === Load dataset ===
    df = pd.read_csv(args.dataset)
    df["text"] = df.apply(construieste_text, axis=1)

    # Split pe sursa held-out
    test_df = df[df["sursa_site"] == args.held_out_source].reset_index(drop=True)
    train_all = df[df["sursa_site"] != args.held_out_source].reset_index(drop=True)

    print(f"[INFO] TEST (held-out {args.held_out_source}): {len(test_df)}")
    print(f"[INFO] TRAIN_SOURCES total: {len(train_all)}")
    print(f"[INFO] TRAIN_SOURCES distribuție:")
    print(train_all["sursa_site"].value_counts().to_string())
    print(f"[INFO] TRAIN_SOURCES distribuție clase:")
    print(train_all["label_numeric"].value_counts().to_string())

    # Verificare echilibru clase in TRAIN_SOURCES
    if train_all["label_numeric"].nunique() < 2:
        raise ValueError(f"TRAIN_SOURCES are o singură clasă după excludere "
                         f"{args.held_out_source}. Experimentul nu e valid.")

    # Split intern TRAIN_SOURCES in train/val pentru early stopping
    # Stratificam pe (label, sursa) din TRAIN_SOURCES
    train_all["strat_key"] = (train_all["label_numeric"].astype(str) + "_"
                               + train_all["sursa_site"])
    train_df, val_df = train_test_split(
        train_all,
        test_size=args.val_frac,
        stratify=train_all["strat_key"],
        random_state=args.seed,
    )
    print(f"\n[INFO] TRAIN intern: {len(train_df)} | VAL intern: {len(val_df)}")

    # === Class weights ===
    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes,
                               y=train_df["label_numeric"].values)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    print(f"[INFO] Class weights: cls0={cw[0]:.4f}, cls1={cw[1]:.4f}")

    # === Tokenizer + Model ===
    print(f"[INFO] Încărcare {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "stire_credibila", 1: "dezinformare_pro_rusa"},
        label2id={"stire_credibila": 0, "dezinformare_pro_rusa": 1},
    )

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length",
                         truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label_numeric"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label_numeric"]])
    train_ds = train_ds.rename_column("label_numeric", "labels")
    val_ds = val_ds.rename_column("label_numeric", "labels")
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # === Training ===
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=False,
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("[INFO] Start antrenare LOSO...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[INFO] Training terminat în {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # === Evaluare pe TEST (held-out source) ===
    print(f"\n[INFO] Evaluare pe held-out source: {args.held_out_source}")
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(test_df), args.batch_size):
            batch_texts = test_df["text"].iloc[i:i + args.batch_size].tolist()
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(probs.argmax(axis=-1).tolist())
            all_probs.extend(probs[:, 1].tolist())

    test_df["pred"] = all_preds
    test_df["prob_cls1"] = all_probs

    y_true = test_df["label_numeric"].values
    y_pred = test_df["pred"].values

    # Toate articolele held-out sunt din aceeasi clasa (cls1 pt veridica/stopfals)
    majority_class = int(y_true[0])
    assert (y_true == majority_class).all(), (
        f"TEST held-out mixat? {pd.Series(y_true).value_counts()}"
    )

    acc = accuracy_score(y_true, y_pred)
    recall_cls1 = float((y_pred == 1).sum() / max(len(y_pred), 1)) if majority_class == 1 \
        else float("nan")

    # Confusion + probabilitati
    n_correct = int((y_pred == y_true).sum())
    n_wrong = int((y_pred != y_true).sum())
    mean_prob_cls1 = float(np.mean(all_probs))
    median_prob_cls1 = float(np.median(all_probs))

    # === Interpretare drop ===
    # Baseline v2 TEST recall cls1 — citim din metrics_baseline_v2.json daca exista
    baseline_recall_cls1 = None
    metrics_baseline_path = Path("findings/metrics_baseline_v2.json")
    if metrics_baseline_path.exists():
        with open(metrics_baseline_path, encoding="utf-8") as f:
            mb = json.load(f)
        baseline_recall_cls1 = mb.get("test", {}).get("cls1", {}).get("recall")

    print(f"\n=== LOSO Results ({args.held_out_source}) ===")
    print(f"  n={len(test_df)}")
    print(f"  Accuracy (vs majority cls{majority_class}): {acc:.4f}")
    if majority_class == 1:
        print(f"  Recall cls1: {recall_cls1:.4f}")
    print(f"  Mean prob_cls1: {mean_prob_cls1:.4f}")
    print(f"  Median prob_cls1: {median_prob_cls1:.4f}")

    drop_interp = "N/A (baseline necunoscut)"
    drop_value = None
    if baseline_recall_cls1 is not None and majority_class == 1:
        drop_value = baseline_recall_cls1 - recall_cls1
        print(f"\n  Baseline v2 test recall cls1: {baseline_recall_cls1:.4f}")
        print(f"  LOSO recall cls1: {recall_cls1:.4f}")
        print(f"  DROP: {drop_value:+.4f} ({drop_value*100:+.2f}pp)")
        if abs(drop_value) < 0.10:
            drop_interp = (f"Drop <10pp — modelul învață GENUIN semnalul de dezinformare, "
                           f"nu depinde critic de sursa {args.held_out_source}.")
        elif abs(drop_value) < 0.30:
            drop_interp = (f"Drop 10-30pp — modelul învață un MIX de dezinformare și "
                           f"stil al sursei {args.held_out_source}.")
        else:
            drop_interp = (f"Drop >30pp — SHORTCUT DE SURSĂ CONFIRMAT. Modelul învață "
                           f"predominant stilul {args.held_out_source}, nu dezinformarea.")
        print(f"\n  Interpretare: {drop_interp}")

    # === Save ===
    # Model
    final_dir = out / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[OK] Model LOSO salvat: {final_dir}")

    # JSON rezultate
    result = {
        "experiment": f"LOSO — held_out={args.held_out_source}",
        "dataset_version": "v2",
        "held_out_source": args.held_out_source,
        "n_test": len(test_df),
        "majority_class_test": majority_class,
        "n_train_sources": len(train_df),
        "n_val_sources": len(val_df),
        "train_sources_distributie": train_df["sursa_site"].value_counts().to_dict(),
        "accuracy_test": float(acc),
        "recall_cls1_test": recall_cls1 if majority_class == 1 else None,
        "mean_prob_cls1": mean_prob_cls1,
        "median_prob_cls1": median_prob_cls1,
        "baseline_recall_cls1_v2": baseline_recall_cls1,
        "drop_vs_baseline": drop_value,
        "interpretare": drop_interp,
        "training_time_seconds": elapsed,
    }
    json_path = Path(args.findings_path).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] Rezultate JSON: {json_path}")

    # CSV predictii complete — util pentru analiza ulterioara
    csv_path = Path(args.findings_path).with_name(
        Path(args.findings_path).stem + "_predictions.csv"
    )
    test_df[["id", "sursa_site", "an", "titlu", "label_numeric", "pred", "prob_cls1"]].to_csv(
        csv_path, index=False
    )
    print(f"[OK] Predicții CSV: {csv_path}")

    # Markdown findings
    md = [
        f"# Findings — LOSO experiment (held-out: {args.held_out_source})",
        "",
        "## Context metodologic",
        "",
        "Scop: discrimina între:",
        "- **(a)** model cu shortcut de sursă stilistic",
        "- **(b)** model care învață genuin dezinformarea",
        "",
        f"Train: toate sursele EXCEPT `{args.held_out_source}` → Test: TOT `{args.held_out_source}`",
        "",
        "## Setup",
        "",
        f"- **n_train (intern)**: {len(train_df)}",
        f"- **n_val (intern, early stopping)**: {len(val_df)}",
        f"- **n_test (held-out)**: {len(test_df)}",
        "",
        "**Distribuție surse TRAIN:**",
        "",
        "| Sursa | n |",
        "|-------|---|",
    ]
    for s, n in train_df["sursa_site"].value_counts().items():
        md.append(f"| {s} | {n} |")
    md.extend([
        "",
        "## Rezultate",
        "",
        f"- **Accuracy**: {acc:.4f}",
        f"- **Recall cls{majority_class}**: {recall_cls1:.4f}" if majority_class == 1 else f"- Majority class: {majority_class}",
        f"- **Mean prob_cls1**: {mean_prob_cls1:.4f}",
        f"- **Median prob_cls1**: {median_prob_cls1:.4f}",
        f"- **Corecte**: {n_correct}/{len(test_df)}",
        f"- **Greșite**: {n_wrong}/{len(test_df)}",
        "",
        "## Comparație cu baseline v2",
        "",
    ])
    if baseline_recall_cls1 is not None:
        md.extend([
            f"- Baseline v2 (test full) recall cls1: **{baseline_recall_cls1:.4f}**",
            f"- LOSO recall cls1: **{recall_cls1:.4f}**",
            f"- **Drop**: {drop_value:+.4f} ({drop_value*100:+.2f}pp)",
            "",
            f"### Interpretare",
            "",
            drop_interp,
            "",
            "### Praguri (din handoff, Secțiunea 6.4):",
            "- Drop <10pp → model învață genuin",
            "- Drop 10-30pp → model învață mix dezinformare + stil sursei",
            "- Drop >30pp → model învață predominant stil sursei",
        ])
    else:
        md.append("_Baseline v2 rulat mai târziu — comparația se face manual._")

    md.extend([
        "",
        "## Timp antrenare",
        "",
        f"{elapsed:.1f}s ({elapsed/60:.1f} min)",
    ])

    with open(args.findings_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[OK] Findings markdown: {args.findings_path}")


if __name__ == "__main__":
    main()
