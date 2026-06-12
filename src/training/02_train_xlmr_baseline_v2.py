"""
Fine-tuning XLM-RoBERTa-base pe datasetul v2.

Configuratie identica cu v1 (pentru comparabilitate directa):
- xlm-roberta-base, max_length=256, batch_size=16, lr=2e-5, epochs=3
- CrossEntropyLoss cu class_weight balanced (WeightedTrainer custom)
- Device: MPS (M2 Pro) auto-detect; fallback CPU
- Seed: 42
- load_best_model_at_end=True pe macro_f1 (VAL)
- EarlyStoppingCallback(patience=2) — la 3 epoci nu se declanseaza practic,
  dar protejeaza re-rularile cu --epochs mai mare (adaugat la audit, iunie 2026)

Telemetrie trunchiere (adaugata la audit): raportam cate articole din TRAIN/VAL
depasesc max_length si ar fi trunchiate — cifra utila pentru Cap. 3 al tezei.

Fix-uri tehnice pastrate din v1:
- accelerate>=1.1.0
- processing_class=tokenizer (NU tokenizer=tokenizer — deprecated)

Usage:
    python src/training/02_train_xlmr_baseline_v2.py \
        --data_dir data/processed \
        --output_dir models/xlmr_baseline_v2
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


# === Reproducibilitate ===
def set_seed(seed: int = 42):
    """Seteaza seed-urile pentru reproducibilitate completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS nu are API public pentru seed, dar torch.manual_seed acopera


# === WeightedTrainer ===
class WeightedTrainer(Trainer):
    """Trainer cu class_weight balanced pentru CrossEntropyLoss.

    Necesar pentru a compensa dezechilibrul cls0/cls1 (737/746 in v2,
    aproape balansat, dar pastram pentru comparabilitate cu v1).
    """

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


# === Metrici ===
def compute_metrics(eval_pred):
    """Macro-F1 + accuracy + precision/recall per-clasa."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": p,
        "macro_recall": r,
        "f1_cls0": f1_per[0],
        "f1_cls1": f1_per[1],
        "recall_cls0": r_per[0],
        "recall_cls1": r_per[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="models/xlmr_baseline_v2")
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_suffix", default="v2",
                        help="Sufixul fișierelor dataset (ex: v2 → dataset_v2_train.csv)")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Device detection ===
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[INFO] Device: {device}")

    # === Incarcare date ===
    train_df = pd.read_csv(data_dir / f"dataset_{args.dataset_suffix}_train.csv")
    val_df = pd.read_csv(data_dir / f"dataset_{args.dataset_suffix}_val.csv")
    print(f"[INFO] TRAIN: {len(train_df)} | VAL: {len(val_df)}")

    # === Class weights (balanced) ===
    classes = np.array([0, 1])
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label_numeric"].values,
    )
    class_weights = torch.tensor(cw, dtype=torch.float32)
    print(f"[INFO] Class weights: cls0={cw[0]:.4f}, cls1={cw[1]:.4f}")

    # === Tokenizer + Model ===
    print(f"[INFO] Încărcare model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "stire_credibila", 1: "dezinformare_pro_rusa"},
        label2id={"stire_credibila": 0, "dezinformare_pro_rusa": 1},
    )

    # === Telemetrie trunchiere ===
    # Cate articole depasesc max_length (si deci sunt trunchiate la antrenare).
    def raporteaza_trunchiere(df: pd.DataFrame, nume: str) -> dict:
        """Numara articolele cu n_tokens > max_length si raporteaza procentul."""
        ids = tokenizer(df["text"].tolist(), truncation=False)["input_ids"]
        n_peste = sum(1 for t in ids if len(t) > args.max_length)
        pct = 100.0 * n_peste / max(len(ids), 1)
        print(f"[INFO] Trunchiere {nume}: {n_peste}/{len(ids)} articole "
              f"({pct:.1f}%) depășesc max_length={args.max_length}")
        return {"n_peste_max_length": n_peste, "n_total": len(ids),
                "procent": round(pct, 2)}

    trunchiere = {
        "train": raporteaza_trunchiere(train_df, "TRAIN"),
        "val": raporteaza_trunchiere(val_df, "VAL"),
    }

    # === Tokenizare ===
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label_numeric"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label_numeric"]])

    train_ds = train_ds.rename_column("label_numeric", "labels")
    val_ds = val_ds.rename_column("label_numeric", "labels")

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # === Training args ===
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        # MPS-specific: fp16 NU functioneaza pe MPS → lasam default fp32
        fp16=False,
        dataloader_num_workers=0,  # MPS pe macOS prefera 0
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  # NU tokenizer=... (deprecated in transformers recent)
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        # Opreste antrenarea daca macro_f1 pe val nu creste 2 epoci la rand.
        # La 3 epoci e practic inert; protejeaza rularile cu --epochs mare.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # === Antrenare ===
    print("[INFO] Start antrenare...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[INFO] Training terminat în {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # === Evaluare finala pe VAL ===
    print("\n[INFO] Evaluare finală pe VAL (best checkpoint):")
    val_metrics = trainer.evaluate()
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # === Salvare model ===
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[OK] Model salvat în: {final_dir}")

    # === Salvare metrici VAL ===
    training_info = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": device,
        "training_time_seconds": elapsed,
        "val_metrics_best_checkpoint": {k: float(v) for k, v in val_metrics.items()
                                         if isinstance(v, (int, float))},
        "class_weights": {"cls0": float(cw[0]), "cls1": float(cw[1])},
        "n_train": len(train_df),
        "n_val": len(val_df),
        "trunchiere": trunchiere,
    }
    with open(output_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    print(f"[OK] Training info salvat.")


if __name__ == "__main__":
    main()
