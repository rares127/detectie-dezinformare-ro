"""
Integrated Gradients (captum) — XAI complementar LIME pentru clasa 1.

Se ruleaza DOAR daca LIME fidelity ramane scazut pe Veridica in v2 (< 0.25).
In caz contrar, LIME e suficient si IG adauga doar complexitate.

Motivatie (din handoff 6.6): IG e metoda standard pentru transformere in
literatura academica. Functioneaza pe modele high-confidence pentru ca
foloseste gradienti raw, nu local-surrogate ca LIME.

Config:
- Steps IG: 50 (standard)
- Baseline: embedding PAD (alternativa: embedding zero)
- Aggregation: sum per-token (dupa normalizare)

Output: top-15 token-uri per exemplu + vizualizare HTML inline color-coded.

Usage:
    python src/training/07_integrated_gradients.py \
        --model_dir models/xlmr_baseline_v2/final \
        --predictions findings/test_predictions_v2.csv \
        --test_data data/processed/dataset_v2_test.csv \
        --output_dir findings
"""
import argparse
import html as html_lib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_dir", default="findings")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n_exemple", type=int, default=6,
                        help="Nr exemple pe clasa 1 (Veridica/Stopfals)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    html_dir = out / "ig_html"
    html_dir.mkdir(parents=True, exist_ok=True)

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
    model.eval()

    test_df = pd.read_csv(args.test_data)
    preds_df = pd.read_csv(args.predictions)
    merged = preds_df.merge(test_df[["id", "text"]], on="id", how="left")

    # Selectam TP pe clasa 1 din Veridica + Stopfals
    tp_cls1 = merged[(merged["correct"] == 1) & (merged["label_numeric"] == 1)].copy()
    tp_cls1["conf"] = tp_cls1["prob_cls1"]

    exemple = []
    for sursa in ["veridica.ro", "stopfals.md"]:
        sub = tp_cls1[tp_cls1["sursa_site"] == sursa].sort_values("conf", ascending=False)
        if len(sub) == 0:
            continue
        # High/mid/low conf
        n_per = args.n_exemple // 2
        indices = np.linspace(0, len(sub) - 1, n_per, dtype=int)
        for idx in indices:
            exemple.append(sub.iloc[idx].to_dict())

    print(f"[INFO] Selectat {len(exemple)} exemple pentru IG")

    # === Setup IG ===
    # Pentru clasificare, folosim LayerIntegratedGradients pe embedding layer
    def forward_fn(input_ids, attention_mask):
        """Forward care intoarce logits pentru IG."""
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Layer-ul de embeddings XLM-R
    # Pentru xlm-roberta-base: model.roberta.embeddings.word_embeddings
    if hasattr(model, "roberta"):
        embedding_layer = model.roberta.embeddings.word_embeddings
    elif hasattr(model, "model") and hasattr(model.model, "embeddings"):
        embedding_layer = model.model.embeddings.word_embeddings
    else:
        raise RuntimeError("Nu pot localiza stratul de embedding pentru IG")

    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    rezultate = []
    for i, ex in enumerate(exemple, 1):
        text = ex["text"]
        sursa = ex["sursa_site"]
        pred = int(ex["pred"])
        print(f"\n[{i}/{len(exemple)}] {sursa} | pred={pred} | conf={ex['conf']:.3f}")

        # Tokenize
        enc = tokenizer(text, padding="max_length", truncation=True,
                         max_length=args.max_length, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Baseline: PAD tokens
        baseline_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        # Primul si ultimul token raman BOS/EOS
        if tokenizer.bos_token_id is not None:
            baseline_input_ids[0, 0] = tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            # Gasim ultimul non-pad din input real
            last_real = attention_mask.sum().item() - 1
            baseline_input_ids[0, last_real] = tokenizer.eos_token_id

        # Compute attributions pentru clasa prezisa
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            additional_forward_args=(attention_mask,),
            target=pred,
            n_steps=args.steps,
            return_convergence_delta=True,
        )

        # Sum over embedding dim, normalize
        attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        # Normalize: abs-max normalization
        if abs(attr).max() > 0:
            attr = attr / abs(attr).max()

        # Mapare la token-uri
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        mask = attention_mask.squeeze(0).cpu().numpy()

        # Pastram doar token-urile reale (attention_mask == 1), exceptand BOS/EOS
        token_scores = []
        for tok, score, m in zip(tokens, attr, mask):
            if m == 0:
                continue
            if tok in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]:
                continue
            token_scores.append((tok, float(score)))

        # Top-15 pozitive (sustin clasa prezisa)
        top_pos = sorted(token_scores, key=lambda x: x[1], reverse=True)[:15]
        # Top-15 negative
        top_neg = sorted(token_scores, key=lambda x: x[1])[:15]

        print(f"   Delta convergență: {delta.item():.4f}")
        print(f"   Top-5 pozitive: " + ", ".join([f"{t}({s:+.3f})" for t, s in top_pos[:5]]))

        # === HTML vizualizare ===
        html_lines = [
            "<html><head><meta charset='utf-8'><style>",
            "body { font-family: Inter, sans-serif; padding: 20px; max-width: 900px; }",
            "span { padding: 2px; border-radius: 3px; }",
            ".neg { background: rgba(100, 100, 255, var(--alpha)); }",
            ".pos { background: rgba(255, 100, 100, var(--alpha)); }",
            "h2 { color: #333; }",
            ".meta { color: #666; font-size: 13px; margin-bottom: 20px; }",
            "</style></head><body>",
            f"<h2>Integrated Gradients — exemplu {i}</h2>",
            f"<div class='meta'>Sursa: {html_lib.escape(sursa)} | Pred: cls{pred} | "
            f"Confidence: {ex['conf']:.3f} | Convergence delta: {delta.item():.4f}</div>",
            "<p style='line-height:2'>",
        ]
        for tok, score in token_scores:
            # Curata prefixul ▁ pentru XLM-R (SentencePiece)
            display_tok = tok.replace("▁", " ").strip()
            if not display_tok:
                display_tok = " "
            alpha = min(abs(score), 1.0)
            cls = "pos" if score > 0 else "neg"
            html_lines.append(
                f"<span class='{cls}' style='--alpha:{alpha:.3f}'>"
                f"{html_lib.escape(display_tok)}</span> "
            )
        html_lines.extend([
            "</p>",
            "<h3>Top-15 token-uri pozitive (susțin clasa prezisă)</h3><ol>",
        ])
        for t, s in top_pos:
            html_lines.append(f"<li><code>{html_lib.escape(t)}</code> → {s:+.4f}</li>")
        html_lines.append("</ol>")
        html_lines.append("<h3>Top-15 token-uri negative</h3><ol>")
        for t, s in top_neg:
            html_lines.append(f"<li><code>{html_lib.escape(t)}</code> → {s:+.4f}</li>")
        html_lines.append("</ol></body></html>")

        fname = f"ig_{i:02d}_{sursa.split('.')[0]}_class{pred}.html"
        with open(html_dir / fname, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))

        rezultate.append({
            "exemplu_num": i,
            "sursa": sursa,
            "confidence": float(ex["conf"]),
            "convergence_delta": float(delta.item()),
            "top_positive": [(t, s) for t, s in top_pos],
            "top_negative": [(t, s) for t, s in top_neg],
            "html_file": fname,
        })

    # === Save ===
    out_json = {
        "config": {"steps": args.steps, "n_exemple": len(rezultate)},
        "rezultate": rezultate,
    }
    with open(out / "ig_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] IG results: {out / 'ig_results_v2.json'}")
    print(f"[OK] HTML vizualizări: {html_dir}/")


if __name__ == "__main__":
    main()
