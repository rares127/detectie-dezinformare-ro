"""
Proba D1: Integrated Gradients pe exemple cls0 (Digi24 + G4Media).

Ratiune: IG pe cls1 (exemplele 1-6 din ig_results_v2.json) a aratat:
- Top tokens dominate de sub-cuvinte morfologice: rul, tul, rice, esc, ova, uni, uri
- Token-uri chirilice: ▁в, Б, ный, ические
- Fragmente de entitati: ▁Zahar, ▁Lug, ▁Sputnik, ▁KP

Intrebare D1: sunt aceste sub-cuvinte specifice cls1 sau sunt artefact
generic al tokenizarii XLM-R pe romana?

Daca IG pe cls0 produce aceleasi tipuri de fragmente morfologice ca IG
pe cls1, atunci IG la nivel de sub-cuvinte NU e interpretabil pe XLM-R
pentru romana — trebuie agregare la nivel de cuvant. Daca pe cls0 vedem
tokens mai „semantici" (fraze lungi, cuvinte intregi), atunci pattern-ul
morfologic de pe cls1 e intr-adevar un semnal diferentiator.

Ruleaza pe BASELINE (singurul model unde cls0 e rezolvata perfect).

Usage:
    python src/training/D1_ig_cls0.py \\
        --model_dir models/xlmr_baseline_v2/final \\
        --predictions findings/test_predictions_v2.csv \\
        --test_data data/processed/dataset_v2_test.csv \\
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
    parser.add_argument("--n_per_sursa", type=int, default=3,
                        help="Nr exemple pe fiecare sursă cls0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    html_dir = out / "ig_html_d1"
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

    # === Selectare exemple cls0 ===
    tp_cls0 = merged[(merged["correct"] == 1) & (merged["label_numeric"] == 0)].copy()
    tp_cls0["conf"] = 1 - tp_cls0["prob_cls1"]  # confidence pe cls0

    exemple = []
    for sursa in ["digi24.ro", "g4media.ro"]:
        sub = tp_cls0[tp_cls0["sursa_site"] == sursa].sort_values("conf", ascending=False)
        if len(sub) == 0:
            continue
        indices = np.linspace(0, len(sub) - 1, args.n_per_sursa, dtype=int)
        for idx in indices:
            exemple.append(sub.iloc[idx].to_dict())

    print(f"[INFO] Selectat {len(exemple)} exemple cls0 pentru IG")

    # === IG setup ===
    def forward_fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    if hasattr(model, "roberta"):
        embedding_layer = model.roberta.embeddings.word_embeddings
    else:
        raise RuntimeError("Nu pot localiza embedding layer")

    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    rezultate = []
    from collections import Counter
    token_counter_pos = Counter()  # sub-cuvinte dominante la top pozitive

    for i, ex in enumerate(exemple, 1):
        text = ex["text"]
        sursa = ex["sursa_site"]
        pred = 0  # toate sunt cls0
        print(f"\n[{i}/{len(exemple)}] {sursa} | conf={ex['conf']:.3f}")

        enc = tokenizer(text, padding="max_length", truncation=True,
                         max_length=args.max_length, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        if tokenizer.bos_token_id is not None:
            baseline_ids[0, 0] = tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            last_real = attention_mask.sum().item() - 1
            baseline_ids[0, last_real] = tokenizer.eos_token_id

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=pred,
            n_steps=args.steps,
            return_convergence_delta=True,
        )

        attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        if abs(attr).max() > 0:
            attr = attr / abs(attr).max()

        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        mask = attention_mask.squeeze(0).cpu().numpy()

        token_scores = []
        for tok, score, m in zip(tokens, attr, mask):
            if m == 0:
                continue
            if tok in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]:
                continue
            token_scores.append((tok, float(score)))

        top_pos = sorted(token_scores, key=lambda x: x[1], reverse=True)[:15]
        top_neg = sorted(token_scores, key=lambda x: x[1])[:15]

        # Agregare: cate din top-15 pozitive sunt sub-cuvinte (fara prefix ▁)?
        subwords_pos = sum(1 for t, _ in top_pos if not t.startswith("▁"))
        full_words_pos = 15 - subwords_pos

        # Adaugam la counter global
        for t, _ in top_pos:
            token_counter_pos[t] += 1

        print(f"   Delta convergență: {delta.item():.4f}")
        print(f"   Top-5 pozitive: " + ", ".join([f"{t}({s:+.3f})" for t, s in top_pos[:5]]))
        print(f"   Sub-cuvinte în top-15: {subwords_pos}/15 | Cuvinte întregi: {full_words_pos}/15")

        # HTML
        html_lines = [
            "<html><head><meta charset='utf-8'><style>",
            "body { font-family: Inter, sans-serif; padding: 20px; max-width: 900px; }",
            "span { padding: 2px; border-radius: 3px; }",
            ".neg { background: rgba(100, 100, 255, var(--alpha)); }",
            ".pos { background: rgba(100, 200, 100, var(--alpha)); }",
            ".meta { color: #666; font-size: 13px; margin-bottom: 20px; }",
            "</style></head><body>",
            f"<h2>IG cls0 — exemplu {i}</h2>",
            f"<div class='meta'>Sursa: {html_lib.escape(sursa)} | Pred: cls0 | "
            f"Confidence: {ex['conf']:.3f} | Convergence delta: {delta.item():.4f}<br>"
            f"Sub-cuvinte în top-15 pozitive: {subwords_pos}/15</div>",
            "<p style='line-height:2'>",
        ]
        for tok, score in token_scores:
            display_tok = tok.replace("▁", " ").strip()
            if not display_tok:
                display_tok = " "
            alpha = min(abs(score), 1.0)
            cls = "pos" if score > 0 else "neg"
            html_lines.append(
                f"<span class='{cls}' style='--alpha:{alpha:.3f}'>"
                f"{html_lib.escape(display_tok)}</span> "
            )
        html_lines.extend(["</p>", "<h3>Top-15 pozitive</h3><ol>"])
        for t, s in top_pos:
            html_lines.append(f"<li><code>{html_lib.escape(t)}</code> → {s:+.4f}</li>")
        html_lines.append("</ol></body></html>")

        fname = f"ig_d1_{i:02d}_{sursa.split('.')[0]}_class0.html"
        with open(html_dir / fname, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))

        rezultate.append({
            "exemplu_num": i,
            "sursa": sursa,
            "confidence": float(ex["conf"]),
            "convergence_delta": float(delta.item()),
            "subwords_in_top15": subwords_pos,
            "full_words_in_top15": full_words_pos,
            "top_positive": [(t, s) for t, s in top_pos],
            "top_negative": [(t, s) for t, s in top_neg],
            "html_file": fname,
        })

    # === Comparatie cu IG cls1 (din ig_results_v2.json) ===
    # Metrici agregate pe cls0
    mean_subwords_cls0 = np.mean([r["subwords_in_top15"] for r in rezultate])
    mean_abs_delta_cls0 = np.mean([abs(r["convergence_delta"]) for r in rezultate])

    # Incarcam rezultatele cls1 daca exista
    comparatie = None
    ig_cls1_path = Path(args.output_dir) / "ig_results_v2.json"
    if ig_cls1_path.exists():
        with open(ig_cls1_path, encoding="utf-8") as f:
            ig_cls1 = json.load(f)
        cls1_subwords = []
        for r in ig_cls1["rezultate"]:
            s = sum(1 for t, _ in r["top_positive"] if not t.startswith("▁"))
            cls1_subwords.append(s)
        mean_subwords_cls1 = float(np.mean(cls1_subwords))
        mean_abs_delta_cls1 = float(np.mean([abs(r["convergence_delta"])
                                               for r in ig_cls1["rezultate"]]))
        comparatie = {
            "cls0": {
                "mean_subwords_top15": float(mean_subwords_cls0),
                "mean_abs_convergence_delta": float(mean_abs_delta_cls0),
            },
            "cls1": {
                "mean_subwords_top15": mean_subwords_cls1,
                "mean_abs_convergence_delta": mean_abs_delta_cls1,
            },
        }

    out_json = {
        "config": {"steps": args.steps, "n_exemple": len(rezultate)},
        "rezultate": rezultate,
        "comparatie_cls1_cls0": comparatie,
    }
    with open(out / "ig_cls0_results.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    # === Markdown ===
    md = [
        "# Proba D1 — IG pe cls0 (comparație cu cls1)",
        "",
        "## Ipoteza testată",
        "",
        "Sub-cuvintele morfologice (`rul`, `tul`, `ova`, `esc`, ...) găsite de IG pe",
        "cls1 reflectă un shortcut real sau sunt artefact generic al tokenizării?",
        "",
        "## Metodologie",
        "",
        f"- Model: baseline v2 (singurul model care rezolvă cls0 perfect)",
        f"- N exemple: {len(rezultate)} (3× digi24 + 3× g4media, stratificat pe confidence)",
        f"- Steps IG: {args.steps}",
        "",
        "## Rezultate sumar",
        "",
        "| Sursa | N | Conf mediu | Sub-cuv în top-15 (mean) | Conv delta (|mean|) |",
        "|-------|---|------------|-------------------------|----------------------|",
    ]
    for sursa in ["digi24.ro", "g4media.ro"]:
        sub_rez = [r for r in rezultate if r["sursa"] == sursa]
        if not sub_rez:
            continue
        n = len(sub_rez)
        mconf = np.mean([r["confidence"] for r in sub_rez])
        msub = np.mean([r["subwords_in_top15"] for r in sub_rez])
        mdelta = np.mean([abs(r["convergence_delta"]) for r in sub_rez])
        md.append(f"| {sursa} | {n} | {mconf:.3f} | {msub:.1f}/15 | {mdelta:.3f} |")

    if comparatie:
        md.extend([
            "",
            "## Comparație cls0 vs cls1",
            "",
            "| Clasă | Sub-cuvinte în top-15 (mean) | |Δ convergence| (mean) |",
            "|-------|------------------------------|------------------------|",
            f"| cls1 (Veridica+Stopfals) | {comparatie['cls1']['mean_subwords_top15']:.1f}/15 | "
            f"{comparatie['cls1']['mean_abs_convergence_delta']:.3f} |",
            f"| cls0 (Digi24+G4Media) | {comparatie['cls0']['mean_subwords_top15']:.1f}/15 | "
            f"{comparatie['cls0']['mean_abs_convergence_delta']:.3f} |",
            "",
            "### Interpretare",
            "",
        ])
        delta_sub = comparatie['cls1']['mean_subwords_top15'] - comparatie['cls0']['mean_subwords_top15']
        if abs(delta_sub) < 2:
            md.append(f"**Sub-cuvintele apar similar în ambele clase** (delta {delta_sub:+.1f}/15).")
            md.append("Asta sugerează că fragmentarea SentencePiece pe română produce natural")
            md.append("sub-cuvinte în top atribuții, INDIFERENT de clasă. **Concluzie:** IG la")
            md.append("nivel de sub-token nu e interpretabil aici — ai nevoie de agregare la")
            md.append("nivel de cuvânt pentru IG, sau să treci la alternative (attention rollout,")
            md.append("SHAP cu agregare). Pattern-ul morfologic de pe cls1 NU e shortcut confirmat.")
        elif delta_sub > 2:
            md.append(f"**Sub-cuvinte MULT mai dense pe cls1** (delta {delta_sub:+.1f}/15).")
            md.append("Asta sugerează că modelul FOLOSEȘTE într-adevăr pattern-uri sub-lexicale")
            md.append("specifice cls1 care nu apar pe cls0 — susține ipoteza shortcut morfologic.")
        else:
            md.append(f"**Sub-cuvinte mai dense pe cls0** (delta {delta_sub:+.1f}/15) — contraintuitiv.")
            md.append("Posibilă cauză: stiluri Digi24/G4Media au propriile particularități")
            md.append("(prefix-uri Agerpres, citate oficiale). Investigare suplimentară necesară.")

    md.extend([
        "",
        "## Top token-uri agregate (cls0)",
        "",
        "| Token | Frecvență în top-15 |",
        "|-------|--------------------|",
    ])
    for t, c in token_counter_pos.most_common(15):
        md.append(f"| `{t}` | {c} |")

    md.extend([
        "",
        "## Vizualizări",
        "",
        f"Fișiere HTML: `{html_dir}/`",
    ])
    with open(out / "findings_d1_ig_cls0.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[OK] Findings: {out / 'findings_d1_ig_cls0.md'}")


if __name__ == "__main__":
    main()
