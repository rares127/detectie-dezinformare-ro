"""
LIME (Local Interpretable Model-agnostic Explanations) pe modelul baseline v2.

Configuratie IDENTICA cu v1 pentru comparabilitate directa:
- num_features=15, num_samples=1000, bow=False

Esantion stratificat (12 exemple True Positives, confidence variat):
- 3 Veridica TP (clasa 1)
- 3 Stopfals TP (clasa 1 — NOU v2)
- 3 Digi24 TP (clasa 0)
- 3 G4Media TP (clasa 0)

Outputs:
- lime_html/ — 12 vizualizari HTML interactive
- findings_lime_baseline_v2.md — raport cu fidelity, token-uri top, comparatie v1

Comparatie critica:
- v1 Veridica fidelity: 0.04-0.09 (foarte scazut)
- v2 tinta: daca ramane scazut → limitare LIME intrinseca confirmata
           daca creste > 0.20 → entity balancing a imbunatatit interpretabilitatea

Usage:
    python src/training/06_lime_xlmr_v2.py \
        --model_dir models/xlmr_baseline_v2/final \
        --test_data data/processed/dataset_v2_test.csv \
        --predictions findings/test_predictions_v2.csv \
        --output_dir findings
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", default="findings")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_features", type=int, default=15)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    html_dir = out / "lime_html"
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

    merged = preds_df.merge(
        test_df[["id", "text", "stire_citata"]],
        on="id", how="left",
    )

    # === Predict function pentru LIME ===
    def predict_proba(texts):
        """Functie de predictie care primeste lista texte, intoarce matrice (N, 2) de probabilitati."""
        all_probs = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                 max_length=args.max_length, return_tensors="pt").to(device)
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    # === Selectare 12 exemple stratificat ===
    # TP = correct predictions; selectam 3 de pe fiecare sursa
    tp = merged[merged["correct"] == 1].copy()
    # Confidence = prob clasei prezise
    tp["conf"] = tp.apply(
        lambda r: r["prob_cls1"] if r["pred"] == 1 else 1 - r["prob_cls1"], axis=1,
    )

    # Pentru fiecare sursa selectam 3 exemple: 1 high-conf, 1 mid-conf, 1 low-conf
    exemple = []
    for sursa in ["veridica.ro", "stopfals.md", "digi24.ro", "g4media.ro"]:
        sub = tp[tp["sursa_site"] == sursa].sort_values("conf", ascending=False)
        if len(sub) == 0:
            print(f"[WARN] Zero TP pentru {sursa} — sar")
            continue
        # Stratificare pe conf: high, mid, low din distributie
        indices = [0, len(sub) // 2, len(sub) - 1]
        indices = list(dict.fromkeys(indices))[:3]  # deduplicare daca n<3
        for idx in indices:
            exemple.append(sub.iloc[idx].to_dict())

    print(f"\n[INFO] Selectat {len(exemple)} exemple pentru LIME")

    # === LIME explainer ===
    explainer = LimeTextExplainer(
        class_names=["stire_credibila", "dezinformare_pro_rusa"],
        bow=False,  # pastreaza ordinea token-urilor (important pentru transformere)
        random_state=args.seed,
    )

    # === Rulare LIME + colectare rezultate ===
    rezultate = []
    token_counter_cls1 = Counter()
    token_counter_cls0 = Counter()

    for i, ex in enumerate(exemple, 1):
        sursa = ex["sursa_site"]
        label_true = ex["label_numeric"]
        label_pred = ex["pred"]
        text = ex["text"]
        titlu = str(ex["titlu"])[:60]

        tip = "TP" if label_true == label_pred else ("FP" if label_pred == 1 else "FN")
        fname = f"exemplu_{i:02d}_{sursa.split('.')[0]}_{tip}_class{label_pred}.html"
        print(f"\n[{i:2d}/{len(exemple)}] {sursa} | true={label_true} pred={label_pred} | "
              f"conf={ex['conf']:.3f} | {titlu}...")

        # Rulare LIME pentru clasa prezisa
        try:
            exp = explainer.explain_instance(
                text,
                predict_proba,
                num_features=args.num_features,
                num_samples=args.num_samples,
                labels=[label_pred],
            )
        except Exception as e:
            print(f"   [EROARE LIME]: {e}")
            continue

        # Fidelity = scor-ul local R² pentru modelul surogat
        fidelity = exp.score

        # Top features (cuvant, weight) pentru clasa prezisa
        top_features = exp.as_list(label=label_pred)

        # Salvare HTML
        html_path = html_dir / fname
        exp.save_to_file(str(html_path))

        # Agregare token-uri per clasa
        for word, weight in top_features:
            word_norm = word.strip().lower()
            if not word_norm or len(word_norm) < 2:
                continue
            if label_pred == 1 and weight > 0:
                token_counter_cls1[word_norm] += 1
            elif label_pred == 0 and weight > 0:
                token_counter_cls0[word_norm] += 1

        rezultate.append({
            "exemplu_num": i,
            "sursa": sursa,
            "titlu": titlu,
            "label_true": int(label_true),
            "label_pred": int(label_pred),
            "confidence": float(ex["conf"]),
            "tip": tip,
            "fidelity": float(fidelity),
            "top_features": [(w, float(s)) for w, s in top_features],
            "html_file": fname,
        })

        print(f"   fidelity={fidelity:.4f} | top_3: " +
              ", ".join([f"{w}({s:+.3f})" for w, s in top_features[:3]]))

    # === Agregari ===
    # Fidelity mediu per sursa
    fid_per_sursa = {}
    for r in rezultate:
        s = r["sursa"]
        fid_per_sursa.setdefault(s, []).append(r["fidelity"])
    fid_summary = {s: {"mean": float(np.mean(fids)),
                        "min": float(np.min(fids)),
                        "max": float(np.max(fids)),
                        "n": len(fids)}
                   for s, fids in fid_per_sursa.items()}

    print("\n=== Fidelity per sursă ===")
    for s, summ in fid_summary.items():
        print(f"  {s:15s}  mean={summ['mean']:.4f}  range=[{summ['min']:.4f}, {summ['max']:.4f}]  n={summ['n']}")

    print("\n=== Top token-uri pro-cls1 (agregat) ===")
    for w, c in token_counter_cls1.most_common(15):
        print(f"  {w:30s}  {c}")

    print("\n=== Top token-uri pro-cls0 (agregat) ===")
    for w, c in token_counter_cls0.most_common(15):
        print(f"  {w:30s}  {c}")

    # === Save JSON ===
    out_json = {
        "config": {
            "num_features": args.num_features,
            "num_samples": args.num_samples,
            "bow": False,
            "seed": args.seed,
        },
        "fidelity_per_sursa": fid_summary,
        "top_tokens_cls1": dict(token_counter_cls1.most_common(20)),
        "top_tokens_cls0": dict(token_counter_cls0.most_common(20)),
        "rezultate_detaliate": rezultate,
    }
    with open(out / "lime_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Rezultate JSON: {out / 'lime_results_v2.json'}")

    # === Markdown findings cu comparatie v1 ↔ v2 ===
    # Fidelity v1 pentru comparatie (din recapitulare v1)
    fid_v1 = {
        "veridica.ro": {"mean": 0.06, "range": "0.036–0.092", "n": 3},
        "digi24.ro":   {"mean": 0.50, "range": "0.317–0.701", "n": 3},
        "g4media.ro":  {"mean": 0.177, "range": "0.177 (n=1 TP)", "n": 1},
    }

    md = [
        "# Findings — LIME pe baseline v2",
        "",
        "## 1. Configurație",
        "",
        f"- num_features = {args.num_features}",
        f"- num_samples = {args.num_samples}",
        f"- bow = False",
        f"- seed = {args.seed}",
        f"- N exemple analizate: {len(rezultate)}",
        "",
        "## 2. Fidelity mediu per sursă",
        "",
        "Fidelity = scorul R² local al modelului-surogat LIME. Valoare înaltă → LIME",
        "explică bine modelul pe exemplul respectiv.",
        "",
        "| Sursa | v2 mean | v2 range | v1 mean | Delta |",
        "|-------|---------|----------|---------|-------|",
    ]
    for s, summ in fid_summary.items():
        v1 = fid_v1.get(s)
        if v1:
            delta = summ["mean"] - v1["mean"]
            md.append(f"| {s} | {summ['mean']:.4f} | [{summ['min']:.4f}, {summ['max']:.4f}] | "
                      f"{v1['mean']:.4f} | {delta:+.4f} |")
        else:
            md.append(f"| {s} | {summ['mean']:.4f} | [{summ['min']:.4f}, {summ['max']:.4f}] | N/A (nou) | N/A |")

    md.extend([
        "",
        "## 3. Interpretare",
        "",
    ])

    veridica_fid_v2 = fid_summary.get("veridica.ro", {}).get("mean")
    if veridica_fid_v2 is not None:
        if veridica_fid_v2 < 0.15:
            md.append(f"**Fidelity pe Veridica v2 = {veridica_fid_v2:.4f}** — rămâne scăzut "
                      f"(ca în v1, 0.04-0.09).")
            md.append("")
            md.append("Concluzie: e limitare INTRINSECĂ a LIME pe transformere high-confidence,")
            md.append("nu bug de dataset. Modelul folosește reprezentări distribuite global,")
            md.append("nu features lexicale localizate. **Recomandare:** adaugă Integrated")
            md.append("Gradients (captum) ca XAI complementar pe cls1.")
        elif veridica_fid_v2 < 0.25:
            md.append(f"**Fidelity pe Veridica v2 = {veridica_fid_v2:.4f}** — îmbunătățire")
            md.append("marginală față de v1. Posibil efect al entity balancing, dar încă")
            md.append("sub pragul recomandat (0.25). Integrated Gradients rămâne util.")
        else:
            md.append(f"**Fidelity pe Veridica v2 = {veridica_fid_v2:.4f}** — îmbunătățire")
            md.append("semnificativă față de v1 (0.06 mean). Entity balancing + Stopfals")
            md.append("au redus dependența de pattern-uri globale. LIME devine suficient.")

    md.extend([
        "",
        "## 4. Token-uri agregate top-15",
        "",
        "### Pro-cls1 (dezinformare)",
        "",
        "| Token | Frecvență în top-features |",
        "|-------|---------------------------|",
    ])
    for w, c in token_counter_cls1.most_common(15):
        md.append(f"| {w} | {c} |")

    md.extend([
        "",
        "### Pro-cls0 (credibil)",
        "",
        "| Token | Frecvență în top-features |",
        "|-------|---------------------------|",
    ])
    for w, c in token_counter_cls0.most_common(15):
        md.append(f"| {w} | {c} |")

    md.extend([
        "",
        "## 5. Vizualizări HTML",
        "",
        f"Cele {len(rezultate)} vizualizări LIME interactive sunt în: `{html_dir}/`",
        "",
        "| # | Sursa | Tip | True/Pred | Confidence | Fidelity | HTML |",
        "|---|-------|-----|-----------|------------|----------|------|",
    ])
    for r in rezultate:
        md.append(f"| {r['exemplu_num']} | {r['sursa']} | {r['tip']} | "
                  f"{r['label_true']}/{r['label_pred']} | {r['confidence']:.3f} | "
                  f"{r['fidelity']:.4f} | `{r['html_file']}` |")

    with open(out / "findings_lime_baseline_v2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[OK] Findings markdown: {out / 'findings_lime_baseline_v2.md'}")


if __name__ == "__main__":
    main()
