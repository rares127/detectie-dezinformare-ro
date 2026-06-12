"""
Proba D3: Ablatie entitati specifice (toponime, media propagandistice, oficiali).

Ipoteza H2: modelul LOSO-V foloseste nume proprii specifice lexicului
pro-Kremlin ca semnal lexical pentru cls1. IG pe cls1 a aratat in top:
- ▁Zahar (Zaharova, purtator Kremlin)
- ▁Lug (Lugansk)
- ▁Sputnik, ▁KP (Komsomolskaya Pravda) — media propagandistice
- ▁Kreml

Test: inlocuieste lista de entitati cu token generic `[ENT]` in toate
articolele Veridica; predict cu modelul LOSO-V; masoara Δ P(cls1).

Lista curata - include 3 categorii:
1. TOPONIME rusificate/contestate: Lugansk, Donbas, Crimeea, Transnistria, etc.
2. OFICIALI rusi si proxy-uri: Putin, Lavrov, Zaharova, Peskov, Soigu, Dodon, etc.
3. MEDIA propagandiste: Sputnik, RT, Russia Today, TASS, Komsomolskaya, RIA

Nota importanta: nu folosim in lista:
- Kremlin, Rusia, Moscova — aparitia lor e normala si in articole Digi24/G4Media
  care relateaza despre Rusia. Ablarea lor ar introduce alt tip de bias.

Usage:
    python src/training/D3_ablatie_entitati.py \\
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


# === Lista de entitati grupate pe categorie (pentru raportare separata) ===
ENTITATI = {
    "toponime": [
        r"Lugansk", r"Luhansk", r"Doneţk", r"Donetsk", r"Donbas", r"Donbass",
        r"Transnistri[ae]", r"Găgăuzi[ae]", r"Comrat", r"Tiraspol",
        r"Mariupol", r"Berdiansk", r"Herson", r"Zaporij[jȋ]e",
        r"Crimeea", r"Sevastopol", r"Belarus",
    ],
    "oficiali_rusi": [
        r"\bPutin\b", r"Vladimir Putin", r"\bLavrov\b", r"Serghei Lavrov",
        r"\bZaharova\b", r"Maria Zaharova", r"\bPeskov\b", r"Dmitri Peskov",
        r"\bŞoigu\b", r"\bShoigu\b", r"Serghei Șoigu",
        r"\bMedvedev\b", r"Dmitri Medvedev",
        r"\bPatrușev\b", r"\bPatrushev\b",
        r"\bPrigojin\b", r"\bPrigozhin\b",
        r"\bKadîrov\b", r"\bKadyrov\b",
        r"\bDodon\b", r"Igor Dodon",
        r"Ilan Șor", r"\bȘor\b",
        r"Lukașenko", r"Lukashenko",
    ],
    "media_propagandiste": [
        r"\bSputnik\b", r"Russia Today", r"\bRT\b(?=\s|$|\.|,)",
        r"\bTASS\b", r"\bRIA\b(?=\s+Novosti|\s*$|[,\.])",
        r"Komsomolskaya", r"Komsomol.skaia",
        r"\bKP\b(?=\s|$|[,\.])",
        r"Pervîi Kanal", r"Perv[iy]i Kanal",
        r"\bNTV\b",
    ],
}


def build_regex(lista):
    """Regex case-insensitive din lista de pattern-uri."""
    return re.compile("|".join(lista), flags=re.IGNORECASE)


REGEX_TOPONIME = build_regex(ENTITATI["toponime"])
REGEX_OFICIALI = build_regex(ENTITATI["oficiali_rusi"])
REGEX_MEDIA = build_regex(ENTITATI["media_propagandiste"])
REGEX_ALL = build_regex(
    ENTITATI["toponime"] + ENTITATI["oficiali_rusi"] + ENTITATI["media_propagandiste"]
)


def construieste_text(row):
    return f"{str(row['titlu']).strip()}\n\n{str(row['stire_citata']).strip()}"


def predict_probs(model, tokenizer, texts, device, max_length=256, batch_size=16):
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
                        help="models/loso_v/final")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", default="findings")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--replacement", default="[ENT]",
                        help="Token de înlocuire pentru entități")
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

    df = pd.read_csv(args.dataset)
    df["text"] = df.apply(construieste_text, axis=1)
    veridica = df[df["sursa_site"] == "veridica.ro"].reset_index(drop=True)
    print(f"[INFO] Articole Veridica: {len(veridica)}")

    # === Contorizare matches per categorie ===
    veridica["n_toponime"] = veridica["text"].apply(
        lambda t: len(REGEX_TOPONIME.findall(str(t)))
    )
    veridica["n_oficiali"] = veridica["text"].apply(
        lambda t: len(REGEX_OFICIALI.findall(str(t)))
    )
    veridica["n_media"] = veridica["text"].apply(
        lambda t: len(REGEX_MEDIA.findall(str(t)))
    )
    veridica["n_total_entitati"] = (veridica["n_toponime"] +
                                      veridica["n_oficiali"] +
                                      veridica["n_media"])

    print(f"\n[INFO] Match-uri entități (sumar):")
    print(f"  Articole cu ≥1 toponim: {(veridica['n_toponime'] > 0).sum()}/{len(veridica)}")
    print(f"  Articole cu ≥1 oficial: {(veridica['n_oficiali'] > 0).sum()}/{len(veridica)}")
    print(f"  Articole cu ≥1 media propagandistă: {(veridica['n_media'] > 0).sum()}/{len(veridica)}")
    print(f"  Articole cu ≥1 entitate orice categorie: "
          f"{(veridica['n_total_entitati'] > 0).sum()}/{len(veridica)}")
    print(f"  Mediana entități total per articol: {veridica['n_total_entitati'].median():.0f}")

    # === Predict text ORIGINAL ===
    print("\n[INFO] Predict ORIGINAL...")
    probs_orig = predict_probs(model, tokenizer, veridica["text"].tolist(),
                                  device, args.max_length)
    veridica["prob_cls1_orig"] = probs_orig
    veridica["pred_orig"] = (probs_orig > 0.5).astype(int)

    # === Predict text ABLATED (toate categoriile) ===
    print("[INFO] Predict ABLATED (toate entitățile)...")
    veridica["text_ablated"] = veridica["text"].apply(
        lambda t: REGEX_ALL.sub(args.replacement, str(t))
    )
    probs_ablated = predict_probs(model, tokenizer, veridica["text_ablated"].tolist(),
                                     device, args.max_length)
    veridica["prob_cls1_ablated"] = probs_ablated
    veridica["pred_ablated"] = (probs_ablated > 0.5).astype(int)

    veridica["delta_prob_cls1"] = veridica["prob_cls1_orig"] - veridica["prob_cls1_ablated"]
    veridica["flip"] = (veridica["pred_orig"] != veridica["pred_ablated"]).astype(int)

    # === Predict ablation PER CATEGORIE (pentru izolarea contributiei) ===
    print("[INFO] Predict ablations PER CATEGORIE...")
    for cat_name, regex in [("toponime", REGEX_TOPONIME),
                              ("oficiali", REGEX_OFICIALI),
                              ("media", REGEX_MEDIA)]:
        text_cat = veridica["text"].apply(lambda t: regex.sub(args.replacement, str(t)))
        probs = predict_probs(model, tokenizer, text_cat.tolist(),
                                device, args.max_length)
        veridica[f"prob_cls1_ablated_{cat_name}"] = probs
        veridica[f"delta_{cat_name}"] = veridica["prob_cls1_orig"] - probs

    # === Rezultate ===
    print("\n=== Rezultate globale (ablație TOATE entitățile) ===")
    print(f"  Prob cls1 orig mean: {veridica['prob_cls1_orig'].mean():.4f}")
    print(f"  Prob cls1 ablated mean: {veridica['prob_cls1_ablated'].mean():.4f}")
    print(f"  Delta mean: {veridica['delta_prob_cls1'].mean():+.4f}")
    print(f"  Flips: {veridica['flip'].sum()}/{len(veridica)} "
          f"({100*veridica['flip'].mean():.1f}%)")
    print(f"  Recall cls1 orig: {veridica['pred_orig'].mean():.4f}")
    print(f"  Recall cls1 ablated: {veridica['pred_ablated'].mean():.4f}")

    print("\n=== Contribuția fiecărei categorii (delta mean per categorie) ===")
    for cat in ["toponime", "oficiali", "media"]:
        delta_mean = veridica[f"delta_{cat}"].mean()
        print(f"  Ablate DOAR {cat}: delta mean = {delta_mean:+.4f}")

    # Subset: doar articolele cu ≥1 entitate in original
    cu_ent = veridica[veridica["n_total_entitati"] > 0]
    fara_ent = veridica[veridica["n_total_entitati"] == 0]
    print(f"\n=== Breakdown pe prezența entităților ===")
    print(f"  Articole CU entități ({len(cu_ent)}):")
    if len(cu_ent) > 0:
        print(f"    Prob cls1 orig mean: {cu_ent['prob_cls1_orig'].mean():.4f}")
        print(f"    Prob cls1 ablated mean: {cu_ent['prob_cls1_ablated'].mean():.4f}")
        print(f"    Delta mean: {cu_ent['delta_prob_cls1'].mean():+.4f}")
        print(f"    Flip rate: {100*cu_ent['flip'].mean():.1f}%")
    print(f"  Articole FĂRĂ entități ({len(fara_ent)}, control):")
    if len(fara_ent) > 0:
        print(f"    Delta mean: {fara_ent['delta_prob_cls1'].mean():+.4f} (ar trebui ~0)")

    # === Interpretare ===
    delta_mean_cu_ent = cu_ent['delta_prob_cls1'].mean() if len(cu_ent) > 0 else 0
    flip_rate_cu_ent = cu_ent['flip'].mean() if len(cu_ent) > 0 else 0

    if delta_mean_cu_ent > 0.20 or flip_rate_cu_ent > 0.20:
        interp = ("H2 CONFIRMATĂ PARȚIAL: ablația entităților pro-Kremlin scade "
                  "semnificativ P(cls1). Modelul folosește numele proprii propagandistice "
                  "drept semnal lexical important. Atenuare: entity masking în preprocessing "
                  "sau augmentare cu exemple cls0 care menționează aceleași entități "
                  "neutral (relatare, nu citare).")
    elif delta_mean_cu_ent > 0.10:
        interp = ("H2 CONFIRMATĂ SLAB: entitățile contribuie, dar nu sunt singurul factor.")
    else:
        interp = ("H2 RESPINSĂ: ablarea entităților propagandistice nu schimbă predicțiile. "
                  "Shortcut-ul nu vine din vocabularul specific.")

    # Categoria cea mai impactanta
    cat_deltas = {cat: veridica[f"delta_{cat}"].mean() for cat in ["toponime", "oficiali", "media"]}
    cat_top = max(cat_deltas, key=cat_deltas.get)
    print(f"\n  Categoria cu cel mai mare impact: {cat_top} (delta {cat_deltas[cat_top]:+.4f})")
    print(f"\n=== Interpretare ===\n{interp}")

    # === Save ===
    cols_csv = (["id", "titlu", "an", "n_toponime", "n_oficiali", "n_media",
                 "n_total_entitati",
                 "prob_cls1_orig", "prob_cls1_ablated", "delta_prob_cls1",
                 "pred_orig", "pred_ablated", "flip"] +
                [f"delta_{c}" for c in ["toponime", "oficiali", "media"]])
    veridica[cols_csv].to_csv(out / "d3_ablatie_entitati_per_articol.csv", index=False)

    result = {
        "model_used": args.model_dir,
        "replacement": args.replacement,
        "n_articole": int(len(veridica)),
        "distributie_entitati": {
            "cu_toponime": int((veridica["n_toponime"] > 0).sum()),
            "cu_oficiali": int((veridica["n_oficiali"] > 0).sum()),
            "cu_media": int((veridica["n_media"] > 0).sum()),
            "cu_orice_entitate": int((veridica["n_total_entitati"] > 0).sum()),
        },
        "global_ablatie_toate": {
            "prob_cls1_orig_mean": float(veridica["prob_cls1_orig"].mean()),
            "prob_cls1_ablated_mean": float(veridica["prob_cls1_ablated"].mean()),
            "delta_mean": float(veridica["delta_prob_cls1"].mean()),
            "flip_rate": float(veridica["flip"].mean()),
            "recall_cls1_orig": float(veridica["pred_orig"].mean()),
            "recall_cls1_ablated": float(veridica["pred_ablated"].mean()),
        },
        "per_categorie": {
            cat: {
                "delta_mean": float(veridica[f"delta_{cat}"].mean()),
            } for cat in ["toponime", "oficiali", "media"]
        },
        "cu_entitati": {
            "n": int(len(cu_ent)),
            "delta_mean": float(cu_ent["delta_prob_cls1"].mean()) if len(cu_ent) > 0 else None,
            "flip_rate": float(cu_ent["flip"].mean()) if len(cu_ent) > 0 else None,
        },
        "categoria_top": cat_top,
        "interpretare": interp,
    }
    with open(out / "d3_ablatie_entitati.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # === Markdown ===
    md = [
        "# Proba D3 — Ablație entități propagandistice",
        "",
        "## Ipoteza H2",
        "",
        "Modelul LOSO-V folosește nume proprii din vocabularul propagandist (toponime",
        "rusificate, oficiali Kremlin, media de stat rusă) ca semnal lexical pentru cls1.",
        "",
        "## Metodologie",
        "",
        f"- **Model**: `{args.model_dir}` (LOSO-V, shortcut activ)",
        f"- **Test set**: {len(veridica)} articole Veridica",
        f"- **Înlocuire**: entități → `{args.replacement}`",
        "- **3 categorii testate** separat + combinat:",
        f"  - Toponime: {len(ENTITATI['toponime'])} pattern-uri (Lugansk, Donbas, Transnistria...)",
        f"  - Oficiali ruși: {len(ENTITATI['oficiali_rusi'])} pattern-uri (Putin, Lavrov, Zaharova...)",
        f"  - Media propagandiste: {len(ENTITATI['media_propagandiste'])} pattern-uri (Sputnik, TASS, RT...)",
        "",
        "**Notă:** NU am inclus 'Kremlin/Rusia/Moscova' — apar natural și în presă credibilă.",
        "",
        "## Distribuție entități",
        "",
        f"- Articole cu ≥1 toponim: {int((veridica['n_toponime'] > 0).sum())}",
        f"- Articole cu ≥1 oficial: {int((veridica['n_oficiali'] > 0).sum())}",
        f"- Articole cu ≥1 media propagandistă: {int((veridica['n_media'] > 0).sum())}",
        f"- Articole cu ≥1 entitate orice categorie: "
        f"{int((veridica['n_total_entitati'] > 0).sum())}/{len(veridica)}",
        "",
        "## Rezultate globale (ablație TOATE)",
        "",
        "| Măsură | Original | Ablated | Delta |",
        "|--------|----------|---------|-------|",
        f"| P(cls1) mean | {veridica['prob_cls1_orig'].mean():.4f} | "
        f"{veridica['prob_cls1_ablated'].mean():.4f} | "
        f"{veridica['delta_prob_cls1'].mean():+.4f} |",
        f"| Recall cls1 | {veridica['pred_orig'].mean():.4f} | "
        f"{veridica['pred_ablated'].mean():.4f} | "
        f"{veridica['pred_orig'].mean() - veridica['pred_ablated'].mean():+.4f} |",
        "",
        f"**Flips**: {int(veridica['flip'].sum())}/{len(veridica)} ({100*veridica['flip'].mean():.1f}%)",
        "",
        "## Contribuția per categorie",
        "",
        "| Categorie ablated | Delta P(cls1) mean |",
        "|-------------------|--------------------|",
    ]
    for cat in ["toponime", "oficiali", "media"]:
        md.append(f"| {cat} | {veridica[f'delta_{cat}'].mean():+.4f} |")

    md.extend([
        "",
        f"**Categoria cu cel mai mare impact**: `{cat_top}` "
        f"(delta {cat_deltas[cat_top]:+.4f})",
        "",
        "## Breakdown pe prezența entităților",
        "",
        "| Subset | N | Delta mean | Flip rate |",
        "|--------|---|------------|-----------|",
    ])
    for nume, sub in [("Cu entități", cu_ent), ("Fără entități (control)", fara_ent)]:
        if len(sub) == 0:
            continue
        md.append(f"| {nume} | {len(sub)} | "
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
        "CSV per articol: `d3_ablatie_entitati_per_articol.csv`",
    ])
    with open(out / "findings_d3_ablatie_entitati.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[OK] Findings: {out / 'findings_d3_ablatie_entitati.md'}")


if __name__ == "__main__":
    main()
