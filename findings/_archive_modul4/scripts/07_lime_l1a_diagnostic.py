"""
LIME L1a — Diagnostic fidelity pe 4 grupuri × 25 articole.

Scop: testăm ipoteza că R² LIME scăzut pe cls1 e cauzat de:
  (i)  saturare softmax (artefact tehnic, fixabil cu logits)
  (ii) stylistic fingerprint (limitare intrinsecă a modelului)

Comparăm 4 grupuri, fiecare cu n=25 articole:
  - Grup A: TP cls0 baseline modul 2 (control: ar trebui R² mare)
  - Grup B: TP cls1 baseline modul 2 (replica finding 0.06 cu N mai mare)
  - Grup C: FN LOSO-V pe Veridica (modelul LOSO-V ratează propaganda)
  - Grup D: TP LOSO-V pe Veridica (modelul LOSO-V prinde propaganda fără amprentă)

Trei metrici per articol:
  (a) R² pe softmax probabilities (ca în scriptul vechi 06_lime_xlmr_v2.py)
  (b) R² pe logits raw (NEW — testează ipoteza saturare)
  (c) Faithfulness deletion AUC (NEW — câtă parte din predicție explică top-k cuvinte)

Ipoteze testate:
  H1: R²(logits) > R²(proba) pe ambele clase (efect saturare prezent global)
  H2: Δ R² între cls0 și cls1 persistă și pe logits (stylistic fingerprint real)
  H3: Faithfulness deletion mai bună pe cls0 vs cls1 (top-k LIME au impact cauzal mai mare)

Output:
  - findings_lime_l1a.json — date raw per articol (pentru re-analiză)
  - findings_lime_l1a.md — raport cu tabele agregate, IC 95%, Mann-Whitney
  - lime_html_l1a/ — vizualizări HTML pe predict_proba (selecție reprezentativă)

Usage:
    python 07_lime_l1a_diagnostic.py \\
        --baseline_model_dir models/xlmr_baseline_v2/final \\
        --loso_model_dir models/xlmr_loso_v/final \\
        --baseline_test_data data/processed/dataset_v2_test.csv \\
        --baseline_predictions findings/test_predictions_v2.csv \\
        --loso_test_data data/processed/dataset_v2_test.csv \\
        --loso_predictions findings/findings_loso_v_v2_predictions.csv \\
        --output_dir findings \\
        --n_per_group 25
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================================
# CONFIGURARE
# ============================================================================

# Coloana cu textul input — clasificatorul a fost antrenat pe `text_curat`
COLOANA_TEXT = "text_curat"

# Configurație LIME — identică cu scriptul vechi pentru comparabilitate
NUM_FEATURES = 15
NUM_SAMPLES = 1000
BOW = False  # păstrează ordinea token-urilor (important pentru transformer)

# K-uri pentru faithfulness deletion
K_VALUES_DELETION = (1, 3, 5, 10)


def alege_device():
    """Detectează device-ul: MPS pe Mac M-series, CUDA pe NVIDIA, altfel CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def construieste_predict_fns(model, tokenizer, device, max_length=256, batch_size=16):
    """
    Returnează două funcții predict pentru LIME:
      - predict_proba: returnează softmax probabilities (matrice N×2)
      - predict_logits: returnează raw logits (matrice N×2)

    Logits-ul evită saturarea softmax pe modele high-confidence:
      - softmax cls1 ~0.9996 → perturbări produc Δ ~10⁻⁴ (zgomot)
      - logits cls1 ~+8 → aceleași perturbări produc Δ ~1-3 (semnal)
    """

    def predict_proba(texts):
        """Predicție pe softmax — interfața standard LIME."""
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def predict_logits(texts):
        """
        Predicție pe logits raw — LIME va antrena regresie liniară pe logits.

        IMPORTANT: LIME se așteaptă la matrice (N, num_classes) de „scoruri".
        Logits-urile satisfac asta. R² rezultat are aceeași semnificație
        (cât de bine aproximează regresia liniară comportamentul modelului),
        dar pe scală non-saturată.
        """
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
                logits = model(**enc).logits.cpu().numpy()
                all_logits.append(logits)
        return np.vstack(all_logits)

    return predict_proba, predict_logits


def calculeaza_faithfulness_deletion(text, top_features, predict_proba, label_pred,
                                       k_values=K_VALUES_DELETION):
    """
    Faithfulness deletion: șterge top-k cele mai importante cuvinte (după LIME)
    și măsoară cât scade probabilitatea predicției inițiale.

    Returnează AUC-ul curbei (k → drop predicție). AUC mai mare = LIME identifică
    cuvinte cu impact cauzal real asupra modelului.

    Args:
        text: textul original (string)
        top_features: list de (cuvânt, weight) sortat descrescător după |weight|
        predict_proba: funcția de predicție (returnează matrice N×2)
        label_pred: indexul clasei prezise (0 sau 1)
        k_values: ce valori de k să testăm pentru deletion

    Returns:
        dict cu prob_initial, drops_per_k, auc_normalized
    """
    prob_initial = float(predict_proba([text])[0, label_pred])

    # Sortăm features după |weight| descrescător (cele mai importante prima)
    features_sorted = sorted(top_features, key=lambda x: abs(x[1]), reverse=True)

    drops = {}
    for k in k_values:
        if k > len(features_sorted):
            drops[k] = None
            continue

        # Construim text cu top-k cuvinte șterse (case-insensitive)
        cuvinte_de_sters = set(w.lower() for w, _ in features_sorted[:k])
        # Tokenizare simplă pe whitespace (consistency cu LIME bow=False)
        tokens = text.split()
        tokens_filtrati = [t for t in tokens
                            if t.lower().strip(".,!?;:\"'") not in cuvinte_de_sters]
        text_deletion = " ".join(tokens_filtrati)

        if not text_deletion.strip():
            drops[k] = None
            continue

        prob_after = float(predict_proba([text_deletion])[0, label_pred])
        drops[k] = prob_initial - prob_after

    # AUC normalizat = drop mediu pe k_values valide
    drops_valide = [d for d in drops.values() if d is not None]
    auc_normalized = float(np.mean(drops_valide)) if drops_valide else 0.0

    return {
        "prob_initial": prob_initial,
        "drops_per_k": drops,
        "auc_normalized": auc_normalized,
    }


def selecteaza_grupuri(baseline_preds, baseline_test, loso_preds, loso_test,
                        n_per_group, seed):
    """
    Selectează cele 4 grupuri de articole stratificat.

    Returnează un dict cu 4 chei (A, B, C, D), fiecare cu DataFrame de
    n_per_group articole îmbogățite cu coloana text.
    """
    # Merge baseline preds + test
    bl_merged = baseline_preds.merge(
        baseline_test[["id", COLOANA_TEXT]], on="id", how="left"
    ).rename(columns={COLOANA_TEXT: "text"})

    # Merge LOSO preds + test
    loso_merged = loso_preds.merge(
        loso_test[["id", COLOANA_TEXT]], on="id", how="left"
    ).rename(columns={COLOANA_TEXT: "text"})

    grupuri = {}

    # Grup A: TP cls0 baseline (Digi24 + G4Media), stratificat pe sursă
    cls0_tp = bl_merged[(bl_merged["label_numeric"] == 0) &
                          (bl_merged["pred"] == 0)].copy()
    digi = cls0_tp[cls0_tp["sursa_site"] == "digi24.ro"]
    g4m = cls0_tp[cls0_tp["sursa_site"] == "g4media.ro"]
    n_total_cls0 = len(digi) + len(g4m)
    if n_total_cls0 > 0:
        n_digi = int(round(n_per_group * len(digi) / n_total_cls0))
        n_g4m = n_per_group - n_digi
    else:
        n_digi = n_g4m = 0
    grup_a = pd.concat([
        digi.sample(n=min(n_digi, len(digi)), random_state=seed),
        g4m.sample(n=min(n_g4m, len(g4m)), random_state=seed),
    ])
    grup_a["grup"] = "A_baseline_TP_cls0"
    grupuri["A"] = grup_a

    # Grup B: TP cls1 baseline (Veridica + Stopfals), stratificat pe sursă
    cls1_tp = bl_merged[(bl_merged["label_numeric"] == 1) &
                          (bl_merged["pred"] == 1)].copy()
    vrd = cls1_tp[cls1_tp["sursa_site"] == "veridica.ro"]
    spf = cls1_tp[cls1_tp["sursa_site"] == "stopfals.md"]
    n_total_cls1 = len(vrd) + len(spf)
    if n_total_cls1 > 0:
        n_vrd = int(round(n_per_group * len(vrd) / n_total_cls1))
        n_spf = n_per_group - n_vrd
    else:
        n_vrd = n_spf = 0
    grup_b = pd.concat([
        vrd.sample(n=min(n_vrd, len(vrd)), random_state=seed),
        spf.sample(n=min(n_spf, len(spf)), random_state=seed),
    ])
    grup_b["grup"] = "B_baseline_TP_cls1"
    grupuri["B"] = grup_b

    # Grup C: FN LOSO-V pe Veridica (label=1, pred=0) — randomizat
    fn_loso = loso_merged[(loso_merged["label_numeric"] == 1) &
                            (loso_merged["pred"] == 0)].copy()
    # Eliminăm articolele fără text disponibil înainte de eșantionare
    fn_loso_cu_text = fn_loso[fn_loso["text"].notna()].copy()
    n_loso_fn_total = len(fn_loso)
    n_loso_fn_cu_text = len(fn_loso_cu_text)
    grup_c = fn_loso_cu_text.sample(n=min(n_per_group, len(fn_loso_cu_text)),
                                       random_state=seed)
    grup_c["grup"] = "C_loso_FN"
    grupuri["C"] = grup_c

    # Grup D: TP LOSO-V pe Veridica (label=1, pred=1) — randomizat
    tp_loso = loso_merged[(loso_merged["label_numeric"] == 1) &
                            (loso_merged["pred"] == 1)].copy()
    tp_loso_cu_text = tp_loso[tp_loso["text"].notna()].copy()
    n_loso_tp_total = len(tp_loso)
    n_loso_tp_cu_text = len(tp_loso_cu_text)
    grup_d = tp_loso_cu_text.sample(n=min(n_per_group, len(tp_loso_cu_text)),
                                       random_state=seed)
    grup_d["grup"] = "D_loso_TP"
    grupuri["D"] = grup_d

    # Verificări și sumar
    print(f"\n=== Eșantionare grupuri (seed={seed}) ===")
    for nume, df in grupuri.items():
        print(f"  Grup {nume}: n={len(df)}")
        if "sursa_site" in df.columns and len(df) > 0:
            print(f"    surse: {dict(df['sursa_site'].value_counts())}")
        n_text_null = df["text"].isna().sum() if len(df) > 0 else 0
        if n_text_null > 0:
            print(f"    [WARN] {n_text_null} articole fără text — vor fi sărite la rulare")

    # Sumar acoperire LOSO (informativ)
    print(f"\n=== Acoperire LOSO ===")
    print(f"  FN LOSO total: {n_loso_fn_total}, cu text disponibil: {n_loso_fn_cu_text} "
            f"({100*n_loso_fn_cu_text/max(1,n_loso_fn_total):.1f}%)")
    print(f"  TP LOSO total: {n_loso_tp_total}, cu text disponibil: {n_loso_tp_cu_text} "
            f"({100*n_loso_tp_cu_text/max(1,n_loso_tp_total):.1f}%)")

    return grupuri


def ruleaza_lime_pe_articol(text, label_pred, predict_proba, predict_logits,
                              explainer_proba, explainer_logits, num_features, num_samples):
    """
    Rulează LIME pe un articol cu ambele funcții predict (proba + logits).
    Returnează dict cu R² și top features pentru ambele.
    """
    rezultat = {
        "r2_proba": None, "top_features_proba": None,
        "r2_logits": None, "top_features_logits": None,
        "exp_proba": None,  # pentru salvare HTML
        "eroare": None,
    }

    try:
        # LIME pe softmax probabilities
        exp_proba = explainer_proba.explain_instance(
            text, predict_proba,
            num_features=num_features, num_samples=num_samples,
            labels=[label_pred],
        )
        rezultat["r2_proba"] = float(exp_proba.score)
        rezultat["top_features_proba"] = [(w, float(s))
                                            for w, s in exp_proba.as_list(label=label_pred)]
        rezultat["exp_proba"] = exp_proba
    except Exception as e:
        rezultat["eroare"] = f"LIME proba: {e}"
        return rezultat

    try:
        # LIME pe logits raw
        exp_logits = explainer_logits.explain_instance(
            text, predict_logits,
            num_features=num_features, num_samples=num_samples,
            labels=[label_pred],
        )
        rezultat["r2_logits"] = float(exp_logits.score)
        rezultat["top_features_logits"] = [(w, float(s))
                                              for w, s in exp_logits.as_list(label=label_pred)]
    except Exception as e:
        rezultat["eroare"] = f"LIME logits: {e}"

    return rezultat


def proceseaza_grup(nume_grup, df_grup, model, tokenizer, device, html_dir,
                      max_length, num_features, num_samples, seed):
    """
    Procesează un grup: rulează LIME pe fiecare articol, calculează metrici,
    salvează HTML-uri pentru câteva exemple reprezentative.
    """
    predict_proba, predict_logits = construieste_predict_fns(
        model, tokenizer, device, max_length=max_length
    )

    explainer_proba = LimeTextExplainer(
        class_names=["stire_credibila", "dezinformare_pro_rusa"],
        bow=False, random_state=seed,
    )
    explainer_logits = LimeTextExplainer(
        class_names=["stire_credibila", "dezinformare_pro_rusa"],
        bow=False, random_state=seed,
    )

    rezultate = []
    print(f"\n{'='*60}")
    print(f"Procesare Grup {nume_grup} (n={len(df_grup)})")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(df_grup.iterrows(), 1):
        text = row["text"]
        if pd.isna(text) or not str(text).strip():
            print(f"  [{i:2d}/{len(df_grup)}] {row['id']} — text gol, sar")
            continue
        text = str(text)

        label_pred = int(row["pred"])
        label_true = int(row["label_numeric"])
        sursa = row.get("sursa_site", "?")
        titlu = str(row.get("titlu", ""))[:50]

        t_start = time.time()
        lime_out = ruleaza_lime_pe_articol(
            text, label_pred, predict_proba, predict_logits,
            explainer_proba, explainer_logits, num_features, num_samples,
        )
        t_lime = time.time() - t_start

        if lime_out["eroare"]:
            print(f"  [{i:2d}/{len(df_grup)}] {row['id']} — EROARE: {lime_out['eroare']}")
            continue

        # Faithfulness deletion (folosim top features de la predict_proba)
        t_start = time.time()
        faith = calculeaza_faithfulness_deletion(
            text, lime_out["top_features_proba"], predict_proba, label_pred,
        )
        t_faith = time.time() - t_start

        # Salvăm HTML pentru primele 5 exemple din fiecare grup (reprezentativ)
        html_file = None
        if i <= 5:
            html_file = f"grup{nume_grup}_{i:02d}_{row['id']}.html"
            try:
                lime_out["exp_proba"].save_to_file(str(html_dir / html_file))
            except Exception as e:
                print(f"     [WARN] save HTML eșuat: {e}")
                html_file = None

        rezultat = {
            "grup": nume_grup,
            "id": row["id"],
            "sursa": sursa,
            "titlu": titlu,
            "label_true": label_true,
            "label_pred": label_pred,
            "prob_cls1": float(row["prob_cls1"]),
            "r2_proba": lime_out["r2_proba"],
            "r2_logits": lime_out["r2_logits"],
            "top_features_proba": lime_out["top_features_proba"],
            "top_features_logits": lime_out["top_features_logits"],
            "faith_prob_initial": faith["prob_initial"],
            "faith_drops_per_k": faith["drops_per_k"],
            "faith_auc": faith["auc_normalized"],
            "html_file": html_file,
            "t_lime_sec": round(t_lime, 2),
            "t_faith_sec": round(t_faith, 2),
        }
        rezultate.append(rezultat)

        print(f"  [{i:2d}/{len(df_grup)}] {row['id']} | {str(sursa)[:12]:12s} | "
              f"R²_p={lime_out['r2_proba']:.3f} R²_l={lime_out['r2_logits']:.3f} "
              f"Fa={faith['auc_normalized']:+.3f} | {t_lime:.1f}s")

    return rezultate


def agrega_rezultate(toate_rezultatele):
    """Calculează statistici agregate per grup și teste statistice între grupuri."""
    df = pd.DataFrame(toate_rezultatele)

    agregari = {}
    for grup in ["A", "B", "C", "D"]:
        sub = df[df["grup"] == grup]
        if len(sub) == 0:
            continue
        agregari[grup] = {
            "n": len(sub),
            "r2_proba": {
                "mean": float(sub["r2_proba"].mean()),
                "std": float(sub["r2_proba"].std()),
                "median": float(sub["r2_proba"].median()),
                "ci95_low": float(sub["r2_proba"].quantile(0.025)),
                "ci95_high": float(sub["r2_proba"].quantile(0.975)),
            },
            "r2_logits": {
                "mean": float(sub["r2_logits"].mean()),
                "std": float(sub["r2_logits"].std()),
                "median": float(sub["r2_logits"].median()),
                "ci95_low": float(sub["r2_logits"].quantile(0.025)),
                "ci95_high": float(sub["r2_logits"].quantile(0.975)),
            },
            "faith_auc": {
                "mean": float(sub["faith_auc"].mean()),
                "std": float(sub["faith_auc"].std()),
                "median": float(sub["faith_auc"].median()),
            },
        }

    # Mann-Whitney U între grupuri (test non-parametric)
    teste = {}
    perechi = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    for g1, g2 in perechi:
        sub1 = df[df["grup"] == g1]
        sub2 = df[df["grup"] == g2]
        if len(sub1) == 0 or len(sub2) == 0:
            continue
        for metric in ["r2_proba", "r2_logits", "faith_auc"]:
            try:
                u, p = stats.mannwhitneyu(sub1[metric], sub2[metric], alternative="two-sided")
                teste[f"{g1}_vs_{g2}__{metric}"] = {
                    "u": float(u), "p": float(p),
                    "diff_median": float(sub1[metric].median() - sub2[metric].median()),
                }
            except Exception as e:
                teste[f"{g1}_vs_{g2}__{metric}"] = {"eroare": str(e)}

    # Wilcoxon pereche pe articol: logits vs proba (în cadrul aceluiași grup)
    teste_pereche = {}
    for grup in ["A", "B", "C", "D"]:
        sub = df[df["grup"] == grup]
        if len(sub) == 0:
            continue
        try:
            w, p = stats.wilcoxon(sub["r2_logits"], sub["r2_proba"])
            teste_pereche[f"grup_{grup}__logits_vs_proba"] = {
                "w": float(w), "p": float(p),
                "diff_median": float((sub["r2_logits"] - sub["r2_proba"]).median()),
            }
        except Exception as e:
            teste_pereche[f"grup_{grup}__logits_vs_proba"] = {"eroare": str(e)}

    return agregari, teste, teste_pereche


def genereaza_markdown(agregari, teste, teste_pereche, n_per_group, seed):
    """Generează raportul markdown cu interpretare automată."""
    md = [
        "# Findings — LIME L1a (diagnostic fidelity, 4 grupuri × 25)",
        "",
        "## 1. Configurație",
        "",
        f"- N per grup: {n_per_group}",
        f"- num_features = {NUM_FEATURES}, num_samples = {NUM_SAMPLES}, bow = {BOW}",
        f"- seed = {seed}",
        f"- Coloana text input: `{COLOANA_TEXT}`",
        "",
        "**Grupuri:**",
        "- A: TP cls0 baseline modul 2 (control — Digi24/G4Media)",
        "- B: TP cls1 baseline modul 2 (replica finding 0.06 — Veridica/Stopfals)",
        "- C: FN LOSO-V pe Veridica (modelul ratează propaganda)",
        "- D: TP LOSO-V pe Veridica (modelul prinde propaganda fără amprentă)",
        "",
        "## 2. Rezultate fidelity (R²)",
        "",
        "### R² pe softmax probabilities",
        "",
        "| Grup | n | mean ± std | median | IC 95% (quantile) |",
        "|---|---:|---:|---:|---:|",
    ]

    for grup, ag in agregari.items():
        rp = ag["r2_proba"]
        md.append(f"| {grup} | {ag['n']} | {rp['mean']:.4f} ± {rp['std']:.4f} | "
                   f"{rp['median']:.4f} | [{rp['ci95_low']:.4f}, {rp['ci95_high']:.4f}] |")

    md.extend([
        "",
        "### R² pe logits raw",
        "",
        "| Grup | n | mean ± std | median | IC 95% (quantile) |",
        "|---|---:|---:|---:|---:|",
    ])
    for grup, ag in agregari.items():
        rl = ag["r2_logits"]
        md.append(f"| {grup} | {ag['n']} | {rl['mean']:.4f} ± {rl['std']:.4f} | "
                   f"{rl['median']:.4f} | [{rl['ci95_low']:.4f}, {rl['ci95_high']:.4f}] |")

    md.extend([
        "",
        "### Faithfulness deletion AUC (drop mediu pe k=1,3,5,10)",
        "",
        "| Grup | n | mean ± std | median |",
        "|---|---:|---:|---:|",
    ])
    for grup, ag in agregari.items():
        fa = ag["faith_auc"]
        md.append(f"| {grup} | {ag['n']} | {fa['mean']:.4f} ± {fa['std']:.4f} | "
                   f"{fa['median']:.4f} |")

    md.extend([
        "",
        "## 3. Teste statistice — comparații între grupuri (Mann-Whitney U)",
        "",
        "| Comparație | Metrică | Diff median (g1 − g2) | p-value |",
        "|---|---|---:|---:|",
    ])
    for cheie, t in teste.items():
        if "eroare" in t:
            continue
        # Format cheie: "A_vs_B__r2_proba"
        parts = cheie.split("__")
        comp_str = parts[0].replace("_vs_", " vs ")
        metric_str = parts[1]
        sig = "***" if t["p"] < 0.001 else ("**" if t["p"] < 0.01 else ("*" if t["p"] < 0.05 else ""))
        md.append(f"| {comp_str} | {metric_str} | {t['diff_median']:+.4f} | "
                   f"{t['p']:.4g} {sig} |")

    md.extend([
        "",
        "## 4. Teste pereche — logits vs proba (Wilcoxon, în cadrul aceluiași grup)",
        "",
        "Testează ipoteza H1: trecerea de la softmax la logits crește R² (efect saturare).",
        "",
        "| Grup | Diff median (logits − proba) | p-value |",
        "|---|---:|---:|",
    ])
    for cheie, t in teste_pereche.items():
        if "eroare" in t:
            continue
        grup = cheie.split("__")[0].split("_")[1]
        sig = "***" if t["p"] < 0.001 else ("**" if t["p"] < 0.01 else ("*" if t["p"] < 0.05 else ""))
        md.append(f"| {grup} | {t['diff_median']:+.4f} | {t['p']:.4g} {sig} |")

    # Interpretare automată
    md.extend([
        "",
        "## 5. Interpretare ipoteze",
        "",
    ])

    # H1: efect saturare
    grupuri_cu_efect = []
    for cheie, t in teste_pereche.items():
        if "eroare" not in t and t.get("diff_median", 0) > 0.05 and t.get("p", 1) < 0.05:
            grupuri_cu_efect.append(cheie.split("__")[0].split("_")[1])

    if len(grupuri_cu_efect) >= 3:
        md.append(f"**H1 CONFIRMATĂ:** logits îmbunătățește R² semnificativ pe "
                   f"{len(grupuri_cu_efect)}/4 grupuri ({', '.join(grupuri_cu_efect)}). "
                   f"Efectul saturare softmax e real — confirmare independentă a unei "
                   f"limitări tehnice cunoscute a LIME pe modele transformer high-confidence.")
    elif len(grupuri_cu_efect) >= 1:
        md.append(f"**H1 PARȚIAL CONFIRMATĂ:** logits îmbunătățește R² doar pe "
                   f"{len(grupuri_cu_efect)}/4 grupuri ({', '.join(grupuri_cu_efect)}). "
                   f"Efectul saturare e prezent dar nu uniform.")
    else:
        md.append("**H1 RESPINSĂ:** logits nu îmbunătățește R² semnificativ pe niciun grup. "
                   "Saturarea softmax nu e cauza R² scăzut.")
    md.append("")

    # H2: stylistic fingerprint persistă pe logits
    if "A" in agregari and "B" in agregari:
        delta_logits = agregari["A"]["r2_logits"]["mean"] - agregari["B"]["r2_logits"]["mean"]
        ab_test = teste.get("A_vs_B__r2_logits", {})
        if ab_test.get("p", 1) < 0.05 and delta_logits > 0.1:
            md.append(f"**H2 CONFIRMATĂ:** Δ R²(A−B) pe logits = {delta_logits:+.4f} "
                       f"(p={ab_test.get('p', 1):.4g}). Asimetria cls0/cls1 persistă "
                       f"și pe logits — stylistic fingerprint e real, dincolo de saturare.")
        else:
            md.append(f"**H2 NECONFIRMATĂ:** Δ R²(A−B) pe logits = {delta_logits:+.4f} "
                       f"(p={ab_test.get('p', 1):.4g}). Asimetria nu persistă "
                       f"semnificativ pe logits — saturarea explică majoritatea diferenței.")
    md.append("")

    # H3: faithfulness diferențială A vs B
    if "A" in agregari and "B" in agregari:
        delta_faith = agregari["A"]["faith_auc"]["mean"] - agregari["B"]["faith_auc"]["mean"]
        ab_faith_test = teste.get("A_vs_B__faith_auc", {})
        if ab_faith_test.get("p", 1) < 0.05 and delta_faith > 0:
            md.append(f"**H3 CONFIRMATĂ:** faithfulness AUC(A) > AUC(B), Δ={delta_faith:+.4f} "
                       f"(p={ab_faith_test.get('p', 1):.4g}). Cuvintele top-k LIME au "
                       f"impact cauzal mai mare pe cls0 decât pe cls1 — confirmă că modelul "
                       f"nu se bazează pe cuvinte localizate pentru cls1.")
        else:
            md.append(f"**H3 NECONFIRMATĂ:** faithfulness AUC(A) − AUC(B) = {delta_faith:+.4f} "
                       f"(p={ab_faith_test.get('p', 1):.4g}).")
    md.append("")

    # Diagnostic LOSO-V (C vs D)
    if "C" in agregari and "D" in agregari:
        md.append("### Diagnostic LOSO-V (Grup C vs D)")
        md.append("")
        md.append(f"- C (FN, model ratează): R²_proba = {agregari['C']['r2_proba']['mean']:.4f}, "
                   f"R²_logits = {agregari['C']['r2_logits']['mean']:.4f}")
        md.append(f"- D (TP, model prinde): R²_proba = {agregari['D']['r2_proba']['mean']:.4f}, "
                   f"R²_logits = {agregari['D']['r2_logits']['mean']:.4f}")
        cd_test = teste.get("C_vs_D__r2_logits", {})
        if cd_test.get("p", 1) < 0.05:
            md.append(f"- Diferența semnificativă (p={cd_test.get('p'):.4g}) — "
                       f"modelul folosește features detectabile prin LIME când prinde propaganda "
                       f"fără amprentă.")
        else:
            md.append(f"- Diferența nesemnificativă (p={cd_test.get('p', 1):.4g}).")
        md.append("")

    md.extend([
        "## 6. Concluzii pentru capitolul Explicabilitate al tezei",
        "",
        "TBD — pe baza rezultatelor de mai sus, decidem strategia hibridă LIME + IG.",
        "",
        "Întrebări deschise pentru pasul L2 (diagnostic detaliat):",
        "- Dacă H1 confirmată: refacem rapoartele LIME oficiale pe `predict_logits`",
        "- Dacă H2 confirmată: documentăm stylistic fingerprint ca limitare LIME intrinsecă",
        "- Indiferent de rezultate: trecem la pasul L3 (Integrated Gradients) pentru triangulare",
        "",
        "*Generat automat de `07_lime_l1a_diagnostic.py`*",
    ])

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_dir", required=True,
                          help="Folder cu modelul XLM-R baseline v2")
    parser.add_argument("--loso_model_dir", required=True,
                          help="Folder cu modelul XLM-R LOSO-V")
    parser.add_argument("--baseline_test_data", required=True,
                          help="CSV cu test set (coloane: id, text_curat, ...)")
    parser.add_argument("--baseline_predictions", required=True,
                          help="CSV cu predicții baseline modul 2")
    parser.add_argument("--loso_test_data", required=True, nargs="+",
                          help="Unul sau mai multe CSV-uri care conțin textul articolelor "
                                "evaluate la LOSO-V (ex: train.csv test.csv val.csv). Concatenate.")
    parser.add_argument("--loso_predictions", required=True,
                          help="CSV cu predicții LOSO-V pe Veridica")
    parser.add_argument("--output_dir", default="findings")
    parser.add_argument("--n_per_group", type=int, default=25)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    html_dir = out / "lime_html_l1a"
    html_dir.mkdir(parents=True, exist_ok=True)

    device = alege_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] N per grup: {args.n_per_group}")

    print(f"\n[INFO] Încărcare CSV-uri...")
    baseline_preds = pd.read_csv(args.baseline_predictions)
    baseline_test = pd.read_csv(args.baseline_test_data)
    loso_preds = pd.read_csv(args.loso_predictions)

    # LOSO test data poate fi compus din mai multe CSV-uri (ex: train + test + val)
    # Concatenăm și deduplicăm pe id (ultima valoare câștigă pentru duplicate)
    loso_test_dfs = []
    for path in args.loso_test_data:
        df = pd.read_csv(path)
        if COLOANA_TEXT not in df.columns:
            raise ValueError(f"Coloana '{COLOANA_TEXT}' lipsește din {path}. "
                              f"Coloane: {list(df.columns)}")
        loso_test_dfs.append(df)
        print(f"  loso source: {path} → {len(df)} rânduri")
    loso_test = pd.concat(loso_test_dfs, ignore_index=True)
    n_inainte = len(loso_test)
    loso_test = loso_test.drop_duplicates(subset=["id"], keep="last")
    print(f"  loso combined: {n_inainte} → {len(loso_test)} după deduplicare pe id")

    # Verificare acoperire LOSO predictions vs textul disponibil
    ids_loso_pred = set(loso_preds["id"])
    ids_loso_text = set(loso_test["id"])
    acoperire = len(ids_loso_pred & ids_loso_text)
    print(f"  Acoperire LOSO: {acoperire}/{len(ids_loso_pred)} = "
            f"{100*acoperire/len(ids_loso_pred):.1f}%")

    print(f"  baseline preds: {len(baseline_preds)}, test: {len(baseline_test)}")
    print(f"  loso preds: {len(loso_preds)}, test combined: {len(loso_test)}")

    # Verificare coloana text (baseline; LOSO deja verificat la încărcare)
    if COLOANA_TEXT not in baseline_test.columns:
        raise ValueError(f"Coloana '{COLOANA_TEXT}' lipsește din {args.baseline_test_data}. "
                          f"Coloane disponibile: {list(baseline_test.columns)}")

    grupuri = selecteaza_grupuri(baseline_preds, baseline_test,
                                    loso_preds, loso_test,
                                    args.n_per_group, args.seed)

    toate_rezultatele = []
    t_total_start = time.time()

    # Procesare A și B (baseline)
    print(f"\n[INFO] Încărcare model baseline: {args.baseline_model_dir}")
    tokenizer_bl = AutoTokenizer.from_pretrained(args.baseline_model_dir)
    model_bl = AutoModelForSequenceClassification.from_pretrained(args.baseline_model_dir).to(device)
    model_bl.eval()

    for grup_nume in ["A", "B"]:
        rez = proceseaza_grup(grup_nume, grupuri[grup_nume],
                                model_bl, tokenizer_bl, device, html_dir,
                                args.max_length, NUM_FEATURES, NUM_SAMPLES, args.seed)
        toate_rezultatele.extend(rez)

    # Eliberare memorie
    del model_bl, tokenizer_bl
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # Procesare C și D (LOSO-V)
    print(f"\n[INFO] Încărcare model LOSO-V: {args.loso_model_dir}")
    tokenizer_loso = AutoTokenizer.from_pretrained(args.loso_model_dir)
    model_loso = AutoModelForSequenceClassification.from_pretrained(args.loso_model_dir).to(device)
    model_loso.eval()

    for grup_nume in ["C", "D"]:
        rez = proceseaza_grup(grup_nume, grupuri[grup_nume],
                                model_loso, tokenizer_loso, device, html_dir,
                                args.max_length, NUM_FEATURES, NUM_SAMPLES, args.seed)
        toate_rezultatele.extend(rez)

    t_total = time.time() - t_total_start
    print(f"\n[INFO] Procesare totală: {t_total/60:.1f} minute pentru {len(toate_rezultatele)} articole")

    # Agregare + statistici
    print(f"\n[INFO] Calculare agregări și teste statistice...")
    agregari, teste, teste_pereche = agrega_rezultate(toate_rezultatele)

    # Salvare JSON
    out_json = {
        "config": {
            "n_per_group": args.n_per_group,
            "num_features": NUM_FEATURES,
            "num_samples": NUM_SAMPLES,
            "bow": BOW,
            "seed": args.seed,
            "coloana_text": COLOANA_TEXT,
        },
        "agregari": agregari,
        "teste_intergrupuri": teste,
        "teste_pereche_logits_vs_proba": teste_pereche,
        "rezultate_per_articol": toate_rezultatele,
        "t_total_min": round(t_total/60, 2),
    }
    json_path = out / "findings_lime_l1a.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON: {json_path}")

    md_text = genereaza_markdown(agregari, teste, teste_pereche,
                                    args.n_per_group, args.seed)
    md_path = out / "findings_lime_l1a.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[OK] Markdown: {md_path}")

    # Sumar consolă
    print(f"\n{'='*60}")
    print("SUMAR FINAL")
    print(f"{'='*60}")
    for grup, ag in agregari.items():
        print(f"  Grup {grup} (n={ag['n']}): "
              f"R²_proba={ag['r2_proba']['mean']:.4f}, "
              f"R²_logits={ag['r2_logits']['mean']:.4f}, "
              f"faith_auc={ag['faith_auc']['mean']:.4f}")
    print(f"\n[OK] HTML-uri reprezentative în: {html_dir}")


if __name__ == "__main__":
    main()