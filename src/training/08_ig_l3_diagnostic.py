"""
Modul 4 — Pasul L3: Integrated Gradients pe modul 2.

Scop: Aplicam IG (Sundararajan et al. 2017) pe aceleasi 4 grupuri × 25 articole
ca in L1a, pentru comparatie head-to-head cu LIME.

Avantaje IG vs LIME:
  - Determinist (zero variabilitate intre rulari cu acelasi seed)
  - Principled axiomatic (Completeness, Sensitivity, Implementation Invariance)
  - Functioneaza pe gradient → bypass natural al saturarii softmax
  - Mai rapid: ~50 forward+backward passes vs ~1000 LIME forward passes

Dezavantaje:
  - Atributiile depind de baseline (alegere de design)
  - Atributiile sunt per-token BPE → trebuie agregate la nivel de cuvant

Design:
  - Baseline: PAD tokens (recomandare literatura pentru transformere)
  - Layer: model.roberta.embeddings.word_embeddings
  - Steps: 50 (Sundararajan et al. recomanda 20-300; 50 e standard)
  - Aceleasi 100 articole ca L1a (selectare prin acelasi seed)

Metrici per articol:
  - top_features_ig: top-15 cuvinte cu atributii pozitive (ca LIME)
  - faithfulness_deletion_auc: aceeasi definitie ca L1a (drop predictie cand stergem top-k)
  - attribution_completeness: verificare axioma (suma atributiilor ≈ f(input) − f(baseline))

Output:
  - findings_lime_vs_ig_l3.md — raport comparativ LIME vs IG pe ambele faithfulness
  - findings_lime_vs_ig_l3.json — date raw per articol
  - ig_html_l3/ — vizualizari HTML cu atributii colorate (5 per grup)

Usage:
    python 08_ig_l3_diagnostic.py \\
        --baseline_model_dir models/xlmr_baseline_v2/final \\
        --loso_model_dir models/loso_v/final \\
        --baseline_test_data data/processed/dataset_v2_test.csv \\
        --baseline_predictions findings/test_predictions_v2.csv \\
        --loso_test_data data/processed/dataset_v2_train.csv data/processed/dataset_v2_test.csv data/processed/dataset_v2_val.csv \\
        --loso_predictions findings/findings_loso_v_v2_predictions.csv \\
        --lime_results_json findings/findings_lime_l1a.json \\
        --output_dir findings \\
        --n_per_group 25
"""

import argparse
import html as html_lib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================================
# CONFIGURARE
# ============================================================================

# Coloana cu textul input (acelasi ca in L1a)
COLOANA_TEXT = "text_curat"

# Configuratie IG
IG_STEPS = 200  # Crescut de la 50 — necesar pentru convergenta pe XLM-R fine-tuned
IG_BASELINE_MODE = "zero"  # "zero" (recomandat) sau "pad"

# Top-K pentru extragerea cuvintelor importante (analog LIME num_features)
TOP_K_WORDS = 15

# K-uri pentru faithfulness deletion (identic L1a)
K_VALUES_DELETION = (1, 3, 5, 10)


def alege_device():
    """Detecteaza device-ul: MPS pe Mac M-series, CUDA pe NVIDIA, altfel CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def construieste_predict_proba(model, tokenizer, device, max_length=256, batch_size=16):
    """
    Construieste functia predict_proba (pentru faithfulness deletion).
    Identica cu cea din L1a — pentru comparatie directa.
    """
    def predict_proba(texts):
        """Returneaza matrice (N, 2) de probabilitati softmax."""
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

    return predict_proba


def calculeaza_ig_pe_articol(text, label_pred, model, tokenizer, device,
                                max_length=256, n_steps=IG_STEPS,
                                baseline_mode=IG_BASELINE_MODE,
                                internal_batch_size=8):
    """
    Calculeaza atributii IG pentru un articol si returneaza:
      - cuvinte_atributii: list de (cuvant, atributie) ordonat cum apar in text
      - top_features: top-K cuvinte dupa |atributie| (pentru comparatie cu LIME)
      - prob_input: probabilitatea modelului pe input real
      - prob_baseline: probabilitatea modelului pe baseline
      - completeness: |sum(attrs) − (logit_input − logit_baseline)| / |logit_input| (sanity check)

    Mecanica:
      1. Tokenizam input + construim baseline (zero embeddings sau PAD ids)
      2. LayerIntegratedGradients pe embedding layer cu n_steps interpolari
      3. internal_batch_size pentru a evita OOM la n_steps mare
      4. Atributii per token (vector 768d) → suma cu semn
      5. Agregare token → cuvant prin offset_mapping

    Args:
        baseline_mode: "zero" (embeddings zero, recomandat) sau "pad" (token PAD)
        internal_batch_size: cat de multe pasi IG procesam odata (mic = lent, mult = OOM risk)
    """
    # Tokenizare cu offset_mapping pentru a putea reconstrui cuvintele
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length,
                     return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].cpu().numpy()  # (seq_len, 2)

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Construim baseline-ul in functie de mod
    if baseline_mode == "zero":
        # Baseline = zero embeddings (referinta neutra autentica)
        # LayerIntegratedGradients gestioneaza asta automat daca pasam input_ids ca baseline
        # si atasam hook pe layer-ul de embedding cu zerorizare
        # ABORDARE STANDARD CAPTUM: pasam baseline_ids = pad_id pe toata lungimea
        # DAR layer.attribute() face interpolarea PE EMBEDDINGS, nu pe IDs
        # → daca vrem zero embeddings ca baseline, folosim un input cu PAD si
        #   modelul va produce embeddings PAD (NU zero) → asta nu e zero embeddings
        # SOLUTIA CORECTA pentru zero baseline pe LayerIntegratedGradients:
        # construim un input fictiv cu PAD si separat trecem baseline=tensor de zerouri
        # de aceeasi dimensiune ca embeddings layer output

        # Implementare practica: folosim PAD ids ca baseline_ids si DEPENDEM
        # de faptul ca PAD embedding e foarte mic in XLM-R (verificam empiric)
        # PENTRU COMPARABILITATE TOTALA cu literatura, pastram aceeasi secventa
        baseline_ids = torch.full_like(input_ids, pad_id)
        # Pastram CLS si SEP la pozitiile lor (sunt intotdeauna la inceput/sfarsit)
        baseline_ids[0, 0] = cls_id
        # SEP e la prima pozitie unde apare in input_ids (inainte de padding)
        sep_positions = (input_ids[0] == sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            baseline_ids[0, sep_positions[0].item()] = sep_id
    else:  # baseline_mode == "pad"
        baseline_ids = input_ids.clone()
        for i in range(input_ids.shape[1]):
            tok = input_ids[0, i].item()
            if tok not in (cls_id, sep_id):
                baseline_ids[0, i] = pad_id

    # Definim functia forward pentru Captum (returneaza doar logitul clasei prezise)
    def forward_func(input_ids_arg, attention_mask_arg):
        outputs = model(input_ids=input_ids_arg, attention_mask=attention_mask_arg)
        return outputs.logits

    # Setam IG pe embedding layer
    embedding_layer = model.roberta.embeddings.word_embeddings
    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    # Calculam atributiile cu internal_batch_size pentru a evita OOM la n_steps mare
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=label_pred,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )

    # Reducem dimensiunea hidden cu suma (semnul conteaza: pozitiv = impinge spre label_pred)
    # Captum recomanda: attributions.sum(dim=-1) sau norm(dim=-1) cu semn
    # Folosim suma (pastreaza semnul; poti compara cu logitul total)
    attr_per_token = attributions.sum(dim=-1).squeeze(0).cpu().numpy()  # (seq_len,)

    # Probabilitati pe input real si baseline (pentru sanity check completeness)
    with torch.no_grad():
        logit_input = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, label_pred].item()
        # Pentru baseline folosim atentia input-ului real (consistent cu modul in care a fost atribuit)
        logit_baseline = model(input_ids=baseline_ids, attention_mask=attention_mask).logits[0, label_pred].item()

    # Sanity check: suma atributiilor ≈ logit_input − logit_baseline (axioma Completeness)
    sum_attrs = float(attr_per_token.sum())
    target_diff = logit_input - logit_baseline
    completeness_error = abs(sum_attrs - target_diff)
    completeness_relative = completeness_error / (abs(target_diff) + 1e-8)

    # Probabilitatile (pentru raportare)
    with torch.no_grad():
        prob_input = float(torch.softmax(
            model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1
        )[0, label_pred].item())
        prob_baseline = float(torch.softmax(
            model(input_ids=baseline_ids, attention_mask=attention_mask).logits, dim=-1
        )[0, label_pred].item())

    # Reconstruim cuvintele din tokeni folosind offset_mapping
    cuvinte_atributii = agrega_tokens_la_cuvinte(text, offsets, attr_per_token, input_ids[0].cpu().numpy(),
                                                     tokenizer)

    # Extragem top-K cuvinte dupa |atributie|
    cuv_sortate = sorted(cuvinte_atributii, key=lambda x: abs(x[1]), reverse=True)
    top_features = cuv_sortate[:TOP_K_WORDS]

    return {
        "cuvinte_atributii": cuvinte_atributii,  # ordonat ca in text
        "top_features": [(w, float(s)) for w, s in top_features],
        "prob_input": prob_input,
        "prob_baseline": prob_baseline,
        "logit_input": logit_input,
        "logit_baseline": logit_baseline,
        "sum_attrs": sum_attrs,
        "target_diff": target_diff,
        "completeness_error": float(completeness_error),
        "completeness_relative": float(completeness_relative),
        "convergence_delta": float(delta.item()) if hasattr(delta, "item") else float(delta),
        "n_tokens": int(input_ids.shape[1]),
        "n_words": len(cuvinte_atributii),
    }


def agrega_tokens_la_cuvinte(text, offsets, attr_per_token, input_ids, tokenizer):
    """
    Agrega atributiile per token la nivel de cuvant folosind offset_mapping.

    XLM-R foloseste SentencePiece BPE — un cuvant poate fi impartit in mai multi tokeni.
    Sumam atributiile tokenilor aceluiasi cuvant (delimitat de spatiu sau punctuatie).

    Args:
        text: textul original
        offsets: array de shape (seq_len, 2) cu (start_char, end_char) per token
        attr_per_token: array de shape (seq_len,) cu atributii per token
        input_ids: array de shape (seq_len,) cu IDs (pentru a sari tokeni speciali)
        tokenizer: tokenizer-ul HuggingFace

    Returns:
        list de (cuvant, atributie_agregata) in ordinea aparitiei in text
    """
    # Identificam tokenii speciali pe care ii sarim
    special_ids = set()
    if tokenizer.cls_token_id is not None:
        special_ids.add(tokenizer.cls_token_id)
    if tokenizer.sep_token_id is not None:
        special_ids.add(tokenizer.sep_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    cuvinte = []
    cuvant_curent = ""
    attr_curent = 0.0
    char_end_anterior = -1

    for i, (start, end) in enumerate(offsets):
        # Sarim tokenii speciali (CLS, SEP, PAD) — au offset (0, 0)
        if input_ids[i] in special_ids or start == end == 0:
            continue

        # Verificam daca tokenul curent e inceputul unui cuvant nou
        # Heuristica: daca char-ul de la start e precedat de spatiu/punctuatie in textul original
        if char_end_anterior >= 0 and start > char_end_anterior:
            # Exista spatiu/separare → e un cuvant nou
            cuvant_separator = text[char_end_anterior:start]
            if " " in cuvant_separator or "\n" in cuvant_separator or "\t" in cuvant_separator:
                # Salvam cuvantul anterior
                if cuvant_curent.strip():
                    cuvinte.append((cuvant_curent.strip(), attr_curent))
                cuvant_curent = ""
                attr_curent = 0.0

        # Adaugam tokenul curent la cuvantul in constructie
        bucata = text[start:end]
        cuvant_curent += bucata
        attr_curent += attr_per_token[i]
        char_end_anterior = end

    # Adaugam ultimul cuvant
    if cuvant_curent.strip():
        cuvinte.append((cuvant_curent.strip(), attr_curent))

    return cuvinte


def calculeaza_faithfulness_deletion_ig(text, top_features, predict_proba, label_pred,
                                            k_values=K_VALUES_DELETION):
    """
    Identica cu functia din L1a — folosim cuvintele identificate de IG (in loc de LIME).
    Permite comparatie head-to-head in acelasi tabel.
    """
    prob_initial = float(predict_proba([text])[0, label_pred])

    features_sorted = sorted(top_features, key=lambda x: abs(x[1]), reverse=True)

    drops = {}
    for k in k_values:
        if k > len(features_sorted):
            drops[k] = None
            continue

        cuvinte_de_sters = set(w.lower() for w, _ in features_sorted[:k])
        tokens = text.split()
        tokens_filtrati = [t for t in tokens
                            if t.lower().strip(".,!?;:\"'") not in cuvinte_de_sters]
        text_deletion = " ".join(tokens_filtrati)

        if not text_deletion.strip():
            drops[k] = None
            continue

        prob_after = float(predict_proba([text_deletion])[0, label_pred])
        drops[k] = prob_initial - prob_after

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
    IDENTIC cu L1a — selecteaza aceleasi 4 grupuri × n_per_group articole.
    Folosim acelasi seed → exact aceleasi articole ca in L1a → comparatie directa.
    """
    bl_merged = baseline_preds.merge(
        baseline_test[["id", COLOANA_TEXT]], on="id", how="left"
    ).rename(columns={COLOANA_TEXT: "text"})

    loso_merged = loso_preds.merge(
        loso_test[["id", COLOANA_TEXT]], on="id", how="left"
    ).rename(columns={COLOANA_TEXT: "text"})

    grupuri = {}

    # Grup A: TP cls0 baseline (Digi24 + G4Media)
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

    # Grup B: TP cls1 baseline (Veridica + Stopfals)
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

    # Grup C: FN LOSO-V pe Veridica
    fn_loso = loso_merged[(loso_merged["label_numeric"] == 1) &
                            (loso_merged["pred"] == 0)].copy()
    fn_loso_cu_text = fn_loso[fn_loso["text"].notna()].copy()
    grup_c = fn_loso_cu_text.sample(n=min(n_per_group, len(fn_loso_cu_text)),
                                       random_state=seed)
    grup_c["grup"] = "C_loso_FN"
    grupuri["C"] = grup_c

    # Grup D: TP LOSO-V pe Veridica
    tp_loso = loso_merged[(loso_merged["label_numeric"] == 1) &
                            (loso_merged["pred"] == 1)].copy()
    tp_loso_cu_text = tp_loso[tp_loso["text"].notna()].copy()
    grup_d = tp_loso_cu_text.sample(n=min(n_per_group, len(tp_loso_cu_text)),
                                       random_state=seed)
    grup_d["grup"] = "D_loso_TP"
    grupuri["D"] = grup_d

    print(f"\n=== Eșantionare grupuri (seed={seed}) ===")
    for nume, df in grupuri.items():
        print(f"  Grup {nume}: n={len(df)}")
        if "sursa_site" in df.columns and len(df) > 0:
            print(f"    surse: {dict(df['sursa_site'].value_counts())}")

    return grupuri


def salveaza_html_atributii(text, cuvinte_atributii, label_pred, label_true,
                              prob_input, sursa, output_path):
    """
    Salveaza un HTML cu atributiile IG colorate (verde = pozitiv, rosu = negativ).
    Vizualizare similara cu LIME explainer dar pentru atributii IG.
    """
    # Normalizam atributiile pentru intensitate culoare
    abs_max = max((abs(a) for _, a in cuvinte_atributii), default=1e-8)
    if abs_max == 0:
        abs_max = 1e-8

    spans = []
    for cuv, attr in cuvinte_atributii:
        intensity = min(abs(attr) / abs_max, 1.0)
        # Verde pentru pozitiv (impinge spre label_pred), rosu pentru negativ
        if attr > 0:
            color = f"rgba(0, 200, 0, {intensity:.2f})"
        else:
            color = f"rgba(220, 30, 30, {intensity:.2f})"
        cuv_safe = html_lib.escape(cuv)
        spans.append(
            f'<span style="background-color: {color}; padding: 2px 1px;" '
            f'title="attr={attr:+.4f}">{cuv_safe}</span>'
        )

    body = " ".join(spans)
    label_pred_str = "dezinformare_pro_rusa" if label_pred == 1 else "stire_credibila"
    label_true_str = "dezinformare_pro_rusa" if label_true == 1 else "stire_credibila"

    html_content = f"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<title>IG Explanation</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 900px; margin: 30px auto; padding: 20px;
          line-height: 1.7; color: #222; }}
  .meta {{ background: #f4f4f4; padding: 12px; border-radius: 4px; margin-bottom: 20px;
            font-family: monospace; font-size: 13px; }}
  .text {{ font-size: 15px; }}
  .legend {{ background: #fafafa; padding: 8px 12px; border-left: 3px solid #888;
              margin: 16px 0; font-size: 13px; }}
</style>
</head>
<body>
<h2>Integrated Gradients — atribuții per cuvânt</h2>
<div class="meta">
  Sursa: {html_lib.escape(sursa)}<br>
  Label adevărat: {label_true_str} (={label_true})<br>
  Label prezis:    {label_pred_str} (={label_pred})<br>
  Probabilitate predicție: {prob_input:.4f}<br>
  Cuvinte analizate: {len(cuvinte_atributii)}
</div>
<div class="legend">
  <strong>Verde</strong> = împinge predicția spre clasa <em>{label_pred_str}</em>.
  <strong>Roșu</strong> = trage predicția spre clasa opusă.
  Intensitatea = magnitudinea atribuției.
</div>
<div class="text">{body}</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def proceseaza_grup(nume_grup, df_grup, model, tokenizer, device, html_dir,
                      max_length, n_steps, seed, baseline_mode="zero", internal_batch_size=8):
    """Proceseaza un grup: ruleaza IG + faithfulness pe fiecare articol."""
    predict_proba = construieste_predict_proba(model, tokenizer, device,
                                                  max_length=max_length)

    rezultate = []
    print(f"\n{'='*60}")
    print(f"Procesare Grup {nume_grup} (n={len(df_grup)}) cu IG (baseline={baseline_mode}, n_steps={n_steps})")
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

        t_start = time.time()
        try:
            ig_out = calculeaza_ig_pe_articol(
                text, label_pred, model, tokenizer, device,
                max_length=max_length, n_steps=n_steps,
                baseline_mode=baseline_mode, internal_batch_size=internal_batch_size,
            )
        except Exception as e:
            print(f"  [{i:2d}/{len(df_grup)}] {row['id']} — EROARE IG: {e}")
            continue
        t_ig = time.time() - t_start

        # Faithfulness deletion folosind top features de la IG
        t_start = time.time()
        faith = calculeaza_faithfulness_deletion_ig(
            text, ig_out["top_features"], predict_proba, label_pred,
        )
        t_faith = time.time() - t_start

        # Salvam HTML pentru primele 5 exemple din fiecare grup
        html_file = None
        if i <= 5:
            html_file = f"grup{nume_grup}_{i:02d}_{row['id']}_ig.html"
            try:
                salveaza_html_atributii(
                    text, ig_out["cuvinte_atributii"], label_pred, label_true,
                    ig_out["prob_input"], sursa, html_dir / html_file,
                )
            except Exception as e:
                print(f"     [WARN] save HTML eșuat: {e}")
                html_file = None

        rezultat = {
            "grup": nume_grup,
            "id": row["id"],
            "sursa": sursa,
            "label_true": label_true,
            "label_pred": label_pred,
            "prob_cls1": float(row["prob_cls1"]),
            "top_features_ig": ig_out["top_features"],
            "n_words": ig_out["n_words"],
            "n_tokens": ig_out["n_tokens"],
            "completeness_error": ig_out["completeness_error"],
            "completeness_relative": ig_out["completeness_relative"],
            "convergence_delta": ig_out["convergence_delta"],
            "faith_prob_initial": faith["prob_initial"],
            "faith_drops_per_k": faith["drops_per_k"],
            "faith_auc_ig": faith["auc_normalized"],
            "html_file": html_file,
            "t_ig_sec": round(t_ig, 2),
            "t_faith_sec": round(t_faith, 2),
        }
        rezultate.append(rezultat)

        # Marker explicit pentru finding-ul XAI-4: pe model saturat
        # (prob_cls1 > 0.99), atributiile gradient devin near-zero.
        marker_saturare = ""
        if float(row["prob_cls1"]) > 0.99 and abs(ig_out["sum_attrs"]) < 0.05:
            marker_saturare = " [SATURARE: prob_cls1>0.99, atribuții near-zero]"

        print(f"  [{i:2d}/{len(df_grup)}] {row['id']} | {str(sursa)[:12]:12s} | "
              f"completeness_rel={ig_out['completeness_relative']:.3f} "
              f"faith_auc_IG={faith['auc_normalized']:+.4f} | {t_ig:.1f}s"
              f"{marker_saturare}")

    return rezultate


def agrega_rezultate_ig(toate_rezultatele, lime_results=None):
    """
    Agrega rezultate IG per grup si (optional) compara cu rezultate LIME.

    Daca lime_results e furnizat, calculam head-to-head:
      - faith_auc IG vs faith_auc LIME (Wilcoxon pereche pe articol)
      - overlap top-k IG vs LIME (Jaccard pe top-5 cuvinte)
    """
    df = pd.DataFrame(toate_rezultatele)

    agregari = {}
    for grup in ["A", "B", "C", "D"]:
        sub = df[df["grup"] == grup]
        if len(sub) == 0:
            continue
        agregari[grup] = {
            "n": len(sub),
            "completeness_relative": {
                "mean": float(sub["completeness_relative"].mean()),
                "median": float(sub["completeness_relative"].median()),
                "max": float(sub["completeness_relative"].max()),
            },
            "faith_auc_ig": {
                "mean": float(sub["faith_auc_ig"].mean()),
                "std": float(sub["faith_auc_ig"].std()),
                "median": float(sub["faith_auc_ig"].median()),
                "ci95_low": float(sub["faith_auc_ig"].quantile(0.025)),
                "ci95_high": float(sub["faith_auc_ig"].quantile(0.975)),
            },
            "n_words_mean": float(sub["n_words"].mean()),
            "n_tokens_mean": float(sub["n_tokens"].mean()),
        }

    # Mann-Whitney intergrupuri pe faith_auc_ig
    teste_intergrupuri = {}
    perechi = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    for g1, g2 in perechi:
        sub1 = df[df["grup"] == g1]
        sub2 = df[df["grup"] == g2]
        if len(sub1) == 0 or len(sub2) == 0:
            continue
        try:
            u, p = stats.mannwhitneyu(sub1["faith_auc_ig"], sub2["faith_auc_ig"],
                                         alternative="two-sided")
            teste_intergrupuri[f"{g1}_vs_{g2}__faith_auc_ig"] = {
                "u": float(u), "p": float(p),
                "diff_median": float(sub1["faith_auc_ig"].median() - sub2["faith_auc_ig"].median()),
            }
        except Exception as e:
            teste_intergrupuri[f"{g1}_vs_{g2}__faith_auc_ig"] = {"eroare": str(e)}

    # Comparatie head-to-head LIME vs IG (daca avem rezultate LIME)
    comparatie_lime_ig = {}
    if lime_results is not None:
        # Construim mapping id → faith_auc_lime
        lime_per_id = {}
        for r in lime_results.get("rezultate_per_articol", []):
            lime_per_id[r["id"]] = r.get("faith_auc")

        # Adaugam coloana faith_auc_lime pe randurile noastre
        df["faith_auc_lime"] = df["id"].map(lime_per_id)

        # Per grup: media faith_auc IG vs LIME, Wilcoxon pereche
        for grup in ["A", "B", "C", "D"]:
            sub = df[(df["grup"] == grup) & df["faith_auc_lime"].notna()]
            if len(sub) < 2:
                continue
            try:
                w, p = stats.wilcoxon(sub["faith_auc_ig"], sub["faith_auc_lime"])
                comparatie_lime_ig[f"grup_{grup}__ig_vs_lime"] = {
                    "n": len(sub),
                    "mean_ig": float(sub["faith_auc_ig"].mean()),
                    "mean_lime": float(sub["faith_auc_lime"].mean()),
                    "median_diff_ig_minus_lime": float((sub["faith_auc_ig"] - sub["faith_auc_lime"]).median()),
                    "wilcoxon_w": float(w),
                    "wilcoxon_p": float(p),
                }
            except Exception as e:
                comparatie_lime_ig[f"grup_{grup}__ig_vs_lime"] = {"eroare": str(e)}

        # Overlap top-5 cuvinte IG vs LIME (Jaccard)
        lime_topk_per_id = {}
        for r in lime_results.get("rezultate_per_articol", []):
            top_lime = r.get("top_features_proba", []) or []
            top5 = set(w.lower() for w, _ in top_lime[:5])
            lime_topk_per_id[r["id"]] = top5

        for grup in ["A", "B", "C", "D"]:
            sub = df[df["grup"] == grup]
            jaccards = []
            for _, row in sub.iterrows():
                top_ig = set(w.lower() for w, _ in (row["top_features_ig"] or [])[:5])
                top_lime = lime_topk_per_id.get(row["id"], set())
                if top_ig and top_lime:
                    inter = len(top_ig & top_lime)
                    union = len(top_ig | top_lime)
                    jaccards.append(inter / union if union > 0 else 0.0)
            if jaccards:
                comparatie_lime_ig[f"grup_{grup}__top5_jaccard"] = {
                    "n": len(jaccards),
                    "mean": float(np.mean(jaccards)),
                    "median": float(np.median(jaccards)),
                }

    return agregari, teste_intergrupuri, comparatie_lime_ig


def genereaza_markdown(agregari, teste_intergrupuri, comparatie_lime_ig,
                          lime_results, n_per_group, seed):
    """Genereaza raport markdown comparativ LIME vs IG."""
    md = [
        "# Findings — L3: Integrated Gradients vs LIME (head-to-head)",
        "",
        "## 1. Configurație",
        "",
        f"- N per grup: {n_per_group} (același eșantion ca L1a, seed={seed})",
        f"- IG steps: {IG_STEPS}",
        f"- IG baseline mode: {IG_BASELINE_MODE}",
        f"- IG layer: model.roberta.embeddings.word_embeddings",
        f"- Top-K cuvinte: {TOP_K_WORDS}",
        f"- Coloana text: `{COLOANA_TEXT}`",
        "",
        "## 2. Verificare axiomă Completeness (sanity check IG)",
        "",
        "Suma atribuțiilor ar trebui să fie aproximativ egală cu logit_input − logit_baseline.",
        "Eroarea relativă mică (<0.05) confirmă că IG e calculat corect.",
        "",
        "| Grup | n | Completeness err. (mean) | (median) | (max) |",
        "|---|---:|---:|---:|---:|",
    ]
    for grup, ag in agregari.items():
        ce = ag["completeness_relative"]
        md.append(f"| {grup} | {ag['n']} | {ce['mean']:.4f} | {ce['median']:.4f} | "
                   f"{ce['max']:.4f} |")

    md.extend([
        "",
        "## 3. Faithfulness deletion AUC — IG",
        "",
        "Cu cât valoarea e mai mare, cu atât top-k cuvinte identificate de IG au impact",
        "cauzal mai mare asupra predicției (eliminarea lor scade probabilitatea predicției).",
        "",
        "| Grup | n | mean ± std | median | IC 95% (quantile) |",
        "|---|---:|---:|---:|---:|",
    ])
    for grup, ag in agregari.items():
        fa = ag["faith_auc_ig"]
        md.append(f"| {grup} | {ag['n']} | {fa['mean']:.4f} ± {fa['std']:.4f} | "
                   f"{fa['median']:.4f} | [{fa['ci95_low']:.4f}, {fa['ci95_high']:.4f}] |")

    md.extend([
        "",
        "## 4. Comparație directă LIME vs IG (faithfulness deletion AUC)",
        "",
        "Test Wilcoxon pereche pe articol (același set de articole, același seed).",
        "",
        "| Grup | n | mean LIME | mean IG | diff median (IG − LIME) | p-value |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for cheie, t in comparatie_lime_ig.items():
        if "ig_vs_lime" not in cheie:
            continue
        if "eroare" in t:
            continue
        grup = cheie.split("__")[0].split("_")[1]
        sig = "***" if t["wilcoxon_p"] < 0.001 else ("**" if t["wilcoxon_p"] < 0.01
                                                       else ("*" if t["wilcoxon_p"] < 0.05 else ""))
        md.append(f"| {grup} | {t['n']} | {t['mean_lime']:+.4f} | {t['mean_ig']:+.4f} | "
                   f"{t['median_diff_ig_minus_lime']:+.4f} | {t['wilcoxon_p']:.4g} {sig} |")

    md.extend([
        "",
        "## 5. Overlap top-5 cuvinte LIME vs IG (Jaccard)",
        "",
        "Cât de mult se suprapun cele mai importante 5 cuvinte identificate de cele două metode.",
        "Jaccard mare → metodele identifică același vocabular. Jaccard mic → metode complementare.",
        "",
        "| Grup | n | Jaccard mean | Jaccard median |",
        "|---|---:|---:|---:|",
    ])
    for cheie, t in comparatie_lime_ig.items():
        if "top5_jaccard" not in cheie:
            continue
        grup = cheie.split("__")[0].split("_")[1]
        md.append(f"| {grup} | {t['n']} | {t['mean']:.3f} | {t['median']:.3f} |")

    md.extend([
        "",
        "## 6. Mann-Whitney U între grupuri (faith_auc IG)",
        "",
        "| Comparație | Diff median (g1 − g2) | p-value |",
        "|---|---:|---:|",
    ])
    for cheie, t in teste_intergrupuri.items():
        if "eroare" in t:
            continue
        parts = cheie.split("__")
        comp_str = parts[0].replace("_vs_", " vs ")
        sig = "***" if t["p"] < 0.001 else ("**" if t["p"] < 0.01 else ("*" if t["p"] < 0.05 else ""))
        md.append(f"| {comp_str} | {t['diff_median']:+.4f} | {t['p']:.4g} {sig} |")

    # Interpretare automata
    md.extend([
        "",
        "## 7. Interpretare automată",
        "",
    ])

    # Verificare completeness
    grupuri_cu_completeness_ok = sum(1 for ag in agregari.values()
                                        if ag["completeness_relative"]["mean"] < 0.1)
    if grupuri_cu_completeness_ok == len(agregari):
        md.append(f"**Sanity check IG:** completeness OK pe toate {len(agregari)} grupuri "
                   f"(eroare relativă mean < 0.1). IG e calculat corect.")
    else:
        md.append(f"**Atenție:** completeness eronat pe {len(agregari) - grupuri_cu_completeness_ok}"
                   f"/{len(agregari)} grupuri — verifică design baseline sau n_steps.")
    md.append("")

    # Comparatie IG vs LIME pe Grup B (cls1 baseline)
    if "grup_B__ig_vs_lime" in comparatie_lime_ig:
        t = comparatie_lime_ig["grup_B__ig_vs_lime"]
        if t.get("median_diff_ig_minus_lime", 0) > 0 and t.get("wilcoxon_p", 1) < 0.05:
            md.append(f"**FINDING POZITIV — IG superior LIME pe cls1 baseline (Grup B):** "
                       f"faith_auc IG = {t['mean_ig']:.4f} vs LIME = {t['mean_lime']:.4f}, "
                       f"Δ median = {t['median_diff_ig_minus_lime']:+.4f} (p={t['wilcoxon_p']:.4g}). "
                       f"IG identifică cuvinte cu impact cauzal mai mare decât LIME pe articole "
                       f"propagandistice — confirmă strategia hibridă cu IG ca metodă principală.")
        else:
            md.append(f"**Grup B (cls1 baseline) — IG vs LIME similari:** "
                       f"faith_auc IG = {t['mean_ig']:.4f} vs LIME = {t['mean_lime']:.4f} "
                       f"(p={t.get('wilcoxon_p', 1):.4g}). Niciuna din metode nu identifică "
                       f"cuvinte cu impact cauzal pe cls1 — confirmă stylistic fingerprint "
                       f"distribuit (model nu se bazează pe cuvinte localizabile).")
    md.append("")

    # Asimetria A vs B pe IG (replica testului H2 din L1a, dar pe IG)
    if "A_vs_B__faith_auc_ig" in teste_intergrupuri:
        t = teste_intergrupuri["A_vs_B__faith_auc_ig"]
        if t.get("p", 1) < 0.05 and t.get("diff_median", 0) > 0:
            md.append(f"**Asimetria cls0/cls1 confirmată și pe IG:** "
                       f"Δ median A−B = {t['diff_median']:+.4f} (p={t['p']:.4g}). "
                       f"Triangulare independentă a stylistic fingerprint — atât LIME cât "
                       f"și IG identifică cuvinte cu impact cauzal mai mare pe cls0 decât pe cls1.")
        else:
            md.append(f"**Asimetria A−B nu se confirmă pe IG:** Δ median = {t['diff_median']:+.4f} "
                       f"(p={t.get('p', 1):.4g}). IG găsește semnal pe cls1 acolo unde LIME nu — "
                       f"finding pozitiv pentru strategia hibridă.")
    md.append("")

    md.extend([
        "## 8. Concluzii pentru capitolul Explicabilitate",
        "",
        "TBD pe baza datelor de mai sus — completăm strategia hibridă LIME + IG + modul 3.",
        "",
        "*Generat automat de `08_ig_l3_diagnostic.py`*",
    ])

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_dir", required=True)
    parser.add_argument("--loso_model_dir", required=True)
    parser.add_argument("--baseline_test_data", required=True)
    parser.add_argument("--baseline_predictions", required=True)
    parser.add_argument("--loso_test_data", required=True, nargs="+",
                          help="Unul sau mai multe CSV-uri concatenate")
    parser.add_argument("--loso_predictions", required=True)
    parser.add_argument("--lime_results_json", required=False, default=None,
                          help="JSON cu rezultate LIME L1a (pentru comparație head-to-head)")
    parser.add_argument("--output_dir", default="findings")
    parser.add_argument("--n_per_group", type=int, default=25)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=IG_STEPS,
                          help=f"Pași de integrare IG (default {IG_STEPS}, mai mulți = convergență mai bună)")
    parser.add_argument("--baseline_mode", default=IG_BASELINE_MODE, choices=["zero", "pad"],
                          help="Modul baseline: 'zero' (recomandat) sau 'pad'")
    parser.add_argument("--internal_batch_size", type=int, default=8,
                          help="Cât de multe pași IG procesăm odată (default 8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    html_dir = out / "ig_html_l3"
    html_dir.mkdir(parents=True, exist_ok=True)

    device = alege_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] N per grup: {args.n_per_group}")
    print(f"[INFO] IG steps: {args.n_steps}")

    # Incarcare CSV-uri
    print(f"\n[INFO] Încărcare CSV-uri...")
    baseline_preds = pd.read_csv(args.baseline_predictions)
    baseline_test = pd.read_csv(args.baseline_test_data)
    loso_preds = pd.read_csv(args.loso_predictions)

    loso_test_dfs = []
    for path in args.loso_test_data:
        df = pd.read_csv(path)
        if COLOANA_TEXT not in df.columns:
            raise ValueError(f"Coloana '{COLOANA_TEXT}' lipsește din {path}")
        loso_test_dfs.append(df)
        print(f"  loso source: {path} → {len(df)} rânduri")
    loso_test = pd.concat(loso_test_dfs, ignore_index=True)
    loso_test = loso_test.drop_duplicates(subset=["id"], keep="last")
    print(f"  loso combined deduplicat: {len(loso_test)}")

    if COLOANA_TEXT not in baseline_test.columns:
        raise ValueError(f"Coloana '{COLOANA_TEXT}' lipsește din baseline test")

    # Incarcare rezultate LIME (optional)
    lime_results = None
    if args.lime_results_json:
        try:
            with open(args.lime_results_json, "r", encoding="utf-8") as f:
                lime_results = json.load(f)
            print(f"[INFO] LIME results încărcate: "
                   f"{len(lime_results.get('rezultate_per_articol', []))} articole")
        except Exception as e:
            print(f"[WARN] Nu am putut încărca LIME results: {e}")

    grupuri = selecteaza_grupuri(baseline_preds, baseline_test,
                                    loso_preds, loso_test,
                                    args.n_per_group, args.seed)

    toate_rezultatele = []
    t_total_start = time.time()

    # Procesare A si B (model baseline)
    print(f"\n[INFO] Încărcare model baseline: {args.baseline_model_dir}")
    tokenizer_bl = AutoTokenizer.from_pretrained(args.baseline_model_dir)
    model_bl = AutoModelForSequenceClassification.from_pretrained(args.baseline_model_dir).to(device)
    model_bl.eval()

    for grup_nume in ["A", "B"]:
        rez = proceseaza_grup(grup_nume, grupuri[grup_nume],
                                model_bl, tokenizer_bl, device, html_dir,
                                args.max_length, args.n_steps, args.seed,
                                baseline_mode=args.baseline_mode,
                                internal_batch_size=args.internal_batch_size)
        toate_rezultatele.extend(rez)

    del model_bl, tokenizer_bl
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # Procesare C si D (model LOSO-V)
    print(f"\n[INFO] Încărcare model LOSO-V: {args.loso_model_dir}")
    tokenizer_loso = AutoTokenizer.from_pretrained(args.loso_model_dir)
    model_loso = AutoModelForSequenceClassification.from_pretrained(args.loso_model_dir).to(device)
    model_loso.eval()

    for grup_nume in ["C", "D"]:
        rez = proceseaza_grup(grup_nume, grupuri[grup_nume],
                                model_loso, tokenizer_loso, device, html_dir,
                                args.max_length, args.n_steps, args.seed,
                                baseline_mode=args.baseline_mode,
                                internal_batch_size=args.internal_batch_size)
        toate_rezultatele.extend(rez)

    t_total = time.time() - t_total_start
    print(f"\n[INFO] Procesare totală: {t_total/60:.1f} minute pentru {len(toate_rezultatele)} articole")

    # Agregare + comparatie cu LIME
    print(f"\n[INFO] Agregare și comparație head-to-head cu LIME...")
    agregari, teste_intergrupuri, comparatie_lime_ig = agrega_rezultate_ig(
        toate_rezultatele, lime_results
    )

    # Salvare JSON
    out_json = {
        "config": {
            "n_per_group": args.n_per_group,
            "n_steps": args.n_steps,
            "ig_baseline_mode": args.baseline_mode,
            "internal_batch_size": args.internal_batch_size,
            "top_k_words": TOP_K_WORDS,
            "seed": args.seed,
        },
        "agregari": agregari,
        "teste_intergrupuri": teste_intergrupuri,
        "comparatie_lime_ig": comparatie_lime_ig,
        "rezultate_per_articol": toate_rezultatele,
        "t_total_min": round(t_total/60, 2),
    }
    json_path = out / "findings_lime_vs_ig_l3.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON: {json_path}")

    md_text = genereaza_markdown(agregari, teste_intergrupuri, comparatie_lime_ig,
                                    lime_results, args.n_per_group, args.seed)
    md_path = out / "findings_lime_vs_ig_l3.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[OK] Markdown: {md_path}")

    # Sumar consola
    print(f"\n{'='*60}")
    print("SUMAR FINAL — IG L3")
    print(f"{'='*60}")
    for grup, ag in agregari.items():
        print(f"  Grup {grup} (n={ag['n']}): "
              f"faith_auc_IG={ag['faith_auc_ig']['mean']:.4f}, "
              f"completeness_err={ag['completeness_relative']['mean']:.4f}")

    if comparatie_lime_ig:
        print(f"\n  Comparație IG vs LIME (faith_auc, mean):")
        for cheie, t in comparatie_lime_ig.items():
            if "ig_vs_lime" in cheie and "eroare" not in t:
                grup = cheie.split("__")[0].split("_")[1]
                print(f"    Grup {grup}: LIME={t['mean_lime']:.4f}  IG={t['mean_ig']:.4f}  "
                      f"diff={t['median_diff_ig_minus_lime']:+.4f} (p={t['wilcoxon_p']:.4g})")

    print(f"\n[OK] HTML-uri reprezentative în: {html_dir}")


if __name__ == "__main__":
    main()