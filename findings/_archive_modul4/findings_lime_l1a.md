# Findings — LIME L1a (diagnostic fidelity, 4 grupuri × 25)

## 1. Configurație

- N per grup: 25
- num_features = 15, num_samples = 1000, bow = False
- seed = 42
- Coloana text input: `text_curat`

**Grupuri:**
- A: TP cls0 baseline modul 2 (control — Digi24/G4Media)
- B: TP cls1 baseline modul 2 (replica finding 0.06 — Veridica/Stopfals)
- C: FN LOSO-V pe Veridica (modelul ratează propaganda)
- D: TP LOSO-V pe Veridica (modelul prinde propaganda fără amprentă)

## 2. Rezultate fidelity (R²)

### R² pe softmax probabilities

| Grup | n | mean ± std | median | IC 95% (quantile) |
|---|---:|---:|---:|---:|
| A | 25 | 0.3492 ± 0.1710 | 0.3494 | [0.0916, 0.5865] |
| B | 25 | 0.1419 ± 0.0500 | 0.1289 | [0.0753, 0.2509] |
| C | 25 | 0.5250 ± 0.1762 | 0.6020 | [0.1435, 0.6898] |
| D | 25 | 0.3316 ± 0.2758 | 0.3892 | [0.0085, 0.7676] |

### R² pe logits raw

| Grup | n | mean ± std | median | IC 95% (quantile) |
|---|---:|---:|---:|---:|
| A | 25 | 0.4378 ± 0.1477 | 0.4692 | [0.1085, 0.6274] |
| B | 25 | 0.4357 ± 0.0631 | 0.4334 | [0.3153, 0.5294] |
| C | 25 | 0.5264 ± 0.1962 | 0.6005 | [0.1201, 0.7253] |
| D | 25 | 0.3896 ± 0.2985 | 0.5427 | [0.0310, 0.7782] |

### Faithfulness deletion AUC (drop mediu pe k=1,3,5,10)

| Grup | n | mean ± std | median |
|---|---:|---:|---:|
| A | 25 | 0.1687 ± 0.2814 | 0.0003 |
| B | 25 | -0.0001 ± 0.0003 | -0.0000 |
| C | 25 | 0.1268 ± 0.1718 | 0.0088 |
| D | 25 | -0.0024 ± 0.0378 | -0.0005 |

## 3. Teste statistice — comparații între grupuri (Mann-Whitney U)

| Comparație | Metrică | Diff median (g1 − g2) | p-value |
|---|---|---:|---:|
| A vs B | r2_proba | +0.2205 | 3.213e-06 *** |
| A vs B | r2_logits | +0.0358 | 0.3721  |
| A vs B | faith_auc | +0.0003 | 0.001194 ** |
| A vs C | r2_proba | -0.2526 | 0.0002646 *** |
| A vs C | r2_logits | -0.1313 | 0.01529 * |
| A vs C | faith_auc | -0.0086 | 0.6004  |
| A vs D | r2_proba | -0.0399 | 0.5475  |
| A vs D | r2_logits | -0.0735 | 0.9381  |
| A vs D | faith_auc | +0.0008 | 0.0004785 *** |
| B vs C | r2_proba | -0.4731 | 2.297e-08 *** |
| B vs C | r2_logits | -0.1671 | 0.001671 ** |
| B vs C | faith_auc | -0.0088 | 2.777e-05 *** |
| B vs D | r2_proba | -0.2603 | 0.2772  |
| B vs D | r2_logits | -0.1093 | 0.8009  |
| B vs D | faith_auc | +0.0005 | 0.06251  |
| C vs D | r2_proba | +0.2128 | 0.005206 ** |
| C vs D | r2_logits | +0.0578 | 0.1511  |
| C vs D | faith_auc | +0.0094 | 3.025e-05 *** |

## 4. Teste pereche — logits vs proba (Wilcoxon, în cadrul aceluiași grup)

Testează ipoteza H1: trecerea de la softmax la logits crește R² (efect saturare).

| Grup | Diff median (logits − proba) | p-value |
|---|---:|---:|
| A | +0.0578 | 0.0004895 *** |
| B | +0.2936 | 5.96e-08 *** |
| C | -0.0004 | 0.9578  |
| D | +0.0374 | 7.498e-05 *** |

## 5. Interpretare ipoteze

**H1 PARȚIAL CONFIRMATĂ:** logits îmbunătățește R² doar pe 2/4 grupuri (A, B). Efectul saturare e prezent dar nu uniform.

**H2 NECONFIRMATĂ:** Δ R²(A−B) pe logits = +0.0021 (p=0.3721). Asimetria nu persistă semnificativ pe logits — saturarea explică majoritatea diferenței.

**H3 CONFIRMATĂ:** faithfulness AUC(A) > AUC(B), Δ=+0.1687 (p=0.001194). Cuvintele top-k LIME au impact cauzal mai mare pe cls0 decât pe cls1 — confirmă că modelul nu se bazează pe cuvinte localizate pentru cls1.

### Diagnostic LOSO-V (Grup C vs D)

- C (FN, model ratează): R²_proba = 0.5250, R²_logits = 0.5264
- D (TP, model prinde): R²_proba = 0.3316, R²_logits = 0.3896
- Diferența nesemnificativă (p=0.1511).

## 6. Concluzii pentru capitolul Explicabilitate al tezei

TBD — pe baza rezultatelor de mai sus, decidem strategia hibridă LIME + IG.

Întrebări deschise pentru pasul L2 (diagnostic detaliat):
- Dacă H1 confirmată: refacem rapoartele LIME oficiale pe `predict_logits`
- Dacă H2 confirmată: documentăm stylistic fingerprint ca limitare LIME intrinsecă
- Indiferent de rezultate: trecem la pasul L3 (Integrated Gradients) pentru triangulare

*Generat automat de `07_lime_l1a_diagnostic.py`*