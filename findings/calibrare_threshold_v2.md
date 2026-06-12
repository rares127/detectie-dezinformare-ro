# Calibrare threshold v2 — CV pe test + calibrare val

Raport unificat. **Cifrele oficiale pentru teză** sunt în **Secțiunea A** (5-fold CV pe test). Secțiunea B (calibrare clasică pe val) e informativă, cu disclaimer despre distribution shift.

## Configurare

- Model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Seed: `42`, Device: `mps`
- Cross-validation: **5-fold stratificat** pe label
- Filtru lungime propoziție: `[7, 54]` cuvinte
- Scor: `diff_mean` (= `scor_cls1_mean − scor_cls0_mean`)

## Volume

| Set | Articole | Propoziții |
|---|---:|---:|
| Val | 222 | 1,807 |
| Test | 167 | 2,066 |
| Corpus cls0 | — | 5,290 |
| Corpus cls1 baseline (test) | — | 5,290 |
| Corpus cls1 no-val (calibrare val) | — | 4,980 |

## Secțiunea A — CV 5-fold pe test (REZULTAT OFICIAL)

Cross-validation stratificat: pentru fiecare fold, calibrăm τ pe celelalte 4 folduri (max F1) și evaluăm pe fold-ul curent. Raportăm mean ± std peste cele 5 folduri.

### Detaliu per fold

| Fold | n_train | n_test | τ | F1_train | F1_eval | Acc_eval | Prec_eval | Rec_eval |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 133 | 34 | -0.0071 | 0.9379 | 0.9787 | 0.9706 | 0.9583 | 1.0000 |
| 2 | 133 | 34 | -0.0072 | 0.9613 | 0.8837 | 0.8529 | 0.9500 | 0.8261 |
| 3 | 134 | 33 | -0.0080 | 0.9385 | 0.9778 | 0.9697 | 0.9565 | 1.0000 |
| 4 | 134 | 33 | -0.0071 | 0.9503 | 0.9302 | 0.9091 | 0.9524 | 0.9091 |
| 5 | 134 | 33 | -0.0071 | 0.9438 | 0.9565 | 0.9394 | 0.9167 | 1.0000 |

### Statistici agregate (mean ± std peste folduri)

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| **accuracy** | 0.9283 | 0.0492 | 0.8529 | 0.9706 |
| **precision_cls1** | 0.9468 | 0.0172 | 0.9167 | 0.9583 |
| **recall_cls1** | 0.9470 | 0.0782 | 0.8261 | 1.0000 |
| **f1_cls1** | 0.9454 | 0.0397 | 0.8837 | 0.9787 |

**τ mediu = -0.007276 ± 0.000397**

Cifre principale pentru teză:

- F1 = **0.9454 ± 0.0397**
- Accuracy = **0.9283 ± 0.0492**
- Precision = **0.9468 ± 0.0172**
- Recall = **0.9470 ± 0.0782**

### Aplicare τ mediu pe TOT test set-ul (retrospectiv)

Această cifră arată cum ar performa sistemul în producție folosind τ_mediu calibrat din CV. Nu e statistica oficială (care e mean ± std), dar e cifra concretă utilizabilă pentru sistem.

- τ_mediu = -0.007276
- Confusion: TP=106, FP=6, FN=6, TN=49
- F1 = 0.9464, Acc = 0.9281, Prec = 0.9464, Rec = 0.9464

### Breakdown test per sursă (la τ_mediu)

| Sursă | n | TP | FP | FN | TN | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hotnews.ro | 45 | 0 | 4 | 0 | 41 | 0.9111 | 0.0000 |
| libertatea.ro | 5 | 0 | 1 | 0 | 4 | 0.8000 | 0.0000 |
| stirileprotv.ro | 5 | 0 | 1 | 0 | 4 | 0.8000 | 0.0000 |
| stopfals.md | 13 | 13 | 0 | 0 | 0 | 1.0000 | 1.0000 |
| veridica.ro | 99 | 93 | 0 | 6 | 0 | 0.9394 | 0.9688 |

## Secțiunea B — Calibrare val (informativă)

> ⚠ **Disclaimer methodological:** calibrarea pe val a expus un distribution shift între val și test:
>
> - **Val cls0** = articole din Digi24 + G4Media (aceleași surse din corpusul cls0!) → similaritate artificial mare → F1 val perfect (1.0000)
> - **Test cls0** = HotNews, Pro TV, Libertatea (surse externe, NU în corpus)
>
> Threshold-ul calibrat pe val nu generalizează la test. De aceea folosim CV pe test (Secțiunea A) ca rezultat oficial. Această secțiune e raportată pentru transparență metodologică.

### Rezultate

| Metric | Val (la τ_val) | Test (la τ_val) |
|---|---:|---:|
| Accuracy | 1.0000 | 0.6707 |
| Precision | 1.0000 | 0.6707 |
| Recall | 1.0000 | 1.0000 |
| F1 | 1.0000 | 0.8029 |

τ_val = -0.189299

### Sanity check oracle

- τ oracle (calibrat direct pe test): -0.007077
- F1 oracle pe test: **0.9464**
- Δ(oracle − τ_val) = **+0.1436** (mare → distribution shift confirmat)
- Δ(oracle − τ_cv)  = **+0.0000** (mic → CV se apropie de optim)

## Secțiunea C — Finding metodologic

**Finding 8 (proaspăt):** Calibrarea pe val cu surse omogene (Digi24+G4Media în val cls0, identice cu sursele din corpus cls0) produce F1 perfect (1.0000) dar nu generalizează la test set cross-source (HotNews/Pro TV/Libertatea). Δ_oracle_vs_τ_val = **+0.1436**.

**Cauza:** corpus cls0 e construit din 2 surse (Digi24+G4Media), același split din care provine val cls0. Articolele val cls0 sunt same-source cu corpus → similaritate artificial mare → diff_mean extrem de negativ → separare perfectă cu cls1.

**Lecție metodologică:** corpusul de referință trebuie să fie sursă-divers ca să generalizeze cross-source. Pentru calibrare robustă pe test independent, CV pe test set (cu raportarea variabilității std) e protocolul corect, în absența unui set de validare cu distribuție similară testului.

## Comparație directă cu modulul 2

| Modul | Setup | Recall cls1 | F1 |
|---|---|---:|---:|
| Modul 2 (XLM-R) | IID standard | 100% | 100% |
| Modul 2 (XLM-R) | LOSO-V | 29.35% | — |
| **Modul 3 (scor D mean)** | **CV 5-fold pe test** | **94.70% ± 7.82%** | **0.9454 ± 0.0397** |

---

*Modul 3 · Pasul A4 · Calibrare threshold v2 (CV + val)*