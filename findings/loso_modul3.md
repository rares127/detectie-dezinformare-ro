# LOSO pe modulul 3 — testul cross-source ultim

Verifică dacă scorul granular cosine generalizează cross-source. Răspunde la întrebarea-cheie din `DOSAR_problema_generalizare.md`: modulul 3 compensează stylistic fingerprint-ul modulului 2 (LOSO-V recall = 29.35%)?

## Configurare

- Model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Seed: `42`, Device: `mps`
- Variantă LOSO: **B (cls0 intact, fără paritate forțată)**
- Test set: `data/processed/subset_benchmark_v3_curat.parquet` (167 articole, 2,066 prop.)

## Scenarii rulate

### `baseline_standard`

_Corpus cls1 complet (Veridica + Stopfals downsampled la 5,290) — referință din benchmark v4 post-curățare._

- Corpus cls0: 5,290 prop.
- Corpus cls1: 5,290 prop.
- Distribuție sursă cls1: {'veridica.ro': 4907, 'stopfals.md': 383}

#### Test A — scor_cls1 izolat

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.7242 | +0.7075 | 0.6634 | 0.7015 | 0.7016 | 0.7015 |
| min | 0.7196 | +0.7835 | 0.5193 | 0.5811 | 0.5839 | 0.5815 |
| p10 | 0.7080 | +0.6966 | 0.5722 | 0.6188 | 0.6270 | 0.6198 |

#### Test B — scor_cls0 izolat (verificare convenție)

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.3054 | -0.7040 | 0.7007 | 0.6634 | 0.6549 | 0.6624 |
| min | 0.5214 | +0.1024 | 0.5420 | 0.5486 | 0.5658 | 0.5506 |
| p10 | 0.4200 | -0.3096 | 0.6076 | 0.5864 | 0.5883 | 0.5866 |

#### Test D — diferență cls1 − cls0 (scor combinat)

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.9690 | +2.4207 | -0.0373 | 0.0380 | 0.0467 | 0.0390 |
| min | 0.7763 | +1.0354 | -0.0227 | 0.0325 | 0.0181 | 0.0309 |
| p10 | 0.8989 | +1.7126 | -0.0354 | 0.0324 | 0.0387 | 0.0332 |

### `loso_v`

_LOSO-V: corpus cls1 doar Stopfals (Veridica scoasă). Test articolele Veridica din test set sunt detectate prin similaritate cu Stopfals?_

- Corpus cls0: 5,290 prop.
- Corpus cls1: 383 prop.
- Distribuție sursă cls1: {'stopfals.md': 383}

#### Test A — scor_cls1 izolat

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.6995 | +0.6586 | 0.5592 | 0.5950 | 0.6352 | 0.5997 |
| min | 0.6924 | +0.7417 | 0.4142 | 0.4747 | 0.4953 | 0.4771 |
| p10 | 0.6792 | +0.6364 | 0.4721 | 0.5137 | 0.5439 | 0.5172 |

#### Test B — scor_cls0 izolat (verificare convenție)

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.3054 | -0.7040 | 0.7007 | 0.6634 | 0.6549 | 0.6624 |
| min | 0.5214 | +0.1024 | 0.5420 | 0.5486 | 0.5658 | 0.5506 |
| p10 | 0.4200 | -0.3096 | 0.6076 | 0.5864 | 0.5883 | 0.5866 |

#### Test D — diferență cls1 − cls0 (scor combinat)

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.8922 | +1.6661 | -0.1415 | -0.0684 | -0.0197 | -0.0628 |
| min | 0.7849 | +1.0399 | -0.1278 | -0.0739 | -0.0705 | -0.0735 |
| p10 | 0.8503 | +1.3708 | -0.1355 | -0.0727 | -0.0445 | -0.0695 |

### `loso_s`

_LOSO-S: corpus cls1 doar Veridica (Stopfals scos). Simetric cu LOSO-V._

- Corpus cls0: 5,290 prop.
- Corpus cls1: 4,907 prop.
- Distribuție sursă cls1: {'veridica.ro': 4907}

#### Test A — scor_cls1 izolat

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.7144 | +0.6570 | 0.6626 | 0.6997 | 0.6826 | 0.6977 |
| min | 0.7146 | +0.7676 | 0.5169 | 0.5780 | 0.5717 | 0.5773 |
| p10 | 0.6893 | +0.6494 | 0.5699 | 0.6160 | 0.6051 | 0.6148 |

#### Test B — scor_cls0 izolat (verificare convenție)

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.3054 | -0.7040 | 0.7007 | 0.6634 | 0.6549 | 0.6624 |
| min | 0.5214 | +0.1024 | 0.5420 | 0.5486 | 0.5658 | 0.5506 |
| p10 | 0.4200 | -0.3096 | 0.6076 | 0.5864 | 0.5883 | 0.5866 |

#### Test D — diferență cls1 − cls0 (scor combinat)

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.9659 | +2.3751 | -0.0381 | 0.0363 | 0.0277 | 0.0353 |
| min | 0.7690 | +0.9998 | -0.0251 | 0.0294 | 0.0059 | 0.0267 |
| p10 | 0.8964 | +1.6743 | -0.0377 | 0.0296 | 0.0168 | 0.0281 |

## Comparație centrală — Test D mean (rezultatul principal)

Folosim `Test D mean` ca metric principal — robust la artefacte (vezi `diagnostic_v4.md` pentru justificare).

| Scenariu | n cls1 | AUC | Δ vs baseline | Cohen's d | μ(V) | μ(S) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_standard` | 5,290 | 0.9690 | — | +2.4207 | 0.0380 | 0.0467 |
| `loso_v` | 383 | 0.8922 | -0.0768 | +1.6661 | -0.0684 | -0.0197 |
| `loso_s` | 4,907 | 0.9659 | -0.0031 | +2.3751 | 0.0363 | 0.0277 |

## Comparație Test A mean (scor_cls1 izolat)

| Scenariu | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(V) | μ(S) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_standard` | 0.7242 | +0.7075 | 0.6634 | 0.7015 | 0.7015 | 0.7016 |
| `loso_v` | 0.6995 | +0.6586 | 0.5592 | 0.5997 | 0.5950 | 0.6352 |
| `loso_s` | 0.7144 | +0.6570 | 0.6626 | 0.6977 | 0.6997 | 0.6826 |

## Comparație cu modulul 2 (clasificator XLM-R)

Modulul 2 (xlmr_baseline_v2): F1 IID standard = 100%, **recall cls1 LOSO-V = 29.35%** (drop 70.65 puncte procentuale).

| Modul | Standard | LOSO-V | Drop |
|---|---:|---:|---:|
| Modul 2 (recall cls1) | 100% | 29.35% | −70.65pp |
| Modul 3 (Test D AUC mean) | 0.9690 | 0.8922 | -0.0768 |

## Concluzie finală

```
Test D (diferență cls1−cls0, agregare mean):
  Baseline standard: AUC = 0.9690
  LOSO-V (corpus = Stopfals only): AUC = 0.8922 (drop +0.0768)
  LOSO-S (corpus = Veridica only): AUC = 0.9659 (drop +0.0031)

Comparație cu modulul 2:
  Modul 2 standard (IID): F1 = 100% → recall cls1 = 100%
  Modul 2 LOSO-V: recall cls1 = 29.35% (drop 70.65pp)

VERDICT: ✓ Modulul 3 (Test D mean) GENERALIZEAZĂ EXCELENT cross-source. AUC LOSO-V = 0.8922 ≥ 0.85, drop minim (0.0768). Compensează problema modulului 2. Opțiunea 1 din DOSAR_problema_generalizare.md (raportare onestă fără remediere modulul 2) devine COMPLET ACCEPTABILĂ.
```

## Implicații pentru teză (capitolul „Evaluare cross-source

- **Finding 7 (proaspăt):** Modulul 3 generalizează cross-source semnificativ mai bine decât modulul 2 (AUC LOSO-V = 0.8922 pe modul 3 vs recall 29% pe modul 2). Confirmă ipoteza că similaritatea semantică e mai puțin vulnerabilă la stylistic fingerprint decât clasificarea.

- **Decizie metodologică:** Opțiunea 1 din `DOSAR_problema_generalizare.md` (raportare onestă fără remediere modulul 2) devine acceptabilă — modulul 3 oferă răspunsul complementar la limitarea modulului 2.

---

*Modul 3 · Pasul A3 · LOSO cross-source*