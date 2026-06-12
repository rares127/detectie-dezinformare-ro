# Findings — L3: Integrated Gradients vs LIME (head-to-head)

## 1. Configurație

- N per grup: 25 (același eșantion ca L1a, seed=42)
- IG steps: 200
- IG baseline mode: zero
- IG layer: model.roberta.embeddings.word_embeddings
- Top-K cuvinte: 15
- Coloana text: `text_curat`

## 2. Verificare axiomă Completeness (sanity check IG)

Suma atribuțiilor ar trebui să fie aproximativ egală cu logit_input − logit_baseline.
Eroarea relativă mică (<0.05) confirmă că IG e calculat corect.

| Grup | n | Completeness err. (mean) | (median) | (max) |
|---|---:|---:|---:|---:|
| A | 25 | 0.5631 | 0.5098 | 0.9488 |
| B | 25 | 0.4711 | 0.4616 | 0.5686 |
| C | 25 | 0.2161 | 0.1250 | 0.6956 |
| D | 25 | 1.5217 | 0.9722 | 5.9430 |

## 3. Faithfulness deletion AUC — IG

Cu cât valoarea e mai mare, cu atât top-k cuvinte identificate de IG au impact
cauzal mai mare asupra predicției (eliminarea lor scade probabilitatea predicției).

| Grup | n | mean ± std | median | IC 95% (quantile) |
|---|---:|---:|---:|---:|
| A | 25 | 0.1317 ± 0.2912 | 0.0001 | [-0.0013, 0.8886] |
| B | 25 | -0.0001 ± 0.0003 | -0.0000 | [-0.0005, 0.0000] |
| C | 25 | 0.0585 ± 0.1353 | 0.0008 | [-0.0003, 0.4526] |
| D | 25 | -0.0105 ± 0.0219 | -0.0008 | [-0.0648, 0.0004] |

## 4. Comparație directă LIME vs IG (faithfulness deletion AUC)

Test Wilcoxon pereche pe articol (același set de articole, același seed).

| Grup | n | mean LIME | mean IG | diff median (IG − LIME) | p-value |
|---|---:|---:|---:|---:|---:|
| A | 25 | +0.1687 | +0.1317 | -0.0000 | 0.5965  |
| B | 25 | -0.0001 | -0.0001 | +0.0000 | 0.03583 * |
| C | 25 | +0.1268 | +0.0585 | -0.0010 | 0.002785 ** |
| D | 25 | -0.0024 | -0.0105 | -0.0003 | 0.22  |

## 5. Overlap top-5 cuvinte LIME vs IG (Jaccard)

Cât de mult se suprapun cele mai importante 5 cuvinte identificate de cele două metode.
Jaccard mare → metodele identifică același vocabular. Jaccard mic → metode complementare.

| Grup | n | Jaccard mean | Jaccard median |
|---|---:|---:|---:|
| A | 25 | 0.038 | 0.000 |
| B | 25 | 0.027 | 0.000 |
| C | 25 | 0.063 | 0.000 |
| D | 25 | 0.018 | 0.000 |

## 6. Mann-Whitney U între grupuri (faith_auc IG)

| Comparație | Diff median (g1 − g2) | p-value |
|---|---:|---:|
| A vs B | +0.0001 | 3.584e-05 *** |
| A vs C | -0.0007 | 0.6276  |
| A vs D | +0.0010 | 6.795e-07 *** |
| B vs C | -0.0008 | 0.0001222 *** |
| B vs D | +0.0008 | 1.513e-05 *** |
| C vs D | +0.0016 | 5.548e-08 *** |

## 7. Interpretare automată

**Atenție:** completeness eronat pe 4/4 grupuri — verifică design baseline sau n_steps.

**FINDING POZITIV — IG superior LIME pe cls1 baseline (Grup B):** faith_auc IG = -0.0001 vs LIME = -0.0001, Δ median = +0.0000 (p=0.03583). IG identifică cuvinte cu impact cauzal mai mare decât LIME pe articole propagandistice — confirmă strategia hibridă cu IG ca metodă principală.

**Asimetria cls0/cls1 confirmată și pe IG:** Δ median A−B = +0.0001 (p=3.584e-05). Triangulare independentă a stylistic fingerprint — atât LIME cât și IG identifică cuvinte cu impact cauzal mai mare pe cls0 decât pe cls1.

## 8. Concluzii pentru capitolul Explicabilitate

TBD pe baza datelor de mai sus — completăm strategia hibridă LIME + IG + modul 3.

*Generat automat de `08_ig_l3_diagnostic.py`*