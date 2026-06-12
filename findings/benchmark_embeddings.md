# Benchmark embeddings — Modulul 3

**Seed:** 42 · **Device:** mps · **Top-K:** 5 · **Percentilă:** p10

**Corpus referință:** 5290 propoziții
**Subset benchmark:** 20 articole, 164 propoziții

## Tabel principal — separabilitate cls0 vs cls1

Convenție: scor mai mare = mai similar cu corpus cls0 = mai credibil.
Deci cls0 ar trebui să aibă scoruri mai mari decât cls1.

AUC > 0.5 ⇒ direcție corectă. Cohen's d > 0 ⇒ direcție corectă.

| Model | Prop-scor | Art-agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(Veridica) | μ(Stopfals) |
|---|---|---|---|---|---|---|---|---|
| minilm | max | mean | 1.000 | +8.48 | 1.000 | 0.632 | 0.624 | 0.664 |
| minilm | max | min | 1.000 | +8.63 | 1.000 | 0.515 | 0.492 | 0.605 |
| minilm | max | p10 | 1.000 | +8.43 | 1.000 | 0.552 | 0.536 | 0.618 |
| minilm | topk_mean | mean | 1.000 | +2.91 | 0.753 | 0.598 | 0.589 | 0.635 |
| mpnet | max | mean | 1.000 | +9.28 | 1.000 | 0.639 | 0.634 | 0.657 |
| mpnet | max | min | 1.000 | +9.71 | 1.000 | 0.532 | 0.514 | 0.602 |
| mpnet | max | p10 | 1.000 | +8.92 | 1.000 | 0.571 | 0.559 | 0.619 |
| mpnet | topk_mean | mean | 1.000 | +3.64 | 0.771 | 0.608 | 0.602 | 0.633 |
| xlmr_ft_mean | max | mean | 1.000 | +6.31 | 1.000 | 0.992 | 0.992 | 0.992 |
| xlmr_ft_mean | max | min | 1.000 | +2.59 | 1.000 | 0.985 | 0.985 | 0.988 |
| xlmr_ft_mean | max | p10 | 1.000 | +5.26 | 1.000 | 0.989 | 0.989 | 0.990 |
| mpnet | topk_mean | p10 | 0.990 | +3.37 | 0.713 | 0.541 | 0.526 | 0.599 |
| minilm | topk_mean | p10 | 0.980 | +2.81 | 0.692 | 0.517 | 0.500 | 0.585 |
| mpnet | topk_mean | min | 0.970 | +2.91 | 0.681 | 0.506 | 0.487 | 0.584 |
| minilm | topk_mean | min | 0.950 | +2.38 | 0.658 | 0.482 | 0.460 | 0.574 |
| xlmr_ft_mean | topk_mean | mean | 0.810 | +1.13 | 0.993 | 0.991 | 0.991 | 0.991 |
| xlmr_ft_mean | topk_mean | p10 | 0.710 | +0.78 | 0.991 | 0.988 | 0.988 | 0.989 |
| xlmr_ft_mean | topk_mean | min | 0.700 | +0.75 | 0.989 | 0.984 | 0.983 | 0.987 |

## Viteză embeddings (corpus cls0, 5,290 propoziții)

| Model | Propoziții/secundă | Timp total corpus (s) |
|---|---|---|
| minilm | 710.2 | 7.4 |
| mpnet | 322.1 | 16.4 |
| xlmr_ft_mean | 239.3 | 22.1 |

## Interpretare rapidă

**Top configurație după AUC:** `minilm` cu propoziție=max, articol=mean (AUC=1.000, d=+8.48).

### Praguri de decizie orientative
- AUC ≥ 0.90: separabilitate foarte bună → scor granular poate fi semnal puternic
- AUC 0.75–0.90: separabilitate bună → util în combinație cu clasificatorul global
- AUC 0.60–0.75: semnal slab → util doar ca feature auxiliar
- AUC < 0.60: aproape aleator → re-gândim abordarea

### Cross-source (relevant pentru problema LOSO-V)
Compară μ(Veridica) vs μ(Stopfals) pentru configurația câștigătoare.
Dacă diferența e mică (≤0.02), modelul tratează ambele surse similar ⇒
semnal cross-source robust. Dacă diferența e mare, avem încă stylistic fingerprint.

*Generat automat.*