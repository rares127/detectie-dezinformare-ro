# Findings — Baseline XLM-RoBERTa v2

**Dataset:** v2 (1483 articole, cu 2022 + Stopfals + entity balancing Moldova)
**Model:** models/xlmr_baseline_v2/final

## 1. Metrici globale

### VAL (load_best_model_at_end=True)
- Macro-F1: **1.0000**
- Accuracy: 1.0000
- Recall cls1 (dezinformare): 1.0000

### TEST (gold, neatins la model selection)
- Macro-F1: **1.0000**
- Accuracy: 1.0000
- Recall cls1: 1.0000

**Confusion matrix TEST:**

|              | pred_0 | pred_1 |
|--------------|--------|--------|
| true_0 (cred) | 111 | 0 |
| true_1 (dez)  | 0 | 112 |

**Stabilitate VAL → TEST:** delta macro_f1 = +0.0000

## 2. Breakdown per-source (TEST)

| Sursa | n | Accuracy | Clasă |
|-------|---|----------|-------|
| digi24.ro | 59 | 1.0000 | cls0 |
| g4media.ro | 52 | 1.0000 | cls0 |
| stopfals.md | 13 | 1.0000 | cls1 |
| veridica.ro | 99 | 1.0000 | cls1 |

## 3. Comparație cu baseline v1

| Metric | v1 (1427 art) | v2 (1483 art) | Delta |
|--------|---------------|---------------|-------|
| VAL macro-F1 | 0.9844 | 1.0000 | +0.0156 |
| TEST macro-F1 | 0.9897 | 1.0000 | +0.0103 |
| TEST recall cls1 | 1.0000 | 1.0000 | +0.0000 |

**Interpretare:** o scădere ușoară a metricilor este de așteptat și DORITĂ dacă
entity balancing Moldova a atenuat un shortcut. Scădere >3pp ar indica probleme.

## 4. Erori pe TEST (total: 0)

_Zero erori pe test set._