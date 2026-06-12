# Findings — Evaluare stratificată Moldova v2

Verifică eficacitatea entity balancing (D10) prin comparația performanței
pe articole cu/fără termeni-Moldova.

## 1. Regex termeni Moldova

```
\bmoldov[aă]\b|\bchi[sș]in[aă]u\b|\bmaia\s+sandu\b|\btransnistri[ae]\b|\bg[aă]g[aă]uzi[ae]\b|\bcomrat\b|\bdodon\b|\bsor\b|\bmd\b(?=\s|$|[,\.])
```

## 2. Distribuție subseturi (TEST)

- Total TEST: 223
- Cu Moldova: 61 (cls0=20, cls1=41)
- Fără Moldova: 162 (cls0=91, cls1=71)

## 3. Metrici comparative

| Subset | n | Accuracy | Macro-F1 | Recall cls0 | Recall cls1 |
|--------|---|----------|----------|-------------|-------------|
| Cu Moldova | 61 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Fără Moldova | 162 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 4. Interpretare

Entity balancing PARE să funcționeze — performanța e comparabilă pe ambele subseturi (delta <5pp).

**Context v1:** în baseline v1, DF diff cls1-cls0 pentru termeni Moldova era +30.8pp.
Entity balancing aplicat în v2 a redus-o la +16.1pp. Acest test măsoară efectul.