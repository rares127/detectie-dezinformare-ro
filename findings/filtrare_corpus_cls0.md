# Filtrare finală corpus cls0 — raport

**Praguri lungime folosite:** [p5=6, p95=54] cuvinte
**Reguli aplicate:** boilerplate CMS, etichete vorbitor, separator slash,
  4+ ghilimele, degenerate alfanumerice, filtru lungime, dedup normalizat

## 1. Pipeline — eliminări pas cu pas

| Pas | Regulă | Propoziții eliminate |
|---|---|---|
| 0 | Input brut | — (6,047) |
| 1a | Prefix boilerplate curățat (nu eliminat) | 0 modificate |
| 1b | Rămas gol după curățare | 0 |
| 2 | Boilerplate CMS full | 68 |
| 3 | Etichete vorbitor (≤6w + `:`) | 50 |
| 4 | Separator slash între titluri | 47 |
| 5 | 4+ ghilimele (titluri concatenate) | 8 |
| 6 | Degenerate alfanumerice | 5 |
| 7a | Prea scurte (< p5=6w) | 169 |
| 7b | Prea lungi (> p95=54w) | 269 |
| 8 | Duplicate (hash normalizat) | 94 |
| — | **Output final** | **5,337** |

**Retenție:** 88.26% (5,337 / 6,047)

## 2. Breakdown pe sursă (input vs output)

| Sursă | Input | Output | Retenție |
|---|---|---|---|
| digi24.ro | 3,297 | 2,847 | 86.35% |
| g4media.ro | 2,750 | 2,490 | 90.55% |

> **Verificare ipoteză.** Dacă digi24 are retenție mult mai mică decât
> g4media, confirmă că filtrarea a atins zgomotul CMS specific digi24.

## 3. Distribuție lungime după filtrare

| Statistică | Input | Output |
|---|---|---|
| min | 1 | 6 |
| 25% | 15 | 15 |
| 50% | 23 | 23 |
| 75% | 34 | 33 |
| max | 142 | 54 |

## 4. Eșantion propoziții eliminate (verificare vizuală)

> Dacă apar FP (propoziții reale eliminate greșit), revizitează regulile.

## 5. Pași următori

1. Verifică raportul — dacă retenția e în zona 90-95%, e în regulă.
   Dacă < 85%, probabil o regulă e prea agresivă.
2. Corpusul curat e gata la `data/processed/propozitii_cls0_corpus.parquet`.
3. Pasul următor: **benchmark model embeddings** (XLM-R mean-pooled vs
   sentence-transformers multilingv) pe acest corpus.
