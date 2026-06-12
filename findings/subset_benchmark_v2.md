# Subset benchmark v2 — raport selecție

**Seed:** 42
**Reparație față de v1:** cls0 din surse externe (anti-contaminare).

## Compoziție subset

| Categorie | Sursă | Nr. articole |
|---|---|---|
| cls0 (extern) | stirileprotv.ro | 5 |
| cls0 (extern) | hotnews.ro | 5 |
| cls0 (extern) | libertatea.ro | 5 |
| cls1 | veridica.ro | 12 |
| cls1 | stopfals.md | 3 |
| **Total** | | **30** |

## Validare anti-contaminare

- Articole cls0 cu id în corpus: **0** ✓
- Surse cls0 externe: 3

## Statistici propoziții

- Brute (post-Stanza): **521**
- După filtru [7, 54]w: **477**
- Eliminate: **44** (8.4%)

## Distribuție propoziții per articol

- Min: **2**
- Mediană: **14**
- Max: **42**
- Medie: **15.9**

## Distribuție propoziții per sursă × clasă

| Sursă | Label | Nr. propoziții |
|---|---|---|
| hotnews.ro | 0 | 93 |
| libertatea.ro | 0 | 121 |
| stirileprotv.ro | 0 | 134 |
| stopfals.md | 1 | 28 |
| veridica.ro | 1 | 101 |

*Generat automat.*