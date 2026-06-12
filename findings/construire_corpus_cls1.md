# Construire corpus cls1 (propagandistic) — raport

**Opțiunea A** — test dacă dezinformarea pro-Kremlin folosește tipare semantice recurente detectabile prin similaritate.

## Sursă și validare

- Articole cls1 din train+val: **634**
- Validare anti-contaminare cu test set: **✓ Zero suprapuneri**

## Compoziție

| Sursă | Nr. articole |
|---|---|
| veridica.ro | 562 |
| stopfals.md | 72 |

## Tratament titluri (din audit)

| Acțiune | Nr. articole |
|---|---|
| OK, folosit integral | 568 |
| Prefix tăiat (Veridica `PROPAGANDĂ DE RĂZBOI:`) | 66 |
| Exclus (pattern meta-jurnalistic) | 0 |

## Procesare

- Propoziții brute (post-Stanza): **6708**
- După filtru lungime [7, 54]w: **6082**
- Eliminate (prea scurte/lungi): **626** (9.3%)

## Corpus final

- **6082 propoziții**

### Distribuție per sursă

| Sursă | Nr. propoziții |
|---|---|
| veridica.ro | 5632 |
| stopfals.md | 450 |

### Distribuție per an

| An | Nr. propoziții |
|---|---|
| 2022 | 1656 |
| 2023 | 1464 |
| 2024 | 1351 |
| 2025 | 1305 |
| 2026 | 306 |

### Statistici lungime (cuvinte)

- Min: **7**
- p5: **8**
- Mediană: **18**
- p95: **41**
- Max: **54**
- Medie: **20.5**

## Note importante pentru benchmark-ul v4

- **Convenție scor cls1**: scor MAI MARE = articol similar cu narațiuni propagandistice cunoscute = mai probabil dezinformare.
- **Direcție inversă** față de corpus cls0 (acolo scor mare = credibil).
- **Opțiunea A izolat**: AUC doar pe `scor_cls1` (fără cls0). Dacă ≥0.75, viabil ca clasificator.
- **Opțiunea D combinat**: AUC pe `scor_cls1 - scor_cls0`, folosind ambele corpusuri. Scorul combinat ar trebui să dea separabilitate mai bună decât oricare singur.

## Recomandări pentru pași următori

1. **Audit rezidual** pe propozițiile cls1 — posibil să existe zgomot specific (cookie banners, etichete vorbitor, link-uri reziduale) pe care îl curățăm cu un pipeline similar cu cel pentru cls0.
2. **Benchmark v4** pe subset-ul v3 existent (167 articole) — fără recolecting.
3. **Decizie finală** după benchmark: Opțiunea A sau D sau C.

*Generat automat.*