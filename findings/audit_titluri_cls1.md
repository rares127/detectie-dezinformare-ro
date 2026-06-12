# Audit titluri cls1 — decizie includere în corpus propagandistic

**Articole analizate:** 634 (train + val)

## Sumar global

| Categorie | Nr. titluri | % din total |
|---|---|---|
| Cu prefix fact-checking | 67 | 10.6% |
| Cu verbe de demascare | 0 | 0.0% |
| Cu voce jurnalistică | 1 | 0.2% |
| Cu ORICE pattern | 68 | 10.7% |
| **Fără niciun pattern** | **566** | **89.3%** |

## Breakdown per sursă

| Sursă | Total | Cu prefix | Cu verbe | Cu voce | Cu ORICE | % ORICE |
|---|---|---|---|---|---|---|
| veridica.ro | 562 | 66 | 0 | 0 | 66 | 11.7% |
| stopfals.md | 72 | 1 | 0 | 1 | 2 | 2.8% |

## Top pattern-uri individuale

| Pattern | Nr. titluri |
|---|---|
| `prefix_propaganda` | 66 |
| `prefix_fals` | 1 |
| `cuvant_context` | 1 |

## Exemple concrete

### Titluri cu prefix fact-checking (RISC MARE)

- [veridica.ro] PROPAGANDĂ DE RĂZBOI: Putin nu a participat la summit-ul G20 pentru că Occidentul pregătea un atac terorist împotriva sa
- [veridica.ro] PROPAGANDĂ DE RĂZBOI: În Ucraina au sosit mercenari din Israel care luptă împotriva Rusiei alături de militarii grupului "Azov"
- [veridica.ro] PROPAGANDĂ DE RĂZBOI: Observatorii din Occident confirmă că referendumurile din Ucraina au respectat standardele democratice
- [veridica.ro] PROPAGANDĂ DE RĂZBOI: SUA vor provoca un dezastru nuclear în Ucraina pentru a ascunde urmele laboratoarelor de arme biologice
- [veridica.ro] PROPAGANDĂ DE RĂZBOI: Referendumurile de alipire la Rusia a teritoriilor din estul și sudul Ucrainei respectă Carta ONU

### Titluri cu verbe de demascare (RISC MEDIU)


### Titluri cu voce jurnalistică (RISC SCĂZUT)

- [stopfals.md] Propagandistul rus Vladimir Soloviov folosește în context fals o fotografie de la Adunarea Națională „Moldova Europeană”

### Titluri FĂRĂ niciun pattern (OK pentru corpus)

- [veridica.ro] Guvernarea de la Chișinău a furat un miliard de euro din ajutorul financiar oferit de UE
- [veridica.ro] Occidentul a declarat război Federației Ruse, iar Moldova riscă să aibă soarta Ucrainei
- [veridica.ro] Referendumurile din Ucraina ocupată sunt legitime, iar Rusia e superioară militar Occidentului
- [stopfals.md] Ce presupun modificările legislative despre „trădarea de patrie” și cum sursele pro-Kremlin manipulează pe această temă
- [veridica.ro] Chișinăul este gata să renunțe la regiunea transnistreană în favoarea aderării la UE

## Recomandare

**Decizie automată:** `include_cu_curatare_minima`

✓ INCLUDE TITLURILE. Doar 10.7% au pattern, risc mic. Opțional: normalizează cele 67 cu prefix.

## Opțiuni de continuare

1. **Exclude toate titlurile** — cel mai sigur, cost = pierdem ~15-20% din propoziții.
2. **Exclude selectiv** (doar cele cu pattern) — compromis rezonabil.
3. **Normalizare prefixe** — tăiem prefixele gen „Narațiune falsă:” și păstrăm restul titlului.

*Generat automat.*