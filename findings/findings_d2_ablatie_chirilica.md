# Proba D2 — Ablație caractere chirilice

## Ipoteza H1

Modelul LOSO-V folosește prezența alfabetului chirilic ca proxy pentru cls1.
Veridica citează frecvent surse rusești în original; articolele credibile și
Stopfals (transliterat) au mai puțină chirilică.

## Metodologie

- **Model**: `models/loso_v/final` (modelul LOSO-V cu shortcut activ)
- **Test set**: 661 articole Veridica
- **Ablație**: regex `[\u0400-\u04FF\u0500-\u052F]+` înlocuit cu spațiu
- **Măsură**: Δ P(cls1) = orig − ablated

## Distribuția chirilicei

- Articole cu chirilică: 3/661
- Mediana caractere chirilice: 0
- Max: 26

## Rezultate globale

| Măsură | Original | Ablated | Delta |
|--------|----------|---------|-------|
| P(cls1) mean | 0.2082 | 0.2082 | +0.0000 |
| P(cls1) median | 0.0065 | 0.0065 | +0.0000 |
| Recall cls1 | 0.2935 | 0.2935 | +0.0000 |

**Flips de predicție**: 0/661 (0.0%)

## Breakdown pe prezența chirilicei

| Subset | N | P(cls1) orig | P(cls1) ablated | Delta | Flip rate |
|--------|---|--------------|-----------------|-------|-----------|
| Cu chirilică | 3 | 0.2322 | 0.2284 | +0.0038 | 0.0% |
| Fără chirilică (control) | 658 | 0.2081 | 0.2081 | +0.0000 | 0.0% |

## Interpretare

H1 RESPINSĂ: ștergerea chirilicei nu modifică semnificativ predicțiile. Shortcut-ul vine din altă parte (H2 entități, H3 stil, H4 sufixe).

## Audit

CSV cu date per-articol: `d2_ablatie_chirilica_per_articol.csv`