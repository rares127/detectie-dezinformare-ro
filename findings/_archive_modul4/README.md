# Modul 4 (Explicabilitate) — arhivă artefacte finale

Data finalizare: aprilie 2026.

## Conținut
- findings_lime_l1a.{md,json}        — LIME pe 100 articole (4×25, seed=42)
- findings_lime_vs_ig_l3.{md,json}   — IG pe 100 articole, head-to-head LIME
- findings_xai_l4.{md,json}          — DL+GS, 4-way LIME+IG+DL+GS
- scripts/                            — versiunile finale 07, 08, 09

## Eșantion comun
4 grupuri × 25 articole, seed=42:
- A: TP cls0 baseline (Digi24 13 + G4Media 12)
- B: TP cls1 baseline (Veridica 22 + Stopfals 3)
- C: FN LOSO-V pe Veridica (n=25)
- D: TP LOSO-V pe Veridica (n=25)

## Cifre cheie
Vezi findings_xai_l4.md secțiunea 4 (tabel HEADLINE 4-way) și 7bis (D<C finding).

## Reproducibilitate
Seed=42 pretutindeni. DL+GS deterministic, LIME stocastic dar reproducible cu seed.
IG deterministic. Coloana text input: text_curat.
