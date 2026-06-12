# Proba D3 — Ablație entități propagandistice

## Ipoteza H2

Modelul LOSO-V folosește nume proprii din vocabularul propagandist (toponime
rusificate, oficiali Kremlin, media de stat rusă) ca semnal lexical pentru cls1.

## Metodologie

- **Model**: `models/loso_v/final` (LOSO-V, shortcut activ)
- **Test set**: 661 articole Veridica
- **Înlocuire**: entități → `[ENT]`
- **3 categorii testate** separat + combinat:
  - Toponime: 17 pattern-uri (Lugansk, Donbas, Transnistria...)
  - Oficiali ruși: 25 pattern-uri (Putin, Lavrov, Zaharova...)
  - Media propagandiste: 11 pattern-uri (Sputnik, TASS, RT...)

**Notă:** NU am inclus 'Kremlin/Rusia/Moscova' — apar natural și în presă credibilă.

## Distribuție entități

- Articole cu ≥1 toponim: 177
- Articole cu ≥1 oficial: 150
- Articole cu ≥1 media propagandistă: 56
- Articole cu ≥1 entitate orice categorie: 314/661

## Rezultate globale (ablație TOATE)

| Măsură | Original | Ablated | Delta |
|--------|----------|---------|-------|
| P(cls1) mean | 0.2082 | 0.2171 | -0.0088 |
| Recall cls1 | 0.2935 | 0.3086 | -0.0151 |

**Flips**: 10/661 (1.5%)

## Contribuția per categorie

| Categorie ablated | Delta P(cls1) mean |
|-------------------|--------------------|
| toponime | -0.0056 |
| oficiali | -0.0029 |
| media | -0.0001 |

**Categoria cu cel mai mare impact**: `media` (delta -0.0001)

## Breakdown pe prezența entităților

| Subset | N | Delta mean | Flip rate |
|--------|---|------------|-----------|
| Cu entități | 314 | -0.0186 | 3.2% |
| Fără entități (control) | 347 | +0.0000 | 0.0% |

## Interpretare

H2 RESPINSĂ: ablarea entităților propagandistice nu schimbă predicțiile. Shortcut-ul nu vine din vocabularul specific.

## Audit

CSV per articol: `d3_ablatie_entitati_per_articol.csv`