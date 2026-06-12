# Proba D1 — IG pe cls0 (comparație cu cls1)

## Ipoteza testată

Sub-cuvintele morfologice (`rul`, `tul`, `ova`, `esc`, ...) găsite de IG pe
cls1 reflectă un shortcut real sau sunt artefact generic al tokenizării?

## Metodologie

- Model: baseline v2 (singurul model care rezolvă cls0 perfect)
- N exemple: 6 (3× digi24 + 3× g4media, stratificat pe confidence)
- Steps IG: 50

## Rezultate sumar

| Sursa | N | Conf mediu | Sub-cuv în top-15 (mean) | Conv delta (|mean|) |
|-------|---|------------|-------------------------|----------------------|
| digi24.ro | 3 | 0.999 | 4.7/15 | 3.255 |
| g4media.ro | 3 | 1.000 | 5.0/15 | 3.880 |

## Comparație cls0 vs cls1

| Clasă | Sub-cuvinte în top-15 (mean) | |Δ convergence| (mean) |
|-------|------------------------------|------------------------|
| cls1 (Veridica+Stopfals) | 8.8/15 | 2.144 |
| cls0 (Digi24+G4Media) | 4.8/15 | 3.567 |

### Interpretare

**Sub-cuvinte MULT mai dense pe cls1** (delta +4.0/15).
Asta sugerează că modelul FOLOSEȘTE într-adevăr pattern-uri sub-lexicale
specifice cls1 care nu apar pe cls0 — susține ipoteza shortcut morfologic.

## Top token-uri agregate (cls0)

| Token | Frecvență în top-15 |
|-------|--------------------|
| `▁în` | 7 |
| `,` | 7 |
| `▁să` | 5 |
| `▁a` | 4 |
| `▁Ucraina` | 3 |
| `ă` | 3 |
| `▁Uniunea` | 3 |
| `▁şi` | 2 |
| `▁că` | 2 |
| `▁pe` | 2 |
| `▁Europeană` | 2 |
| `▁ajut` | 2 |
| `▁care` | 2 |
| `:` | 2 |
| `▁luni` | 2 |

## Vizualizări

Fișiere HTML: `findings/ig_html_d1/`