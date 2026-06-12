# Proba D4 — Analiză lungime și structură citate

## Ipoteza H3

Veridica și Stopfals diferă structural (lungime, ghilimele, non-ASCII). Modelul
LOSO-V eșuează pe Veridica pentru că Veridica are pattern STRUCTURAL diferit
de cls1 văzută la antrenare (Stopfals) — nu pentru că conținutul diferă.

## Features măsurate

Pe coloana `stire_citata`:
- `nr_cuvinte` — lungimea citatului
- `nr_ghilimele` — deschideri+închideri de citate
- `densitate_ghilimele` — normalizat la lungime
- `nr_paragrafe` — segmentare
- `nr_non_ascii`, `pct_non_ascii` — proxy pentru diacritice + chirilică

## Statistici (median) per sursă

| Feature | Veridica | Stopfals | Digi24 | G4Media |
|---------|----------|----------|--------|---------|
| nr_cuvinte | 184.00 | 130.00 | 216.00 | 218.00 |
| nr_ghilimele | 4.00 | 2.00 | 6.00 | 6.00 |
| densitate_ghilimele | 0.02 | 0.03 | 0.03 | 0.03 |
| nr_paragrafe | 1.00 | 1.00 | 1.00 | 1.00 |
| nr_non_ascii | 60.00 | 64.00 | 73.00 | 74.00 |
| pct_non_ascii | 5.10 | 6.16 | 5.45 | 5.40 |

## Test Kolmogorov-Smirnov: Veridica vs Stopfals

| Feature | KS statistic | p-value | Semnificativ |
|---------|--------------|---------|--------------|
| nr_cuvinte | 0.3965 | 3.701e-11 | *** |
| nr_ghilimele | 0.1961 | 0.005144 | ** |
| densitate_ghilimele | 0.2058 | 0.002805 | ** |
| nr_paragrafe | 0.0000 | 1 | n.s. |
| nr_non_ascii | 0.3178 | 2.944e-07 | *** |
| pct_non_ascii | 0.4327 | 2.737e-13 | *** |

## Corelații cu predicția LOSO-V pe Veridica

| Feature | r(prob_cls1) | r(pred) | mean@pred=0 | mean@pred=1 | Delta |
|---------|--------------|---------|-------------|-------------|-------|
| nr_cuvinte | +0.0851 | +0.0810 | 209.06 | 230.42 | +21.36 |
| nr_ghilimele | -0.1040 | -0.0940 | 5.16 | 4.34 | -0.83 |
| densitate_ghilimele | -0.2204 | -0.2099 | 0.03 | 0.02 | -0.01 |
| nr_paragrafe | +nan | +nan | 1.00 | 1.00 | +0.00 |
| nr_non_ascii | +0.1456 | +0.1405 | 68.36 | 81.87 | +13.51 |
| pct_non_ascii | +0.3208 | +0.3138 | 4.91 | 5.53 | +0.63 |

## Candidați pentru shortcut structural

Criterii: KS p < 0.01 (Veridica ≠ Stopfals) ȘI |corr(pred)| > 0.15.

- **densitate_ghilimele** (KS p=0.002805, corr=-0.2099)
- **pct_non_ascii** (KS p=2.737e-13, corr=+0.3138)

## Interpretare

H3 CONFIRMATĂ PARȚIAL: 2 feature(s) candidate (densitate_ghilimele, pct_non_ascii). Explică O PARTE din shortcut, dar probabil se combină cu alte cauze.