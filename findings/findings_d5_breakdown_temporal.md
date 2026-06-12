# Proba D5 — Breakdown LOSO-V per an

## Ipoteza H6

Drop-ul recall cls1 pe LOSO-V are o componentă temporală: Stopfals (singura
sursă cls1 în train LOSO-V) are distribuție temporală diferită de Veridica,
iar modelul eșuează pe anii sub-reprezentați în train.

## Distribuție Stopfals (train cls1 LOSO-V)

| An | N | % |
|----|---|---|
| 2022 | 36 | 42.4% |
| 2023 | 35 | 41.2% |
| 2024 | 14 | 16.5% |

## Distribuție Veridica (test LOSO-V)

| An | N | % |
|----|---|---|
| 2022 | 177 | 26.8% |
| 2023 | 136 | 20.6% |
| 2024 | 143 | 21.6% |
| 2025 | 168 | 25.4% |
| 2026 | 37 | 5.6% |

## Recall cls1 LOSO-V per an

| An | N Veridica | Corecte | Recall | Mean P(cls1) | Median P(cls1) | Stopfals@an (train) |
|----|-----------|---------|--------|--------------|-----------------|---------------------|
| 2022 | 177 | 45 | 0.2542 | 0.1830 | 0.0055 | 36 |
| 2023 | 136 | 41 | 0.3015 | 0.2101 | 0.0069 | 35 |
| 2024 | 143 | 51 | 0.3566 | 0.2482 | 0.0103 | 14 |
| 2025 | 168 | 46 | 0.2738 | 0.1977 | 0.0042 | 0 |
| 2026 | 37 | 11 | 0.2973 | 0.2160 | 0.0133 | 0 |

**Range recall**: [0.2542, 0.3566] (delta 10.2pp)

## Ani CU vs FĂRĂ Stopfals în training

- **Ani fără Stopfals@train** (2025, 2026): n=205, recall agregat = **0.2780**
- **Ani cu Stopfals@train** (2022, 2023, 2024): n=456, recall agregat = **0.3004**
- **Gap** (cu − fără): **+0.0224**

## Interpretare

H6 CONFIRMATĂ SLAB: variație recall 10.2pp între ani. Shift temporal există dar e secundar față de alte cauze.