# Findings — LIME pe baseline v2

## 1. Configurație

- num_features = 15
- num_samples = 1000
- bow = False
- seed = 42
- N exemple analizate: 12

## 2. Fidelity mediu per sursă

Fidelity = scorul R² local al modelului-surogat LIME. Valoare înaltă → LIME
explică bine modelul pe exemplul respectiv.

| Sursa | v2 mean | v2 range | v1 mean | Delta |
|-------|---------|----------|---------|-------|
| veridica.ro | 0.1102 | [0.0908, 0.1349] | 0.0600 | +0.0502 |
| stopfals.md | 0.0882 | [0.0467, 0.1255] | N/A (nou) | N/A |
| digi24.ro | 0.2904 | [0.1145, 0.4037] | 0.5000 | -0.2096 |
| g4media.ro | 0.3594 | [0.2122, 0.4717] | 0.1770 | +0.1824 |

## 3. Interpretare

**Fidelity pe Veridica v2 = 0.1102** — rămâne scăzut (ca în v1, 0.04-0.09).

Concluzie: e limitare INTRINSECĂ a LIME pe transformere high-confidence,
nu bug de dataset. Modelul folosește reprezentări distribuite global,
nu features lexicale localizate. **Recomandare:** adaugă Integrated
Gradients (captum) ca XAI complementar pe cls1.

## 4. Token-uri agregate top-15

### Pro-cls1 (dezinformare)

| Token | Frecvență în top-features |
|-------|---------------------------|
| de | 4 |
| și | 2 |
| este | 2 |
| ales | 1 |
| lagărul | 1 |
| estul | 1 |
| rusia | 1 |
| ucraineni | 1 |
| erau | 1 |
| gheața | 1 |
| propagandă | 1 |
| lagăre | 1 |
| practic | 1 |
| occidentul | 1 |
| poporul | 1 |

### Pro-cls0 (credibil)

| Token | Frecvență în top-features |
|-------|---------------------------|
| în | 7 |
| declarat | 2 |
| că | 2 |
| luni | 2 |
| ministrul | 2 |
| ajute | 2 |
| care | 2 |
| cu | 2 |
| relatează | 1 |
| ua | 1 |
| au | 1 |
| său | 1 |
| şi | 1 |
| cazuri | 1 |
| săptămâna | 1 |

## 5. Vizualizări HTML

Cele 12 vizualizări LIME interactive sunt în: `findings/lime_html/`

| # | Sursa | Tip | True/Pred | Confidence | Fidelity | HTML |
|---|-------|-----|-----------|------------|----------|------|
| 1 | veridica.ro | TP | 1/1 | 1.000 | 0.0908 | `exemplu_01_veridica_TP_class1.html` |
| 2 | veridica.ro | TP | 1/1 | 1.000 | 0.1349 | `exemplu_02_veridica_TP_class1.html` |
| 3 | veridica.ro | TP | 1/1 | 0.993 | 0.1049 | `exemplu_03_veridica_TP_class1.html` |
| 4 | stopfals.md | TP | 1/1 | 1.000 | 0.0924 | `exemplu_04_stopfals_TP_class1.html` |
| 5 | stopfals.md | TP | 1/1 | 1.000 | 0.1255 | `exemplu_05_stopfals_TP_class1.html` |
| 6 | stopfals.md | TP | 1/1 | 0.994 | 0.0467 | `exemplu_06_stopfals_TP_class1.html` |
| 7 | digi24.ro | TP | 0/0 | 1.000 | 0.4037 | `exemplu_07_digi24_TP_class0.html` |
| 8 | digi24.ro | TP | 0/0 | 1.000 | 0.3530 | `exemplu_08_digi24_TP_class0.html` |
| 9 | digi24.ro | TP | 0/0 | 0.998 | 0.1145 | `exemplu_09_digi24_TP_class0.html` |
| 10 | g4media.ro | TP | 0/0 | 1.000 | 0.3943 | `exemplu_10_g4media_TP_class0.html` |
| 11 | g4media.ro | TP | 0/0 | 1.000 | 0.2122 | `exemplu_11_g4media_TP_class0.html` |
| 12 | g4media.ro | TP | 0/0 | 0.999 | 0.4717 | `exemplu_12_g4media_TP_class0.html` |