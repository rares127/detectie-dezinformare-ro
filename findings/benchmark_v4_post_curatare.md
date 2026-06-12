# Benchmark v4 — POST-CURĂȚARE test set extern

Re-rulare benchmark v4 pe `subset_benchmark_v3_curat.parquet` (test set curățat de cookie banners HotNews/Pro TV/Libertatea + comentarii). Vezi `curatare_test_extern_cookies.md` pentru detalii curățare. Acest raport e comparabil direct cu `benchmark_v4.md` pentru a cuantifica artefactul de cookie banners.

## Configurare

- Model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Device: `mps`
- Seed: `42`
- Downsample cls1: `5,290` (paritate cu cls0)

## Volume

- Corpus cls0: **5,290** prop. (congelat)
- Corpus cls1 full: 6,048 prop. → downsampled la **5,290** prop. (seed=42)
- Test set: **167** articole (2,066 prop.) — 55 cls0 + 112 cls1

## Validare anti-contaminare

- Suprapunere test ∩ cls0: **0** articole
- Suprapunere test ∩ cls1: **0** articole
- ✓ Zero contaminare — benchmark valid

## Rezultate

**Convenție:** scor mare = predicție cls1 (propagandist). AUC > 0.5 = semnal util; 0.5 = aleator; < 0.5 = semnal inversat.

### **Test A — scor_cls1 izolat** (ipoteza principală Opțiunea A): articolele propagandiste au similaritate mai mare cu corpusul propagandistic decât cele credibile?

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | Gap V−S |
|---|---:|---:|---:|---:|---:|
| mean | 0.7242 | +0.7038 | 0.6634 | 0.7015 | -0.0002 |
| min | 0.7196 | +0.7792 | 0.5193 | 0.5815 | -0.0028 |
| p10 | 0.7080 | +0.6928 | 0.5722 | 0.6198 | -0.0082 |

### **Test B — scor_cls0 izolat** (reproducere v3 pentru comparație): rezultatul din v3 a fost AUC = 0.552 pe această configurație. Replicare cu corpusul extins vs v3 pentru sanity-check.

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | Gap V−S |
|---|---:|---:|---:|---:|---:|
| mean | 0.3054 | -0.7004 | 0.7007 | 0.6624 | +0.0085 |
| min | 0.5214 | +0.1018 | 0.5420 | 0.5506 | -0.0172 |
| p10 | 0.4200 | -0.3080 | 0.6076 | 0.5866 | -0.0019 |

### **Test D — diferență cls1 − cls0** (Opțiunea D combinată): scor compus care folosește ambele corpusuri. Dacă `scor_cls1 > scor_cls0`, articolul e mai aproape de propagandă decât de presă credibilă.

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | Gap V−S |
|---|---:|---:|---:|---:|---:|
| mean | 0.9690 | +2.4070 | -0.0373 | 0.0390 | -0.0087 |
| min | 0.7763 | +1.0294 | -0.0227 | 0.0309 | +0.0144 |
| p10 | 0.8989 | +1.7028 | -0.0354 | 0.0332 | -0.0062 |

## Interpretare automată

- **Best Test A:** AUC = 0.7242 (agregare `mean`)
- **Best Test D:** AUC = 0.9690 (agregare `mean`)

### ⚠ Decizie: zona gri

AUC = 0.7242 e între 0.65 și 0.75. Semnalul există dar e slab. Trade-off timp vs rigurozitate: investigăm dacă un top-k mean (k=5) îmbunătățește separabilitatea (secundar), sau acceptăm ca scor complementar în sistemul combinat cu modulul 2 (primar).

## Comparație cu v3

| Metric | v3 (cls0-only) | v4 best (cls1-only) | v4 best (diff) |
|---|---:|---:|---:|
| AUC | 0.552 (minilm/max/mean) | 0.7242 (mpnet/max/mean) | 0.9690 (mpnet/max/mean) |

Notă: v3 a raportat 0.552 cu minilm (model mai mic). v4 folosește mpnet (model mai mare, ales câștigător în v3).

---

*Modul 3 · Pasul A2 · Benchmark v4*