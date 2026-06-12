# Benchmark v4 — scor granular vs corpus propagandistic

## Configurare

- Model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Device: `mps`
- Seed: `42`
- Downsample cls1: `5,290` (paritate cu cls0)

## Volume

- Corpus cls0: **5,290** prop. (congelat)
- Corpus cls1 full: 6,048 prop. → downsampled la **5,290** prop. (seed=42)
- Test set: **167** articole (2,181 prop.) — 55 cls0 + 112 cls1

## Validare anti-contaminare

- Suprapunere test ∩ cls0: **0** articole
- Suprapunere test ∩ cls1: **0** articole
- ✓ Zero contaminare — benchmark valid

## Rezultate

**Convenție:** scor mare = predicție cls1 (propagandist). AUC > 0.5 = semnal util; 0.5 = aleator; < 0.5 = semnal inversat.

### **Test A — scor_cls1 izolat** (ipoteza principală Opțiunea A): articolele propagandiste au similaritate mai mare cu corpusul propagandistic decât cele credibile?

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | Gap V−S |
|---|---:|---:|---:|---:|---:|
| mean | 0.8313 | +1.2081 | 0.6369 | 0.7015 | -0.0002 |
| min | 0.9774 | +2.8080 | 0.3578 | 0.5815 | -0.0028 |
| p10 | 0.8742 | +1.5494 | 0.5185 | 0.6198 | -0.0082 |

### **Test B — scor_cls0 izolat** (reproducere v3 pentru comparație): rezultatul din v3 a fost AUC = 0.552 pe această configurație. Replicare cu corpusul extins vs v3 pentru sanity-check.

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | Gap V−S |
|---|---:|---:|---:|---:|---:|
| mean | 0.4763 | -0.1784 | 0.6721 | 0.6624 | +0.0085 |
| min | 0.9435 | +2.2337 | 0.3709 | 0.5506 | -0.0172 |
| p10 | 0.6904 | +0.6035 | 0.5464 | 0.5866 | -0.0019 |

### **Test D — diferență cls1 − cls0** (Opțiunea D combinată): scor compus care folosește ambele corpusuri. Dacă `scor_cls1 > scor_cls0`, articolul e mai aproape de propagandă decât de presă credibilă.

| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | Gap V−S |
|---|---:|---:|---:|---:|---:|
| mean | 0.9739 | +2.4041 | -0.0352 | 0.0390 | -0.0087 |
| min | 0.7784 | +0.9722 | -0.0131 | 0.0309 | +0.0144 |
| p10 | 0.8995 | +1.6065 | -0.0279 | 0.0332 | -0.0062 |

## Interpretare automată

- **Best Test A:** AUC = 0.9774 (agregare `min`)
- **Best Test D:** AUC = 0.9739 (agregare `mean`)

### ✓ Decizie: Opțiunea A validată

Scorul pe corpus propagandistic separă semnificativ articolele propagandiste de cele credibile. Continuăm cu Test D pentru rigurozitate finală și integrare în scor combinat (modulul 5).

## Comparație cu v3

| Metric | v3 (cls0-only) | v4 best (cls1-only) | v4 best (diff) |
|---|---:|---:|---:|
| AUC | 0.552 (minilm/max/mean) | 0.9774 (mpnet/max/min) | 0.9739 (mpnet/max/mean) |

Notă: v3 a raportat 0.552 cu minilm (model mai mic). v4 folosește mpnet (model mai mare, ales câștigător în v3).

---

*Modul 3 · Pasul A2 · Benchmark v4*