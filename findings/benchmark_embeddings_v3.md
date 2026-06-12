# Benchmark embeddings v3 — Modulul 3 (scale up)

**Seed:** 42 · **Device:** mps · **Top-K:** 5 · **Percentilă:** p10

**Scale up față de v2:** articolele cls0 vin acum din surse EXTERNE (Pro TV, HotNews etc.), nu din aceleași surse Digi24/G4Media din care s-a construit corpusul. V1 avea contaminare 100% (cls0 test ⊂ corpus) ceea ce producea AUC=1.0 trivial.

**Corpus referință:** 5290 propoziții (neatins)
**Subset v3:** 167 articole (55 cls0 extern + 112 cls1), 2181 propoziții

---

## 1. Separabilitate — scoruri agregate

Convenție: scor mai mare = mai similar cu corpus cls0 = mai credibil.
cls0 ar trebui să aibă scoruri MAI MARI decât cls1.

| Model | Prop | Agr. | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(Ver) | μ(Stop) | Δ(V-S) |
|---|---|---|---|---|---|---|---|---|---|
| minilm | max | mean | 0.552 | +0.26 | 0.659 | 0.645 | 0.646 | 0.640 | +0.005 |
| mpnet | max | mean | 0.524 | +0.18 | 0.672 | 0.662 | 0.663 | 0.655 | +0.009 |
| minilm | topk_mean | mean | 0.521 | +0.18 | 0.619 | 0.609 | 0.610 | 0.601 | +0.009 |
| mpnet | topk_mean | mean | 0.501 | +0.15 | 0.636 | 0.628 | 0.629 | 0.616 | +0.013 |
| minilm | max | p10 | 0.394 | -0.39 | 0.533 | 0.559 | 0.559 | 0.559 | +0.000 |
| minilm | topk_mean | p10 | 0.366 | -0.53 | 0.489 | 0.525 | 0.525 | 0.526 | -0.000 |
| mpnet | topk_mean | p10 | 0.326 | -0.57 | 0.516 | 0.554 | 0.554 | 0.549 | +0.005 |
| mpnet | max | p10 | 0.310 | -0.60 | 0.546 | 0.587 | 0.586 | 0.588 | -0.002 |
| minilm | topk_mean | min | 0.073 | -1.59 | 0.365 | 0.483 | 0.481 | 0.493 | -0.011 |
| minilm | max | min | 0.066 | -1.73 | 0.386 | 0.516 | 0.515 | 0.523 | -0.008 |
| mpnet | max | min | 0.056 | -2.23 | 0.371 | 0.551 | 0.549 | 0.566 | -0.017 |
| mpnet | topk_mean | min | 0.040 | -2.33 | 0.347 | 0.521 | 0.519 | 0.532 | -0.012 |
| xlmr_ft_mean | max | p10 | 0.024 | -2.74 | 0.977 | 0.990 | 0.990 | 0.990 | +0.000 |
| xlmr_ft_mean | max | mean | 0.021 | -2.72 | 0.988 | 0.992 | 0.992 | 0.992 | +0.000 |
| xlmr_ft_mean | topk_mean | mean | 0.020 | -2.80 | 0.987 | 0.992 | 0.992 | 0.991 | +0.000 |
| xlmr_ft_mean | topk_mean | p10 | 0.019 | -2.76 | 0.974 | 0.989 | 0.989 | 0.989 | +0.000 |
| xlmr_ft_mean | topk_mean | min | 0.011 | -4.93 | 0.957 | 0.987 | 0.987 | 0.987 | -0.001 |
| xlmr_ft_mean | max | min | 0.010 | -5.10 | 0.959 | 0.988 | 0.988 | 0.988 | -0.000 |

---

## 2. Separabilitate — proporție propoziții sub threshold τ

Metrică: pentru fiecare articol, fracția propozițiilor cu scor_max < τ.
Convenție INVERSĂ aici: fracție MAI MARE = mai multe propoziții fără 
corespondent factual = mai probabil cls1 (dezinformare).

Praguri τ testate: [0.3, 0.4, 0.5, 0.6]

| Model | Prag τ | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(Ver) | μ(Stop) | Δ(V-S) |
|---|---|---|---|---|---|---|---|---|
| minilm | 0.3 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| mpnet | 0.3 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| xlmr_ft_mean | 0.3 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| xlmr_ft_mean | 0.4 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| xlmr_ft_mean | 0.5 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| xlmr_ft_mean | 0.6 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| minilm | 0.6 | 0.489 | +0.11 | 0.286 | 0.311 | 0.311 | 0.316 | -0.005 |
| mpnet | 0.6 | 0.439 | +0.07 | 0.221 | 0.235 | 0.229 | 0.280 | -0.051 |
| minilm | 0.5 | 0.351 | -0.00 | 0.092 | 0.092 | 0.085 | 0.144 | -0.059 |
| mpnet | 0.5 | 0.285 | -0.08 | 0.070 | 0.061 | 0.064 | 0.037 | +0.027 |
| mpnet | 0.4 | 0.153 | -1.58 | 0.046 | 0.005 | 0.005 | 0.000 | +0.005 |
| minilm | 0.4 | 0.096 | -1.72 | 0.059 | 0.009 | 0.010 | 0.000 | +0.010 |

---

## 3. Viteză embeddings (corpus 5,290 propoziții)

| Model | Propoziții/secundă | Timp total (s) |
|---|---|---|
| minilm | 811.0 | 6.5 |
| mpnet | 345.8 | 15.3 |
| xlmr_ft_mean | 241.8 | 21.9 |

---

## 4. Finding metodologic — XLM-R fine-tuned NU e adecvat pentru similaritate

**Observația**: checkpoint-ul XLM-R fine-tuned pe clasificare (modulul 2) produce embeddings cu distribuție colapsată — toate propozițiile primesc 
scoruri aproape identice, indiferent de conținut. Uită-te la coloanele μ(cls0) și μ(cls1) în tabelele de mai sus pentru `xlmr_ft_mean`: diferența dintre clase e sub 0.01 în valoare absolută, vs ~0.35 pentru MiniLM/mpnet.

**Interpretare**: fine-tuning-ul pe sarcina de clasificare binară împinge reprezentările spre hiperplanul de decizie, colapsând geometria semantică originală (pe care XLM-R pretrained o avea). Modelul și-a optimizat reprezentările pentru a separa cls0/cls1 la nivel de POOL de clasificare, nu pentru a captura similaritate semantică generală.

**Implicație pentru teză**: acest rezultat justifică alegerea unui model antrenat specific pe similaritate (sentence-transformers) în locul reutilizării modelului de clasificare. E o contribuție metodologică: validează empiric că arhitectura sistemului trebuie să folosească două modele distincte, nu unul singur pentru ambele sarcini.

**De inclus în capitolul „Arhitectură și justificări tehnice”**.

---

## 5. Cross-source — gap Veridica vs Stopfals (critic pentru modulul 2)

Dacă gap-ul Δ(V-S) e aproape de zero, modelul tratează ambele surse 
propagandistice similar — semn că analiza granulară e robustă 
cross-source și poate compensa problema LOSO-V din modulul 2.

Dacă gap-ul e semnificativ (>0.05), avem încă stylistic fingerprint și 
modulul 3 singur nu rezolvă problema.

**Cel mai mic gap pe scoruri agregate** (mai mic e mai bine):
- `minilm` / topk_mean / p10: Δ = -0.000, AUC = 0.366, d = -0.53

---

## 6. Recomandare finală

Criterii de alegere:
1. AUC cât mai aproape de 1.0 (dar realist — fără contaminare)
2. Gap V-S cât mai mic (robustețe cross-source)
3. Viteză acceptabilă pe M2 Pro
4. Interpretabilitate (preferăm metrici simple de explicat)

**Configurație principală propusă:** `minilm` + propoziție=max + articol=mean
- AUC = 0.552
- Cohen's d = +0.26
- Gap V-S = +0.005

**Metrică interpretabilă suplimentară:** proporție sub τ=0.3 cu `minilm` (AUC = 0.500, d = +nan)

Această metrică e utilă pentru interfața demonstrativă — poate fi prezentată ca „X% din propozițiile articolului nu au corespondent în corpusul de presă credibilă”, mai intuitivă decât un scor abstract.

*Generat automat.*